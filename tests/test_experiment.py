"""CPU integration tests on the actual HF OLMo-3 architecture (random tiny weights)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Olmo3Config, Olmo3ForCausalLM

from episetter.core import orthogonalize, protected_basis_from_gradients, replace_coordinate
from episetter.model import Runner
from episetter.data import validate_bundle
from episetter.experiments import fit, probe, protection_basis


class ByteTokenizer:
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True, return_tensors="pt"):
        return {"input_ids": torch.tensor([([0] if add_special_tokens else []) + [x + 2 for x in text.encode()]])}

    def decode(self, ids, skip_special_tokens=True):
        return bytes([x-2 for x in ids if x >= 2]).decode(errors="replace")


def make_runner(dtype=torch.float32):
    torch.manual_seed(23)
    config = Olmo3Config(hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                        num_attention_heads=4, num_key_value_heads=4, vocab_size=258,
                        max_position_embeddings=512, pad_token_id=0, eos_token_id=1,
                        layer_types=["full_attention"] * 3, sliding_window=64,
                        rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 10000.}})
    config._attn_implementation = "eager"
    return Runner(Olmo3ForCausalLM(config).to(dtype), ByteTokenizer())


def fixture():
    pairs, protected = [], []
    for i, split in enumerate(("train", "validation", "localization", "evaluation")):
        pairs.append({"id": str(i), "fact_id": str(i), "split": split,
                      "question": f"q{i}", "context": f"c{i}",
                      "prior_prompt": f"Prior c{i} q{i}:", "context_prompt": f"Context c{i} q{i}:",
                      "prior_answer": "a", "context_answer": "bc"})
        if split != "localization":
            protected.append({"id": str(i), "fact_id": f"p{i}", "split": split,
                              "task": "test", "prompt": f"calculate {i}:", "answer": "de"})
    return {"schema": "episetter-1", "quality": "synthetic_test_only", "model_path": "/tmp/tiny-olmo",
            "pairs": pairs, "protected": protected}


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_projection_constraints(self):
        g = torch.randn(5, 32)
        q = protected_basis_from_gradients(g, None, torch)
        u = orthogonalize(torch.randn(32), q)
        self.assertLess(float((g @ u).abs().max()), 3e-6)
        h, source = torch.randn(2, 32), torch.randn(2, 32)
        changed = replace_coordinate(h, u, source)
        torch.testing.assert_close(changed @ u, source @ u)
        torch.testing.assert_close(changed - (changed @ u)[:, None]*u, h - (h @ u)[:, None]*u)
        torch.testing.assert_close(replace_coordinate(h, u, h), h)
        with self.assertRaises(ValueError):
            orthogonalize(torch.ones(3), torch.eye(3))

    def test_full_answer_scoring_matches_reference(self):
        runner = make_runner()
        prompt, answer = "hi:", "abc"
        result = runner.run(prompt, answer)
        p, a = runner.ids(prompt), runner.ids(" " + answer, False)
        ids = torch.cat([p, a], 1)
        with torch.no_grad():
            logits = runner.model(input_ids=ids, use_cache=False).logits
            lp = logits[:, p.shape[1]-1:-1].float().log_softmax(-1).gather(-1, a[..., None]).squeeze(-1)
        torch.testing.assert_close(result["token_logp"], lp)

    def test_coordinate_position_gradient_and_hook_cleanup(self):
        runner = make_runner()
        prompt = "Question:"
        baseline = runner.run(prompt, capture=True)
        source = runner.run("Other:", capture=True)["trace"]["L1.residual"]
        u = torch.nn.Parameter(torch.randn(32))
        unit = u / u.norm()
        changed = runner.run(prompt, "a long answer", capture=True, layer=1,
                             direction=unit, donor=source, grad=True)
        torch.testing.assert_close(changed["trace"]["L1.residual"] @ unit.detach(), source @ unit.detach())
        changed["nll"].backward()
        self.assertGreater(float(u.grad.norm()), 0)
        native = baseline["trace"]["L1.residual"]
        restored = runner.run(prompt, layer=1, direction=unit.detach(), donor=native)
        torch.testing.assert_close(restored["first_logits"], baseline["first_logits"])
        # An exception within a registered hook must not leave hooks behind.
        with self.assertRaises(ValueError):
            runner.run(prompt, layer=1, direction=unit.detach())
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in runner.model.modules()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in runner.model.parameters()))

    def test_native_mapping_and_head_self_patch(self):
        for dtype, tolerance in ((torch.float32, 2e-5), (torch.bfloat16, .2)):
            runner = make_runner(dtype)
            base = runner.run("small:", capture=True)
            u = orthogonalize(torch.randn(32), None)
            mapping = runner.native_mapping(base["trace"], 2, u)
            self.assertLess(abs(mapping["residual_reconstruction_error"]), tolerance)
            self.assertTrue(all(abs(v["reconstruction_error"]) < tolerance for v in mapping["detail"].values()))
            head = base["trace"]["L0.z"].reshape(1, 4, -1)[:, 2]
            same = runner.run("small:", patches={"L0.head2": head})
            torch.testing.assert_close(base["first_logits"], same["first_logits"])

    def test_bfloat16_direction_backprop(self):
        runner = make_runner(torch.bfloat16)
        source = runner.run("source", capture=True)["trace"]["L1.residual"]
        v = torch.nn.Parameter(torch.randn(32))
        result = runner.run("target", "multi token", grad=True, layer=1,
                            direction=v/v.norm(), donor=source)
        result["nll"].backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        self.assertGreater(float(v.grad.norm()), 0)

    def test_protected_gradients_against_finite_difference(self):
        runner = make_runner()
        result = runner.run("gradient:", "z", grad=True, layer=1, leaf=True)
        g, = torch.autograd.grad(result["nll"], result["trace"]["leaf"])
        u = orthogonalize(torch.randn(32), None)
        s0 = result["trace"]["leaf"].detach() @ u
        plus = runner.run("gradient:", "z", layer=1, direction=u, target=s0 + .002)["nll"]
        minus = runner.run("gradient:", "z", layer=1, direction=u, target=s0 - .002)["nll"]
        self.assertAlmostEqual(float(g @ u), float((plus-minus)/.004), delta=.002)
        basis, audit = protection_basis(runner, fixture()["protected"][:1], 1)
        self.assertGreater(audit["retained_rank"], 0)
        self.assertLess(float((basis @ orthogonalize(torch.randn(32), basis)).norm()), 1e-5)

    def test_data_leakage_rejection(self):
        data = fixture()
        validate_bundle(data, data["model_path"])
        bad = copy.deepcopy(data)
        bad["pairs"][1]["fact_id"] = bad["pairs"][0]["fact_id"]
        with self.assertRaises(ValueError):
            validate_bundle(bad, bad["model_path"])

    def test_end_to_end_tiny_model(self):
        runner, bundle = make_runner(), fixture()
        with tempfile.TemporaryDirectory(prefix="episetter-test-") as directory:
            root = Path(directory)
            report = fit(runner, bundle, root / "fit", layers=[1], steps=2)
            state = torch.load(root / "fit/direction.pt", weights_only=True)
            self.assertEqual(report["quality"], "synthetic_test_only")
            result = probe(runner, bundle, state, root / "probe", top_k=1,
                           heads_per_attention=1, generation_tokens=1)
            self.assertIn("signed_removed_effect", result)
            rows = json.loads((root / "probe/causal_rows.json").read_text())
            for row in rows:
                c = row["conditions"]
                self.assertAlmostEqual(c["self_patch"]["margin"], c["baseline"]["margin"], places=5)
                self.assertAlmostEqual(c["patch_restore_coordinate"]["coordinate"], c["baseline"]["coordinate"], places=5)
                self.assertAlmostEqual(c["coordinate_only"]["coordinate"], c["circuit_patch"]["coordinate"], places=5)


if __name__ == "__main__":
    unittest.main()
