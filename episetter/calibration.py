"""Frozen-axis, closed-loop calibration of the last prompt token (method Eqs. 10–11)."""
from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .data import normalize, source_targets, write_json
from .experiments import bootstrap_mean, mean, subset


class LayerCalibrator(nn.Module):
    def __init__(self, directions, *, width=32, rho_multiplier=1.0):
        super().__init__()
        if not directions or width < 1 or not math.isfinite(rho_multiplier) or rho_multiplier <= 0:
            raise ValueError("Calibration requires feasible directions and positive width/radius")
        self.layers = tuple(sorted(d["layer"] for d in directions))
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("Duplicate calibration layer")
        self.width = width
        self.predictors = nn.ModuleDict()
        for state in directions:
            if state.get("status") != "validation_feasible" or state.get("mode", "protected") != "protected":
                raise ValueError("Only validation-feasible protected directions may be calibrated")
            layer, u = state["layer"], state["direction"].detach().float().clone()
            rho = float(state["scale"]) * rho_multiplier
            if u.ndim != 1 or not torch.isfinite(u).all() or abs(float(u.norm()) - 1) > 1e-4:
                raise ValueError("Invalid calibration direction")
            if not math.isfinite(rho) or rho <= 0:
                raise ValueError("Invalid calibration radius")
            self.register_buffer(f"axis_{layer}", u)
            self.register_buffer(f"rho_{layer}", torch.tensor(rho, dtype=torch.float32))
            network = nn.Sequential(nn.Linear(u.numel(), width), nn.Tanh(), nn.Linear(width, 1))
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
            self.predictors[str(layer)] = network

    def apply(self, layer, hidden):
        alpha = getattr(self, f"rho_{layer}") * self.predictors[str(layer)](hidden.float()).squeeze(-1).tanh()
        changed = hidden.float() + alpha.unsqueeze(-1) * getattr(self, f"axis_{layer}")
        return changed.to(hidden.dtype), alpha

    def export(self):
        return {"schema": "episetter-calibrator-1", "width": self.width,
                "directions": [{"layer": layer, "direction": getattr(self, f"axis_{layer}").detach().cpu(),
                                "scale": float(getattr(self, f"rho_{layer}")),
                                "mode": "protected", "status": "validation_feasible"} for layer in self.layers],
                "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}

    @classmethod
    def restore(cls, state, device="cpu"):
        if state.get("schema") != "episetter-calibrator-1":
            raise ValueError("Invalid calibration checkpoint")
        module = cls(state["directions"], width=state["width"])
        module.load_state_dict(state["state_dict"], strict=True)
        return module.to(device)


def prefix_kl(base, changed):
    """Teacher || calibrated distribution, uniformly over fixed reference prefixes."""
    p, q = base["log_probs"].detach(), changed["log_probs"]
    return (p.exp() * (p - q)).sum(-1).mean()


def amplitude_cost(result):
    terms = [v.square().mean() for k, v in result["trace"].items() if k.endswith(".alpha")]
    return torch.stack(terms).sum() if terms else result["first_logits"].new_zeros(())


def calibration_metrics(runner, bundle, calibrator, split, generation_tokens=0, limit=None):
    """Never chooses layers or hyperparameters; caller names the held-out split."""
    records = {"source": [], "protected": [], "pairs": []}
    source_rows = source_targets(bundle, split)
    protected_rows = subset(bundle, "protected", split)
    if limit is not None and limit > 0:
        source_rows, protected_rows = source_rows[:limit], protected_rows[:limit]
    for kind, rows in (("source", source_rows), ("protected", protected_rows)):
        for row in tqdm(rows, desc=f"{split} {kind} metrics", unit="row", leave=False, dynamic_ncols=True):
            base = runner.run(row["prompt"], row["answer"])
            changed = runner.run(row["prompt"], row["answer"], calibrator=calibrator)
            rec = {"id": row["id"], "fact_id": row["fact_id"], "task": row["task"],
                   "baseline_nll": float(base["nll"]), "nll": float(changed["nll"]),
                   "baseline_sequence_nll": float(-base["token_logp"].sum()),
                   "sequence_nll": float(-changed["token_logp"].sum()),
                   "nll_increase": float(changed["nll"] - base["nll"]),
                   "reference_prefix_kl": max(0.0, float(prefix_kl(base, changed))),
                   "alphas": {k: float(v) for k, v in changed["trace"].items() if k.endswith(".alpha")}}
            if generation_tokens:
                rec["baseline_generation"] = runner.generate(row["prompt"], generation_tokens)
                rec["generation"] = runner.generate(row["prompt"], generation_tokens, calibrator=calibrator)
                rec["baseline_exact_match"] = normalize(rec["baseline_generation"]) == normalize(row["answer"])
                rec["exact_match"] = normalize(rec["generation"]) == normalize(row["answer"])
            records[kind].append(rec)
    pair_rows = subset(bundle, "pairs", split)
    if limit is not None and limit > 0:
        pair_rows = pair_rows[:limit]
    for row in tqdm(pair_rows, desc=f"{split} pair margins", unit="pair", leave=False, dynamic_ncols=True):
        for intent, sign in (("prior", -1), ("context", 1)):
            prompt = row[f"{intent}_prompt"]
            baseline = runner.margin(prompt, row["prior_answer"], row["context_answer"])
            changed = runner.margin(prompt, row["prior_answer"], row["context_answer"], calibrator=calibrator)
            records["pairs"].append({"fact_id": row["fact_id"], "intent": intent,
                                     "baseline_margin": baseline, "margin": changed,
                                     "intent_margin_gain": sign * (changed - baseline)})
    protected = records["protected"]
    grouped = {}
    for r in records["pairs"]:
        grouped.setdefault(r["fact_id"], []).append(r["intent_margin_gain"])
    records["summary"] = {
        "source_sequence_nll": mean([r["sequence_nll"] for r in records["source"]]),
        "baseline_source_sequence_nll": mean([r["baseline_sequence_nll"] for r in records["source"]]),
        "worst_protected_nll_increase": max(r["nll_increase"] for r in protected),
        "worst_protected_prefix_kl": max(r["reference_prefix_kl"] for r in protected),
        "intent_margin_gain": bootstrap_mean([mean(v) for v in grouped.values()]),
    }
    records.update({"split": split, "quality": bundle["quality"], "layers": list(calibrator.layers),
                    "generation_metric": "strict exact match; integration/explanation requires semantic review"})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def train_calibrator(runner, bundle, direction_bank, output, *, steps=40, lr=1e-3, seed=42,
                     width=32, rho_multiplier=1.0, lambda_kl=1.0, lambda_amplitude=0.01,
                     max_nll_increase=0.05, max_kl=0.05, validation_every=10):
    if steps < 1 or validation_every < 1 or lr <= 0 or min(lambda_kl, lambda_amplitude, max_nll_increase, max_kl) < 0:
        raise ValueError("Invalid calibration training parameters")
    torch.manual_seed(seed)
    calibrator = LayerCalibrator(direction_bank["directions"], width=width, rho_multiplier=rho_multiplier).to(runner.device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    targets, protected = source_targets(bundle, "train"), subset(bundle, "protected", "train")
    if not targets or not protected:
        raise ValueError("Both source and protection training data are required")
    rng = random.Random(seed)
    rng.shuffle(targets)
    rng.shuffle(protected)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=lr)
    source_weight = len(targets) / (len(targets) + len(protected))
    # Identity is a real fallback, never a failed direction presented as a trained module.
    baseline = calibration_metrics(runner, bundle, calibrator, "validation")
    best_score, best_step = baseline["summary"]["source_sequence_nll"], 0
    best_state = copy.deepcopy(calibrator.state_dict())
    history, validations = [], [{"step": 0, **baseline["summary"]}]
    progress = tqdm(range(1, steps + 1), desc="calibration training", unit="step", dynamic_ncols=True)
    for step in progress:
        src, keep = targets[(step-1) % len(targets)], protected[(step-1) % len(protected)]
        optimizer.zero_grad(set_to_none=True)
        source = runner.run(src["prompt"], src["answer"], grad=True, calibrator=calibrator)
        source_loss = -source["token_logp"].sum()
        source_amp = amplitude_cost(source)
        (source_loss + source_weight * lambda_amplitude * source_amp).backward()
        base = runner.run(keep["prompt"], keep["answer"])
        kept = runner.run(keep["prompt"], keep["answer"], grad=True, calibrator=calibrator)
        kl, keep_amp = prefix_kl(base, kept), amplitude_cost(kept)
        (lambda_kl * kl + (1-source_weight) * lambda_amplitude * keep_amp).backward()
        gradients = [p.grad for p in calibrator.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError("Missing/nonfinite calibration gradients")
        torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step, "source_sequence_nll": float(source_loss.detach()),
                        "reference_prefix_kl": float(kl.detach()),
                        "mean_amplitude_cost": float(source_weight*source_amp.detach()+(1-source_weight)*keep_amp.detach())})
        del source, kept, base, source_loss, source_amp, kl, keep_amp
        if step % validation_every == 0 or step == steps:
            metrics = calibration_metrics(runner, bundle, calibrator, "validation")
            summary = metrics["summary"]
            feasible = (summary["worst_protected_nll_increase"] <= max_nll_increase and
                        summary["worst_protected_prefix_kl"] <= max_kl)
            validations.append({"step": step, "feasible": feasible, **summary})
            if feasible and summary["source_sequence_nll"] < best_score:
                best_score, best_step = summary["source_sequence_nll"], step
                best_state = copy.deepcopy(calibrator.state_dict())
            print(f"calibration step={step}/{steps} validation_nll={summary['source_sequence_nll']:.4f} feasible={feasible}", flush=True)
            progress.set_postfix(validation_nll=f"{summary['source_sequence_nll']:.4f}", feasible=feasible, refresh=False)
    calibrator.load_state_dict(best_state)
    if any(p.requires_grad or p.grad is not None for p in runner.model.parameters()):
        raise AssertionError("Backbone is not frozen")
    report = {"status": "validation_selected" if best_step else "identity_fallback_no_validation_improvement",
              "selected_step": best_step, "layers": list(calibrator.layers), "quality": bundle["quality"],
              "selection_split": "validation", "history": history, "validation": validations,
              "parameters": {"steps": steps, "lr": lr, "seed": seed, "width": width,
                             "rho_multiplier": rho_multiplier, "lambda_kl": lambda_kl,
                             "lambda_amplitude": lambda_amplitude, "max_nll_increase": max_nll_increase,
                             "max_kl": max_kl, "validation_every": validation_every}}
    checkpoint = calibrator.export()
    checkpoint.update({"status": report["status"], "selected_step": best_step, "seed": seed,
                       "quality": bundle["quality"], "model_path": bundle["model_path"],
                       "training_parameters": report["parameters"]})
    torch.save(checkpoint, output / "calibrator.pt")
    write_json(output / "training.json", report)
    write_json(output / "validation.json", calibration_metrics(runner, bundle, calibrator, "validation"))
    return report
