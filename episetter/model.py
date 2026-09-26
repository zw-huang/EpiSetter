"""OLMo-3 hooks: interventions use the fixed last PROMPT token, not answer tokens."""
from __future__ import annotations

from contextlib import contextmanager
import torch
from tqdm.auto import tqdm

from .core import set_coordinate


def load_olmo(model_path: str, device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda"):
        torch.empty(1, device=device)  # Fail before loading weights if the driver is unavailable.
    load_progress = tqdm(total=3, desc=f"load model {model_path}", unit="phase", dynamic_ncols=True)
    print(f"[load_olmo] tokenizer: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    load_progress.update(1); load_progress.set_postfix(phase="tokenizer", refresh=False)
    print("[load_olmo] weight shards: starting", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=getattr(torch, dtype),
        attn_implementation="eager", low_cpu_mem_usage=True
    )
    load_progress.update(1); load_progress.set_postfix(phase="weights", refresh=False)
    print(f"[load_olmo] moving model to {device}", flush=True)
    model = model.to(device)
    load_progress.update(1); load_progress.set_postfix(phase="device", refresh=False); load_progress.close()
    runner = Runner(model, tokenizer, device)
    runner.model_path = str(__import__('pathlib').Path(model_path).resolve())
    return runner


class Runner:
    def __init__(self, model, tokenizer, device="cpu"):
        if model.config.model_type not in ("olmo3", "qwen3"):
            raise ValueError("This implementation requires OLMo-3 or Qwen3")
        self.model, self.tokenizer, self.device = model.eval(), tokenizer, device
        for p in model.parameters():
            p.requires_grad_(False)
        self.layers = model.model.layers

    def ids(self, text, special=True):
        if special and self.model.config.model_type == "qwen3":
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            special = False
        return self.tokenizer(text, add_special_tokens=special, return_tensors="pt")["input_ids"].to(self.device)

    @contextmanager
    def hooks(self, position, *, capture=False, layer=None, direction=None,
              target=None, donor=None, patches=None, leaf=False, calibrator=None, offsets=None):
        """Patch post-norm L{k}.attn/mlp, or pre-o_proj L{k}.head{j}.

        Head patching recomputes normalization. Donors use their own last prompt token.
        """
        patches = patches or {}
        offsets = offsets or {}
        if any(not isinstance(i, int) or not 0 <= i < len(self.layers) for i in offsets):
            raise ValueError("Invalid offset layer")
        if calibrator is not None:
            if direction is not None or leaf or offsets:
                raise ValueError("Calibration cannot be combined with coordinate/leaf/offset interventions")
            if any(not 0 <= i < len(self.layers) for i in calibrator.layers):
                raise ValueError("Invalid calibration layer")
        allowed = {f"L{i}.{kind}" for i in range(len(self.layers)) for kind in ("attn", "mlp")}
        allowed.update(f"L{i}.head{j}" for i in range(len(self.layers))
                       for j in range(self.model.config.num_attention_heads))
        if set(patches) - allowed:
            raise ValueError(f"Unknown component patches: {set(patches) - allowed}")
        if (direction is not None or leaf) and (layer is None or not 0 <= layer < len(self.layers)):
            raise ValueError("Invalid actuator layer")
        if direction is not None:
            if direction.shape != (self.model.config.hidden_size,) or not torch.isfinite(direction).all():
                raise ValueError("Invalid direction")
            if abs(float(direction.detach().norm()) - 1) > 1e-4:
                raise ValueError("Direction must have unit norm")
        trace, handles = {}, []

        def output_hook(key, actuator=False, block_index=None):
            def hook(_module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                if key in patches:
                    value = value.clone()
                    value[:, position] = patches[key].to(value.device, value.dtype)
                if actuator and leaf:
                    point = value[:, position].detach().float().requires_grad_(True)
                    trace["leaf"] = point
                    value = value.clone()
                    value[:, position] = point.to(value.dtype)
                if actuator and direction is not None:
                    desired = target
                    if donor is not None:
                        desired = donor.to(direction.device).float() @ direction
                    if desired is None:
                        raise ValueError("Coordinate intervention needs target or donor")
                    replacement = set_coordinate(value[:, position].clone(), direction, desired)
                    value = value.clone()
                    value[:, position] = replacement
                if block_index in offsets:
                    value = value.clone()
                    value[:, position] = value[:, position] + offsets[block_index].to(value.device, value.dtype)
                if calibrator is not None and block_index in calibrator.layers:
                    replacement, alpha = calibrator.apply(block_index, value[:, position])
                    trace[f"L{block_index}.alpha"] = alpha
                    value = value.clone()
                    value[:, position] = replacement
                if capture:
                    trace[key] = value[:, position].detach().float().cpu().clone()
                    if key.endswith((".attn", ".mlp")) and self.model.config.model_type == "olmo3":
                        trace[key + ".pre_norm"] = args[0][:, position].detach().float().cpu().clone()
                return (value,) + output[1:] if isinstance(output, tuple) else value
            return hook

        def input_hook(key, heads=False):
            def hook(_module, args):
                value = args[0]
                if heads:
                    n = self.model.config.num_attention_heads
                    d = value.shape[-1] // n
                    for j in range(n):
                        name = key.replace(".z", f".head{j}")
                        if name in patches:
                            value = value.clone()
                            value[:, position, j*d:(j+1)*d] = patches[name].to(value.device, value.dtype)
                if capture:
                    trace[key] = value[:, position].detach().float().cpu().clone()
                return (value,) + args[1:]
            return hook

        try:
            if capture:
                handles.append(self.model.model.embed_tokens.register_forward_hook(output_hook("embedding")))
            for i, block in enumerate(self.layers):
                components = (("attn", block.post_attention_layernorm),
                              ("mlp", block.post_feedforward_layernorm)) if self.model.config.model_type == "olmo3" else (
                              ("attn", block.self_attn), ("mlp", block.mlp))
                for kind, module in components:
                    key = f"L{i}.{kind}"
                    if capture or key in patches:
                        handles.append(module.register_forward_hook(output_hook(key)))
                if capture or any(key.startswith(f"L{i}.head") for key in patches):
                    handles.append(block.self_attn.o_proj.register_forward_pre_hook(input_hook(f"L{i}.z", True)))
                if capture:
                    handles.append(block.mlp.down_proj.register_forward_pre_hook(input_hook(f"L{i}.product")))
                if capture or i in offsets or (calibrator is not None and i in calibrator.layers) or (i == layer and (direction is not None or leaf)):
                    handles.append(block.register_forward_hook(output_hook(f"L{i}.residual", i == layer, i)))
            yield trace
        finally:
            for handle in handles:
                handle.remove()

    def run(self, prompt, answer=None, *, grad=False, **intervention):
        prompt_ids = self.ids(prompt)
        position = prompt_ids.shape[1] - 1
        if position < 0:
            raise ValueError("Empty prompt")
        answer_ids = self.ids(" " + answer.strip(), False) if answer is not None else None
        if answer_ids is not None and answer_ids.numel() == 0:
            raise ValueError("Empty answer")
        ids = prompt_ids if answer_ids is None else torch.cat([prompt_ids, answer_ids[:, :-1]], 1)
        if ids.shape[1] > min(self.model.config.max_position_embeddings, 2048):
            raise ValueError("Input exceeds experiment limit (2048); no silent truncation")
        with torch.set_grad_enabled(grad), self.hooks(position, **intervention) as trace:
            hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
            logits = self.model.get_output_embeddings()(hidden[:, position:]).float()
            logp = logits.log_softmax(-1)
            token_logp = None if answer_ids is None else logp.gather(-1, answer_ids[..., None]).squeeze(-1)
            nll = None if token_logp is None else -token_logp.mean()
        return {"nll": nll, "token_logp": token_logp, "trace": trace,
                "log_probs": logp, "answer_state": hidden[:, position].detach().float(),
                "first_logits": logits[:, 0], "prompt_tokens": position + 1,
                "answer_tokens": None if answer_ids is None else answer_ids[0].tolist()}

    def generate(self, prompt, max_new_tokens=16, **intervention):
        """Greedy, no-cache reference implementation with a fixed prompt index."""
        ids = self.ids(prompt)
        position, generated = ids.shape[1] - 1, []
        with torch.no_grad(), self.hooks(position, **intervention):
            for _ in range(max_new_tokens):
                if ids.shape[1] >= min(self.model.config.max_position_embeddings, 2048):
                    break
                hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1:]
                token = self.model.get_output_embeddings()(hidden).argmax(-1)
                item = int(token.item())
                generated.append(item)
                if item == self.tokenizer.eos_token_id:
                    break
                ids = torch.cat([ids, token], 1)
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def margin(self, prompt, prior, context, **intervention):
        p = self.run(prompt, prior, **intervention)
        c = self.run(prompt, context, **intervention)
        return float(p["nll"] - c["nll"])

    def native_mapping(self, trace, layer, direction):
        """Residual-addition mapping; head/neuron values condition on observed RMS scale.

        These are algebraic attributions, not causal effects. BF16 rounding is audited.
        """
        if self.model.config.model_type != "olmo3":
            raise NotImplementedError("Post-normalization native component attribution is OLMo3-specific")
        u = direction.detach().float().cpu()
        terms = {"embedding": float(trace["embedding"] @ u)}
        detail = {}
        for i in range(layer + 1):
            block = self.layers[i]
            for kind, norm, projection, key in (
                ("attn", block.post_attention_layernorm, block.self_attn.o_proj, "z"),
                ("mlp", block.post_feedforward_layernorm, block.mlp.down_proj, "product"),
            ):
                name = f"L{i}.{kind}"
                terms[name] = float(trace[name] @ u)
                raw = trace[name + ".pre_norm"]
                eps = getattr(norm, "variance_epsilon", None)
                if eps is None:
                    eps = norm.eps
                scale = torch.rsqrt(raw.pow(2).mean(-1) + eps)
                effective_u = u.to(norm.weight.device) * norm.weight.detach().float()
                weights = (projection.weight.detach().float().T @ effective_u).cpu()
                contributions = trace[f"L{i}.{key}"][0] * weights * scale.item()
                if kind == "attn":
                    contributions = contributions.reshape(self.model.config.num_attention_heads, -1).sum(-1)
                bias = 0.0 if projection.bias is None else float(projection.bias.detach().float() @ effective_u) * scale.item()
                detail[name] = {"contributions": contributions.tolist(), "bias": bias,
                                "reconstruction_error": float(contributions.sum()) + bias - terms[name]}
        coordinate = float(trace[f"L{layer}.residual"] @ u)
        return {"coordinate": coordinate, "terms": terms, "detail": detail,
                "residual_reconstruction_error": sum(terms.values()) - coordinate}
