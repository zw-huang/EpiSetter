"""Direction interchange learning, protection checks, and causal component probes."""
from __future__ import annotations

import random
import json
import gc
from pathlib import Path
import torch
from tqdm.auto import tqdm

from .core import orthogonalize, protected_basis_from_gradients
from .data import normalize, write_json


def subset(bundle, kind, split):
    return [row for row in bundle[kind] if row["split"] == split]


def paired_runs(rows):
    for row in rows:
        for recipient, donor, sign in (("prior", "context", 1), ("context", "prior", -1)):
            yield row, recipient, donor, sign


def mean(values):
    return sum(values) / len(values) if values else None


def bootstrap_mean(values, seed=0):
    """Input values must already be aggregated by fact, never duplicated by intent."""
    if len(values) < 2:
        return {"mean": mean(values), "ci95": None, "n_facts": len(values)}
    rng = random.Random(seed)
    resamples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(1000))
    return {"mean": mean(values), "ci95": [resamples[24], resamples[974]], "n_facts": len(values)}


def protection_basis(runner, rows, layer, rank=None):
    gradients = []
    # Long answers can keep tens of thousands of token-level VJP graphs alive
    # long enough to exhaust a 32 GiB card.  The low-memory resume route uses
    # one exact gradient of the complete protected-answer NLL per row.  The
    # historical token-level route remains the default unless explicitly set.
    row_sum = __import__('os').environ.get('R5_GRADIENT_MODE') == 'per_row_sum'
    progress = tqdm(rows, desc=f"layer {layer} protection gradients", unit="row", dynamic_ncols=True)
    for row_index, row in enumerate(progress):
        result = runner.run(row["prompt"], row["answer"], grad=True, layer=layer, leaf=True)
        token_losses = -result["token_logp"].flatten()
        if row_sum:
            gradient, = torch.autograd.grad(token_losses.sum(), result["trace"]["leaf"],
                                            retain_graph=False)
            gradients.append(gradient[0].detach().cpu())
            del gradient
        else:
            for index, loss in enumerate(token_losses):
                gradient, = torch.autograd.grad(loss, result["trace"]["leaf"],
                                               retain_graph=index + 1 < len(token_losses))
                gradients.append(gradient[0].detach().cpu())
                del gradient
        progress.set_postfix(gradient_rows=len(gradients), refresh=False)
        # Release the complete transformer graph before moving to the next
        # protected row. Periodic cache trimming prevents long fits from
        # exhausting the allocator through fragmentation.
        # `gradient` is only bound when at least one answer token exists.
        # The row-level result and token losses are sufficient for cleanup;
        # avoid an unbound-local failure on empty-token edge cases.
        del result, token_losses
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gradients = torch.stack(gradients)
    basis = protected_basis_from_gradients(gradients, rank, torch)
    return basis.to(runner.device), {"gradient_rows": len(gradients), "retained_rank": len(basis),
                                    "definition": "per-answer-token NLL gradient at last prompt residual"}


def learn_direction(runner, rows, layer, basis, steps, lr, seed):
    cache, differences = {}, []
    for row in rows:
        for intent in ("prior", "context"):
            prompt = row[f"{intent}_prompt"]
            # Residual donors are reused across optimization steps; keep them
            # on CPU so a broad layer sweep does not pin one GPU copy per row.
            cache[prompt] = runner.run(prompt, capture=True)["trace"][f"L{layer}.residual"].cpu()
        differences.append(cache[row["context_prompt"]] - cache[row["prior_prompt"]])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raw = torch.stack(differences).mean((0, 1)).to(runner.device)
    if float(raw.norm()) < 1e-8:
        raise ValueError("No paired intent difference at the candidate interface")
    vector = torch.nn.Parameter(orthogonalize(raw, basis).clone())
    optimizer = torch.optim.Adam([vector], lr=lr)
    examples = list(paired_runs(rows))
    rng, losses = random.Random(seed), []
    progress = tqdm(range(steps), desc=f"layer {layer} direction learning", unit="step", dynamic_ncols=True)
    for step in progress:
        if step % len(examples) == 0:
            rng.shuffle(examples)
        row, recipient, donor, _ = examples[step % len(examples)]
        optimizer.zero_grad(set_to_none=True)
        u = orthogonalize(vector, basis)
        result = runner.run(row[f"{recipient}_prompt"], row[f"{donor}_answer"], grad=True,
                            layer=layer, direction=u, donor=cache[row[f"{donor}_prompt"]])
        loss = -result["token_logp"].sum()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        if vector.grad is None or not torch.isfinite(vector.grad).all():
            raise ValueError("Missing/non-finite direction gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % 10 == 0:
            print(f"layer={layer} protection_rank={len(basis)} step={step+1}/{steps} nll={losses[-1]:.4f}", flush=True)
            progress.set_postfix(nll=f"{losses[-1]:.4f}", refresh=False)
        del result, loss, u
        if torch.cuda.is_available() and (step + 1) % 4 == 0:
            gc.collect(); torch.cuda.empty_cache()
    u = orthogonalize(vector.detach(), basis)
    if float(raw @ u) < 0:
        u = -u
    scale = torch.stack([(d.to(u.device) @ u).abs().reshape(()) for d in differences]).median().item()
    if scale < 1e-6:
        raise ValueError("Learned direction has a degenerate training coordinate range")
    if any(p.grad is not None or p.requires_grad for p in runner.model.parameters()):
        raise AssertionError("Backbone was not frozen")
    return u, scale, losses


def direction_effects(runner, rows, layer, u, generation_tokens=0, shift_direction=None):
    records = []
    for row, recipient, donor, sign in tqdm(paired_runs(rows), desc=f"layer {layer} direction test", unit="pair", leave=False, dynamic_ncols=True):
        prompt = row[f"{recipient}_prompt"]
        source = runner.run(row[f"{donor}_prompt"], capture=True)["trace"][f"L{layer}.residual"].to(runner.device)
        native = runner.run(prompt, capture=True)["trace"][f"L{layer}.residual"].to(runner.device)
        baseline = runner.margin(prompt, row["prior_answer"], row["context_answer"])
        intervention = {"layer": layer, "direction": u, "donor": source}
        shift = (source-native) @ u
        if shift_direction is not None:
            shift = (source-native) @ shift_direction
            intervention = {"layer": layer, "direction": u, "target": native @ u + shift}
        patched = runner.margin(prompt, row["prior_answer"], row["context_answer"], **intervention)
        record = {"fact_id": row["fact_id"], "recipient": recipient, "donor": donor,
                  "baseline_margin": baseline, "swap_margin": patched,
                  "native_coordinate": float(native @ u), "donor_coordinate": float(source @ u),
                  "coordinate_shift": float(shift),
                  "shift_policy": "matched_reference_shift" if shift_direction is not None else "own_donor_coordinate",
                  "signed_margin_gain": sign * (patched - baseline)}
        if generation_tokens:
            record["baseline_generation"] = runner.generate(prompt, generation_tokens)
            record["swap_generation"] = runner.generate(prompt, generation_tokens, **intervention)
            for condition in ("baseline", "swap"):
                # Conservative first-line exact match; preserve raw text for semantic review.
                text = normalize(record[f"{condition}_generation"].strip().split("\n")[0])
                matches = [intent for intent in ("prior", "context") if text in {
                    normalize(row[f"{intent}_answer"]), *[normalize(a) for a in row.get(f"{intent}_aliases", [])]}]
                record[f"{condition}_source"] = matches[0] if len(matches) == 1 else "ambiguous" if matches else "other"
        records.append(record)
    return records


def protection_effects(runner, rows, layer, u, scale, generation_tokens=0):
    records = []
    for row in tqdm(rows, desc=f"layer {layer} protection test", unit="row", leave=False, dynamic_ncols=True):
        base = runner.run(row["prompt"], row["answer"], capture=True)
        s0 = base["trace"][f"L{layer}.residual"].to(runner.device) @ u
        base_lp = base["first_logits"].log_softmax(-1)
        for sign in (-1, 1):
            intervention = {"layer": layer, "direction": u, "target": s0 + sign * scale}
            changed = runner.run(row["prompt"], row["answer"], **intervention)
            kl = (base_lp.exp() * (base_lp - changed["first_logits"].log_softmax(-1))).sum()
            rec = {"id": row["id"], "task": row["task"], "sign": sign,
                   "shift": sign * scale, "baseline_nll": float(base["nll"]),
                   "nll_increase": float(changed["nll"] - base["nll"]),
                   "first_token_kl": max(0., float(kl))}
            if generation_tokens:
                rec["baseline_generation"] = runner.generate(row["prompt"], generation_tokens)
                rec["generation"] = runner.generate(row["prompt"], generation_tokens, **intervention)
                rec["baseline_exact_match"] = normalize(rec["baseline_generation"]) == normalize(row["answer"])
                rec["exact_match"] = normalize(rec["generation"]) == normalize(row["answer"])
            records.append(rec)
    return records


def fit(runner, bundle, output, *, layers, steps=40, lr=0.01, seed=42,
        max_nll_increase=0.05, max_kl=0.05, source_limit=None, protection_limit=None,
        skip_protection=False, protection_basis_cache=None):
    """Select on validation only; no localization/evaluation data consumed here."""
    torch.manual_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates, states = [], {}
    completed_layers = set()
    # Layer checkpoints are independently reusable after interruption.
    for layer in tqdm(layers, desc="scan layer checkpoints", unit="layer", dynamic_ncols=True):
        files = [output / f"L{layer}-{mode}.json" for mode in ("unconstrained", "protected")]
        states_files = [output / f"L{layer}-{mode}.pt" for mode in ("unconstrained", "protected")]
        if all(p.is_file() for p in files + states_files):
            candidates.extend(json.loads(p.read_text()) for p in files)
            for p in states_files:
                state = torch.load(p, map_location="cpu", weights_only=True)
                states[f"L{layer}-{state['mode']}"] = state
            completed_layers.add(layer)
    write_json(output / "fit_progress.json", {"completed_layers": sorted(completed_layers),
                                                "requested_layers": list(layers)})
    training = subset(bundle, "pairs", "train")
    protection = subset(bundle, "protected", "train")
    # Deterministic caps reduce repeated direction fitting cost for broad layer
    # sweeps. Validation and test remain untouched; the default preserves the
    # original full-data behavior.
    if source_limit is not None and source_limit > 0:
        training = training[:source_limit]
    if protection_limit is not None and protection_limit > 0:
        protection = protection[:protection_limit]
    for layer in tqdm(layers, desc="fit layers", unit="layer", dynamic_ncols=True):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if layer in completed_layers:
            print(f"Reusing completed layer {layer}", flush=True)
            continue
        if skip_protection:
            basis = torch.empty((0, runner.model.config.hidden_size), device=runner.device)
            basis_audit = {"gradient_rows": 0, "retained_rank": 0,
                           "definition": "skipped in fast directional probe"}
            print(f"Skipping protection gradients at layer {layer} (fast directional probe)", flush=True)
        else:
            cache_path = None if protection_basis_cache is None else Path(protection_basis_cache) / f"L{layer}-basis.pt"
            if cache_path is not None and cache_path.is_file():
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                basis = cached["basis"].to(runner.device)
                basis_audit = cached["audit"]
                print(f"Reusing cached protection basis at layer {layer}", flush=True)
            else:
                print(f"Computing protection gradients at layer {layer}", flush=True)
                basis, basis_audit = protection_basis(runner, protection, layer)
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"basis": basis.detach().cpu(), "audit": basis_audit}, cache_path)
        for mode in ("unconstrained", "protected"):
            used_basis = basis if mode == "protected" else basis[:0]
            key = f"L{layer}-{mode}"
            try:
                u, scale, losses = learn_direction(runner, training, layer, used_basis, steps, lr, seed)
            except ValueError as exc:
                if not any(message in str(exc) for message in (
                    "vanished in the protected nullspace", "No paired intent difference",
                    "degenerate training coordinate range")):
                    raise
                rec = {"key": key, "mode": mode, "layer": layer, "feasible": False,
                       "failure": str(exc), "functional_basis_audit": basis_audit}
                candidates.append(rec)
                write_json(output / f"{key}.json", rec)
                continue
            selection = direction_effects(runner, subset(bundle, "pairs", "validation"), layer, u)
            protection_validation = [] if skip_protection else protection_effects(
                runner, subset(bundle, "protected", "validation"), layer, u, scale)
            gain = mean([row["signed_margin_gain"] for row in selection])
            worst_nll = 0.0 if skip_protection else max(row["nll_increase"] for row in protection_validation)
            worst_kl = 0.0 if skip_protection else max(row["first_token_kl"] for row in protection_validation)
            directional_gains = {intent: mean([r["signed_margin_gain"] for r in selection
                                               if r["recipient"] == intent]) for intent in ("prior", "context")}
            rec = {"key": key, "mode": mode, "layer": layer, "scale": scale,
                   "validation_signed_margin_gain": gain, "worst_protected_nll_increase": worst_nll,
                   "worst_protected_first_token_kl": worst_kl,
                   "functional_basis_audit": basis_audit,
                   "basis_dot_u_max": float((basis @ u).abs().max()) if len(basis) else 0.,
                   "directional_gains": directional_gains,
                   # Direction screening is source-side only. Protection metrics
                   # remain recorded for Pareto selection during calibration.
                   "feasible": all(g > 0 for g in directional_gains.values()),
                   "training_losses": losses, "selection_rows": selection, "protection_rows": protection}
            candidates.append(rec)
            states[key] = {"direction": u.cpu(), "basis": basis.cpu(), "scale": scale, "layer": layer,
                           "mode": mode, "seed": seed, "model_path": bundle["model_path"],
                           "quality": bundle["quality"], "status": "validation_feasible" if rec["feasible"] else "diagnostic_only"}
            write_json(output / f"{key}.json", rec)
            torch.save(states[key], output / f"{key}.pt")
        write_json(output / "fit_progress.json", {"completed_layers": sorted(completed_layers | {layer}),
                                                    "requested_layers": list(layers)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    feasible = [c for c in candidates if c["mode"] == "protected" and c["feasible"]]
    pool = feasible or [c for c in candidates if c["mode"] == "protected" and c["key"] in states]
    chosen = max(pool, key=lambda c: c["validation_signed_margin_gain"]) if pool else None
    state = dict(states[chosen["key"]]) if chosen else {}
    state.update({"key": chosen["key"] if chosen else None, "quality": bundle["quality"],
                  "model_path": bundle["model_path"], "seed": seed,
                  "status": "validation_feasible" if feasible else "no_feasible_direction_diagnostic_only" if chosen else "no_valid_direction"})
    torch.save(state, output / "direction.pt")
    torch.save({"schema": "episetter-directions-2", "directions": [states[c["key"]] for c in feasible],
                "status": "validation_feasible" if feasible else "no_feasible_direction",
                "model_path": bundle["model_path"], "quality": bundle["quality"], "seed": seed}, output / "directions.pt")
    report = {"status": state["status"], "selected": state["key"], "quality": bundle["quality"],
              "feasible_layers": [c["layer"] for c in feasible],
              "thresholds": {"max_nll_increase": max_nll_increase, "max_first_token_kl": max_kl},
              "selection_policy": "source-positive direction; Pareto protection selection deferred to calibration",
              "selection_split": "validation", "candidate_summaries": [
                  {k: v for k, v in c.items() if k not in ("training_losses", "selection_rows", "protection_rows")}
                  for c in candidates]}
    write_json(output / "selection.json", report)
    return report


def patch_values(names, trace, n_heads):
    result = {}
    for name in names:
        if ".head" in name:
            prefix, index = name.split(".head")
            result[name] = trace[prefix + ".z"].reshape(1, n_heads, -1)[:, int(index)]
        else:
            result[name] = trace[name]
    return result


def localize(runner, rows, layer, names):
    results = {name: [] for name in names}
    for row, recipient, donor, sign in paired_runs(rows):
        source = runner.run(row[f"{donor}_prompt"], capture=True)["trace"]
        prompt = row[f"{recipient}_prompt"]
        baseline = runner.margin(prompt, row["prior_answer"], row["context_answer"])
        for name in names:
            patch = patch_values([name], source, runner.model.config.num_attention_heads)
            margin = runner.margin(prompt, row["prior_answer"], row["context_answer"], patches=patch)
            results[name].append({"fact_id": row["fact_id"], "recipient": recipient,
                                  "signed_margin_gain": sign * (margin - baseline)})
    return [{"name": name, "score": mean([r["signed_margin_gain"] for r in records]), "rows": records}
            for name, records in results.items()]


def probe(runner, bundle, state, output, *, top_k=2, heads_per_attention=2, generation_tokens=16):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    layer, u = state["layer"], state["direction"].to(runner.device)
    names = [f"L{i}.{kind}" for i in range(layer + 1) for kind in ("attn", "mlp")]
    if top_k > len(names) // 2 or heads_per_attention > runner.model.config.num_attention_heads // 2:
        raise ValueError("Selection leaves insufficient disjoint matched control components")
    # Independent circuit selection uses only localization behavior, not evaluation or u.
    ranking = localize(runner, subset(bundle, "pairs", "localization"), layer, names)
    ranked = sorted(ranking, key=lambda row: row["score"], reverse=True)
    selected = [row["name"] for row in ranked[:top_k]]
    head_ranking = []
    if heads_per_attention:
        for name in list(selected):
            if name.endswith(".attn"):
                head_names = [name.replace(".attn", f".head{h}") for h in range(runner.model.config.num_attention_heads)]
                ranking_h = localize(runner, subset(bundle, "pairs", "localization"), layer, head_names)
                head_ranking.extend(ranking_h)
                selected.remove(name)
                selected.extend(row["name"] for row in sorted(ranking_h, key=lambda r: r["score"], reverse=True)[:heads_per_attention])
    rng = random.Random(state["seed"])
    controls = []
    for name in selected:
        kind = "head" if ".head" in name else name.split(".")[1]
        pool = [f"L{i}.head{h}" for i in range(layer + 1) for h in range(runner.model.config.num_attention_heads)] if kind == "head" else [f"L{i}.{kind}" for i in range(layer + 1)]
        pool = [n for n in pool if n not in selected and n not in controls]
        if not pool:
            raise ValueError("Insufficient components for a disjoint matched random control")
        controls.append(rng.choice(pool))
    write_json(output / "localization.json", {"split": "localization", "ranking": ranking,
               "head_ranking": head_ranking, "frozen_components": selected, "random_components": controls,
               "positive_localization_effect": all(r["score"] > 0 for r in ranked[:top_k])})
    records, mappings = [], []
    for row, recipient, donor, sign in paired_runs(subset(bundle, "pairs", "evaluation")):
        prompt = row[f"{recipient}_prompt"]
        native = runner.run(prompt, capture=True)["trace"]
        source = runner.run(row[f"{donor}_prompt"], capture=True)["trace"]
        s0 = native[f"L{layer}.residual"].to(runner.device) @ u
        patches = patch_values(selected, source, runner.model.config.num_attention_heads)
        patched_trace = runner.run(prompt, capture=True, patches=patches)["trace"]
        sp = patched_trace[f"L{layer}.residual"].to(runner.device) @ u
        settings = {
            "baseline": {}, "circuit_patch": {"patches": patches},
            "patch_restore_coordinate": {"patches": patches, "layer": layer, "direction": u, "target": s0},
            "coordinate_only": {"layer": layer, "direction": u, "target": sp},
            "self_patch": {"patches": patch_values(selected, native, runner.model.config.num_attention_heads)},
            "random_patch": {"patches": patch_values(controls, source, runner.model.config.num_attention_heads)},
        }
        conditions = {}
        for name, intervention in settings.items():
            trace = runner.run(prompt, capture=True, **intervention)["trace"]
            margin = runner.margin(prompt, row["prior_answer"], row["context_answer"], **intervention)
            conditions[name] = {"coordinate": float(trace[f"L{layer}.residual"].to(runner.device) @ u), "margin": margin}
            if generation_tokens:
                conditions[name]["generation"] = runner.generate(prompt, generation_tokens, **intervention)
        b, p, r, c = [conditions[k]["margin"] for k in ("baseline", "circuit_patch", "patch_restore_coordinate", "coordinate_only")]
        records.append({"fact_id": row["fact_id"], "recipient": recipient, "conditions": conditions,
                        "signed_total_effect": sign * (p-b), "signed_restored_effect": sign * (r-b),
                        "signed_removed_effect": sign * (p-r), "signed_coordinate_only_effect": sign * (c-b),
                        "coordinate_change": float(sp-s0)})
        mappings.append({"fact_id": row["fact_id"], "intent": recipient,
                         "mapping": runner.native_mapping(native, layer, u)})
        print(f"probe {row['id']} {recipient}: total={sign*(p-b):.4f}, removed={sign*(p-r):.4f}", flush=True)
    write_json(output / "causal_rows.json", records)
    write_json(output / "native_mapping.json", mappings)
    summary = {}
    for key in ("signed_total_effect", "signed_restored_effect", "signed_removed_effect", "signed_coordinate_only_effect"):
        grouped = {}
        for rec in records:
            grouped.setdefault(rec["fact_id"], []).append(rec[key])
        summary[key] = bootstrap_mean([mean(v) for v in grouped.values()], state["seed"])
    summary.update({"quality": bundle["quality"], "direction_status": state["status"],
                    "components": selected, "scope": "candidate component set; not a recovered complete circuit",
                    "ratio_policy": "Absolute effects reported; no unstable division by a near-zero total effect."})
    x = torch.tensor([r["conditions"]["baseline"]["coordinate"] for r in records])
    y = torch.tensor([r["conditions"]["baseline"]["margin"] for r in records])
    xc, yc = x-x.mean(), y-y.mean()
    denom = xc.norm() * yc.norm()
    summary["native_coordinate_margin_correlation_descriptive"] = float(xc @ yc / denom) if float(denom) > 1e-8 else None
    summary["correlation_warning"] = "Descriptive association across both intents; shared intent can confound it. Causal rows are separate."
    write_json(output / "summary.json", summary)
    return summary
