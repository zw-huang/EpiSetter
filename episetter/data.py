"""Schema-2 pair loading and matched intent prompt construction."""

from __future__ import annotations

import json
import hashlib
import re
import copy
from pathlib import Path


PRIOR_INSTRUCTION = "Instruction: Rely on your stored knowledge."
CONTEXT_INSTRUCTION = "Instruction: Rely on the supplied context."


def render_prompt(question, context, instruction):
    """Serialize already-reviewed factors; never infer intent from quoted context."""
    for name, value in (("question", question), ("instruction", instruction)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing {name}")
    if not isinstance(context, str):
        raise ValueError("context must be a string")
    return f"Task: {instruction}\nContext:\n{context}\nQuestion: {question}\nAnswer:"


def materialize_bundle(bundle):
    """Canonical prompts for v2; v1 remains a legacy, explicit-prompt format."""
    bundle = copy.deepcopy(bundle)
    if bundle.get("schema") == "episetter-2":
        for row in bundle.get("pairs", []):
            for intent in ("prior", "context"):
                prompt = render_prompt(row["question"], row["context"], row[f"{intent}_instruction"])
                key = f"{intent}_prompt"
                if key in row and row[key] != prompt:
                    raise ValueError("Stored prompt differs from structured q/c/I fields")
                row[key] = prompt
        for row in bundle.get("targets", []):
            prompt = render_prompt(row["question"], row["context"], row["instruction"])
            if "prompt" in row and row["prompt"] != prompt:
                raise ValueError("Target prompt differs from structured q/c/I fields")
            row["prompt"] = prompt
    return bundle


def source_targets(bundle, split):
    """Endpoint answers plus non-binary integration/refusal targets, for Eq. 11."""
    result = []
    for row in bundle["pairs"]:
        if row["split"] == split:
            for intent in ("prior", "context"):
                result.append({"id": f"{row['id']}:{intent}", "fact_id": row["fact_id"],
                               "prompt": row[f"{intent}_prompt"], "answer": row[f"{intent}_answer"],
                               "task": intent})
    result.extend(row for row in bundle.get("targets", []) if row["split"] == split)
    return result


def read_pairs(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        if row.get("schema_version") != "2.0" or row.get("pair_status") != "ready_for_behavior":
            raise ValueError("direction learning requires reviewed schema-2 ready pairs")
    return rows[:limit]


def intent_prompt(row: dict, intent: str) -> str:
    if intent not in ("prior", "context"):
        raise ValueError("Unknown intent")
    instruction = PRIOR_INSTRUCTION if intent == "prior" else CONTEXT_INSTRUCTION
    return render_prompt(row["question"], row["contexts"]["conflicting"], instruction)


def training_examples(rows: list[dict]) -> list[dict]:
    examples = []
    for row in rows:
        prior = intent_prompt(row, "prior")
        context = intent_prompt(row, "context")
        examples.extend(
            [
                {"source": prior, "source_answer": row["parametric_answer"], "target": context},
                {"source": context, "source_answer": row["context_answer"], "target": prior},
            ]
        )
    return examples


SPLITS = ("train", "validation", "localization", "evaluation")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_manifest(path):
    """Hash small model/tokenizer files and record shard size/mtime (not weight hashes)."""
    path = Path(path)
    small = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]
    return {"small_file_sha256": {name: digest(path / name) for name in small if (path / name).is_file()},
            "shard_stat_identity": {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                    for p in sorted(path.glob("*.safetensors"))}}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def normalize(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def validate_bundle(bundle, model_path):
    bundle = materialize_bundle(bundle)
    if bundle.get("schema") not in ("episetter-1", "episetter-2"):
        raise ValueError("Expected episetter-1 or episetter-2 bundle")
    if Path(bundle["model_path"]).resolve() != Path(model_path).resolve():
        raise ValueError("Data/model binding differs")
    groups, questions, identifiers = {}, {}, set()
    for row in bundle["pairs"]:
        for field in ("id", "fact_id", "question", "context", "prior_prompt", "context_prompt",
                      "prior_answer", "context_answer", "split"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing pair field: {field}")
        if row["id"] in identifiers or row["split"] not in SPLITS:
            raise ValueError("Duplicate id or invalid split")
        identifiers.add(row["id"])
        if normalize(row["prior_answer"]) == normalize(row["context_answer"]):
            raise ValueError("Answers must differ")
        if row["prior_prompt"] == row["context_prompt"]:
            raise ValueError("Intents need distinct prompts")
        for prompt in (row["prior_prompt"], row["context_prompt"]):
            if row["question"] not in prompt or row["context"] not in prompt:
                raise ValueError("Each intent must retain the same question and context")
        for bank, key in ((groups, row["fact_id"]), (questions, normalize(row["question"]))):
            if key in bank and bank[key] != row["split"]:
                raise ValueError("Fact/question leakage across splits")
            bank[key] = row["split"]
    for split in SPLITS:
        if not any(row["split"] == split for row in bundle["pairs"]):
            raise ValueError(f"Empty pair split: {split}")
    for row in bundle.get("targets", []):
        for field in ("id", "fact_id", "question", "context", "instruction", "prompt", "answer", "task", "split"):
            if not isinstance(row.get(field), str) or (field != "context" and not row[field].strip()):
                raise ValueError(f"Missing target field: {field}")
        if row["id"] in identifiers or row["split"] not in SPLITS:
            raise ValueError("Duplicate target id or invalid split")
        identifiers.add(row["id"])
        for bank, key in ((groups, row["fact_id"]), (questions, normalize(row["question"]))):
            if key in bank and bank[key] != row["split"]:
                raise ValueError("Target fact/question leakage across splits")
            bank[key] = row["split"]
    seen_protect = {}
    for row in bundle["protected"]:
        if row["split"] not in ("train", "validation", "evaluation"):
            raise ValueError("Invalid protected split")
        for field in ("id", "fact_id", "prompt", "answer", "task"):
            if not row.get(field):
                raise ValueError(f"Missing protected field: {field}")
        for key in (row["fact_id"], normalize(row["prompt"])):
            if key in seen_protect and seen_protect[key] != row["split"]:
                raise ValueError("Protected task leakage")
            seen_protect[key] = row["split"]
        if row["fact_id"] in groups and groups[row["fact_id"]] != row["split"]:
            raise ValueError("Source/protection fact leakage")
    for split in ("train", "validation", "evaluation"):
        if not any(row["split"] == split for row in bundle["protected"]):
            raise ValueError(f"Empty protection split: {split}")
    prompts, contexts, entities = {}, {}, {}
    for kind in ("pairs", "targets", "protected"):
        for row in bundle.get(kind, []):
            keys = [row[f"{intent}_prompt"] for intent in ("prior", "context")] if kind == "pairs" else [row["prompt"]]
            checks = [(prompts, normalize(key)) for key in keys]
            if row.get("context", "").strip():
                checks.append((contexts, normalize(row["context"])))
            checks.extend((entities, key) for key in row.get("entity_ids", []))
            for seen, key in checks:
                if key in seen and seen[key] != row["split"]:
                    raise ValueError("Shared prompt/context/entity leakage across splits")
                seen[key] = row["split"]


def prepare_pilot(source, model_path, output):
    """Export ONLY existing development facts; preserve their known quality limitation."""
    if "test" in Path(source).name.lower():
        raise ValueError("Pilot preparation refuses a test file")
    rows = read_pairs(source)
    unique = {}
    for row in rows:
        if Path(row["parameter_knowledge"]["model_name"]).resolve() != Path(model_path).resolve():
            raise ValueError("Source prior was discovered with a different model")
        # Same questions with different source IDs are one group.
        unique.setdefault(normalize(row["question"]), row)
    keys = sorted(unique, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    if len(keys) < 8:
        raise ValueError("At least 8 distinct questions are needed even for this pilot")
    pairs = []
    cycle = ("train", "train", "validation", "localization", "evaluation")
    for i, key in enumerate(keys):
        row = unique[key]
        pairs.append({"id": row["pair_id"], "fact_id": key, "split": cycle[i % 5],
                      "question": row["question"], "context": row["contexts"]["conflicting"],
                      "prior_instruction": PRIOR_INSTRUCTION, "context_instruction": CONTEXT_INSTRUCTION,
                      "prior_prompt": intent_prompt(row, "prior"),
                      "context_prompt": intent_prompt(row, "context"),
                      "prior_answer": row["parametric_answer"], "context_answer": row["context_answer"],
                      "prior_aliases": row.get("parametric_aliases", []),
                      "context_aliases": row.get("context_aliases", []),
                      "prior_provenance": row["parameter_knowledge"], "context_variant": "counterfactual"})
    protected = []
    for split, offset in (("train", 0), ("validation", 10), ("evaluation", 20)):
        for k in range(2):
            n = offset + k + 2
            fixtures = [
                ("arithmetic", f"Question: What is {n} + {n+1}?\nAnswer:", str(2*n+1)),
                ("extraction", f"Document: The parcel code is Z{n}Q.\nQuestion: What is the parcel code?\nAnswer:", f"Z{n}Q"),
                ("instruction", f"Write only the first item of this list: token{n}, token{n+1}.\nAnswer:", f"token{n}"),
                ("multihop", f"Document: Box {n} is inside box {n+1}. Box {n+1} is in room {n+2}.\nQuestion: In which room is box {n}?\nAnswer:", str(n+2)),
            ]
            for task, prompt, answer in fixtures:
                protected.append({"id": f"{task}-{n}", "fact_id": f"{task}-{n}", "split": split,
                                  "task": task, "prompt": prompt, "answer": answer})
    bundle = {"schema": "episetter-2", "quality": "exploratory_only", "model_path": str(Path(model_path).resolve()),
              "source": str(Path(source).resolve()), "source_sha256": digest(source),
              "limitations": ["Legacy development calibration does not satisfy the current 0.5 Wilson lower bound.",
                              "Intent prompts are programmatic, not personD human labels.",
                              "Protection fixtures are synthetic engineering examples, not a capability benchmark.",
                              "No meaningless-character matched control from personC is available.",
                              "All splits are subdivisions of legacy development; no frozen test data was read."],
              "pairs": pairs, "protected": protected}
    validate_bundle(bundle, model_path)
    write_json(output, bundle)
    return {"quality": bundle["quality"], "pairs": len(pairs), "protected": len(protected),
            "splits": {s: sum(r["split"] == s for r in pairs) for s in SPLITS}}
