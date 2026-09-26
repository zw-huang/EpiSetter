# EpiSetter: OLMo3-7B L11--15 Release

This repository contains the released EpiSetter implementation, the seed-42
L11--15 calibrator, frozen evaluation inputs, per-example predictions, and
verification code. The paper manuscript is intentionally kept outside this
code release.

## Contents

- `episetter/`: direction learning, bounded layer calibration, inference, and scoring.
- `checkpoints/l11_15/seed42/`: the released calibrator and validation metadata.
- `data/`: frozen training/evaluation inputs and exclusion metadata.
- `results/`: matched MRQA and capability predictions and summaries.
- `provenance/`: input, checkpoint, and source hashes.
- `REPRODUCIBILITY.md`: verification and inference instructions.

The OLMo3-7B backbone is not redistributed. The large direction-learning cache
is also excluded from GitHub; it is not required for inference with the frozen
calibrator.

## Install and verify

The reference environment used Python 3.12.14, PyTorch 2.13.0+cu130, and
Transformers 5.16.1. Install the package and run the offline checks:

```bash
pip install -e .
python scripts/verify_release.py
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

The checks validate release hashes, frozen data, saved predictions, matched
question IDs, checkpoint restoration, and bounded five-layer interventions.

## Run inference

Provide a local OLMo3-7B Base checkpoint whose fingerprints match
`checkpoints/l11_15/seed42/run_manifest.json`:

```bash
export EPISETTER_MODEL=/path/to/olmo
python scripts/generate.py --model "$EPISETTER_MODEL" \
  --question 'How many sides does a triangle have?' \
  --context 'A triangle has seven sides.' --intent context
```

For MRQA evaluation, use the frozen calibrator and the matching input bundle:

```bash
python scripts/evaluate_mrqa.py --model "$EPISETTER_MODEL" \
  --method ours \
  --checkpoint checkpoints/l11_15/seed42/calibration/calibrator.pt \
  --bundle data/mrqa_exclusion_bundle.json \
  --output runs/hotpot_seed42 --device cuda:0 --limit 0 \
  --max-new-tokens 32 --batch-size 2 --datasets HotpotQA.jsonl
```

The calibrator has 655,685 trainable setter parameters and intervenes at blocks
11--15. Inference uses the frozen backbone, BF16, greedy decoding, and the
generation budgets recorded in the manifests.

