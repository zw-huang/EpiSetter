# Reproducibility

This repository contains the released OLMo3-7B Base + EpiSetter L11--15
checkpoint (seed 42), code, frozen inputs, per-example predictions, summaries,
and verification scripts. The paper source is maintained separately and is
not part of this release snapshot.

## Fast verification

From the repository root:

```bash
python scripts/verify_release.py
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

These checks validate repository hashes, checkpoint restoration, bounded
intervention geometry, frozen MRQA predictions, matched question IDs, and the
published metric summaries. They do not require GPU inference.

## Inference reproduction

Install the package in the documented environment, then provide a local copy
of the OLMo3-7B Base weights:

```bash
pip install -e .
export EPISETTER_MODEL=/path/to/olmo
python scripts/generate.py --model "$EPISETTER_MODEL" \
  --question 'How many sides does a triangle have?' \
  --context 'A triangle has seven sides.' --intent context
```

The model path must match the fingerprints in
`checkpoints/l11_15/seed42/run_manifest.json`. The calibrator is frozen and
contains 655,685 trainable setter parameters; the backbone is not redistributed
by this repository. The manifest records Python, PyTorch, Transformers, model
file fingerprints, layer indices, seed, and data hashes.

## Published evaluation evidence

`results/mrqa/` contains matched MRQA predictions and summaries. The
`results/capability/` directory contains the final matched ARC-Challenge,
IFEval, and GSM8K predictions and summaries for Base and L11--15. Each result
directory includes the input manifest and the exact scoring protocol. All
reported intervals are exploratory percentile intervals; bootstrap units and
seeds are recorded in the corresponding summary or manifest.

## Environment and data

Before a public release, export the exact environment and record GPU/driver
information:

```bash
conda env export --no-builds > environment.yml
python -m pip freeze > requirements-lock.txt
python --version
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.version.cuda)'
```

Publish or link the licensed OLMo weights and every input file listed by the
manifests. Do not silently regenerate or reshuffle frozen evaluation IDs.
Model-screened data and the remaining human semantic audit limitation are
documented in the release notes.

Full retraining is optional and may vary slightly across CUDA and library
versions. Reproduction of the paper numbers is supported directly from the
frozen checkpoint and saved predictions.
