# Release status

The repository snapshot corresponds to the OLMo3-7B Base + EpiSetter L11--15
calibrator, seed 42, selected at calibration step 1800.

| Component | Status |
|---|---|
| Seed-42 training and validation | Complete; `validation_selected`, step 1800 |
| ConFiQA-QA paired evaluation | Complete; 417 held-out examples |
| MRQA development evaluation | Complete for the six matched datasets |
| Independent capability evaluation | Complete for ARC-Challenge, IFEval, GSM8K, and a SQuAD-dev slice |
| External comparisons | Local protocol-matched CK-PLUG, SHIFT, and Knowledgeable-R1 adaptations |
| Human semantic audit of model-screened data | Incomplete |

Base predictions are reused by frozen example ID for paired comparisons. The
saved summaries contain exploratory percentile intervals; they do not support
claims about cross-seed variance or universal capability preservation.

Run `python scripts/verify_release.py` to validate the checkpoint, input hashes,
saved predictions, and published scores.
