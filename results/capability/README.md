# Matched capability evaluation

These are frozen Base and EpiSetter L11--15 seed-42 predictions on the same
example IDs. The evaluation uses BF16, greedy decoding, a 2,048-token input
budget, and stage-specific generation limits recorded in `manifest.json`.

| Dataset | Examples | Reported metric |
|---|---:|---|
| ARC-Challenge | 1,170 | accuracy and normalized accuracy |
| IFEval | 541 | strict and loose instruction following |
| GSM8K | 1,319 | exact match |

Each directory contains `predictions.jsonl` and `summary.json`. The prediction
records retain example IDs and both methods, so the published point estimates
and paired intervals can be recomputed without model inference. The source
input files and checkpoint are identified by SHA-256 in `manifest.json`.
