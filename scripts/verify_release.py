"""Offline checks of release bytes, model module and published prediction scores."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from episetter.data import digest
from episetter.mrqa_metrics import (extract_answer, exact_match_score, f1_score,
                                    acc_score, metric_max_over_ground_truths)

ROOT = Path(__file__).resolve().parents[1]

def main():
    seal = ROOT / 'provenance/SHA256SUMS.json'
    for relative, expected in json.loads(seal.read_text()).items():
        assert digest(ROOT / relative) == expected, relative
    freeze = json.loads((ROOT / 'data/freeze.json').read_text())
    for name, expected in freeze['files'].items():
        assert digest(ROOT / 'data' / name) == expected, name
    evaluation = json.loads((ROOT / 'provenance/evaluation_inputs.json').read_text())
    assert digest(ROOT / 'checkpoints/l11_15/seed42/calibration/calibrator.pt') == evaluation['seeds']['42']['sha256']
    assert digest(ROOT / 'data/test417.json') == evaluation['test417']['sha256']
    assert digest(ROOT / 'data/mrqa_exclusion_bundle.json') == evaluation['mrqa_bundle']['sha256']
    for entry in json.loads((ROOT / 'results/mrqa/matched_summary.json').read_text()):
        ids = []
        for method, key in [('base', 'base'), ('seed42', 'l11_15')]:
            path = ROOT / f'results/mrqa/{method}' / (entry['dataset'] + '.predictions.jsonl')
            records = [json.loads(line) for line in path.read_text().splitlines()]
            qids = [r['qid'] for r in records]
            assert len(qids) == len(set(qids)) == entry['n']
            assert hashlib.sha256('\n'.join(qids).encode()).hexdigest() == entry['selected_qids_sha256']
            ids.append(qids)
            for metric, fn in [('EM', exact_match_score), ('F1', f1_score), ('ACC', acc_score)]:
                values = []
                for r in records:
                    parsed = extract_answer(r['response'])
                    answer = parsed if parsed is not None else r['response'].strip()
                    value = metric_max_over_ground_truths(fn, answer, r['references'])
                    assert abs(value - r['scores'][metric]) < 1e-8
                    values.append(value)
                assert abs(100 * sum(values) / len(values) - entry[key][metric]) < 1e-8
        assert ids[0] == ids[1]
    import torch
    from episetter.calibration import LayerCalibrator
    state = torch.load(ROOT / 'checkpoints/l11_15/seed42/calibration/calibrator.pt',
                       map_location='cpu', weights_only=True)
    cal = LayerCalibrator.restore(state, 'cpu').eval()
    assert list(cal.layers) == [11, 12, 13, 14, 15]
    assert sum(p.numel() for p in cal.parameters()) == 655685
    assert state['selected_step'] == 1800 and state['status'] == 'validation_selected'
    for layer in cal.layers:
        axis = getattr(cal, f'axis_{layer}')
        h = torch.randn(2, axis.numel())
        out, alpha = cal.apply(layer, h)
        assert torch.isfinite(out).all()
        assert (alpha.abs() <= getattr(cal, f'rho_{layer}') + 1e-6).all()
        assert torch.allclose(out, h + alpha.unsqueeze(-1) * axis, atol=1e-5)
    print('PASS: release hashes, frozen data, 11784 predictions rescored, matched qids, checkpoint restore, five-layer bounded intervention. No GPU experiment rerun.')

if __name__ == '__main__':
    main()
