"""Generate with the released L11–15 model and frozen dual-intent prompt."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from episetter.release import check_model

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--context', required=True)
    p.add_argument('--intent', choices=['prior', 'context'], required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    check_model(a.model)
    import torch
    from episetter.model import load_olmo
    from episetter.calibration import LayerCalibrator
    from episetter.protocol import prompt
    runner = load_olmo(a.model, a.device, 'bfloat16')
    cp = Path(__file__).resolve().parents[1] / 'checkpoints/l11_15/seed42/calibration/calibrator.pt'
    cal = LayerCalibrator.restore(torch.load(cp, map_location='cpu', weights_only=True), a.device).eval()
    print(runner.generate(prompt(a.question, a.context, a.intent, 'fewshot'), 32, calibrator=cal))

if __name__ == '__main__':
    main()
