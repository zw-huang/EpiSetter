"""Validate the available training-time model identity, independent of location."""
import json
from pathlib import Path
from .data import digest

def check_model(model):
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / 'checkpoints/l11_15/seed42/run_manifest.json').read_text())
    model = Path(model)
    manifest = receipt['model_manifest']
    for name, expected in manifest['small_file_sha256'].items():
        if digest(model / name) != expected:
            raise ValueError('Base model identity mismatch: ' + name)
    for name, identity in manifest['shard_stat_identity'].items():
        if (model / name).stat().st_size != identity['size']:
            raise ValueError('Base model shard size mismatch: ' + name)
