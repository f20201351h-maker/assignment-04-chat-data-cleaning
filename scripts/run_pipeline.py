"""Run the whole Session 4 pipeline.

python scripts/run_pipeline.py [--config config/pipeline.toml] [--out artifacts]

Inputs: data/raw/* and data/eval/* (from scripts/download.py).
Outputs: data/clean/*.jsonl, data/quarantine/*.jsonl, data/audit/* (local only),
artifacts/metrics.json, artifacts/manifest.json, artifacts/examples_raw.json.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from s4clean.pipeline import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "pipeline.toml"))
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    a = ap.parse_args()
    m = run(Path(a.config), Path(a.out))
    print("corpus", m["corpus_id"])
