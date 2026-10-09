"""Reproducibility check (same input + same config -> same output).

python scripts/check_determinism.py

Runs the whole pipeline a second time with its artifacts written to a
temporary folder, then compares against artifacts/ from the first run:
corpus id, every shard's content sha256, doc and token counts, and every
stage counter. Writes artifacts/audits/determinism.json.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


first_manifest = load(ROOT / "artifacts" / "manifest.json")
first_metrics = load(ROOT / "artifacts" / "metrics.json")

with tempfile.TemporaryDirectory() as tmp:
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--out", tmp],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        sys.exit("second run failed")
    second_manifest = load(Path(tmp) / "manifest.json")
    second_metrics = load(Path(tmp) / "metrics.json")

results = []


def same(name, a, b):
    results.append({"check": name, "identical": a == b, "first": a if not isinstance(a, (dict, list)) else None})
    print(("SAME " if a == b else "DIFF ") + name)


same("corpus_id", first_manifest["corpus_id"], second_manifest["corpus_id"])
for s1, s2 in zip(first_manifest["shards"], second_manifest["shards"]):
    for k in ("shard_id", "content_sha256", "docs", "token_count", "languages", "config_sha256", "cleaning_scripts"):
        same(f"{s1['source'].rsplit('/', 1)[-1]}.{k}", s1[k], s2[k])
for st1, st2 in zip(first_metrics["stages"], second_metrics["stages"]):
    same(f"stage {st1['stage']} counters", st1, st2)
same("dedup info", first_metrics["dedup"], second_metrics["dedup"])
same("decontam info", first_metrics["decontam"], second_metrics["decontam"])

out = {"all_identical": all(r["identical"] for r in results), "checks": results}
(ROOT / "artifacts" / "audits").mkdir(parents=True, exist_ok=True)
(ROOT / "artifacts" / "audits" / "determinism.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
sys.exit(0 if out["all_identical"] else 1)
