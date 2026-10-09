"""Run the unit tests and record the result for the site (artifacts/audits/tests.json)."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                   capture_output=True, text=True, encoding="utf-8")
tail = (r.stdout.strip().splitlines() or [""])[-1]
passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
errors = int(m.group(1)) if (m := re.search(r"(\d+) error", tail)) else 0
out = {"passed": passed, "failed": failed, "errors": errors, "summary": tail, "exit_code": r.returncode}
(ROOT / "artifacts" / "audits").mkdir(parents=True, exist_ok=True)
(ROOT / "artifacts" / "audits" / "tests.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(tail)
sys.exit(r.returncode)
