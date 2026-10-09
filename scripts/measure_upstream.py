"""Measure the backslash / markup loss in the raw Glaive file (an upstream defect).

python scripts/measure_upstream.py  ->  artifacts/audits/upstream_glaive_markup.json

Reads the pinned raw file only, so the numbers do not depend on any cleaning step.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "data/raw/glaive-function-calling-v2.json").read_text(encoding="utf-8"))
BS = chr(92)
LATEX = re.compile(r"(?<![\\\w])(?:begin|end|section\*?|subsection\*?|textbf|textit|emph|usepackage|documentclass|frac|"
                   r"sqrt|hline|caption|includegraphics|maketitle|toprule|midrule|bottomrule|multicolumn)\{")
rows_with_bs = sum(1 for r in rows if BS in (r.get("chat") or "") or BS in (r.get("system") or ""))
total_bs = sum((r.get("chat") or "").count(BS) + (r.get("system") or "").count(BS) for r in rows)
code_rows = sum(1 for r in rows if "```" in (r.get("chat") or ""))
code_rows_bs = sum(1 for r in rows if "```" in (r.get("chat") or "") and BS in (r.get("chat") or ""))
latex_rows = sum(1 for r in rows if len(LATEX.findall(r.get("chat") or "")) >= 2)
intact_latex = sum(1 for r in rows if (BS + "begin{") in (r.get("chat") or "") or (BS + "section") in (r.get("chat") or ""))
out = {
    "file": "data/raw/glaive-function-calling-v2.json",
    "rows": len(rows),
    "rows_containing_any_backslash": rows_with_bs,
    "backslashes_total": total_bs,
    "rows_with_code_block": code_rows,
    "rows_with_code_block_and_any_backslash": code_rows_bs,
    "rows_with_2plus_backslashless_latex_commands": latex_rows,
    "rows_with_intact_backslash_begin_or_section": intact_latex,
}
(ROOT / "artifacts/audits").mkdir(parents=True, exist_ok=True)
(ROOT / "artifacts/audits/upstream_glaive_markup.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
print(out)
