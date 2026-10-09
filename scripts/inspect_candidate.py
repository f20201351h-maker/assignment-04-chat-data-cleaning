"""Quick look at a candidate Hugging Face dataset without downloading all of it.

usage: python scripts/inspect_candidate.py <dataset_id> [file_substring] [max_rows]

Prints the pinned revision, licence/gating from the card, the data files with
sizes, then downloads ONE data file (the first matching file_substring) and
prints its schema and a few truncated rows. Used during dataset selection.
"""
import json
import sys

from huggingface_hub import HfApi, hf_hub_download

ds = sys.argv[1]
sub = sys.argv[2] if len(sys.argv) > 2 else ""
max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 3

api = HfApi()
info = api.dataset_info(ds, files_metadata=True)
card = info.card_data.to_dict() if info.card_data else {}
print(f"{ds} @ {info.sha}")
print("license:", card.get("license"), "| gated:", info.gated, "| langs:", card.get("language"))
files = [(s.rfilename, s.size or 0) for s in info.siblings
         if s.rfilename.endswith((".parquet", ".jsonl", ".json", ".jsonl.gz", ".arrow"))]
total = sum(sz for _, sz in files)
print(f"{len(files)} data files, {total/1e6:.1f} MB total")
for fn, sz in files[:25]:
    print(f"   {sz/1e6:9.1f} MB  {fn}")
pick = [f for f in files if sub in f[0]]
if not pick:
    sys.exit("no matching file")
fn = pick[0][0]
p = hf_hub_download(ds, fn, repo_type="dataset", revision=info.sha)
print("downloaded", fn)
if fn.endswith(".parquet"):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(p)
    print("rows in file:", pf.metadata.num_rows)
    print(pf.schema_arrow)
    rows = pf.read_row_group(0).slice(0, max_rows).to_pylist()
else:
    rows = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            rows.append(json.loads(line))
for r in rows:
    print(json.dumps(r, ensure_ascii=False, default=str)[:1500])
    print("-" * 60)
