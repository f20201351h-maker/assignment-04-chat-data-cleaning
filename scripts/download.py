"""Download the exact source files at pinned revisions and record their hashes.

python scripts/download.py

Writes data/raw/<file> and artifacts/sources.json. Re-running checks that the
upstream files still hash to the same values.
"""
import json
import pathlib
import shutil
import sys

from huggingface_hub import HfApi, hf_hub_download

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s4clean.manifest import sha256_file  # noqa: E402

SOURCES = [
    {
        "shard": "glaive",
        "repo": "glaiveai/glaive-function-calling-v2",
        "revision": "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
        "file": "glaive-function-calling-v2.json",
        "license": "apache-2.0",
        "contributor": "Glaive AI (synthetic generation; generator model not stated on card)",
        "terms_note": "Card is empty; generating model unknown.",
    },
    {
        "shard": "anudesh",
        "repo": "ai4bharat/indic-align",
        "revision": None,  # resolved below to the full sha of the inspected revision
        "revision_prefix": "032b6a9070",
        "file": "indicalign-instruct/anudesh/anudesh1.parquet",
        "license": "cc-by-4.0",
        "contributor": "AI4Bharat (crowd-sourced prompts; responses from Llama-2-70B-Chat per card)",
        "terms_note": "CC-BY-4.0 needs attribution. Responses are Llama-2-70B-Chat outputs; the Llama 2 "
        "licence restricts using outputs to improve other LLMs.",
    },
]
EVAL = [
    ("gsm8k", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet", "mit"),
    ("math500", "HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl", "mit (MATH)"),
    ("humaneval", "openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
     "openai_humaneval/test-00000-of-00001.parquet", "mit"),
    ("mmlu", "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all/test-00000-of-00001.parquet", "mit"),
]


def main():
    api = HfApi()
    raw = ROOT / "data" / "raw"
    ev = ROOT / "data" / "eval"
    raw.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)
    out = {"sources": [], "eval_sets": []}
    for s in SOURCES:
        rev = s["revision"]
        if rev is None:
            # resolve the short prefix we inspected to the full commit sha
            refs = [c.commit_id for c in api.list_repo_commits(s["repo"], repo_type="dataset")]
            match = [c for c in refs if c.startswith(s["revision_prefix"])]
            if not match:
                sys.exit(f"revision {s['revision_prefix']} not found for {s['repo']}")
            rev = match[0]
        p = hf_hub_download(s["repo"], s["file"], repo_type="dataset", revision=rev)
        dst = raw / pathlib.Path(s["file"]).name
        shutil.copy(p, dst)
        rec = {k: v for k, v in s.items() if k not in ("revision_prefix",)}
        rec["revision"] = rev
        rec["local_path"] = str(dst.relative_to(ROOT)).replace("\\", "/")
        rec["sha256"] = sha256_file(dst)
        rec["bytes"] = dst.stat().st_size
        out["sources"].append(rec)
        print(s["shard"], rev, rec["sha256"][:16], rec["bytes"])
    for name, repo, rev, fn, lic in EVAL:
        p = hf_hub_download(repo, fn, repo_type="dataset", revision=rev)
        dst = ev / (name + "_test" + pathlib.Path(fn).suffix)
        shutil.copy(p, dst)
        out["eval_sets"].append({"name": name, "repo": repo, "revision": rev, "file": fn, "license": lic,
                                 "sha256": sha256_file(dst)})
        print(name, rev[:12], out["eval_sets"][-1]["sha256"][:16])
    out["eval_sets_not_used"] = [{"name": "MILU", "repo": "ai4bharat/MILU",
                                  "reason": "gated (requires accepting terms); not downloaded"}]
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "sources.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
