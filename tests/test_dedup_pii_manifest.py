import random

import numpy as np

from s4clean.dedup import (
    MinHasher,
    estimate_jaccard,
    exact_key,
    jaccard_arrays,
    lsh_catch_probability,
    lsh_threshold,
    near_duplicates,
    shingle_hashes,
    shingles,
)
from s4clean.manifest import doc_id, manifest_status
from s4clean.pii import scrub


def _doc(seed, n=300):
    r = random.Random(seed)
    return " ".join(f"w{r.randrange(5000)}" for _ in range(n))


def test_exact_key_is_content_based():
    assert exact_key("abc") == exact_key("abc")
    assert exact_key("abc") != exact_key("abd")


def test_shingles_word_level():
    assert shingles("A b c d e f", 5) == {"a b c d e", "b c d e f"}
    assert shingles("short doc", 5) == {"short doc"}


def test_minhash_estimate_tracks_true_jaccard():
    mh = MinHasher(256, seed=1)
    base = _doc(0).split()
    r = random.Random(1)
    for edits in (0, 10, 40, 120):
        w = list(base)
        for k in r.sample(range(len(w)), edits):
            w[k] = "zz" + str(k)
        a = shingle_hashes(shingles(" ".join(base), 5))
        b = shingle_hashes(shingles(" ".join(w), 5))
        tj = jaccard_arrays(a, b)
        est = estimate_jaccard(mh.signature(a), mh.signature(b))
        assert abs(tj - est) < 0.1


def test_minhash_reproducible():
    a = shingle_hashes(shingles(_doc(3), 5))
    assert np.array_equal(MinHasher(64, seed=7).signature(a), MinHasher(64, seed=7).signature(a))


def test_lsh_formula_matches_widget5():
    # widget 5 default b=6 r=4 -> 0.639; FineWeb preset b=14 r=8 -> 0.719
    assert round(lsh_threshold(6, 4), 3) == 0.639
    assert round(lsh_threshold(14, 8), 3) == 0.719
    assert lsh_catch_probability(1.0, 14, 8) == 1.0


def test_near_duplicates_found_and_unrelated_kept():
    base = _doc(10).split()
    near = list(base)
    near[5] = "CHANGED"
    near[200] = "ALSO"
    docs = [" ".join(base), " ".join(near), _doc(11), _doc(12)]
    sh = [shingle_hashes(shingles(d, 5)) for d in docs]
    mh = MinHasher(128, seed=1)
    sigs = np.stack([mh.signature(s) for s in sh])
    res = near_duplicates(sh, sigs, bands=16, rows=8, threshold=0.7)
    assert res.clusters == [[0, 1]]


def test_pii_masks_real_keeps_placeholders():
    t = "a.b@gmail.com x@example.com +91 98450 12345 8.8.8.8 127.0.0.1 answer 1234567890"
    out, c, _ = scrub(t)
    assert "[EMAIL]" in out and "x@example.com" in out
    assert "[PHONE]" in out and "[IP]" in out
    assert "127.0.0.1" in out and "1234567890" in out
    assert c["email"] == 1 and c["phone"] == 1 and c["ip"] == 1


def test_pii_does_not_touch_math_or_versions():
    t = "x = 3.14159265358979, 2^31 - 1 = 2147483647, version 10.0.19045.1"
    out, c, _ = scrub(t)
    assert out == t


def test_doc_id_deterministic():
    assert doc_id("hello") == doc_id("hello")
    assert doc_id("hello") != doc_id("hello ")


def test_manifest_blocks_unknown_license_and_missing_fields():
    good = dict(source="x", license="apache-2.0", contributor="me", cleaning_scripts=[{"n": 1}],
                content_sha256="ab", token_count=1, languages={"en": 1.0})
    assert manifest_status(good)[0] == "clean"
    assert manifest_status({**good, "license": "unknown"})[0] == "blocked"
    assert manifest_status({**good, "token_count": None})[0] == "blocked"


def test_greedy_keep_does_not_chain():
    from s4clean.dedup import greedy_keep
    # 0~1 and 1~2 are near copies, but 0 and 2 are not similar
    pairs = [(0, 1, 0.8, 0.8), (1, 2, 0.8, 0.8)]
    removed = greedy_keep(3, pairs)
    assert removed == {1: (0, 0.8)}  # 2 survives: its only near copy (1) was removed


def test_decontam_coverage_separates_copy_from_shared_quote():
    from s4clean.decontam import EvalIndex
    item = ("Both authors below were speaking of the French Revolution. It was the best of times, it was the "
            "worst of times, it was the age of wisdom, it was the age of foolishness. Which of the following best "
            "describes the attitude of the first author toward the revolution and its effect on ordinary people?")
    idx = EvalIndex()
    idx.add("mmlu", 7, item)
    copied = idx.scan("Question: " + item)[0]
    quote = idx.scan("Summarise the novel. It was the best of times, it was the worst of times, it was the age of "
                     "wisdom, it was the age of foolishness, and so on.")[0]
    assert copied[3] > 0.9 and quote[3] < 0.5


def test_pii_ignores_slices_cloze_and_versions():
    for t in ("x = a[0::2]", "arr[::2, ::2]", "{{c1::Paris}} is the capital",
              "Mozilla/5.0 Chrome/91.0.4.12 Safari"):
        assert scrub(t)[0] == t, t
    assert scrub("server 2001:4860:4860::8888 down")[0] == "server [IP] down"


def test_function_call_with_stray_extra_brace_is_repaired():
    from collections import Counter
    from s4clean.formats import parse_function_call
    raw = ' {"name": "f", "arguments": ' + "'" + '{"a": 1}' + "'" + '}} '
    assert parse_function_call(raw, Counter()) == {"name": "f", "arguments": {"a": 1}}
