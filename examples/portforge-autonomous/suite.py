"""Frozen, backend-independent workloads; no previous TT port is used."""
import hashlib
from common import digest

MODEL = "facebook/nllb-200-distilled-600M"
REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
WEIGHT_SHA256 = "c266c2cfd19758b6d09c1fc31ecdf1e485509035f6b51dfe84f1ada83eefcc42"
CORPUS_SHA256 = "c9b03ca2525db184b6feaf341a52383ec1d9d74311682462c29ba036e74327d1"
LANGUAGES = ("fra_Latn", "uzn_Latn", "arb_Arab", "zho_Hans")
GATES = {"nrmse": 0.04, "max_chrf_loss": 0.5, "behavior_exact": True}


def make_suite(corpus, stage, heldout_ids=()):
    if digest(corpus) != CORPUS_SHA256:
        raise ValueError("corpus identity mismatch")
    if stage not in ("smoke", "development", "qualification", "full", "final"):
        raise ValueError("unknown stage")
    ordered = sorted(range(1012), key=lambda i: hashlib.sha256(
        f"portforge-autonomous-v1:{i}".encode()).digest())
    selected = [i for i in ordered if i not in heldout_ids]
    counts = {"smoke": 1, "development": 4, "qualification": 128}
    ids = list(heldout_ids) if stage == "final" else (list(range(1012)) if stage == "full"
                                                     else selected[:counts[stage]])
    directions = [("eng_Latn", l) for l in LANGUAGES] + [(l, "eng_Latn") for l in LANGUAGES]
    if stage == "smoke":
        directions = directions[:1]
    cases = []
    for src, tgt in directions:
        for offset in range(0, len(ids), 4):
            chosen = ids[offset:offset + 4]
            cases.append(dict(name=f"{src}-{tgt}-{offset}", kind="quality", src=src, tgt=tgt,
                              texts=[corpus[src][i] for i in chosen],
                              references=[corpus[tgt][i] for i in chosen],
                              numerical=offset == 0, max_new_tokens=64))
        if stage == "smoke":
            cases[-1]["max_new_tokens"] = 8
        if stage in ("development", "qualification", "full", "final") and len(ids) >= 4:
            first = next(c for c in cases if c["name"] == f"{src}-{tgt}-0")
            for tag, rows in [("single0", [0]), ("single1", [1]), ("single2", [2]),
                              ("single3", [3]), ("pair", [0, 1]), ("triple", [0, 1, 2]),
                              ("reverse", [3, 2, 1, 0]), ("repeat", [0, 1, 2, 3]),
                              ("padding", [0, 1, 2, 3])]:
                cases.append(dict(name=first["name"] + "-" + tag, kind="behavior", src=src, tgt=tgt,
                                  texts=[first["texts"][i] for i in rows], numerical=tag == "padding",
                                  max_new_tokens=64, extra_padding=32 if tag == "padding" else 0,
                                  compare_to=first["name"], compare_rows=rows))
    if stage != "smoke":
        for length in (2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256):
            cases.append(dict(name=f"length-{length}", kind="boundary", src="eng_Latn", tgt="fra_Latn",
                              texts=["The students are testing multilingual translation. " * 100],
                              token_length=length, numerical=True, max_new_tokens=8))
        for cap in (2, 3, 31, 32, 33):
            cases.append(dict(name=f"cap-{cap}", kind="boundary", src="eng_Latn", tgt="fra_Latn",
                              texts=["Hello world."], numerical=False, max_new_tokens=cap))
        cases.append(dict(name="unicode-empty", kind="boundary", src="eng_Latn", tgt="fra_Latn",
                          texts=["", " \n\t", "Hello 👋! Café — €50.", "<script>alert('hello')</script>"],
                          numerical=True, max_new_tokens=32))
    if stage == "final":
        # Full covers every original row. Final therefore tests NEW transformed
        # inputs rather than falsely calling previously seen corpus rows held-out.
        for number, case in enumerate(cases):
            if "compare_to" not in case and case["kind"] == "quality":
                marker = str(100000 + heldout_ids[number % len(heldout_ids)] * 37) + ". "
                case["texts"] = [marker + x for x in case["texts"]]
                case["references"] = [marker + x for x in case["references"]]
        by_name = {c["name"]: c for c in cases}
        for case in cases:
            if "compare_to" in case:
                case["texts"] = [by_name[case["compare_to"]]["texts"][i] for i in case["compare_rows"]]
    return dict(version=1, stage=stage, model=MODEL, revision=REVISION,
                weight_sha256=WEIGHT_SHA256, corpus_sha256=CORPUS_SHA256,
                dataset_license="CC-BY-SA-4.0", model_license="CC-BY-NC-4.0",
                max_source_tokens=256, gates=GATES.copy(), cases=cases)
