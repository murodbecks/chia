"""Compare artifacts outside the candidate namespace; policy is never agent-writable."""
import json
import hashlib
import math
from pathlib import Path
import statistics


def canonical_tokens(row, target, cap):
    row = [int(x) for x in row]
    if len(row) < 2 or len(row) > cap + 1 or row[:2] != [2, target]:
        raise ValueError("invalid start, forced language or generation length")
    if any(x < 0 or x >= 256206 for x in row):
        raise ValueError("out-of-vocabulary token")
    if 2 in row[2:]:
        end = row.index(2, 2)
        if 1 in row[2:end]:
            raise ValueError("padding before EOS")
        if any(x != 1 for x in row[end + 1:]):
            raise ValueError("non-padding after EOS")
        return row[:end + 1]
    if 1 in row[2:] or len(row) != cap + 1:
        raise ValueError("padding before EOS or premature termination")
    return row


def latency_summary(samples):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("invalid timing samples")
    ordered = sorted(samples)
    return {"p50_seconds": statistics.median(samples),
            "p95_seconds": ordered[max(0, math.ceil(len(ordered) * .95) - 1)],
            "samples": len(samples)}


def assess(directory, include_quality=False):
    import numpy as np
    directory = Path(directory)
    spec = json.loads((directory / "suite.json").read_text())
    actual = np.load(directory / "output/actual.npz", allow_pickle=False)
    expected = np.load(directory / "expected.npz", allow_pickle=False)
    measured = json.loads((directory / "output/measurements.json").read_text())
    reference = json.loads((directory / "reference.json").read_text())
    failures, checks, canonical = [], [], {}
    expected_keys = {c["name"] + "__" + k for c in spec["cases"]
                     for k in (["tokens", "encoder", "logits"] if c["numerical"] else ["tokens"])}
    if set(actual.files) != expected_keys:
        failures.append("missing or unexpected output arrays")
    measurements = {c["name"]: c for c in measured["cases"]}
    if len(measurements) != len(spec["cases"]) or set(measurements) != {c["name"] for c in spec["cases"]}:
        failures.append("missing/duplicate timing entries")
    directions = {}
    for case in spec["cases"]:
        name = case["name"]
        check = {"name": name}
        try:
            values = actual[name + "__tokens"]
            if values.ndim != 2 or values.shape[0] != len(case["texts"]) or values.dtype.kind not in "iu":
                raise ValueError("invalid token array shape/type")
            tokens = [canonical_tokens(row, case["target_id"], case["max_new_tokens"]) for row in values]
            canonical[name] = tokens
            check["tokens_sha256"] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            reference_tokens = [canonical_tokens(row, case["target_id"], case["max_new_tokens"])
                                for row in expected[name + "__tokens"]]
            check["exact_reference_rows"] = sum(a == b for a, b in zip(tokens, reference_tokens))
            check["rows"] = len(tokens)
            check["capped_rows"] = sum(2 not in row[2:] for row in tokens)
            if case["numerical"]:
                for key in ("encoder", "logits"):
                    a = actual[name + "__" + key].astype(np.float64)
                    b = expected[name + "__" + key].astype(np.float64)
                    if a.shape != b.shape or not np.isfinite(a).all():
                        raise ValueError(f"{key}: shape or finite check failed")
                    error = float(np.sqrt(np.mean((a - b) ** 2)) / max(1e-12, np.sqrt(np.mean(b ** 2))))
                    check[key + "_nrmse"] = error
                    if error > spec["gates"]["nrmse"]:
                        failures.append(f"{name}: {key} NRMSE {error:.6f}")
            if "compare_to" in case:
                if tokens != [canonical[case["compare_to"]][i] for i in case["compare_rows"]]:
                    failures.append(name + ": batch/order/padding/repeat semantics changed")
            check.update(latency_summary(measurements[name]["seconds"]))
            check["generated_tokens"] = sum(len(row) - 1 for row in tokens)
            check["tokens_per_second"] = check["generated_tokens"] / check["p50_seconds"]
            if case["kind"] == "quality":
                directions.setdefault(case["src"] + "->" + case["tgt"], []).append((case, tokens))
        except (KeyError, ValueError, IndexError) as exc:
            failures.append(name + ": " + str(exc))
        checks.append(check)
    quality = {}
    # Model/tokenizer cache and human references are visible only to this process.
    if include_quality and spec["stage"] in ("qualification", "full", "final"):
        from transformers import AutoTokenizer
        from sacrebleu.metrics import CHRF
        tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"],
                                                  trust_remote_code=False, local_files_only=True)
        reference_by_name = {x["name"]: x for x in reference["cases"]}
        metric = CHRF(word_order=2)
        for direction, batches in directions.items():
            hypotheses, baseline, labels = [], [], []
            for case, tokens in batches:
                hypotheses.extend(tokenizer.batch_decode(tokens, skip_special_tokens=True))
                baseline.extend(reference_by_name[case["name"]]["translations"])
                labels.extend(case["references"])
            score = metric.corpus_score(hypotheses, [labels]).score
            ref_score = metric.corpus_score(baseline, [labels]).score
            quality[direction] = {"chrf++": score, "cuda_chrf++": ref_score, "rows": len(labels)}
            if ref_score - score > spec["gates"]["max_chrf_loss"]:
                failures.append(direction + ": chrF++ loss exceeds frozen 0.5 gate")
    quality_done = include_quality or spec["stage"] in ("smoke", "development")
    return {"passed": not failures and quality_done, "stage": spec["stage"], "failures": failures,
            "quality_evaluated": quality_done, "numerical_behavior_passed": not failures,
            "checks": checks, "quality": quality, "measurements": measured,
            "cuda_reference": reference, "production_certified": False,
            "limitations": ["Python source review still required; not a hostile-code proof",
                            "memory views are allocator snapshots, not measured hardware peaks",
                            "p95 from five repeats is descriptive, not a tail-latency SLA"]}
