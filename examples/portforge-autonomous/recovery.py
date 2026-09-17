"""Durable controller recovery; never import or rewrite candidate code."""
import json
from pathlib import Path

from common import digest


def restore_jobs(run_dir):
    """Recover completed evidence; orphaned Ray references are explicitly lost."""
    root = Path(run_dir)
    events = root / "events.jsonl"
    jobs = {}
    if not events.exists():
        return jobs
    for line in events.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") != "started":
            continue
        name = event["job"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("invalid archived job ID")
        trial = root / "trials" / name
        source = json.loads((trial / "source.json").read_text())
        if digest(source) != event["source_sha256"]:
            raise ValueError("archived source checksum mismatch")
        result_file = trial / "result.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {
            "returncode": -1, "interrupted": True,
            "error": "Previous controller ended without a durable result. This job is not "
                     "running in the restored controller. After remote-worker reconciliation, "
                     "request a new evaluation; this record is not a model correctness verdict."}
        jobs[name] = {"kind": event["kind"], "source_sha256": event["source_sha256"],
                      "result": result}
    return jobs


def next_round(run_dir):
    numbers = [int(p.stem) for p in (Path(run_dir) / "rounds").glob("*.json")
               if p.stem.isdigit()]
    return max(numbers, default=0) + 1


def capacity_failure(record):
    return not record.get("success") and "at capacity" in record.get("error", "").lower()


def recover_capacity_error(run_dir, record):
    """The CLI wrapper truncates exception text; retain a logged terminal error."""
    if record.get("success"):
        return record
    for log in (Path(run_dir) / "agent-logs" / str(record["round"])).glob("*.log"):
        text = log.read_text()
        if "[Error]\nSelected model is at capacity." in text:
            return {**record, "error": "Selected model is at capacity. Please try again later."}
    return record


def retry_delay(attempt):
    return min(30 * 2 ** min(max(attempt - 1, 0), 4), 300)


def load_checkpoint(campaign, corpus):
    campaign = Path(campaign).resolve()
    manifest = json.loads((campaign / "manifest.json").read_text())
    if digest(corpus) != manifest["corpus_sha256"]:
        raise ValueError("resume corpus differs from original campaign")
    workspace = Path(manifest["workspace"])
    if not workspace.is_dir():
        raise ValueError("saved workspace missing")
    frozen = campaign / "harness"
    if digest({p.name: p.read_text() for p in frozen.glob("*.py")}) != manifest["harness_sha256"]:
        raise ValueError("frozen harness checksum mismatch")
    # Evaluator, model identity, input selection and gates must remain unchanged.
    for name in ("suite.py", "assess.py", "candidate_worker.py", "reference_worker.py", "remote.py"):
        if (frozen / name).read_bytes() != (Path(__file__).parent / name).read_bytes():
            raise ValueError("resume would change frozen evaluation: " + name)
    if (campaign / "submission.json").exists():
        raise ValueError("submitted campaign requires final-validation recovery, not more editing")
    return manifest, workspace
