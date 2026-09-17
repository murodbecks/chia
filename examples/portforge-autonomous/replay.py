"""Replay an archived candidate without involving Codex or changing its source."""
import argparse
import json
from pathlib import Path
import uuid
import ray
from chia.base.ChiaFunction import get
from common import digest, write_json
from suite import make_suite
from workflow import evaluation_node


def load_trial(campaign, trial):
    campaign = Path(campaign).resolve()
    if not trial or Path(trial).name != trial or trial in (".", ".."):
        raise ValueError("invalid trial identifier")
    events = [json.loads(line) for line in (campaign / "events.jsonl").read_text().splitlines()]
    event = next(x for x in events if x.get("job") == trial and x["event"] == "started")
    if event["kind"] not in ("smoke", "development", "qualification", "full"):
        raise ValueError("only evaluation trials can be replayed")
    files = json.loads((campaign / "trials" / trial / "source.json").read_text())
    if digest(files) != event["source_sha256"]:
        raise ValueError("archived source checksum mismatch")
    return files, event["kind"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    files, stage = load_trial(args.campaign, args.trial)
    manifest = json.loads((args.campaign / "manifest.json").read_text())
    corpus = json.loads(args.corpus.read_text())
    suite = make_suite(corpus, stage, manifest["heldout_ids"])
    ray.init(num_cpus=4, include_dashboard=False, _node_ip_address="127.0.0.1",
             resources={"reference_submitter": 1, "tt_evaluator": 1, "evaluation_coordinator": 1})
    try:
        trial = "replay-" + uuid.uuid4().hex
        write_json(args.campaign / "trials" / trial / "source.json", files)
        result = get(evaluation_node.chia_remote(files, suite, str(args.campaign.resolve()), trial))
        print(json.dumps({"trial": trial, "returncode": result["returncode"],
                          "passed": result.get("evaluation", {}).get("passed", False)}))
    finally:
        ray.shutdown()
