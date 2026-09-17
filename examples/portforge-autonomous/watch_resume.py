"""Bounded capacity-recovery supervisor for an already-running continuation.

Never interrupts a live controller or resets hardware. In particular, a long
device evaluation is not mistaken for an agent-capacity failure.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from recovery import capacity_failure, next_round, recover_capacity_error, retry_delay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=3)
    args = parser.parse_args()
    deadline = time.monotonic() + args.hours * 3600
    attempts = 0
    while time.monotonic() < deadline:
        state = json.loads((args.campaign / "status.json").read_text())
        if state["status"] in ("running", "retry_wait", "preflight"):
            time.sleep(10)
            continue
        if state["status"] != "agent_error":
            return
        number = next_round(args.campaign) - 1
        record = json.loads((args.campaign / "rounds" / f"{number:03d}.json").read_text())
        if not capacity_failure(recover_capacity_error(args.campaign, record)):
            return
        pid = state.get("controller_pid")
        if pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                time.sleep(10)
                continue
        attempts += 1
        until = min(deadline, time.monotonic() + retry_delay(attempts))
        while time.monotonic() < until:
            time.sleep(min(10, max(0, until - time.monotonic())))
        remaining = (deadline - time.monotonic()) / 3600
        if remaining <= 0:
            return
        print(json.dumps({"action": "resume_after_capacity", "attempt": attempts}), flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("resume.py")),
            "--campaign", str(args.campaign.resolve()), "--corpus", str(args.corpus.resolve()),
            "--hours", str(remaining), "--detach"], check=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
