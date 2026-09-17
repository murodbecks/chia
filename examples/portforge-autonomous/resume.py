"""Continue saved agent work with a new CHIA controller and the original evaluator."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import digest, snapshot, write_json
from recovery import capacity_failure, load_checkpoint, next_round, restore_jobs, retry_delay, recover_capacity_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.hours <= 0 or args.rounds <= 0:
        parser.error("positive budget required")
    campaign = args.campaign.resolve()
    corpus = json.loads(args.corpus.read_text())
    manifest, workspace = load_checkpoint(campaign, corpus)
    # Refuse a second controller; preserve the original manifest and trial files.
    prior = json.loads((campaign / "status.json").read_text())
    if prior.get("status") in ("running", "retry_wait", "preflight"):
        parser.error("campaign reports an active controller; reconcile it before resuming")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recovery_dir = campaign / "recoveries" / stamp
    if args.detach:
        log_path = campaign / ("resume-" + stamp + ".log")
        command = [sys.executable, str(Path(__file__).resolve()), "--campaign", str(campaign),
                   "--corpus", str(args.corpus.resolve()), "--hours", str(args.hours),
                   "--rounds", str(args.rounds)]
        with log_path.open("wb") as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        if sys.platform == "darwin":
            subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(proc.pid)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        record = {"pid": proc.pid, "log": str(log_path), "campaign": str(campaign)}
        write_json(campaign / ("resume-" + stamp + ".launch.json"), record)
        print(json.dumps(record), flush=True)
        return
    lock = (campaign / "resume.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    recovery_dir.mkdir(parents=True)
    write_json(recovery_dir / "previous-status.json", prior)
    controller = {p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}
    write_json(recovery_dir / "controller-source.json", controller)
    jobs = restore_jobs(campaign)
    write_json(recovery_dir / "manifest.json", {
        "controller_pid": os.getpid(), "controller_sha256": digest(controller),
        "workspace_sha256": digest(snapshot(workspace)), "workspace": str(workspace),
        "intervention": "Controller resume after capacity failure; device recovery by operator agent. "
                        "No candidate source or acceptance gate changes.",
        "recovered_jobs": len(jobs),
        "missing_results": [j for j, s in jobs.items() if s["result"].get("interrupted")],
        "hours": args.hours, "rounds": args.rounds})

    def status(value, **extra):
        write_json(campaign / "status.json", {"status": value, "workspace": str(workspace),
                   "controller_pid": os.getpid(), "recovery": str(recovery_dir),
                   "updated": datetime.now(timezone.utc).isoformat(),
                   "production_certified": False, **extra})

    import ray
    from chia.base.ChiaFunction import get
    from chia.trace.profiler import start_collector
    from workflow import PortingTools, agent_node, device_node, evaluation_node
    from suite import make_suite
    status("preflight")
    ray.init(num_cpus=8, include_dashboard=False, _node_ip_address="127.0.0.1",
             resources={"codex_creds": 1, "reference_submitter": 1,
                        "tt_evaluator": 4, "evaluation_coordinator": 4})
    start_collector(log_dir=str(recovery_dir / "chia-profile"))
    tools = PortingTools(name="portforge_resume_" + stamp, workspace=str(workspace),
                        run_dir=str(campaign), corpus=corpus, heldout_ids=manifest["heldout_ids"])
    deadline = time.monotonic() + args.hours * 3600
    try:
        # Original frozen sandbox/worker, real device compute, no earlier private port.
        probe = get(device_node.chia_remote({"kind": "run", "files": {}, "timeout": 180,
            "command": "python -c 'import os,torch,ttnn; "
            "assert not os.path.exists(\"/home/abror/portforge-nllb\"); "
            "assert not os.path.exists(\"/home/abror/.ssh\"); "
            "d=ttnn.open_device(device_id=0); "
            "x=ttnn.from_torch(torch.ones(1,1,32,32),device=d,layout=ttnn.TILE_LAYOUT); "
            "y=ttnn.add(x,x); assert torch.equal(ttnn.to_torch(y),torch.full((1,1,32,32),2.)); "
            "ttnn.close_device(d); print(\"RESUME_DEVICE_OK\")'"},
            str(campaign), "resume-preflight-" + stamp))
        write_json(recovery_dir / "device-preflight.json", probe)
        if probe["returncode"] != 0:
            raise RuntimeError("device recovery preflight failed")
        capacity_attempts = failures = 0
        outcome = "budget_exhausted"
        for number in range(next_round(campaign), next_round(campaign) + args.rounds):
            if time.monotonic() >= deadline:
                break
            status("running", round=number)
            record = get(agent_node.chia_remote(str(workspace), str(campaign), [tools], number,
                "CONTINUATION of your existing NLLB port, not a fresh experiment. Read STATE.md, "
                "then call portforge_status. The previous controller stopped on model capacity; "
                "it has been restored and idle Blackhole cards reset by the operator. "
                "Old jobs without durable results now report interrupted, not running; request "
                "new evaluations as needed. Your source and gates are unchanged. Prioritize "
                "the selected backend's smoke then development evaluation now. Preserve the "
                "first passing development baseline. Diagnose actual failures; do not spend "
                "rounds only adding host tests while reference validation is missing. Proceed "
                "to qualification/full/final when ready. Save STATE.md. About 25 minutes per "
                "round. If device initialization fails, report it immediately and avoid "
                "repeated failing hardware jobs. No device resets from candidate tools."))
            record = recover_capacity_error(campaign, record)
            print(json.dumps({"round": number, "success": record["success"],
                              "error": record.get("error", "")}), flush=True)
            if (campaign / "submission.json").exists():
                submission = json.loads((campaign / "submission.json").read_text())
                final = get(evaluation_node.chia_remote(submission["files"],
                    make_suite(corpus, "final", manifest["heldout_ids"]), str(campaign), "final"))
                outcome = "accepted_research_port" if final.get("evaluation", {}).get("passed") else "final_failed"
                break
            if capacity_failure(record):
                capacity_attempts += 1
                delay = retry_delay(capacity_attempts)
                status("retry_wait", reason="model_capacity", delay_seconds=delay, round=number)
                until = min(deadline, time.monotonic() + delay)
                while time.monotonic() < until:
                    time.sleep(min(10, max(0, until - time.monotonic())))
                continue
            capacity_attempts = 0
            failures = 0 if record["success"] else failures + 1
            if failures >= 2:
                outcome = "agent_error"
                break
        status(outcome)
    except BaseException as exc:
        status("harness_error", error=str(exc)[-3000:])
        raise
    finally:
        tools.stop()
        ray.shutdown()
        lock.close()


if __name__ == "__main__":
    main()
