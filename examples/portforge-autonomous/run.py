"""Launch a fresh autonomous CHIA campaign; no prior model implementation is seeded."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import ray
from chia.base.ChiaFunction import get
from common import digest, ssh, upload, write_json
from suite import make_suite, MODEL, REVISION, WEIGHT_SHA256
from workflow import (HERE, TT_ROOT, PortingTools, agent_node, device_node,
                      evaluation_node, reference_node)


def prepare(run_dir, corpus_path):
    corpus = json.loads(Path(corpus_path).read_text())
    workspace = Path(tempfile.mkdtemp(prefix="portforge-fresh-"))
    for source in (HERE / "agent").glob("*.md"):
        (workspace / source.name).write_bytes(source.read_bytes())
    (workspace / "TASK.md").write_text(f"""# NLLB bring-up task

Build {MODEL} at revision {REVISION} from scratch for a Tenstorrent Blackhole.
Weight SHA256: {WEIGHT_SHA256}. Start with the smallest model; larger variants
are future tasks, not a reason to compromise this model's correctness.

The controller creates a fresh FP32 PyTorch reference on an NVIDIA Slurm compute
node. The login node never runs inference. Request smoke/development evaluations
to compare your implementation; reference jobs may take a few minutes to queue.
The provided target runtime is TT-Metal ba9340e3a45ac5ba51c752a49341f2def28d0514.
Inspect installed source and small experiments to learn the exact API.

Read CONTRACT.md and TT_PORTING_GUIDE.md. Research public prior work, implement
backend.py, write your own tests, pass development, then optimize against the
first passing TT snapshot. Test broad translation quality before submission.
No existing private TT implementation or previous optimization results are
provided. You own the engineering, diagnosis and optimization decisions.

You may use public web search and the PortForge tools. No host shell is exposed.
The run tool executes a copied workspace in a network-isolated TT environment;
only write persists source changes. Up to four device jobs can execute in
parallel, each with a physical-card lease. Never invoke device reset utilities.
Write STATE.md after each milestone so a new agent round can continue.

Model weights are CC-BY-NC-4.0; this campaign does not grant commercial serving
rights. Keep code, data and model licenses distinct in your final report.
""")
    heldout = random.SystemRandom().sample(range(1012), 32)
    identity = ssh("tt_box", "git -C /home/abror/portforge-tt/tt-metal rev-parse HEAD").decode().strip()
    if identity != "ba9340e3a45ac5ba51c752a49341f2def28d0514":
        raise RuntimeError("TT runtime revision changed; review the task pin before launch")
    weight_hash = ssh("tt_box", "sha256sum /home/abror/portforge-nllb/cache/" + REVISION + "/pytorch_model.bin", timeout=120).decode().split()[0]
    if weight_hash != WEIGHT_SHA256:
        raise RuntimeError("model checkpoint hash mismatch")
    suite = make_suite(corpus, "smoke", heldout)
    harness = {p.name: p.read_text() for p in HERE.glob("*.py")}
    harness_hash = digest(harness)
    (run_dir / "harness").mkdir()
    for name, content in harness.items():
        (run_dir / "harness" / name).write_text(content)
    remote_harness = TT_ROOT + "/harness/" + harness_hash
    write_json(run_dir / "manifest.json", {"created": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace), "model": MODEL, "revision": REVISION,
        "seed_files": sorted(p.name for p in workspace.iterdir()), "heldout_ids": heldout,
        "corpus_sha256": digest(corpus), "tt_metal_commit": identity, "weight_sha256": weight_hash,
        "harness_sha256": harness_hash, "remote_harness": remote_harness,
        "controller_pid": os.getpid(),
        "status": "prepared"})
    upload("tt_box", remote_harness, {name: harness[name]
                                           for name in ("remote.py", "candidate_worker.py", "assess.py")})
    return corpus, heldout, workspace, suite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--hours", type=float, default=6)
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--detach", action="store_true", help="run in background with durable log and process ID")
    args = p.parse_args()
    run_dir = (args.run_dir or HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    if args.detach:
        if run_dir.exists():
            p.error("run directory already exists")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--corpus", str(args.corpus.resolve()),
                   "--hours", str(args.hours), "--rounds", str(args.rounds), "--run-dir", str(run_dir)]
        if args.preflight_only:
            command.append("--preflight-only")
        log_path = run_dir.with_suffix(".log")
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        if sys.platform == "darwin":
            subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(process.pid)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        record = {"pid": process.pid, "run_dir": str(run_dir), "log": str(log_path)}
        write_json(run_dir.with_suffix(".launch.json"), record)
        print(json.dumps(record), flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    corpus, heldout, workspace, smoke = prepare(run_dir, args.corpus)
    # This is a local native Ray controller, with explicit SSH/Slurm execution
    # adapters. It does not pretend remote hosts are Docker/Ray workers.
    ray.init(num_cpus=8, include_dashboard=False, _node_ip_address="127.0.0.1",
             resources={"codex_creds": 1, "reference_submitter": 1,
                        "tt_evaluator": 4, "evaluation_coordinator": 4})
    from chia.trace.profiler import start_collector
    start_collector(log_dir=str(run_dir / "chia-profile"))
    tools = PortingTools(name="portforge_" + run_dir.name.replace("-", "_"),
                         workspace=str(workspace), run_dir=str(run_dir), corpus=corpus,
                         heldout_ids=heldout)
    try:
        probe = get(device_node.chia_remote({"kind": "run", "files": {}, "timeout": 180,
            "command": "python -c 'import os,socket,ttnn,torch; "
            "assert not os.path.exists(\"/home/abror/portforge-nllb\"); "
            "assert not os.path.exists(\"/home/abror/.ssh\"); "
            "d=ttnn.open_device(device_id=0); "
            "x=ttnn.from_torch(torch.ones(1,1,32,32),device=d,layout=ttnn.TILE_LAYOUT); "
            "assert torch.equal(ttnn.to_torch(x),torch.ones(1,1,32,32)); "
            "import importlib.util; spec=importlib.util.spec_from_file_location(\"runner\",\"/runner.py\"); "
            "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            "v=m.memory_snapshot(ttnn,d); assert v[\"DRAM\"][\"total_allocated_bytes\"]>0; print(v); "
            "ttnn.close_device(d); print(\"ISOLATION_DEVICE_OK\")'"}, str(run_dir), "preflight"))
        if probe["returncode"] != 0:
            raise RuntimeError("device preflight failed: " + json.dumps(probe))
        # Test CLI configuration/auth/tool visibility before any engineering round.
        preflight = get(agent_node.chia_remote(str(workspace), str(run_dir), [tools], 0,
            "This is ONLY an infrastructure preflight. Call portforge_files, read CONTRACT.md, "
            "and report which tools you have. Do NOT implement the model yet. Confirm whether "
            "host shell/exec, filesystem/image, apps, other MCP servers or prior sessions are "
            "accessible. Try portforge_read('../outside-canary') and confirm it rejects traversal. "
            "Do not access credentials. End with PREFLIGHT_OK only if the scoped tools work."))
        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()] if (run_dir / "events.jsonl").exists() else []
        observed = {x["event"] for x in events}
        if (not preflight["success"] or preflight["response"].strip().splitlines()[-1] != "PREFLIGHT_OK"
                or not {"files", "read", "read_rejected"} <= observed):
            raise RuntimeError("Codex isolation preflight did not pass")
        write_json(run_dir / "preflight.json", {"device": probe, "agent": preflight})
        if args.preflight_only:
            print(json.dumps({"status": "preflight_passed", "run_dir": str(run_dir)}), flush=True)
            return
        write_json(run_dir / "status.json", {"status": "running", "workspace": str(workspace),
                    "production_certified": False, "budget_hours": args.hours, "max_rounds": args.rounds})
        # Reference baseline executes while the agent researches public upstream work.
        reference = reference_node.chia_remote(smoke, str(run_dir))
        deadline = time.monotonic() + args.hours * 3600
        status = "budget_exhausted"
        for number in range(1, args.rounds + 1):
            if time.monotonic() >= deadline:
                break
            if (run_dir / "submission.json").exists():
                status = "submitted"
                break
            print(json.dumps({"round": number, "workspace": str(workspace), "run_dir": str(run_dir)}), flush=True)
            result = get(agent_node.chia_remote(str(workspace), str(run_dir), [tools], number,
                "Continue the autonomous porting task. Read STATE.md if present, then take "
                "concrete engineering actions with the tools. Save progress before ending this "
                "round. You have up to 25 minutes in this round; do not just propose a plan. "
                "Begin with research and reference validation if this is the first round. "
                "Use async job IDs to overlap useful work. Aim for a correct, thoroughly "
                "optimized and reproducible backend, and submit when all stages pass."))
            print(json.dumps({"round": number, "success": result["success"]}), flush=True)
            if not result["success"]:
                # One failed/expired CLI round need not discard persisted source.
                # Two consecutive failures stop rather than looping on auth errors.
                if number > 1 and not json.loads((run_dir / "rounds" / f"{number - 1:03d}.json").read_text())["success"]:
                    status = "agent_error"
                    break
        if (run_dir / "submission.json").exists():
            submission = json.loads((run_dir / "submission.json").read_text())
            final = get(evaluation_node.chia_remote(submission["files"],
                make_suite(corpus, "final", heldout), str(run_dir), "final"))
            status = "accepted_research_port" if final.get("evaluation", {}).get("passed") else "final_failed"
        write_json(run_dir / "status.json", {"status": status, "workspace": str(workspace),
                    "production_certified": False, "finished": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"status": status, "run_dir": str(run_dir)}), flush=True)
    except BaseException as exc:
        write_json(run_dir / "status.json", {"status": "harness_error", "error": str(exc)[-3000:],
                    "workspace": str(workspace), "production_certified": False})
        raise
    finally:
        tools.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
