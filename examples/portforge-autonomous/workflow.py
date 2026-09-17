"""CHIA agentic graph: Codex -> tools -> scheduled reference/TT work -> evaluator."""
import json
import os
from pathlib import Path
import shlex
import time
import uuid

import ray
from chia.base.ChiaFunction import ChiaFunction, get
from chia.base.tools.ChiaTool import ChiaTool
from chia.models.codex import CodexLLM
from common import digest, safe_path, snapshot, ssh, upload, write_json
from suite import make_suite
from recovery import restore_jobs

HERE = Path(__file__).resolve().parent
TT_ROOT = "/home/abror/portforge-autonomous"


def harness_root(run_dir):
    frozen = Path(run_dir) / "harness"
    return frozen if frozen.is_dir() else HERE


@ChiaFunction(resources={"reference_submitter": 1}, max_retries=0)
def reference_node(suite, run_dir):
    key = digest(suite)
    local = Path(run_dir) / "references" / key
    if (local / "complete.json").exists():
        return {"id": key, "directory": str(local)}
    relative = "portforge-autonomous/references/" + key
    # Resolve the account home without trusting arbitrary shell interpolation.
    home = ssh("student_lab", "printf '%s' \"$HOME\"").decode()
    remote = home + "/" + relative
    script = '''#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=portforge-auto-ref
#SBATCH --output=slurm.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
trap 'rc=$?; printf "%s\\n" "$rc" > exit-code.txt' EXIT
export OMP_NUM_THREADS=12 PYTHONNOUSERSITE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$HOME/portforge-reference/.cache/huggingface"
"$HOME/portforge-reference/.venv/bin/python" reference_worker.py
'''
    upload("student_lab", remote, {"suite.json": json.dumps(suite), "reference.sbatch": script,
                                  "reference_worker.py": (harness_root(run_dir) / "reference_worker.py").read_text()})
    # Fresh job directory removes stale completion ambiguity; never cancel unrelated jobs.
    ssh("student_lab", f"rm -f {shlex.quote(remote + '/exit-code.txt')}")
    job = ssh("student_lab", f"cd {shlex.quote(remote)} && sbatch --parsable reference.sbatch").decode().strip()
    if not job.isdigit():
        raise RuntimeError("invalid Slurm job ID: " + job)
    write_json(local / "job.json", {"job_id": job, "directory": remote, "suite_sha256": key})
    try:
        deadline = time.monotonic() + 10800
        while time.monotonic() < deadline:
            state = ssh("student_lab", f"if test -f {shlex.quote(remote + '/exit-code.txt')}; then cat {shlex.quote(remote + '/exit-code.txt')}; else echo pending; fi").decode().strip()
            if state != "pending":
                log = ssh("student_lab", f"tail -c 16000 {shlex.quote(remote + '/slurm.log')}").decode()
                (local / "slurm.log").write_text(log)
                if state != "0":
                    raise RuntimeError("reference failed: " + log)
                files = {}
                for source, target in (("inputs.npz", "inputs.npz"), ("expected.npz", "expected.npz"),
                                       ("prepared-suite.json", "suite.json"), ("reference.json", "reference.json"),
                                       ("config.json", "config.json")):
                    value = ssh("student_lab", "cat " + shlex.quote(remote + "/" + source), timeout=300)
                    (local / target).write_bytes(value)
                    files[target] = value
                upload("tt_box", TT_ROOT + "/references/" + key, files)
                write_json(local / "complete.json", {"job_id": job, "suite_sha256": key})
                return {"id": key, "directory": str(local)}
            time.sleep(10)
        raise TimeoutError("reference allocation/compute exceeded three hours")
    except BaseException:
        ssh("student_lab", "scancel " + job)
        raise


@ChiaFunction(resources={"tt_evaluator": 1}, max_retries=0)
def device_node(request, run_dir, trial_id):
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    remote_harness = manifest.get("remote_harness", TT_ROOT + "/harness")
    result = json.loads(ssh("tt_box",
        "/home/abror/portforge-tt/tt-metal/python_env/bin/python " + remote_harness + "/remote.py",
        json.dumps(request).encode(), timeout=13000))
    write_json(Path(run_dir) / "trials" / trial_id / "result.json", result)
    return result


@ChiaFunction(resources={"evaluation_coordinator": 1}, max_retries=0)
def evaluation_node(files, suite, run_dir, trial_id):
    reference = get(reference_node.chia_remote(suite, run_dir))
    result = get(device_node.chia_remote({"kind": "evaluate", "files": files,
                                         "reference_id": reference["id"]}, run_dir, trial_id))
    if suite["stage"] in ("qualification", "full", "final") and result.get("evaluation", {}).get("numerical_behavior_passed"):
        result["evaluation"] = get(quality_node.chia_remote(result["job"], reference,
                                                            run_dir, trial_id))
    write_json(Path(run_dir) / "trials" / trial_id / "result.json", result)
    return result


@ChiaFunction(resources={"reference_submitter": 1}, max_retries=0)
def quality_node(tt_job, reference, run_dir, trial_id):
    """Use existing pinned SacreBLEU/tokenizer on a Slurm compute node, never login."""
    local = Path(reference["directory"])
    files = {name: (local / name).read_bytes() for name in ("suite.json", "expected.npz", "reference.json")}
    for name in ("actual.npz", "measurements.json"):
        files["output/" + name] = ssh("tt_box", f"cat {TT_ROOT}/jobs/{tt_job}/output/{name}", timeout=300)
    files["assess.py"] = (harness_root(run_dir) / "assess.py").read_text()
    files["score.py"] = 'import json\nfrom pathlib import Path\nfrom assess import assess\nPath("assessment.json").write_text(json.dumps(assess(".", include_quality=True)))\n'
    files["score.sbatch"] = '''#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=portforge-auto-score
#SBATCH --output=slurm.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
trap 'rc=$?; printf "%s\\n" "$rc" > exit-code.txt' EXIT
export HF_HOME="$HOME/portforge-reference/.cache/huggingface"
export PYTHONPATH="$HOME/portforge-reference/qualification/deps:$PWD"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
"$HOME/portforge-reference/.venv/bin/python" score.py
'''
    home = ssh("student_lab", "printf '%s' \"$HOME\"").decode()
    remote = home + "/portforge-autonomous/scores/" + uuid.uuid4().hex
    upload("student_lab", remote, files)
    job = ssh("student_lab", f"cd {shlex.quote(remote)} && sbatch --parsable score.sbatch").decode().strip()
    if not job.isdigit():
        raise RuntimeError("invalid scorer job ID")
    try:
        deadline = time.monotonic() + 5400
        while time.monotonic() < deadline:
            state = ssh("student_lab", f"if test -f {shlex.quote(remote + '/exit-code.txt')}; then cat {shlex.quote(remote + '/exit-code.txt')}; else echo pending; fi").decode().strip()
            if state == "0":
                return json.loads(ssh("student_lab", "cat " + shlex.quote(remote + "/assessment.json"), timeout=300))
            if state != "pending":
                raise RuntimeError(ssh("student_lab", "tail -c 5000 " + shlex.quote(remote + "/slurm.log")).decode())
            time.sleep(10)
        raise TimeoutError("quality job allocation exceeded budget")
    except BaseException:
        ssh("student_lab", "scancel " + job)
        raise


class PortingTools(ChiaTool):
    def setup(self, workspace, run_dir, corpus, heldout_ids):
        self.workspace, self.run_dir = Path(workspace), Path(run_dir)
        self.corpus, self.heldout_ids = corpus, heldout_ids
        self.jobs = restore_jobs(self.run_dir)
        self.final_requested = False
        for name in ("files", "read", "write", "run", "evaluate", "poll", "status", "submit"):
            self.mcp.add_tool(getattr(self, name), name="portforge_" + name)

    def record(self, event):
        with (self.run_dir / "events.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), **event}) + "\n")

    def status(self) -> dict:
        """Recover job IDs and trial kinds across agent rounds without repeating work."""
        return {job: {"kind": state["kind"], "source_sha256": state["source_sha256"],
                      "collected": "result" in state} for job, state in self.jobs.items()}

    def files(self) -> list[str]:
        """List your source files. No other workspace is accessible."""
        self.record({"event": "files"})
        return list(snapshot(self.workspace))

    def read(self, path: str, start_line: int = 1, max_lines: int = 250) -> str:
        """Read one of your source/instruction files with bounded output."""
        try:
            target = safe_path(self.workspace, path)
        except ValueError:
            self.record({"event": "read_rejected", "path": path})
            raise
        self.record({"event": "read", "path": path})
        lines = target.read_text().splitlines()
        return "\n".join(lines[max(0, start_line - 1):max(0, start_line - 1) + min(max_lines, 500)])

    def write(self, path: str, content: str) -> dict:
        """Create/replace a UTF-8 candidate file. Evaluation policy is outside this workspace."""
        if self.final_requested:
            raise ValueError("source frozen after final submission")
        if Path(path).as_posix() in ("AGENTS.md", "TASK.md", "CONTRACT.md", "TT_PORTING_GUIDE.md"):
            raise ValueError("seed instructions are read-only")
        if len(content.encode()) > 2_000_000:
            raise ValueError("file exceeds limit")
        target = safe_path(self.workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        version = digest(content)
        archive = self.run_dir / "edits" / version
        archive.parent.mkdir(exist_ok=True)
        archive.write_text(content)
        self.record({"event": "write", "path": path, "sha256": version})
        return {"path": path, "characters": len(content)}

    def start(self, kind, ref, files):
        job = uuid.uuid4().hex
        write_json(self.run_dir / "trials" / job / "source.json", files)
        self.jobs[job] = {"ref": ref(job), "kind": kind, "source_sha256": digest(files)}
        self.record({"event": "started", "job": job, "kind": kind, "source_sha256": digest(files)})
        return {"job_id": job, "kind": kind, "source_sha256": digest(files), "status": "running"}

    def run(self, command: str, timeout_seconds: int = 120) -> dict:
        """Run a shell command in the isolated Blackhole sandbox. No network, credentials or
        other ports. Public TT source=/opt/tt-metal; weights=/weights/pytorch_model.bin.
        Python/torch/ttnn available. Your source is copied to /work; shell edits are
        ephemeral, so persist code with write. Returns an async job ID; poll for output.
        """
        files = snapshot(self.workspace)
        return self.start("run", lambda job: device_node.chia_remote(
            {"kind": "run", "files": files, "command": command, "timeout": timeout_seconds},
            str(self.run_dir), job), files)

    def evaluate(self, stage: str = "smoke") -> dict:
        """Freeze your current source and evaluate on NVIDIA-reference inputs. Stages:
        smoke, development, qualification, full. Results include failures, timings,
        memory and (broad stages) chrF++. First reference may queue in Slurm.
        """
        if stage not in ("smoke", "development", "qualification", "full"):
            raise ValueError("use submit for final")
        files = snapshot(self.workspace)
        if "backend.py" not in files:
            raise ValueError("implement backend.py before evaluation")
        suite = make_suite(self.corpus, stage, self.heldout_ids)
        return self.start(stage, lambda job: evaluation_node.chia_remote(files, suite,
                          str(self.run_dir), job), files)

    def poll(self, job_id: str) -> dict:
        """Collect a job without blocking. Repeated polling does not rerun an experiment."""
        job = self.jobs[job_id]
        if "result" not in job:
            ready, _ = ray.wait([job["ref"]], timeout=0)
            if not ready:
                return {"job_id": job_id, "status": "running"}
            try:
                result = get(job["ref"])
            except Exception as exc:
                result = {"returncode": -1, "error": str(exc)[-5000:]}
            write_json(self.run_dir / "trials" / job_id / "result.json", result)
            job["result"] = result
            self.record({"event": "finished", "job": job_id, "kind": job["kind"],
                         "source_sha256": job["source_sha256"],
                         "passed": result.get("evaluation", {}).get("passed", False)})
            if job["kind"] == "development" and result.get("evaluation", {}).get("passed"):
                baseline = self.run_dir / "baseline.json"
                if not baseline.exists():
                    write_json(baseline, {"job_id": job_id, "source_sha256": job["source_sha256"]})
        result = job["result"]
        # Full artifacts stay in the trusted run directory; bound prompt size.
        evaluation = result.get("evaluation")
        if evaluation:
            result = {k: v for k, v in result.items() if k != "evaluation"}
            result["evaluation"] = {k: v for k, v in evaluation.items()
                                    if k not in ("measurements", "cuda_reference", "checks")}
            result["evaluation"]["checks"] = evaluation["checks"][:30]
            result["evaluation"]["check_count"] = len(evaluation["checks"])
            result["evaluation"]["load_seconds"] = evaluation["measurements"]["load_seconds"]
            result["evaluation"]["resident_memory"] = evaluation["measurements"]["resident_memory"]
            result["evaluation"]["host_peak_rss_bytes"] = evaluation["measurements"]["host_peak_rss_bytes"]
            reference = {x["name"]: x for x in evaluation["cuda_reference"]["cases"]}
            result["evaluation"]["cuda_gpu"] = evaluation["cuda_reference"]["gpu"]
            result["evaluation"]["cuda_load_seconds"] = evaluation["cuda_reference"]["load_seconds"]
            import statistics
            result["evaluation"]["cuda_comparison"] = [
                {"name": c["name"], "cuda_seconds": statistics.median(reference[c["name"]]["seconds"]),
                 "tt_seconds": c["p50_seconds"],
                 "cuda_peak_allocated_bytes": reference[c["name"]]["peak_allocated_bytes"]}
                for c in evaluation["checks"][:30] if "p50_seconds" in c]
            baseline_file = self.run_dir / "baseline.json"
            if baseline_file.exists():
                baseline_id = json.loads(baseline_file.read_text())["job_id"]
                baseline_result = json.loads((self.run_dir / "trials" / baseline_id / "result.json").read_text())
                base = {x["name"]: x for x in baseline_result["evaluation"]["checks"]}
                comparison = [{"name": c["name"], "speedup": base[c["name"]]["p50_seconds"] / c["p50_seconds"],
                               "identical_outputs": base[c["name"]]["tokens_sha256"] == c["tokens_sha256"],
                               "baseline_tokens": base[c["name"]]["generated_tokens"],
                               "current_tokens": c["generated_tokens"]}
                              for c in evaluation["checks"] if c["name"] in base and "p50_seconds" in c]
                write_json(self.run_dir / "trials" / job_id / "comparison.json", comparison)
                result["evaluation"]["tt_baseline_comparison"] = comparison[:30]
        return {"job_id": job_id, "status": "complete", **result}

    def submit(self) -> dict:
        """Freeze final source after qualification and full pass. Controller evaluates final
        independently, without exposing held-out feedback for further optimization."""
        files = snapshot(self.workspace)
        if "REPORT.md" not in files:
            raise ValueError("write REPORT.md first")
        source = digest(files)
        # Documentation edits may change the snapshot: compare executable source only.
        code = {k: v for k, v in files.items() if not k.endswith(".md")}
        for stage in ("qualification", "full"):
            candidates = [j for j, state in self.jobs.items() if state["kind"] == stage]
            passed = False
            for j in candidates:
                self.poll(j)
                state = self.jobs[j]
                prior = json.loads((self.run_dir / "trials" / j / "source.json").read_text())
                if ({k: v for k, v in prior.items() if not k.endswith(".md")} == code
                        and state.get("result", {}).get("evaluation", {}).get("passed")):
                    passed = True
            if not passed:
                raise ValueError(f"current executable source needs a passing {stage} evaluation")
        write_json(self.run_dir / "submission.json", {"source_sha256": source, "files": files})
        self.final_requested = True
        return {"status": "submitted", "source_sha256": source,
                "message": "source frozen; controller will run independent final validation"}


class IsolatedCodex(CodexLLM):
    """Use a restricted named profile instead of the provider's legacy sandbox flag."""
    def _build_cmd(self, *args, **kwargs):
        command = super()._build_cmd(*args, **kwargs)
        index = command.index("--sandbox")
        del command[index:index + 2]
        return command

    def _mcp_config_args(self, tools):
        args = super()._mcp_config_args(tools)
        for tool in tools:
            # User explicitly authorized unattended operation of these scoped
            # tools. This is per-run configuration, not a global policy change.
            args += ["-c", f'mcp_servers.{tool.name}.default_tools_approval_mode="approve"',
                     "-c", f"mcp_servers.{tool.name}.required=true"]
        return args


def codex_options(workspace):
    filesystem = {":minimal": "read", ":workspace_roots": "read"}
    for path in (str(HERE.parents[1]), str(Path.home() / ".codex"), str(Path.home() / ".ssh")):
        filesystem[path] = "deny"
    fs_toml = "{" + ",".join(json.dumps(k) + "=" + json.dumps(v) for k, v in filesystem.items()) + "}"
    options = ["--ignore-user-config", "-c", 'default_permissions="portforge"',
               "-c", "permissions.portforge.filesystem=" + fs_toml,
               "-c", "permissions.portforge.network.enabled=false", "-c", 'web_search="live"',
               "-c", "project_doc_max_bytes=0", "-c", 'history.persistence="none"',
               "-c", "memories.use_memories=false", "-c", "memories.generate_memories=false",
               "-c", "agents.enabled=false",
               "-c", "features.skip_host_skill_discovery=true"]
    for name in ("shell_tool", "unified_exec", "view_image", "apps", "plugins", "remote_plugin",
                 "browser_use", "computer_use", "hooks", "memories", "shell_snapshot",
                 "image_generation", "multi_agent", "multi_agent_v2", "skill_search"):
        options += ["-c", f"features.{name}=false"]
    return options


@ChiaFunction(resources={"codex_creds": 1}, max_retries=0)
def agent_node(workspace, run_dir, tools, round_number, prompt):
    agent = IsolatedCodex(model="gpt-6-astra", work_dir=workspace, timeout_seconds=1800,
        reasoning_effort="high", retries=1, sandbox="read-only", approval_policy="never",
        dangerously_bypass_approvals_and_sandbox=False, ephemeral=True,
        log_dir=str(Path(run_dir) / "agent-logs" / str(round_number)),
        extra_cli_args=codex_options(workspace))
    instructions = (Path(workspace) / "AGENTS.md").read_text()
    result = agent.prompt(instructions + "\n\n" + prompt, tools=tools)
    record = {"round": round_number, "success": result.success, "response": result.result,
              "error": getattr(result, "stderr", "")[-5000:] if not result.success else "",
              "usage": agent._last_metadata}
    write_json(Path(run_dir) / "rounds" / f"{round_number:03d}.json", record)
    return record
