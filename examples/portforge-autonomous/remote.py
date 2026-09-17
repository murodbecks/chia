"""Trusted TT host dispatcher; no secrets or earlier solution paths enter the sandbox."""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

ROOT = Path.home() / "portforge-autonomous"
METAL = Path.home() / "portforge-tt/tt-metal"
PYTHON_ENV = Path.home() / "miniconda3/envs/chia_env"
WEIGHTS = Path.home() / "portforge-nllb/cache/f8d333a098d19b4fd9a8b18f94170487ad3f821d"
def discover_cards():
    cards = []
    for node in Path("/dev/tenstorrent").iterdir():
        if node.name.isdigit():
            pci = (Path("/sys/class/tenstorrent") / ("tenstorrent!" + node.name) / "device").resolve().name
            if pci not in ("0000:01:00.0", "0000:41:00.0", "0000:42:00.0", "0000:c1:00.0"):
                raise ValueError("unexpected physical device")
            cards.append((pci, node.name))
    return sorted(cards)


def workspace_file(root, name):
    p = Path(name)
    if p.is_absolute() or not p.parts or ".." in p.parts or "\\" in name:
        raise ValueError("invalid candidate path")
    target = root / p
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def sandbox_args(directory, pci, node):
    directory = Path(directory)
    argv = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--clearenv",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin", "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
            "--ro-bind", "/sys", "/sys", "--tmpfs", "/tmp",
            "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--ro-bind", str(METAL), str(METAL),
            "--ro-bind", str(METAL), "/opt/tt-metal",
            "--ro-bind", str(PYTHON_ENV), str(PYTHON_ENV),
            "--ro-bind", str(WEIGHTS), "/weights",
            "--bind", str(directory / "work"), "/work",
            "--ro-bind", str(directory / "inputs"), "/inputs",
            "--bind", str(directory / "output"), "/output",
            "--ro-bind", str(Path(__file__).resolve().parent / "candidate_worker.py"), "/runner.py",
            "--dev-bind", "/dev/tenstorrent/" + node, "/dev/tenstorrent/" + node]
    # Hide prior generated artifacts and give this job its own generated outputs.
    for alias in (str(METAL), "/opt/tt-metal"):
        argv += ["--tmpfs", alias + "/generated", "--tmpfs", alias + "/.git"]
    for config in ("/etc/hosts", "/etc/fonts"):
        if Path(config).exists():
            argv += ["--ro-bind", config, config]
    for huge in ("/dev/hugepages", "/dev/hugepages-1G"):
        if Path(huge).exists():
            argv += ["--bind", huge, huge]
    for runtime in ("/opt/openmpi-v5.0.7-ulfm", "/opt/tenstorrent"):
        if Path(runtime).exists():
            argv += ["--ro-bind", runtime, runtime]
    env = {"PATH": str(METAL / "python_env/bin") + ":/usr/bin:/bin",
           "HOME": "/work", "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": "/work:" + str(METAL), "TT_METAL_HOME": str(METAL),
           "TT_METAL_CACHE_DIR": "/work/.tt-cache", "TT_VISIBLE_DEVICES": pci,
           "OMP_NUM_THREADS": "12", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    for k, v in env.items():
        argv += ["--setenv", k, v]
    return argv + ["--chdir", "/work"]


def main():
    request = json.load(sys.stdin)
    directory = ROOT / "jobs" / uuid.uuid4().hex
    for sub in ("work", "inputs", "output"):
        (directory / sub).mkdir(parents=True)
    for name, value in request["files"].items():
        workspace_file(directory / "work", name).write_text(value)
    if request["kind"] == "evaluate":
        reference = ROOT / "references" / request["reference_id"]
        if len(request["reference_id"]) != 64 or any(c not in "0123456789abcdef" for c in request["reference_id"]):
            raise ValueError("invalid reference identifier")
        for name in ("suite.json", "expected.npz", "reference.json"):
            (directory / name).write_bytes((reference / name).read_bytes())
        suite = json.loads((directory / "suite.json").read_text())
        # Candidate sees only the request contract, never text labels/reference outputs.
        public_suite = {k: v for k, v in suite.items() if k != "gates"}
        public_suite["cases"] = [{k: v for k, v in c.items() if k not in ("references", "texts")}
                                 for c in suite["cases"]]
        (directory / "inputs/suite.json").write_text(json.dumps(public_suite))
        for name in ("inputs.npz", "config.json"):
            (directory / "inputs" / name).write_bytes((reference / name).read_bytes())
        command = [str(METAL / "python_env/bin/python"), "/runner.py"]
        timeout = 10800 if suite["stage"] == "full" else 3600
    else:
        command = ["/bin/bash", "--noprofile", "--norc", "-c", request["command"]]
        timeout = min(max(int(request.get("timeout", 120)), 1), 600)
    lease = None
    deadline = time.monotonic() + 1800
    while lease is None and time.monotonic() < deadline:
        for pci, node in discover_cards():
            lock = open(Path.home() / "portforge-nllb" / ("device-" + pci.replace(":", "_") + ".lock"), "a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lease = lock
                break
            except BlockingIOError:
                lock.close()
        if lease is None:
            time.sleep(2)
    if lease is None:
        raise TimeoutError("no free Blackhole card")
    try:
        argv = ["timeout", "--signal=TERM", "--kill-after=10s", str(timeout)] + sandbox_args(directory, pci, node) + command
        with open(directory / "stdout.log", "wb") as stdout, open(directory / "stderr.log", "wb") as stderr:
            proc = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=timeout + 30)
        result = {"returncode": proc.returncode, "pci": pci, "job": directory.name,
                  "stdout": (directory / "stdout.log").read_text(errors="replace")[-16000:],
                  "stderr": (directory / "stderr.log").read_text(errors="replace")[-16000:]}
        if request["kind"] == "evaluate" and proc.returncode == 0:
            from assess import assess
            result["evaluation"] = assess(directory)
            (directory / "assessment.json").write_text(json.dumps(result["evaluation"]))
        print(json.dumps(result))
    finally:
        lease.close()


if __name__ == "__main__":
    main()
