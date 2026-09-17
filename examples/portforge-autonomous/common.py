"""Controller utilities. No candidate implementation is imported here."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def safe_path(root, relative):
    name = PurePosixPath(relative)
    if not relative or name.is_absolute() or ".." in name.parts or "\\" in relative:
        raise ValueError("expected a relative workspace path without traversal")
    root = Path(root).resolve()
    path = root.joinpath(*name.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("path escapes workspace")
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("symlinks are not allowed")
    return path


def snapshot(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("candidate symlink")
        if path.is_file() and not any(p.startswith(".") or p == "__pycache__" for p in path.parts):
            if path.suffix not in (".py", ".md", ".json", ".txt", ".cpp", ".hpp", ".h"):
                continue
            if path.stat().st_size > 2_000_000:
                raise ValueError("candidate file too large")
            result[relative] = path.read_text()
    if sum(len(x.encode()) for x in result.values()) > 8_000_000:
        raise ValueError("candidate snapshot too large")
    return result


def ssh(host, command, data=None, timeout=60):
    if host not in ("tt_box", "student_lab"):
        raise ValueError("unconfigured host")
    p = subprocess.run(["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=15", host, command], input=data,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{host}: exit {p.returncode}: {p.stderr.decode(errors='replace')[-5000:]}")
    return p.stdout


def upload(host, directory, files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in files.items():
            safe_path("/tmp/portforge-upload", name)
            payload = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(payload))
    ssh(host, f"mkdir -p {shlex.quote(directory)} && tar -xf - -C {shlex.quote(directory)}",
        buffer.getvalue(), timeout=180)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)
