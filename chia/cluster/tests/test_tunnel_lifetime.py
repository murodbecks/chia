import os
import signal
import subprocess
import sys
import time

import pytest

from chia.cluster import tunnel
from chia.cluster.config import SSHAuthConfig, TunnelConfig


def test_tunnel_survives_launcher_exit_and_can_write_diagnostics(tmp_path):
    ssh = tmp_path / "ssh"
    ssh.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "time.sleep(0.3)\n"
        "print('still connected', file=sys.stderr, flush=True)\n"
        "time.sleep(30)\n"
    )
    ssh.chmod(0o700)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    script = """
import sys
from chia.cluster import tunnel
from chia.cluster.config import SSHAuthConfig, TunnelConfig
tunnel._PID_DIR = sys.argv[1]
manager = tunnel.TunnelManager()
manager.start_tunnel('127.0.0.2', '192.0.2.1', SSHAuthConfig(ssh_user='test'),
    TunnelConfig(tunnel_ip='127.0.0.2', kill_orphaned_tunnels=False))
"""
    pid = None
    try:
        subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                       env=env, capture_output=True, check=True, timeout=10)
        pid = int((tmp_path / "chia_tunnel_127-0-0-2.pid").read_text())
        log = tmp_path / "chia_tunnel_127-0-0-2.pid.log"
        deadline = time.monotonic() + 5
        while "still connected" not in log.read_text() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert "still connected" in log.read_text()
        assert os.getsid(pid) == pid
        os.kill(pid, 0)
    finally:
        if pid is not None:
            os.kill(pid, signal.SIGTERM)


def test_failed_tunnel_reports_file_backed_diagnostics(tmp_path, monkeypatch):
    ssh = tmp_path / "ssh"
    ssh.write_text("#!/bin/sh\necho 'forward collision' >&2\nexit 23\n")
    ssh.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setattr(tunnel, "_PID_DIR", str(tmp_path))
    manager = tunnel.TunnelManager()
    manager.start_tunnel('127.0.0.2', '192.0.2.1', SSHAuthConfig(ssh_user='test'),
                         TunnelConfig(tunnel_ip='127.0.0.2', kill_orphaned_tunnels=False))
    with pytest.raises(RuntimeError, match="forward collision"):
        manager.wait_for_tunnel('127.0.0.2')
    manager.stop_all()
