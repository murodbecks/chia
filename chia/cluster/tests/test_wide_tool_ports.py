from types import SimpleNamespace

from chia.cluster.config import TunnelConfig
from chia.cluster.node_setup import allocate_worker_tunnels


def test_wide_tool_ranges_do_not_overlap_across_six_workers():
    config=SimpleNamespace(worker_ips=['203.0.113.1'], is_tunneled=lambda ip: True,
        get_tunnel_config=lambda ip: TunnelConfig(tool_port_min=19000,tool_port_max=19127,
            ray_worker_port_min=54000,ray_worker_port_max=54191))
    assignments=[SimpleNamespace(ip='203.0.113.1',node_type=SimpleNamespace(name=f'worker{i}'),worker_index=0) for i in range(6)]
    allocations=list(allocate_worker_tunnels(config,assignments).values())
    ports=[p for a in allocations for p in range(a.tool_port_min,a.tool_port_max+1)]
    assert len(ports)==len(set(ports))==6*128
    assert allocations[-1].ray_worker_port_max==59191


def test_small_tool_ranges_keep_existing_hundred_port_stride():
    config=SimpleNamespace(worker_ips=['203.0.113.1'], is_tunneled=lambda ip: True,
        get_tunnel_config=lambda ip: TunnelConfig())
    assignments=[SimpleNamespace(ip='203.0.113.1',node_type=SimpleNamespace(name=f'worker{i}'),worker_index=0) for i in range(2)]
    allocations=list(allocate_worker_tunnels(config,assignments).values())
    assert allocations[1].tool_port_min-allocations[0].tool_port_min==100
