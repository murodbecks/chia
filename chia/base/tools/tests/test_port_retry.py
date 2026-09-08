import importlib
import socket
import time


def test_occupied_port_is_retried_without_waiting_for_startup_deadline():
    module=importlib.import_module('chia.base.tools.ChiaTool')
    tool=module.ChiaTool('port_retry_check')
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1',0))
        occupied.listen()
        port=occupied.getsockname()[1]
        # Confirm the adjacent port is available before using a two-port range.
        with socket.socket() as adjacent:
            adjacent.bind(('127.0.0.1',port+1))
        start=time.monotonic()
        try:
            actual=module.start_router(tool,'127.0.0.1',base_port=port,max_tries=2)
            assert actual==port+1
            assert time.monotonic()-start<3
        finally:
            module.stop_router(tool.name)
