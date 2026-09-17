"""Trusted measurement wrapper, mounted read-only inside the TT sandbox.

Candidate Python shares a process with this wrapper: isolation protects the host
and reference labels, not against a deliberately malicious Python monkeypatch.
Source review and operation evidence remain required for production acceptance.
"""
import importlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time


def memory_snapshot(ttnn, device):
    """Allocator accounting, including bank multiplicity; snapshots are not peaks."""
    views = {}
    fields = ("num_banks", "total_bytes_allocated_per_bank", "total_bytes_free_per_bank",
              "largest_contiguous_bytes_free_per_bank", "total_bytes_per_bank")
    for kind in ("DRAM", "L1", "TRACE"):
        try:
            view = ttnn.device.get_memory_view(device, getattr(ttnn.BufferType, kind))
            values = {key: int(getattr(view, key)) for key in fields if hasattr(view, key)}
            if "num_banks" in values and "total_bytes_allocated_per_bank" in values:
                values["total_allocated_bytes"] = values["num_banks"] * values["total_bytes_allocated_per_bank"]
            views[kind] = values
        except Exception as exc:
            views[kind] = {"unavailable": str(exc)[:200]}
    return views


def main():
    import numpy as np
    import torch
    import ttnn
    from torch.utils._python_dispatch import TorchDispatchMode

    class RejectCPUCompute(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            forbidden = ("aten.mm.", "aten.bmm.", "aten.addmm.", "aten.matmul.",
                         "aten.linear.", "aten.embedding.", "aten.native_layer_norm.",
                         "aten.layer_norm.", "aten._softmax.", "aten.softmax.",
                         "aten.gelu.", "aten.scaled_dot_product")
            if name.startswith(forbidden):
                raise RuntimeError("CPU learned-compute fallback forbidden: " + name)
            return func(*args, **(kwargs or {}))

    # Lightweight call evidence is a fallback detector, not a hardware profiler
    # or a proof against a deliberately malicious Python implementation.
    op_counts = {}
    for op_name in ("matmul", "linear", "embedding", "layer_norm", "rms_norm", "execute_trace"):
        original = getattr(ttnn, op_name, None)
        if original is not None:
            def traced(*args, _name=op_name, _op=original, **kwargs):
                op_counts[_name] = op_counts.get(_name, 0) + 1
                return _op(*args, **kwargs)
            setattr(ttnn, op_name, traced)
    spec = json.loads(Path("/inputs/suite.json").read_text())
    config = json.loads(Path("/inputs/config.json").read_text())
    inputs = np.load("/inputs/inputs.npz", allow_pickle=False)
    sys.path.insert(0, "/work")
    backend_module = importlib.import_module("backend")
    options = getattr(backend_module, "DEVICE_OPTIONS", {})
    bounds = {"trace_region_size": (0, 128 * 1024 * 1024), "l1_small_size": (0, 128 * 1024),
              "num_hw_cqs": (1, 2)}
    if not isinstance(options, dict) or set(options) - set(bounds):
        raise ValueError("unsupported DEVICE_OPTIONS")
    if any(type(v) is not int or not bounds[k][0] <= v <= bounds[k][1] for k, v in options.items()):
        raise ValueError("DEVICE_OPTIONS outside resource limits")
    device = ttnn.open_device(device_id=0, **options)
    result, outputs = {"cases": [], "ttnn": importlib.metadata.version("ttnn")}, {}
    result["device_options"] = options

    try:
        start = time.perf_counter()
        backend = backend_module.create_backend("/weights/pytorch_model.bin", config, device)
        ttnn.synchronize_device(device)
        result["load_seconds"] = time.perf_counter() - start
        result["resident_memory"] = memory_snapshot(ttnn, device)
        banks = result["resident_memory"].get("DRAM", {}).get("num_banks")
        result["trace_reserved_bytes"] = banks * options.get("trace_region_size", 0) if banks else None
        for case in spec["cases"]:
            name = case["name"]
            ids = inputs[name + "__input_ids"]
            mask = inputs[name + "__attention_mask"]
            prefix = inputs[name + "__decoder_input_ids"]
            if case["numerical"]:
                with RejectCPUCompute():
                    forward = backend.forward(ids.copy(), mask.copy(), prefix.copy())
                ttnn.synchronize_device(device)
                outputs[name + "__encoder"] = np.asarray(forward["encoder"], dtype=np.float32)
                logits = np.asarray(forward["logits"], dtype=np.float32)
                if logits.ndim != 3 or logits.shape[:2] != prefix.shape:
                    raise ValueError("forward logits must have shape [B,T,V]")
                outputs[name + "__logits"] = logits[:, -1]
            samples = []
            first = None
            repeats = 5 if case["numerical"] else 1
            for repeat in range(repeats + 1):
                ttnn.synchronize_device(device)
                start = time.perf_counter()
                before_ops = sum(op_counts.values())
                with RejectCPUCompute():
                    tokens = np.asarray(backend.generate(ids.copy(), mask.copy(), case["target_id"],
                                                         case["max_new_tokens"]))
                ttnn.synchronize_device(device)
                elapsed = time.perf_counter() - start
                if sum(op_counts.values()) == before_ops:
                    raise RuntimeError("generation produced no observed TT compute/trace calls")
                if tokens.dtype.kind not in "iu":
                    raise ValueError("generation must return integers")
                if first is None:
                    first = tokens.copy()
                    cold = elapsed
                elif not np.array_equal(first, tokens):
                    raise ValueError("candidate nondeterminism across repeated calls")
                else:
                    samples.append(elapsed)
            outputs[name + "__tokens"] = first
            result["cases"].append(dict(name=name, seconds=samples, first_seconds=cold,
                                         memory=memory_snapshot(ttnn, device)))
            print(json.dumps({"completed": name}), flush=True)
        np.savez_compressed("/output/actual.npz", **outputs)
        result["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result["observed_tt_calls"] = op_counts
        result["bandwidth"] = {"measured": False, "reason": "no hardware counter capture in wrapper"}
        Path("/output/measurements.json").write_text(json.dumps(result))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
