"""Trusted FP32 CUDA baseline. Must run under Slurm on a compute node."""
import json
import os
from pathlib import Path
import resource
import socket
import time


def main():
    if not os.getenv("SLURM_JOB_ID") or socket.gethostname().startswith(("login", "lo-")):
        raise RuntimeError("reference requires a Slurm compute allocation")
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    spec = json.loads(Path("suite.json").read_text())
    torch.set_num_threads(12)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    started = time.perf_counter()
    model = AutoModelForSeq2SeqLM.from_pretrained(spec["model"], revision=spec["revision"],
        dtype=torch.float32, attn_implementation="eager", trust_remote_code=False,
        weights_only=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"], trust_remote_code=False)
    model.config.to_json_file("config.json")
    meta = dict(load_seconds=time.perf_counter() - started, gpu=torch.cuda.get_device_name(0),
                hostname=socket.gethostname(), job_id=os.environ["SLURM_JOB_ID"],
                torch=torch.__version__, transformers=transformers.__version__, cases=[])
    inputs, expected = {}, {}
    with torch.inference_mode():
        for case in spec["cases"]:
            name = case["name"]
            tokenizer.src_lang = case["src"]
            values = tokenizer(case["texts"], padding=True, truncation=True,
                               max_length=256, return_tensors="pt")
            if "token_length" in case:
                values = {k: v[:, :case["token_length"]].clone() for k, v in values.items()}
                values["input_ids"][:, -1] = 2
            if case.get("extra_padding"):
                # Stay inside the 256-token support envelope.
                pad = min(case["extra_padding"], 256 - values["input_ids"].shape[1])
                values = {k: torch.nn.functional.pad(v, (0, pad), value=1 if k == "input_ids" else 0)
                          for k, v in values.items()}
            target = tokenizer.convert_tokens_to_ids(case["tgt"])
            prefix = torch.tensor([[2, target]] * len(case["texts"]))
            for key, value in {**values, "decoder_input_ids": prefix}.items():
                inputs[name + "__" + key] = value.numpy()
            case["target_id"] = target
            gpu_values = {k: v.cuda() for k, v in values.items()}
            if case["numerical"]:
                output = model(**gpu_values, decoder_input_ids=prefix.cuda(), use_cache=False)
                expected[name + "__encoder"] = output.encoder_last_hidden_state.float().cpu().numpy()
                expected[name + "__logits"] = output.logits[:, -1].float().cpu().numpy()
            kwargs = dict(forced_bos_token_id=target, max_new_tokens=case["max_new_tokens"],
                          do_sample=False, num_beams=1, use_cache=True)
            generated = model.generate(**gpu_values, **kwargs)
            torch.cuda.synchronize()
            samples = []
            torch.cuda.reset_peak_memory_stats()
            # Timed region includes input H2D, complete generation and output D2H.
            repeats = 5 if case["numerical"] else 1
            for _ in range(repeats):
                t0 = time.perf_counter()
                result = model.generate(**{k: v.cuda() for k, v in values.items()}, **kwargs).cpu()
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - t0)
                if not torch.equal(result, generated.cpu()):
                    raise RuntimeError("CUDA reference nondeterminism")
            expected[name + "__tokens"] = generated.cpu().numpy()
            meta["cases"].append(dict(name=name, seconds=samples,
                translations=tokenizer.batch_decode(generated, skip_special_tokens=True),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            print(json.dumps({"completed": name}), flush=True)
    np.savez_compressed("inputs.npz", **inputs)
    np.savez_compressed("expected.npz", **expected)
    Path("prepared-suite.json").write_text(json.dumps(spec))
    meta["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    Path("reference.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
