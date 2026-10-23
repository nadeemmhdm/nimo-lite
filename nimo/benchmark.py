"""Reproducible local inference benchmark for Nimo's own randomly initialized model.

The benchmark intentionally downloads no model weights. It measures kernel/runtime speed,
not language quality. Run on the same hardware before and after performance changes.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import torch

from .models import build_text_model


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / 1024.0 if platform.system() != "Darwin" else value / (1024.0 * 1024.0))


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _uncached(model, prompt, steps: int, device: str) -> dict:
    sequence = prompt
    _sync(device); start = time.perf_counter(); first = None
    with torch.inference_mode():
        for i in range(steps):
            logits = model(sequence)
            token = logits[:, -1:].argmax(dim=-1)
            if i == 0:
                _sync(device); first = time.perf_counter() - start
            sequence = torch.cat((sequence, token), dim=1)
    _sync(device); elapsed = time.perf_counter() - start
    return {"first_token_s": first, "decode_tokens_per_s": steps / elapsed, "elapsed_s": elapsed}


def _cached(model, prompt, steps: int, device: str) -> dict:
    _sync(device); prefill_start = time.perf_counter()
    with torch.inference_mode():
        logits, cache = model(prompt, use_cache=True)
        token = logits[:, -1:].argmax(dim=-1)
        _sync(device); first = time.perf_counter() - prefill_start
        decode_start = time.perf_counter()
        for _ in range(max(0, steps - 1)):
            logits, cache = model(token, cache=cache, use_cache=True)
            token = logits[:, -1:].argmax(dim=-1)
        _sync(device)
    decode_elapsed = time.perf_counter() - decode_start
    return {
        "first_token_s": first,
        "prefill_tokens_per_s": prompt.numel() / max(first, 1e-9),
        "decode_tokens_per_s": max(0, steps - 1) / max(decode_elapsed, 1e-9),
        "elapsed_s": first + decode_elapsed,
    }


def run(profile="small", context=1024, prompt_tokens=128, new_tokens=24, repeats=2, threads=None) -> dict:
    device = _device()
    if threads and device == "cpu":
        torch.set_num_threads(max(1, int(threads)))
    torch.manual_seed(7)
    load_start = time.perf_counter()
    model = build_text_model(profile=profile, context=context, vocab_size=8000).to(device).eval()
    _sync(device)
    load_s = time.perf_counter() - load_start
    prompt = torch.randint(8, 7999, (1, min(prompt_tokens, context - new_tokens - 1)), device=device)
    with torch.inference_mode():
        model(prompt[:, : min(8, prompt.shape[1])], use_cache=True)
    cached = [_cached(model, prompt, new_tokens, device) for _ in range(repeats)]
    uncached = [_uncached(model, prompt, new_tokens, device) for _ in range(repeats)]
    def median(key, rows): return statistics.median(float(x[key]) for x in rows)
    result = {
        "schema": 1,
        "note": "Runtime/kernel benchmark with Nimo random weights; not a language-quality benchmark.",
        "hardware": {
            "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
        "model": {"profile": profile, "parameters": model.parameter_count(), "context": context, "vocab_size": 8000},
        "workload": {"prompt_tokens": int(prompt.shape[1]), "new_tokens": new_tokens, "repeats": repeats},
        "model_load_s": load_s,
        "cached": {k: median(k, cached) for k in cached[0]},
        "uncached": {k: median(k, uncached) for k in uncached[0]},
        "peak_rss_mb": _rss_mb(),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark Nimo local inference")
    p.add_argument("--profile", default="small", choices=("micro", "tiny", "small", "medium"))
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--new-tokens", type=int, default=24)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--threads", type=int)
    p.add_argument("--output", default="benchmark-results.json")
    args = p.parse_args()
    result = run(args.profile, args.context, args.prompt_tokens, args.new_tokens, args.repeats, args.threads)
    text = json.dumps(result, indent=2, sort_keys=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
