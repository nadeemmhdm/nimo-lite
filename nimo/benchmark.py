"""Reproducible local inference benchmark for Nimo's own randomly initialized model.

The benchmark intentionally downloads no model weights. It measures tokenizer/runtime
speed, not language quality. Run on the same hardware before and after performance changes.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import psutil
import torch

from .fast_inference import get_engine
from .models import build_text_model, quantize_for_inference
from .tokenizer import train_tokenizer


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / 1024.0 if platform.system() != "Darwin" else value / (1024.0 * 1024.0))


def _current_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tokenizer_benchmark(repeats: int) -> dict:
    corpus = [
        "Nimo is a fully local generative assistant with a native tokenizer.",
        "നിമോ ഒരു ലോക്കൽ ജനറേറ്റീവ് അസിസ്റ്റന്റാണ്.",
        "Nimo fast ayi local reply generate cheyyunnu.",
        "def add(a, b):\n    return a + b\n",
    ] * 16
    tokenizer = train_tokenizer(corpus, vocab_size=1024)
    text = "\n".join(corpus) * 8
    encoded = tokenizer.encode(text)
    encode_samples = []
    decode_samples = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        ids = tokenizer.encode(text)
        encode_samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        tokenizer.decode(ids)
        decode_samples.append(time.perf_counter() - started)
    return {
        "vocab_size": tokenizer.vocab_size,
        "input_characters": len(text),
        "encoded_tokens": len(encoded),
        "encode_tokens_per_s": len(encoded) / max(statistics.median(encode_samples), 1e-9),
        "decode_tokens_per_s": len(encoded) / max(statistics.median(decode_samples), 1e-9),
    }


def _uncached(model, prompt, steps: int, device: str) -> dict:
    sequence = prompt
    _sync(device)
    start = time.perf_counter()
    first = None
    with torch.inference_mode():
        for index in range(steps):
            logits = model(sequence)
            token = logits[:, -1:].argmax(dim=-1)
            if index == 0:
                _sync(device)
                first = time.perf_counter() - start
            sequence = torch.cat((sequence, token), dim=1)
    _sync(device)
    elapsed = time.perf_counter() - start
    return {"first_token_s": first, "decode_tokens_per_s": steps / elapsed, "elapsed_s": elapsed}


def _cached(model, prompt, steps: int, device: str) -> dict:
    engine = get_engine(model)
    engine.reset()
    ids = prompt[0].tolist()
    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits, metadata = engine.prefill(ids)
        token = logits.argmax(dim=-1)
        _sync(device)
        first = time.perf_counter() - started
        decode_start = time.perf_counter()
        for _ in range(max(0, steps - 1)):
            logits = engine.decode(int(token.item()))
            token = logits.argmax(dim=-1)
        _sync(device)
    decode_elapsed = time.perf_counter() - decode_start
    return {
        "first_token_s": first,
        "prefill_tokens_per_s": len(ids) / max(first, 1e-9),
        "decode_tokens_per_s": max(0, steps - 1) / max(decode_elapsed, 1e-9),
        "elapsed_s": first + decode_elapsed,
        "kv_cache_mb": engine.session.cache.bytes / (1024 * 1024),
        "processed_prompt_tokens": metadata["processed_prompt_tokens"],
    }


def run(profile="small", context=1024, prompt_tokens=128, new_tokens=24, repeats=2, threads=None, quantization="none") -> dict:
    device = _device()
    if threads and device == "cpu":
        os.environ["NIMO_INFERENCE_THREADS"] = str(max(1, int(threads)))
        torch.set_num_threads(max(1, int(threads)))
    torch.manual_seed(7)
    load_start = time.perf_counter()
    model = build_text_model(profile=profile, context=context, vocab_size=8000).to(device).eval()
    original_parameters = model.parameter_count()
    selected_quantization = "fp32" if quantization == "none" else quantization
    if quantization != "none":
        model, selected_quantization = quantize_for_inference(model, quantization, device)
    engine = get_engine(model)
    _sync(device)
    gc.collect()
    load_s = time.perf_counter() - load_start
    steady_after_load_mb = _current_rss_mb()
    prompt = torch.randint(8, 7999, (1, min(prompt_tokens, context - new_tokens - 1)), device=device)
    with torch.inference_mode():
        engine.reset()
        engine.prefill(prompt[0, : min(8, prompt.shape[1])].tolist())
        engine.reset()
    cached = [_cached(model, prompt, new_tokens, device) for _ in range(repeats)]
    uncached = [_uncached(model, prompt, new_tokens, device) for _ in range(repeats)]

    def median(key, rows):
        return statistics.median(float(item[key]) for item in rows)

    gc.collect()
    result = {
        "schema": 4,
        "note": "Runtime/tokenizer benchmark with Nimo random weights; not a language-quality benchmark.",
        "hardware": {
            "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
        "model": {
            "profile": profile,
            "parameters": int(getattr(model, "_nimo_parameter_count", original_parameters)),
            "context": context,
            "vocab_size": 8000,
            "requested_quantization": quantization,
            "selected_quantization": selected_quantization,
            "quantization_decision": getattr(model, "_nimo_quantization_decision", None),
        },
        "workload": {"prompt_tokens": int(prompt.shape[1]), "new_tokens": new_tokens, "repeats": repeats},
        "model_load_s": load_s,
        "steady_rss_after_load_mb": steady_after_load_mb,
        "steady_rss_after_benchmark_mb": _current_rss_mb(),
        "tokenizer": _tokenizer_benchmark(repeats),
        "cached": {key: median(key, cached) for key in cached[0]},
        "uncached": {key: median(key, uncached) for key in uncached[0]},
        "peak_rss_mb": _peak_rss_mb(),
        "engine": "preallocated-kv-v1",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Nimo local inference")
    parser.add_argument("--profile", default="small", choices=("micro", "tiny", "small", "medium"))
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--quantization", default="none", choices=("none", "auto", "fp32", "bf16", "fp16", "int8"))
    parser.add_argument("--output", default="benchmark-results.json")
    args = parser.parse_args()
    result = run(args.profile, args.context, args.prompt_tokens, args.new_tokens, args.repeats, args.threads, args.quantization)
    text = json.dumps(result, indent=2, sort_keys=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
