import os
import platform
import shutil

import psutil


PARAMETER_ESTIMATES = {
    "tiny": 13_000_000,
    "small": 31_400_000,
    "medium": 88_000_000,
}


def _memory_limits():
    mem = psutil.virtual_memory()
    limits = []
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            raw = open(path).read().strip()
            if raw != "max":
                value = int(raw)
                if value > 0:
                    limits.append(value)
        except (OSError, ValueError):
            pass
    total = min([mem.total] + limits)
    available = min(mem.available, total)
    try:
        used = int(open("/sys/fs/cgroup/memory.current").read())
        available = min(available, max(0, total - used))
    except (OSError, ValueError):
        pass
    return total, available


def hardware(root="."):
    total, available = _memory_limits()
    gib = total / 2**30
    logical = os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    tier = "low" if gib < 10 else "standard" if gib < 24 else "performance"
    defaults = {
        "low": {
            "model_profile": "tiny",
            "context": 512,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "tokenizer_vocab": 8000,
        },
        "standard": {
            "model_profile": "small",
            "context": 1024,
            "batch_size": 2,
            "gradient_accumulation": 4,
            "tokenizer_vocab": 8000,
        },
        "performance": {
            "model_profile": "medium",
            "context": 2048,
            "batch_size": 4,
            "gradient_accumulation": 2,
            "tokenizer_vocab": 16000,
        },
    }[tier]
    override = os.environ.get("NIMO_INFERENCE_THREADS", "").strip()
    inference_threads = (
        max(1, int(override))
        if override.isdigit()
        else max(1, min(4, int(physical)))
    )
    return dict(
        os=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        logical_cores=logical,
        physical_cores=physical,
        cores=logical,
        ram_gb=round(gib, 2),
        available_gb=round(available / 2**30, 2),
        disk_free_gb=round(shutil.disk_usage(root).free / 2**30, 2),
        tier=tier,
        profile=tier,
        threads=inference_threads,
        inference_threads=inference_threads,
        inference_threads_source="override" if override.isdigit() else "safe default; runtime microbenchmarks on first CPU model load",
        image_size=32,
        process_ram_mb=round(psutil.Process().memory_info().rss / 2**20, 1),
        cpu_percent=psutil.cpu_percent(),
        **defaults,
    )


def accelerator():
    import torch

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        bf16 = bool(torch.cuda.is_bf16_supported())
        return dict(
            device="cuda",
            name=torch.cuda.get_device_name(),
            vram_gb=round(total / 2**30, 2),
            free_vram_gb=round(free / 2**30, 2),
            fp16=True,
            bf16=bf16,
            recommended_dtype="bf16" if bf16 else "fp16",
        )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return dict(
            device="mps",
            name="Apple Metal",
            vram_gb=None,
            free_vram_gb=None,
            fp16=True,
            bf16=False,
            recommended_dtype="fp16",
        )
    return dict(
        device="cpu",
        name="CPU",
        vram_gb=0,
        free_vram_gb=0,
        fp16=False,
        bf16=False,
        recommended_dtype="int8-or-fp32-benchmark",
    )


def _estimated_weight_mb(profile, bytes_per_parameter):
    parameters = PARAMETER_ESTIMATES.get(profile)
    if not parameters:
        return None
    return round(parameters * bytes_per_parameter / 2**20, 1)


def recommended_profile(root="."):
    hw = hardware(root)
    acc = accelerator()
    profile = hw["model_profile"]
    if acc["device"] == "cpu" and profile == "medium":
        profile = "small"
    if acc["device"] == "cpu":
        quantization = "auto"
        estimated = {
            "fp32_mb": _estimated_weight_mb(profile, 4),
            "int8_weight_mb": _estimated_weight_mb(profile, 1),
        }
    else:
        quantization = acc["recommended_dtype"]
        estimated = {
            "fp32_mb": _estimated_weight_mb(profile, 4),
            "fp16_bf16_mb": _estimated_weight_mb(profile, 2),
        }
    return {
        "hardware_profile": hw["profile"],
        "model_profile": profile,
        "context": hw["context"],
        "batch_size": hw["batch_size"],
        "gradient_accumulation": hw["gradient_accumulation"],
        "tokenizer_vocab": hw["tokenizer_vocab"],
        "inference_threads": hw["inference_threads"],
        "quantization": quantization,
        "estimated_weights": estimated,
        "accelerator": acc,
    }
