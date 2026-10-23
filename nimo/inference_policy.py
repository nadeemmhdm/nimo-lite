"""One-time local inference policy for quantization and warmup.

Auto mode never assumes INT8 is faster. On CPU it compares Nimo's own FP32 model with a
dynamic-INT8 candidate on a tiny local probe and keeps INT8 only when it is measurably
faster and preserves the probe's next-token decision within a conservative error bound.
"""
from __future__ import annotations

import gc
import os
import time

import torch

_INSTALLED = False


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _timed_forward(model, probe, device: str) -> tuple[float, torch.Tensor]:
    samples = []
    output = None
    with torch.inference_mode():
        output = model(probe)
        _sync(device)
        for _ in range(2):
            started = time.perf_counter()
            output = model(probe)
            _sync(device)
            samples.append(time.perf_counter() - started)
    return min(samples), output[:, -1, :].float()


def _quality_guard(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if reference.shape != candidate.shape or not torch.isfinite(candidate).all():
        return {"passed": False, "reason": "invalid candidate logits"}
    same_choice = int(reference.argmax()) == int(candidate.argmax())
    scale = float(reference.std().clamp_min(1e-6))
    normalized_mae = float((reference - candidate).abs().mean()) / scale
    return {
        "passed": bool(same_choice and normalized_mae <= 0.08),
        "same_next_token": bool(same_choice),
        "normalized_mae": normalized_mae,
    }


def _architecture_count(model) -> int | None:
    try:
        return int(getattr(model, "_nimo_parameter_count", model.parameter_count()))
    except Exception:
        return None


def _attach_metadata(model, parameter_count, decision) -> None:
    if parameter_count is not None:
        setattr(model, "_nimo_parameter_count", int(parameter_count))
    setattr(model, "_nimo_quantization_decision", decision)


def _warmup(model) -> None:
    setting = os.environ.get("NIMO_WARMUP", "auto").strip().lower()
    if setting in {"0", "off", "false", "no"}:
        return
    try:
        from .fast_inference import get_engine

        engine = get_engine(model)
        token = min(8, max(0, int(getattr(model, "vocab_size", 9)) - 1))
        with torch.inference_mode():
            engine.reset()
            engine.prefill([token])
            engine.reset()
    except Exception:
        # Warmup is an optimization only and must never make model loading fail.
        return


def install_inference_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from . import models as model_module

    original = model_module.quantize_for_inference

    def benchmarked_quantize(model, mode="auto", device="cpu"):
        requested = (mode or "auto").lower()
        original_parameter_count = _architecture_count(model)
        if requested != "auto" or device != "cpu":
            selected, qmode = original(model, requested, device)
            _attach_metadata(
                selected,
                original_parameter_count,
                {
                    "requested": requested,
                    "selected": qmode,
                    "benchmark": "explicit mode" if requested != "auto" else "accelerator capability",
                },
            )
            _warmup(selected)
            return selected, qmode

        fp32 = model.float().eval()
        candidate, candidate_mode = original(fp32, "int8", "cpu")
        if candidate_mode != "int8":
            _attach_metadata(
                fp32,
                original_parameter_count,
                {"requested": "auto", "selected": "fp32", "reason": "INT8 unavailable"},
            )
            _warmup(fp32)
            return fp32, "fp32"

        vocab = max(9, int(getattr(fp32, "vocab_size", 512)))
        probe = torch.tensor(
            [[8 % vocab, 9 % vocab, 10 % vocab, 11 % vocab, 12 % vocab, 13 % vocab, 14 % vocab, 15 % vocab]]
        )
        try:
            fp_time, reference = _timed_forward(fp32, probe, "cpu")
            int8_time, quantized = _timed_forward(candidate, probe, "cpu")
            quality = _quality_guard(reference, quantized)
            faster = int8_time < fp_time * 0.95
            use_int8 = bool(quality["passed"] and faster)
            selected = candidate if use_int8 else fp32
            qmode = "int8" if use_int8 else "fp32"
            decision = {
                "requested": "auto",
                "selected": qmode,
                "fp32_probe_s": fp_time,
                "int8_probe_s": int8_time,
                "int8_speedup": fp_time / max(int8_time, 1e-9),
                "quality_guard": quality,
                "minimum_speed_gain": 0.05,
            }
            _attach_metadata(selected, original_parameter_count, decision)
            # Drop the rejected representation before warmup. The process peak may still
            # include the short-lived comparison copy, but steady-state chat memory does not.
            if use_int8:
                fp32 = None
            else:
                candidate = None
            gc.collect()
            _warmup(selected)
            return selected, qmode
        except Exception as exc:
            candidate = None
            gc.collect()
            _attach_metadata(
                fp32,
                original_parameter_count,
                {
                    "requested": "auto",
                    "selected": "fp32",
                    "reason": "benchmark fallback",
                    "error": type(exc).__name__,
                },
            )
            _warmup(fp32)
            return fp32, "fp32"

    model_module.quantize_for_inference = benchmarked_quantize
    _INSTALLED = True
