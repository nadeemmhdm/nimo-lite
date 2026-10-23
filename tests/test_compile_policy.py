import gc
import weakref

import torch

from nimo.fast_inference import FastInferenceEngine, get_engine
from nimo.models import NimoLMV2


def _model():
    torch.manual_seed(73)
    return NimoLMV2(
        vocab_size=256,
        width=48,
        layers=2,
        heads=4,
        ff=96,
        context=48,
        dropout=0.0,
    ).eval()


def test_compile_off_is_explicit_and_correct(monkeypatch):
    monkeypatch.setenv("NIMO_COMPILE", "off")
    model = _model()
    engine = FastInferenceEngine(model)
    with torch.inference_mode():
        actual, _ = engine.prefill([10, 11, 12, 13])
        expected = model(torch.tensor([[10, 11, 12, 13]]))[:, -1, :]
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    assert engine.compile_info["requested"] == "off"
    assert engine.compile_info["selected"] == "off"


def test_compile_auto_does_not_blindly_enable_on_cpu(monkeypatch):
    monkeypatch.setenv("NIMO_COMPILE", "auto")
    model = _model()
    engine = FastInferenceEngine(model)
    engine.prefill([20, 21, 22])
    assert engine.device == "cpu"
    assert engine.compile_info["requested"] == "auto"
    assert engine.compile_info["selected"] == "off"
    assert "eager" in engine.compile_info["reason"].lower()


def test_compile_on_uses_explicit_compiled_callable(monkeypatch):
    monkeypatch.setenv("NIMO_COMPILE", "on")
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = _model()
    engine = FastInferenceEngine(model)
    with torch.inference_mode():
        actual, _ = engine.prefill([30, 31, 32])
        expected = model(torch.tensor([[30, 31, 32]]))[:, -1, :]
    assert calls
    assert engine.compile_info["selected"] == "on"
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_inference_engine_does_not_prevent_model_unload(monkeypatch):
    monkeypatch.setenv("NIMO_COMPILE", "off")
    model = _model()
    ref = weakref.ref(model)
    engine = get_engine(model)
    engine.prefill([10, 11])
    del model
    gc.collect()
    assert ref() is None
