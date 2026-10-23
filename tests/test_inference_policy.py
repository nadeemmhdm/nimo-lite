import torch

from nimo.conversation import ConversationStore
from nimo.fast_inference import get_engine
from nimo.models import NimoLMV2, quantize_for_inference
from nimo.store import Store


def _model():
    torch.manual_seed(12)
    return NimoLMV2(
        vocab_size=512,
        width=64,
        layers=1,
        heads=4,
        ff=128,
        context=64,
        dropout=0.0,
    ).eval()


def test_explicit_int8_or_safe_fp32_fallback_can_prefill():
    model = _model()
    selected, mode = quantize_for_inference(model, "int8", "cpu")
    assert mode in {"int8", "fp32"}
    engine = get_engine(selected)
    engine.reset()
    logits, meta = engine.prefill([10, 11, 12, 13])
    assert logits.shape == (1, 512)
    assert meta["prompt_tokens"] == 4


def test_auto_quantization_records_measured_decision():
    model = _model()
    selected, mode = quantize_for_inference(model, "auto", "cpu")
    assert mode in {"int8", "fp32"}
    decision = getattr(selected, "_nimo_quantization_decision", {})
    assert decision.get("requested") == "auto"
    assert decision.get("selected") == mode
    if "quality_guard" in decision:
        assert "normalized_mae" in decision["quality_guard"]
        assert decision["minimum_speed_gain"] == 0.05


def test_new_conversation_resets_inference_prefix_cache(tmp_path):
    model = _model()
    engine = get_engine(model)
    engine.reset()
    engine.prefill([10, 11, 12, 13])
    assert engine.session.cache.length == 4

    chats = ConversationStore(Store(tmp_path))
    chats.create()
    assert engine.session.cache.length == 0
