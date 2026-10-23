import torch

from nimo.fast_inference import get_engine, reset_inference_cache
from nimo.models import NimoLMV2
from nimo.tokenizer import NimoTokenizer, train_tokenizer


def _model():
    torch.manual_seed(41)
    return NimoLMV2(
        vocab_size=512,
        width=64,
        layers=2,
        heads=4,
        ff=128,
        context=64,
        dropout=0.0,
    ).eval()


def _tokenizer():
    return train_tokenizer(
        [
            "Hello Nimo local inference cache.",
            "നിമോ വേഗത്തിൽ മറുപടി നൽകുന്നു.",
            "Nimo fast ayi reply cheyyunnu.",
            "def add(a, b): return a + b",
        ] * 20,
        vocab_size=512,
    )


def test_fast_prefill_and_decode_match_full_forward():
    model = _model()
    engine = get_engine(model)
    engine.reset()
    prompt = torch.tensor([[10, 11, 12, 13, 14]])
    with torch.inference_mode():
        expected = model(prompt)[:, -1, :]
        actual, meta = engine.prefill(prompt[0].tolist())
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
        assert meta["processed_prompt_tokens"] == prompt.shape[1]

        token = int(expected.argmax(dim=-1).item())
        expected_next = model(torch.cat((prompt, torch.tensor([[token]])), dim=1))[:, -1, :]
        actual_next = engine.decode(token)
        assert torch.allclose(actual_next, expected_next, atol=1e-5, rtol=1e-4)


def test_incremental_prompt_prefix_reuse_and_edit_invalidation():
    model = _model()
    engine = get_engine(model)
    engine.reset()
    with torch.inference_mode():
        _, first = engine.prefill([20, 21, 22])
        assert first["reused_prefix_tokens"] == 0

        extended = [20, 21, 22, 23, 24]
        logits, second = engine.prefill(extended)
        assert second["reused_prefix_tokens"] == 3
        assert second["processed_prompt_tokens"] == 2
        expected = model(torch.tensor([extended]))[:, -1, :]
        assert torch.allclose(logits, expected, atol=1e-5, rtol=1e-4)

        edited = [20, 21, 99, 23]
        logits, third = engine.prefill(edited)
        assert third["reused_prefix_tokens"] == 2
        expected = model(torch.tensor([edited]))[:, -1, :]
        assert torch.allclose(logits, expected, atol=1e-5, rtol=1e-4)


def test_preallocated_kv_storage_does_not_reallocate_during_decode():
    model = _model()
    engine = get_engine(model)
    engine.reset()
    pointers = [(layer.key.data_ptr(), layer.value.data_ptr()) for layer in engine.session.cache.layers]
    logits, _ = engine.prefill([10, 11, 12, 13])
    for _ in range(5):
        token = int(logits.argmax(dim=-1).item())
        logits = engine.decode(token)
    after = [(layer.key.data_ptr(), layer.value.data_ptr()) for layer in engine.session.cache.layers]
    assert pointers == after
    assert engine.session.cache.length == 9


def test_fast_deterministic_generation_matches_uncached_greedy():
    model = _model()
    tok = _tokenizer()
    prompt = tok.encode_chat([{"role": "user", "content": "Hello"}], True)
    engine = get_engine(model)
    engine.reset()
    fast = list(model.generate_ids(prompt, tok, limit=6, deterministic=True))

    ids = list(prompt)
    slow = []
    eos = tok.special_id("EOS")
    pad = tok.special_id("PAD")
    with torch.inference_mode():
        for _ in range(6):
            logits = model(torch.tensor([ids]))
            token = int(logits[0, -1].argmax())
            if token in (eos, pad):
                break
            slow.append(token)
            ids.append(token)
            if len(ids) >= model.context:
                break
    assert fast == slow


def test_tokenizer_artifact_is_reused_and_model_sessions_are_separate():
    tok = _tokenizer()
    artifact = tok.artifact()
    one = NimoTokenizer.from_artifact(artifact)
    two = NimoTokenizer.from_artifact(artifact)
    assert one is two

    a = _model()
    b = _model()
    ea = get_engine(a)
    eb = get_engine(b)
    assert ea is not eb
    ea.prefill([10, 11, 12])
    assert ea.session.cache.length == 3
    assert eb.session.cache.length == 0
    reset_inference_cache(a)
    assert ea.session.cache.length == 0
