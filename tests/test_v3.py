import pytest
import torch

from nimo.models import MODEL_PROFILES, NimoLMV2, build_text_model
from nimo.runtime import _safe_calculate, create_app
from nimo.web_access import _public_host


def test_v2_model_uses_cached_incremental_decode_shape():
    model = build_text_model('micro', 128)
    first = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    logits, cache = model(first, use_cache=True)
    assert logits.shape == (1, 4, 259)
    assert len(cache) == MODEL_PROFILES['micro']['layers']
    second = torch.tensor([[4]], dtype=torch.long)
    next_logits, next_cache = model(second, cache=cache, use_cache=True)
    assert next_logits.shape == (1, 1, 259)
    assert next_cache[0][0].shape[2] == 5


def test_standard_profile_is_target_30_to_50m_parameters():
    model = build_text_model('standard', 256)
    parameters = sum(p.numel() for p in model.parameters())
    assert 30_000_000 <= parameters <= 50_000_000


def test_safe_calculator_has_no_eval_or_calls():
    assert _safe_calculate('2 + 3 * 4') == 14
    assert _safe_calculate('2 ** 8') == 256
    with pytest.raises(ValueError):
        _safe_calculate('__import__("os").system("id")')
    with pytest.raises(ValueError):
        _safe_calculate('2 ** 1000')


def test_ssrf_guard_blocks_loopback_and_private_networks():
    for url in ('http://127.0.0.1/', 'http://10.0.0.1/', 'http://[::1]/'):
        with pytest.raises(ValueError):
            _public_host(url)


def test_v3_runtime_replaces_chat_and_search_routes(tmp_path):
    app = create_app(tmp_path)
    routes = [(getattr(r, 'path', ''), getattr(r, 'methods', set())) for r in app.router.routes]
    assert sum(path == '/api/chat' and 'POST' in methods for path, methods in routes) == 1
    assert sum(path == '/api/search' and 'POST' in methods for path, methods in routes) == 1
    assert any(path == '/api/chat/stream' and 'POST' in methods for path, methods in routes)
    assert any(path == '/api/web/read' and 'POST' in methods for path, methods in routes)
