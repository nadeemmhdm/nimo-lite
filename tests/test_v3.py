import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from nimo.memory import LocalMemory, extract_memories
from nimo.models import MODEL_PROFILES, build_text_model, load_model
from nimo.runtime import REASONING, _safe_calculate, create_app
from nimo.store import Store
from nimo.training import run_job, start_job
from nimo.web_access import _public_host


GENERAL = Path(__file__).parents[1] / 'nimo/resources/general-basics.json'


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
    assert model.head.weight is model.token.weight


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


def test_general_training_corpus_covers_requested_basics():
    rows = json.loads(GENERAL.read_text())
    categories = {row['category'] for row in rows}
    required = {
        'qa:language','qa:world','qa:math','qa:science','qa:technology','qa:programming',
        'qa:reasoning','qa:practical','qa:business','qa:education','qa:health','qa:safety',
        'qa:social','qa:current','qa:uncertainty','qa:conversation'
    }
    assert required <= categories
    questions = {row['question'].casefold() for row in rows}
    assert 'good morning' in questions and len(rows) >= 70
    assert all(row['source'].startswith('Nimo Lite contributors: original') for row in rows)
    assert all(row['license'] == 'CC0-1.0' for row in rows)


def test_general_corpus_can_train_v2_text_model(tmp_path):
    store = Store(tmp_path)
    for item in json.loads(GENERAL.read_text())[:18]:
        row = store.add_text(**item)
        store.update(row['id'], {'status':'approved'})
    ident = start_job(store, 'text', epochs=1, context=128, profile='micro')
    run_job(tmp_path, ident)
    model, checkpoint = load_model(tmp_path/'models'/f'{ident}.pt')
    assert checkpoint['task'] == 'text' and checkpoint['epoch'] == 1
    assert checkpoint['architecture']['version'] == 2
    assert model.head.weight is model.token.weight


def test_local_memory_is_useful_and_filters_sensitive_data(tmp_path):
    memory = LocalMemory(Store(tmp_path))
    assert extract_memories('I prefer concise technical answers')
    saved = memory.remember_from_message('I prefer concise technical answers')
    assert saved and memory.list()[0]['kind'] == 'preference'
    before = len(memory.list())
    assert memory.remember_from_message('my password is correct-horse-battery-staple') == []
    assert memory.remember_from_message('I was diagnosed with a medical condition') == []
    assert len(memory.list()) == before
    assert memory.clear() == before
    assert memory.list() == []


def test_runtime_replaces_routes_and_exposes_reasoning_memory(tmp_path):
    app = create_app(tmp_path)
    routes = [(getattr(r, 'path', ''), getattr(r, 'methods', set())) for r in app.router.routes]
    assert sum(path == '/api/chat' and 'POST' in methods for path, methods in routes) == 1
    assert sum(path == '/api/search' and 'POST' in methods for path, methods in routes) == 1
    assert any(path == '/api/chat/stream' and 'POST' in methods for path, methods in routes)
    assert any(path == '/api/web/read' and 'POST' in methods for path, methods in routes)
    assert any(path == '/api/memory' and 'GET' in methods for path, methods in routes)
    assert set(REASONING) == {'fast', 'balanced', 'deep'}

    with TestClient(app) as client:
        client.headers['X-Nimo-Token'] = client.get('/api/bootstrap').json()['token']
        result = client.post('/api/chat', json={'text':'What is 2 + 3 * 4?','reasoning':'fast'}).json()
        assert result['answer'] == '14' and result['mode'].startswith('deterministic calculator')
        result = client.post('/api/chat', json={'text':'I prefer concise technical answers','reasoning':'balanced'}).json()
        assert result['memory_saved'] == 1
        memories = client.get('/api/memory').json()['memories']
        assert any(m['value'] == 'concise technical answers' for m in memories)
