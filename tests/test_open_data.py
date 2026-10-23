from pathlib import Path

from nimo.corpus import CorpusStore
from nimo.open_data import (
    _train_tokenizer_from_corpus,
    default_data_root,
    import_dataset,
    prepare_types,
)
from nimo.open_data_sources import (
    DATASETS,
    DATASET_TYPES,
    STARTER_BUNDLE,
    TYPE_ORDER,
    expand_types,
    oasst_paths,
)
from nimo.store import Store


def test_four_selectable_types_cover_default_bundle():
    assert TYPE_ORDER == ("general", "malayalam", "code", "math")
    assert set(DATASET_TYPES) == set(TYPE_ORDER)
    assert expand_types(("all",)) == list(STARTER_BUNDLE)
    assert "wikipedia-en" in DATASET_TYPES["general"].datasets
    assert DATASET_TYPES["malayalam"].datasets == ("wikipedia-ml",)
    assert DATASET_TYPES["code"].datasets == ("code-alpaca",)
    assert DATASET_TYPES["math"].datasets == ("gsm8k",)
    assert DATASETS["fineweb-edu"].default is False
    assert "Common Crawl" in DATASETS["fineweb-edu"].terms_note


def test_default_data_root_is_user_local(monkeypatch, tmp_path):
    monkeypatch.delenv("NIMO_DATA", raising=False)
    monkeypatch.setenv("NIMO_HOME", str(tmp_path / "nimo-home"))
    assert default_data_root() == tmp_path / "nimo-home"
    monkeypatch.setenv("NIMO_DATA", str(tmp_path / "explicit-data"))
    assert default_data_root() == tmp_path / "explicit-data"


def test_wikipedia_import_is_approved_and_exact_deduped(tmp_path, monkeypatch):
    rows = [
        {"title": "One", "url": "https://example.test/one", "text": "A useful article. " * 20},
        {"title": "Two", "url": "https://example.test/two", "text": "Another useful article. " * 20},
    ]
    monkeypatch.setattr("nimo.open_data.load_source", lambda spec: rows)
    store = Store(tmp_path)
    first = import_dataset(store, "wikipedia-en", limit=2)
    second = import_dataset(store, "wikipedia-en", limit=2)
    assert first["added"] == 2
    assert second["duplicates"] == 2
    imported = CorpusStore(store).list(status="approved")
    assert len(imported) == 2
    assert {row["language"] for row in imported} == {"en"}
    assert all(row["license"] == "CC-BY-SA-3.0 AND GFDL" for row in imported)


def test_dolly_becomes_instruction_messages(tmp_path, monkeypatch):
    rows = [{
        "instruction": "Explain photosynthesis",
        "context": "For a school student",
        "response": "Plants convert light energy into chemical energy.",
        "category": "open_qa",
    }]
    monkeypatch.setattr("nimo.open_data.load_source", lambda spec: rows)
    store = Store(tmp_path)
    result = import_dataset(store, "dolly-15k", limit=1)
    assert result["added"] == 1
    row = CorpusStore(store).list(kind="instruction")[0]
    messages = row["metadata"]["messages"]
    assert messages[0]["role"] == "user"
    assert "Context:" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"


def test_code_alpaca_becomes_code_training_data(tmp_path, monkeypatch):
    rows = [{
        "instruction": "Write a Python function that adds two integers.",
        "input": "a=2, b=3",
        "output": "def add(a, b):\n    return a + b",
    }]
    monkeypatch.setattr("nimo.open_data.load_source", lambda spec: rows)
    store = Store(tmp_path)
    result = import_dataset(store, "code-alpaca", limit=1)
    assert result["added"] == 2
    code_row = CorpusStore(store).list(kind="code")[0]
    instruction_row = CorpusStore(store).list(kind="instruction")[0]
    assert code_row["license"] == "Apache-2.0"
    assert code_row["metadata"]["dataset_type"] == "code"
    assert code_row["metadata"]["messages"][1]["role"] == "assistant"
    assert "def add" in code_row["text"]
    assert code_row["split"] == instruction_row["split"]


def test_gsm8k_becomes_math_training_data(tmp_path, monkeypatch):
    rows = [{"question": "If 2 apples cost $4, what does one cost?", "answer": "$2"}]
    monkeypatch.setattr("nimo.open_data.load_source", lambda spec: rows)
    store = Store(tmp_path)
    result = import_dataset(store, "gsm8k", limit=1)
    assert result["added"] == 2
    math_row = CorpusStore(store).list(kind="math")[0]
    instruction_row = CorpusStore(store).list(kind="instruction")[0]
    assert math_row["license"] == "MIT"
    assert math_row["metadata"]["dataset_type"] == "math"
    assert "Math problem:" in math_row["text"]
    assert math_row["split"] == instruction_row["split"]


def test_prepare_types_uses_cache_under_selected_data_root(tmp_path, monkeypatch):
    calls = []

    def fake_import(store, key, limit, status, cache_dir):
        calls.append((key, limit, status, Path(cache_dir)))
        return {
            "dataset": key,
            "dataset_type": DATASETS[key].group,
            "added": 0,
            "duplicates": 0,
            "skipped": 0,
            "license": DATASETS[key].license,
            "corpus": CorpusStore(store).stats(),
        }

    monkeypatch.setattr("nimo.open_data.import_dataset", fake_import)
    result = prepare_types(tmp_path, ("code", "math"), "quick")
    assert [item[0] for item in calls] == ["code-alpaca", "gsm8k"]
    assert all(item[3] == tmp_path / "datasets-cache" for item in calls)
    assert result["types"] == ["code", "math"]
    assert (tmp_path / "open-data" / "manifest.json").exists()


def test_oasst_reconstructs_non_overlapping_leaf_path():
    rows = [
        {"message_id": "u1", "parent_id": None, "role": "user", "text": "Hello", "deleted": False},
        {"message_id": "a1", "parent_id": "u1", "role": "assistant", "text": "Hi", "deleted": False, "rank": 0},
        {"message_id": "u2", "parent_id": "a1", "role": "user", "text": "Explain rain", "deleted": False},
        {
            "message_id": "a2",
            "parent_id": "u2",
            "role": "assistant",
            "text": "Water condenses and falls.",
            "deleted": False,
            "rank": 0,
        },
    ]
    assert oasst_paths(rows) == [[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Explain rain"},
        {"role": "assistant", "content": "Water condenses and falls."},
    ]]


def test_open_tokenizer_reads_approved_corpus(tmp_path):
    store = Store(tmp_path)
    corpus = CorpusStore(store)
    for index in range(6):
        corpus.add(
            f"Document {index} with Malayalam മലയാളം and English training text repeated repeated repeated.",
            f"local:test:{index}",
            "CC0-1.0",
            status="approved",
        )
    info = _train_tokenizer_from_corpus(store, vocab_size=512)
    assert info["ready"] is True
    assert info["metadata"]["training_documents"] >= 1
