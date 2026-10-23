from nimo.corpus import CorpusStore
from nimo.open_data import _train_tokenizer_from_corpus, import_dataset
from nimo.open_data_sources import DATASETS, STARTER_BUNDLE, oasst_paths
from nimo.store import Store


def test_starter_bundle_keeps_fineweb_opt_in():
    assert STARTER_BUNDLE == ("wikipedia-en", "wikipedia-ml", "oasst1", "dolly-15k")
    assert DATASETS["fineweb-edu"].default is False
    assert "Common Crawl" in DATASETS["fineweb-edu"].terms_note


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


def test_oasst_reconstructs_non_overlapping_leaf_path():
    rows = [
        {"message_id": "u1", "parent_id": None, "role": "user", "text": "Hello", "deleted": False},
        {"message_id": "a1", "parent_id": "u1", "role": "assistant", "text": "Hi", "deleted": False, "rank": 0},
        {"message_id": "u2", "parent_id": "a1", "role": "user", "text": "Explain rain", "deleted": False},
        {"message_id": "a2", "parent_id": "u2", "role": "assistant", "text": "Water condenses and falls.", "deleted": False, "rank": 0},
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
