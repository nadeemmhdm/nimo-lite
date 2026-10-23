from nimo.corpus import CorpusStore
from nimo.memory import LocalMemory
from nimo.retrieval import retrieve_documents, retrieve_knowledge
from nimo.store import Store


def test_knowledge_and_corpus_use_fts_indexes_when_available(tmp_path):
    store = Store(tmp_path)
    corpus = CorpusStore(store)
    for index in range(20):
        store.add_knowledge(f"unrelated banana note {index}", f"noise-{index}")
    store.add_knowledge("Nimo KV cache accelerates incremental token decoding", "perf-note")
    corpus.add(
        "SQLite FTS makes local document retrieval efficient for Nimo",
        "local-doc",
        "CC0",
        kind="knowledge",
        status="approved",
    )

    knowledge = retrieve_knowledge(store, "Nimo incremental KV cache", 3)
    documents = retrieve_documents(corpus, "Nimo SQLite retrieval", 3)
    assert knowledge and knowledge[0]["source"] == "perf-note"
    assert documents and documents[0]["source"] == "local-doc"

    with store.db() as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    # Python's normal SQLite builds include FTS5. If a custom build lacks it, retrieval
    # still passes above through the bounded BM25 fallback.
    if "knowledge_fts" in names:
        assert "documents_fts" in names


def test_memory_retrieval_does_not_require_full_memory_list_scan(tmp_path):
    store = Store(tmp_path)
    memory = LocalMemory(store)
    saved = memory.remember_from_message("I am working on Nimo inference speed optimizer")
    assert saved

    # The indexed path and bounded fallback query SQLite directly; relevant() should not
    # call the UI-oriented list() method that materializes every memory row.
    memory.list = lambda: (_ for _ in ()).throw(AssertionError("full list scan"))
    rows = memory.relevant("Nimo inference optimizer", 3)
    assert rows
    assert "Nimo" in rows[0]["value"]
