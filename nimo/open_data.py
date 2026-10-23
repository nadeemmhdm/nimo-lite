"""Import openly licensed corpora and train Nimo locally from scratch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .corpus import MAX_ITEM_TEXT, CorpusStore, detect_language, simhash64, split_for_checksum
from .lm_training import run_job, start_job, tokenizer_path, tokenizer_status
from .open_data_sources import DATASETS, PRESETS, STARTER_BUNDLE, catalog, load_source, records
from .store import Store, now, uid
from .tokenizer import train_tokenizer


def _fast_add(corpus: CorpusStore, record: dict, spec, status: str) -> dict:
    text = str(record["text"]).replace("\r\n", "\n").replace("\r", "\n")
    source, kind = str(record["source"]).strip(), str(record["kind"])
    language, category = str(record["language"]), str(record["category"])[:120]
    if kind not in {"raw", "knowledge", "code", "math", "instruction", "conversation"}:
        raise ValueError("Unsupported document kind")
    if status not in {"pending", "approved"}:
        raise ValueError("Invalid review status")
    if not text.strip() or not source or not spec.license.strip():
        raise ValueError("Text, source and license are required")
    if len(text.encode("utf-8")) > MAX_ITEM_TEXT:
        raise ValueError("Single document exceeds 2 MiB")
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if language == "auto":
        language = detect_language(text)
    with corpus.store.db() as db:
        exact = db.execute("SELECT id FROM documents WHERE checksum=?", (checksum,)).fetchone()
        if exact:
            return {"id": exact["id"], "duplicate": "exact"}
        if db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] >= corpus._limit():
            raise ValueError("Configured document limit reached")
        ident, split, stamp = uid(), split_for_checksum(checksum), now()
        metadata = {**record["metadata"], "license": spec.license, "terms_note": spec.terms_note}
        db.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ident, kind, text, checksum, f"{simhash64(text):016x}", source, spec.license, language, category,
             status, split, 1, json.dumps(metadata, ensure_ascii=False), stamp, stamp),
        )
    return {"id": ident, "duplicate": False, "split": split, "language": language}


def import_dataset(store: Store, key: str, limit: int | None = None, status: str = "approved") -> dict:
    if key not in DATASETS:
        raise ValueError(f"Unknown open dataset: {key}")
    if status not in {"approved", "pending"}:
        raise ValueError("status must be approved or pending")
    if limit is not None and int(limit) < 1:
        raise ValueError("limit must be positive")
    spec, corpus = DATASETS[key], CorpusStore(store)
    result = {"dataset": key, "added": 0, "duplicates": 0, "skipped": 0, "license": spec.license}
    for record in records(spec, load_source(spec), int(limit) if limit is not None else None):
        try:
            item = _fast_add(corpus, record, spec, status)
        except ValueError:
            result["skipped"] += 1
            continue
        result["duplicates" if item.get("duplicate") else "added"] += 1
    result["corpus"] = corpus.stats()
    return result


def prepare_bundle(root: str | Path, preset: str = "quick", include_fineweb: bool = False,
                   status: str = "approved") -> dict:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    store = Store(root)
    CorpusStore(store)
    keys = list(STARTER_BUNDLE) + (["fineweb-edu"] if include_fineweb else [])
    results = []
    for key in keys:
        limit = PRESETS[preset].get(key)
        if key == "fineweb-edu":
            limit = limit or PRESETS[preset]["wikipedia-en"]
        results.append(import_dataset(store, key, limit, status))
    return {"preset": preset, "datasets": results, "stats": CorpusStore(store).stats()}


def _train_tokenizer_from_corpus(store: Store, vocab_size: int) -> dict:
    CorpusStore(store)

    def texts():
        with store.db() as db:
            rows = db.execute("SELECT kind,text,metadata FROM documents WHERE status='approved' ORDER BY checksum LIMIT 50000")
            for row in rows:
                if row["kind"] in {"instruction", "conversation"}:
                    try:
                        metadata = json.loads(row["metadata"])
                    except Exception:
                        metadata = {}
                    for message in metadata.get("messages") or []:
                        value = message.get("content") if isinstance(message, dict) else None
                        if isinstance(value, str) and value.strip():
                            yield value
                else:
                    yield row["text"]
        for row in store.samples("text"):
            if row["status"] == "approved":
                yield row["content"]["question"]
                yield row["content"]["answer"]

    tokenizer = train_tokenizer(texts(), vocab_size=vocab_size)
    if tokenizer.metadata.get("training_documents", 0) == 0:
        raise ValueError("Approve text/corpus data before training a tokenizer")
    tokenizer.save(tokenizer_path(store))
    return tokenizer_status(store)


def _run_checked(root: str, ident: str):
    try:
        run_job(root, ident)
    except Exception as exc:
        store = Store(root)
        with store.db() as db:
            db.execute("UPDATE jobs SET status='failed',progress=?,modified=? WHERE id=?",
                       (json.dumps({"error": str(exc)}), now(), ident))
        raise


def train_open_model(root: str | Path, *, preset="quick", prepare=True, include_fineweb=False,
                     vocab_size=8000, profile=None, context=None, pretrain_epochs=1,
                     instruction_epochs=1, precision="auto") -> dict:
    root = str(root)
    prepared = prepare_bundle(root, preset, include_fineweb, "approved") if prepare else None
    store = Store(root)
    corpus = CorpusStore(store)
    if not corpus.stats()["approved"]:
        raise ValueError("No approved corpus data. Prepare or import open datasets first.")
    tokenizer = _train_tokenizer_from_corpus(store, int(vocab_size))
    pretrain_id = start_job(store, "pretrain", epochs=int(pretrain_epochs), profile=profile, context=context, precision=precision)
    _run_checked(root, pretrain_id)
    instruction_id = start_job(store, "instruction", epochs=int(instruction_epochs), profile=profile, context=context,
                               precision=precision, base_model=pretrain_id)
    _run_checked(root, instruction_id)
    with store.db() as db:
        pretrain = dict(db.execute("SELECT status,progress FROM jobs WHERE id=?", (pretrain_id,)).fetchone())
        instruction = dict(db.execute("SELECT status,progress FROM jobs WHERE id=?", (instruction_id,)).fetchone())
    return {
        "prepared": prepared,
        "tokenizer": tokenizer,
        "pretrain": {"id": pretrain_id, "status": pretrain["status"], "progress": json.loads(pretrain["progress"])},
        "instruction": {"id": instruction_id, "status": instruction["status"], "progress": json.loads(instruction["progress"])},
    }


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Import open datasets and train Nimo locally")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data", default="data")
    prepare.add_argument("--preset", choices=PRESETS, default="quick")
    prepare.add_argument("--include-fineweb", action="store_true")
    prepare.add_argument("--pending", action="store_true")

    one = sub.add_parser("import")
    one.add_argument("dataset", choices=DATASETS)
    one.add_argument("--data", default="data")
    one.add_argument("--limit", type=int)
    one.add_argument("--pending", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("--data", default="data")
    train.add_argument("--preset", choices=PRESETS, default="quick")
    train.add_argument("--skip-prepare", action="store_true")
    train.add_argument("--include-fineweb", action="store_true")
    train.add_argument("--vocab-size", type=int, default=8000)
    train.add_argument("--profile", choices=["tiny", "small", "medium"])
    train.add_argument("--context", type=int, choices=[256, 512, 1024, 2048, 4096])
    train.add_argument("--pretrain-epochs", type=int, default=1)
    train.add_argument("--instruction-epochs", type=int, default=1)
    train.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")

    args = parser.parse_args(argv)
    if args.command == "list":
        return _print({"starter_bundle": STARTER_BUNDLE, "presets": PRESETS, "datasets": catalog()})
    if args.command == "prepare":
        return _print(prepare_bundle(args.data, args.preset, args.include_fineweb, "pending" if args.pending else "approved"))
    if args.command == "import":
        return _print(import_dataset(Store(args.data), args.dataset, args.limit, "pending" if args.pending else "approved"))
    return _print(train_open_model(
        args.data, preset=args.preset, prepare=not args.skip_prepare, include_fineweb=args.include_fineweb,
        vocab_size=args.vocab_size, profile=args.profile, context=args.context,
        pretrain_epochs=args.pretrain_epochs, instruction_epochs=args.instruction_epochs, precision=args.precision,
    ))


if __name__ == "__main__":
    main()
