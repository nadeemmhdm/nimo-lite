"""Download openly licensed corpora and train Nimo locally from scratch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .corpus import MAX_ITEM_TEXT, CorpusStore, detect_language, simhash64, split_for_checksum
from .lm_training import run_job, start_job, tokenizer_path, tokenizer_status
from .open_data_sources import (
    DATASETS,
    DATASET_TYPES,
    PRESETS,
    STARTER_BUNDLE,
    TYPE_ORDER,
    catalog,
    expand_types,
    load_source,
    records,
    type_catalog,
)
from .store import Store, now, uid
from .tokenizer import train_tokenizer


def default_data_root() -> Path:
    """Resolve a user-local data root so downloaded datasets never need to live in the repo."""
    explicit = os.environ.get("NIMO_DATA")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("NIMO_HOME")
    if home:
        return Path(home).expanduser()
    return Path.home() / ".nimo-lite"


def dataset_cache(root: str | Path) -> Path:
    path = Path(root).expanduser() / "datasets-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path(root: str | Path) -> Path:
    path = Path(root).expanduser() / "open-data"
    path.mkdir(parents=True, exist_ok=True)
    return path / "manifest.json"


def _write_manifest(root: str | Path, payload: dict) -> None:
    path = _manifest_path(root)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


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
        metadata = {
            **record["metadata"],
            "license": spec.license,
            "terms_note": spec.terms_note,
            "dataset_key": spec.key,
            "dataset_type": spec.group,
        }
        split_key = str(metadata.get("split_key") or checksum)
        ident, split, stamp = uid(), split_for_checksum(split_key), now()
        db.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ident, kind, text, checksum, f"{simhash64(text):016x}", source, spec.license, language, category,
                status, split, 1, json.dumps(metadata, ensure_ascii=False), stamp, stamp,
            ),
        )
    return {"id": ident, "duplicate": False, "split": split, "language": language}


def import_dataset(
    store: Store,
    key: str,
    limit: int | None = None,
    status: str = "approved",
    cache_dir: str | Path | None = None,
) -> dict:
    if key not in DATASETS:
        raise ValueError(f"Unknown open dataset: {key}")
    if status not in {"approved", "pending"}:
        raise ValueError("status must be approved or pending")
    if limit is not None and int(limit) < 1:
        raise ValueError("limit must be positive")
    spec, corpus = DATASETS[key], CorpusStore(store)
    result = {
        "dataset": key,
        "dataset_type": spec.group,
        "added": 0,
        "duplicates": 0,
        "skipped": 0,
        "license": spec.license,
    }
    source_rows = load_source(spec, cache_dir) if cache_dir is not None else load_source(spec)
    for record in records(spec, source_rows, int(limit) if limit is not None else None):
        try:
            item = _fast_add(corpus, record, spec, status)
        except ValueError:
            result["skipped"] += 1
            continue
        result["duplicates" if item.get("duplicate") else "added"] += 1
    result["corpus"] = corpus.stats()
    return result


def prepare_types(
    root: str | Path,
    dataset_types=("all",),
    preset: str = "quick",
    *,
    include_fineweb: bool = False,
    status: str = "approved",
) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    root = Path(root).expanduser()
    store = Store(root)
    CorpusStore(store)
    requested = list(TYPE_ORDER) if not dataset_types or "all" in dataset_types else [str(v).lower() for v in dataset_types]
    keys = expand_types(requested)
    if include_fineweb and "fineweb-edu" not in keys:
        keys.append("fineweb-edu")
    cache = dataset_cache(root)
    results = []
    for key in keys:
        limit = PRESETS[preset].get(key)
        if key == "fineweb-edu":
            limit = limit or PRESETS[preset]["wikipedia-en"]
        results.append(import_dataset(store, key, limit, status, cache))
    payload = {
        "schema": 1,
        "updated": now(),
        "root": str(root.resolve()),
        "cache": str(cache.resolve()),
        "preset": preset,
        "types": requested,
        "datasets": results,
        "stats": CorpusStore(store).stats(),
    }
    _write_manifest(root, payload)
    return payload


def prepare_bundle(
    root: str | Path,
    preset: str = "quick",
    include_fineweb: bool = False,
    status: str = "approved",
) -> dict:
    """Compatibility wrapper: the starter bundle now covers the four selectable dataset types."""
    return prepare_types(root, TYPE_ORDER, preset, include_fineweb=include_fineweb, status=status)


def dataset_status(root: str | Path) -> dict:
    root = Path(root).expanduser()
    manifest = _manifest_path(root)
    saved = {}
    if manifest.exists():
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = {"warning": "Open-data manifest exists but could not be read"}
    store = Store(root)
    return {
        "root": str(root.resolve()),
        "cache": str(dataset_cache(root).resolve()),
        "manifest": saved,
        "corpus": CorpusStore(store).stats(),
    }


def _train_tokenizer_from_corpus(store: Store, vocab_size: int) -> dict:
    CorpusStore(store)

    def texts():
        with store.db() as db:
            rows = db.execute(
                "SELECT kind,text,metadata FROM documents WHERE status='approved' ORDER BY checksum LIMIT 50000"
            )
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
            db.execute(
                "UPDATE jobs SET status='failed',progress=?,modified=? WHERE id=?",
                (json.dumps({"error": str(exc)}), now(), ident),
            )
        raise


def train_open_model(
    root: str | Path,
    *,
    preset="quick",
    prepare=True,
    dataset_types=("all",),
    include_fineweb=False,
    vocab_size=8000,
    profile=None,
    context=None,
    pretrain_epochs=1,
    instruction_epochs=1,
    precision="auto",
) -> dict:
    root = str(Path(root).expanduser())
    prepared = (
        prepare_types(root, dataset_types, preset, include_fineweb=include_fineweb, status="approved")
        if prepare else None
    )
    store = Store(root)
    corpus = CorpusStore(store)
    if not corpus.stats()["approved"]:
        raise ValueError("No approved corpus data. Prepare or import open datasets first.")
    tokenizer = _train_tokenizer_from_corpus(store, int(vocab_size))
    pretrain_id = start_job(
        store, "pretrain", epochs=int(pretrain_epochs), profile=profile, context=context, precision=precision
    )
    _run_checked(root, pretrain_id)
    instruction_id = start_job(
        store,
        "instruction",
        epochs=int(instruction_epochs),
        profile=profile,
        context=context,
        precision=precision,
        base_model=pretrain_id,
    )
    _run_checked(root, instruction_id)
    with store.db() as db:
        pretrain = dict(db.execute("SELECT status,progress FROM jobs WHERE id=?", (pretrain_id,)).fetchone())
        instruction = dict(db.execute("SELECT status,progress FROM jobs WHERE id=?", (instruction_id,)).fetchone())
    return {
        "prepared": prepared,
        "tokenizer": tokenizer,
        "pretrain": {
            "id": pretrain_id,
            "status": pretrain["status"],
            "progress": json.loads(pretrain["progress"]),
        },
        "instruction": {
            "id": instruction_id,
            "status": instruction["status"],
            "progress": json.loads(instruction["progress"]),
        },
    }


def _parse_types(value: str | None) -> tuple[str, ...]:
    if not value:
        return ("all",)
    items = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    expand_types(items)
    return items or ("all",)


def _interactive_types() -> tuple[str, ...]:
    print("Choose Nimo open dataset types (comma-separated numbers, Enter = all):")
    for index, item in enumerate(type_catalog(), 1):
        sources = ", ".join(item["datasets"])
        print(f"  {index}. {item['label']} - {item['description']} [{sources}]")
    raw = input("Selection [all]: ").strip().lower()
    if not raw or raw in {"all", "a", "*"}:
        return tuple(TYPE_ORDER)
    selected = []
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if part.isdigit() and 1 <= int(part) <= len(TYPE_ORDER):
            selected.append(TYPE_ORDER[int(part) - 1])
        elif part in DATASET_TYPES:
            selected.append(part)
        else:
            raise ValueError(f"Unknown selection: {part}")
    if not selected:
        return tuple(TYPE_ORDER)
    return tuple(dict.fromkeys(selected))


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _add_download_args(parser):
    parser.add_argument("--data", default=None, help="Local Nimo data root; defaults to NIMO_DATA/NIMO_HOME/~/.nimo-lite")
    parser.add_argument("--preset", choices=PRESETS, default="quick")
    parser.add_argument("--types", default="all", help="all or comma-separated: general,malayalam,code,math")
    parser.add_argument("--include-fineweb", action="store_true")
    parser.add_argument("--pending", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download/select open datasets and train Nimo locally")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--data", default=None)

    bootstrap = sub.add_parser(
        "bootstrap",
        help="Download all four dataset types with one command, or select them with --types",
    )
    _add_download_args(bootstrap)

    select = sub.add_parser("select", help="Interactively choose which of the four dataset types to download")
    select.add_argument("--data", default=None)
    select.add_argument("--preset", choices=PRESETS, default="quick")
    select.add_argument("--include-fineweb", action="store_true")
    select.add_argument("--pending", action="store_true")

    prepare = sub.add_parser("prepare")
    _add_download_args(prepare)

    one = sub.add_parser("import")
    one.add_argument("dataset", choices=DATASETS)
    one.add_argument("--data", default=None)
    one.add_argument("--limit", type=int)
    one.add_argument("--pending", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("--data", default=None)
    train.add_argument("--preset", choices=PRESETS, default="quick")
    train.add_argument("--types", default="all", help="all or comma-separated: general,malayalam,code,math")
    train.add_argument("--skip-prepare", action="store_true")
    train.add_argument("--include-fineweb", action="store_true")
    train.add_argument("--vocab-size", type=int, default=8000)
    train.add_argument("--profile", choices=["tiny", "small", "medium"])
    train.add_argument("--context", type=int, choices=[256, 512, 1024, 2048, 4096])
    train.add_argument("--pretrain-epochs", type=int, default=1)
    train.add_argument("--instruction-epochs", type=int, default=1)
    train.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")

    args = parser.parse_args(argv)
    root = Path(getattr(args, "data", None) or default_data_root()).expanduser()

    if args.command == "list":
        return _print({
            "default_data_root": str(default_data_root()),
            "starter_bundle": STARTER_BUNDLE,
            "types": type_catalog(),
            "presets": PRESETS,
            "datasets": catalog(),
        })
    if args.command == "status":
        return _print(dataset_status(root))
    if args.command in {"bootstrap", "prepare"}:
        return _print(
            prepare_types(
                root,
                _parse_types(args.types),
                args.preset,
                include_fineweb=args.include_fineweb,
                status="pending" if args.pending else "approved",
            )
        )
    if args.command == "select":
        return _print(
            prepare_types(
                root,
                _interactive_types(),
                args.preset,
                include_fineweb=args.include_fineweb,
                status="pending" if args.pending else "approved",
            )
        )
    if args.command == "import":
        return _print(
            import_dataset(
                Store(root),
                args.dataset,
                args.limit,
                "pending" if args.pending else "approved",
                dataset_cache(root),
            )
        )
    return _print(
        train_open_model(
            root,
            preset=args.preset,
            prepare=not args.skip_prepare,
            dataset_types=_parse_types(args.types),
            include_fineweb=args.include_fineweb,
            vocab_size=args.vocab_size,
            profile=args.profile,
            context=args.context,
            pretrain_epochs=args.pretrain_epochs,
            instruction_epochs=args.instruction_epochs,
            precision=args.precision,
        )
    )


if __name__ == "__main__":
    main()
