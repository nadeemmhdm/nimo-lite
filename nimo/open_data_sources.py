"""Registry and transforms for Nimo's optional openly licensed training corpora."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator

from .corpus import detect_language


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset: str
    config: str | None
    split: str
    license: str
    purpose: str
    language: str
    transform: str
    group: str
    streaming: bool = True
    default: bool = True
    terms_note: str = ""


@dataclass(frozen=True)
class DatasetType:
    key: str
    label: str
    description: str
    datasets: tuple[str, ...]


DATASETS = {
    "wikipedia-en": DatasetSpec(
        "wikipedia-en", "wikimedia/wikipedia", "20231101.en", "train",
        "CC-BY-SA-3.0 AND GFDL", "pretrain", "en", "wikipedia", "general",
    ),
    "wikipedia-ml": DatasetSpec(
        "wikipedia-ml", "wikimedia/wikipedia", "20231101.ml", "train",
        "CC-BY-SA-3.0 AND GFDL", "pretrain", "ml", "wikipedia", "malayalam",
    ),
    "oasst1": DatasetSpec(
        "oasst1", "OpenAssistant/oasst1", None, "train",
        "Apache-2.0", "instruction", "multi", "oasst1", "general", streaming=False,
    ),
    "dolly-15k": DatasetSpec(
        "dolly-15k", "databricks/databricks-dolly-15k", None, "train",
        "CC-BY-SA-3.0", "instruction", "en", "dolly", "general",
    ),
    "code-alpaca": DatasetSpec(
        "code-alpaca", "thisisanshgupta/CodeAlpaca", None, "train",
        "Apache-2.0", "pretrain+instruction", "en", "alpaca-code", "code",
    ),
    "gsm8k": DatasetSpec(
        "gsm8k", "openai/gsm8k", "main", "train",
        "MIT", "pretrain+instruction", "en", "gsm8k", "math",
    ),
    "fineweb-edu": DatasetSpec(
        "fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train",
        "ODC-BY-1.0", "pretrain", "en", "fineweb", "general",
        default=False, terms_note="Dataset use is also subject to Common Crawl Terms of Use.",
    ),
}


DATASET_TYPES = {
    "general": DatasetType(
        "general",
        "General knowledge",
        "English Wikipedia plus general assistant/instruction conversations.",
        ("wikipedia-en", "oasst1", "dolly-15k"),
    ),
    "malayalam": DatasetType(
        "malayalam",
        "Malayalam",
        "Malayalam Wikipedia for native-script language and knowledge coverage.",
        ("wikipedia-ml",),
    ),
    "code": DatasetType(
        "code",
        "Code",
        "Apache-2.0 CodeAlpaca programming instruction/response examples.",
        ("code-alpaca",),
    ),
    "math": DatasetType(
        "math",
        "Math & reasoning",
        "MIT-licensed GSM8K grade-school math problems with worked solutions.",
        ("gsm8k",),
    ),
}

TYPE_ORDER = ("general", "malayalam", "code", "math")
STARTER_BUNDLE = tuple(
    key for group in TYPE_ORDER for key in DATASET_TYPES[group].datasets
)

PRESETS = {
    "quick": {
        "wikipedia-en": 1000,
        "wikipedia-ml": 1000,
        "oasst1": 2000,
        "dolly-15k": 2000,
        "code-alpaca": 2500,
        "gsm8k": 2500,
    },
    "balanced": {
        "wikipedia-en": 5000,
        "wikipedia-ml": 5000,
        "oasst1": 10000,
        "dolly-15k": 10000,
        "code-alpaca": 10000,
        "gsm8k": 6000,
    },
    "large": {
        "wikipedia-en": 25000,
        "wikipedia-ml": 25000,
        "oasst1": 40000,
        "dolly-15k": 15000,
        "code-alpaca": 20000,
        "gsm8k": 7473,
    },
}


def catalog() -> list[dict]:
    return [asdict(DATASETS[key]) for key in DATASETS]


def type_catalog() -> list[dict]:
    return [asdict(DATASET_TYPES[key]) for key in TYPE_ORDER]


def expand_types(dataset_types: Iterable[str]) -> list[str]:
    requested = [str(value).strip().lower() for value in dataset_types if str(value).strip()]
    if not requested or requested == ["all"] or "all" in requested:
        requested = list(TYPE_ORDER)
    invalid = sorted(set(requested) - set(DATASET_TYPES))
    if invalid:
        raise ValueError(f"Unknown dataset type(s): {', '.join(invalid)}")
    keys: list[str] = []
    seen: set[str] = set()
    for group in TYPE_ORDER:
        if group not in requested:
            continue
        for key in DATASET_TYPES[group].datasets:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def load_source(spec: DatasetSpec, cache_dir=None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('Install open-data support with: pip install -e ".[open-data]"') from exc
    kwargs = {"split": spec.split, "streaming": spec.streaming}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return load_dataset(spec.dataset, spec.config, **kwargs) if spec.config else load_dataset(spec.dataset, **kwargs)


def _chunks(text: str, max_chars: int = 250_000) -> Iterator[str]:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    start = 0
    while text and start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = text.rfind("\n\n", start, end)
            if cut <= start + max_chars // 3:
                cut = text.rfind("\n", start, end)
            if cut > start:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        start = max(end, start + 1)


def _text_records(spec: DatasetSpec, rows: Iterable[dict], limit: int | None) -> Iterator[dict]:
    emitted = 0
    for row in rows:
        text = str(row.get("text") or "").strip()
        if len(text) < 80:
            continue
        for index, chunk in enumerate(_chunks(text)):
            if limit is not None and emitted >= limit:
                return
            yield {
                "text": chunk,
                "kind": "raw",
                "language": spec.language,
                "category": "wikipedia" if spec.transform == "wikipedia" else "educational-web",
                "source": str(row.get("url") or f"https://huggingface.co/datasets/{spec.dataset}"),
                "metadata": {
                    "dataset": spec.dataset,
                    "config": spec.config,
                    "dataset_type": spec.group,
                    "title": str(row.get("title") or "")[:500],
                    "chunk": index,
                },
            }
            emitted += 1


def _dolly_records(spec: DatasetSpec, rows: Iterable[dict], limit: int | None) -> Iterator[dict]:
    emitted = 0
    for row in rows:
        instruction = str(row.get("instruction") or "").strip()
        response = str(row.get("response") or "").strip()
        context = str(row.get("context") or "").strip()
        if not instruction or not response:
            continue
        user = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": response}]
        yield {
            "text": json.dumps(messages, ensure_ascii=False),
            "kind": "instruction",
            "language": "en",
            "category": str(row.get("category") or "instruction")[:120],
            "source": f"https://huggingface.co/datasets/{spec.dataset}",
            "metadata": {"dataset": spec.dataset, "dataset_type": spec.group, "messages": messages},
        }
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def oasst_paths(rows: Iterable[dict]) -> list[list[dict]]:
    clean = []
    for raw in rows:
        row = dict(raw)
        ident = str(row.get("message_id") or "")
        text = str(row.get("text") or "").strip()
        role = str(row.get("role") or "").lower()
        if ident and text and role in {"user", "assistant"} and not bool(row.get("deleted")):
            row.update(message_id=ident, text=text, role=role)
            clean.append(row)
    by_id = {r["message_id"]: r for r in clean}
    children: dict[str, list[str]] = {}
    for row in clean:
        parent = row.get("parent_id")
        if parent in by_id:
            children.setdefault(str(parent), []).append(row["message_id"])
    leaves = []
    for row in clean:
        if row["role"] != "assistant" or row.get("review_result") is False or row.get("rank") not in (None, 0):
            continue
        if not any(child in by_id for child in children.get(row["message_id"], [])):
            leaves.append(row)
    paths = []
    for leaf in sorted(leaves, key=lambda r: r["message_id"]):
        chain, seen, current = [], set(), leaf
        while current and current["message_id"] not in seen:
            seen.add(current["message_id"])
            chain.append(current)
            parent = current.get("parent_id")
            current = by_id.get(str(parent)) if parent else None
        messages = [{"role": r["role"], "content": r["text"]} for r in reversed(chain)]
        if len(messages) < 2 or messages[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in messages):
            continue
        if any(a["role"] == b["role"] for a, b in zip(messages, messages[1:])):
            continue
        paths.append(messages)
    return paths


def _oasst_records(spec: DatasetSpec, rows: Iterable[dict], limit: int | None) -> Iterator[dict]:
    for index, messages in enumerate(oasst_paths(rows)):
        if limit is not None and index >= limit:
            return
        yield {
            "text": json.dumps(messages, ensure_ascii=False),
            "kind": "conversation",
            "language": detect_language("\n".join(m["content"] for m in messages)),
            "category": "assistant-conversation",
            "source": f"https://huggingface.co/datasets/{spec.dataset}",
            "metadata": {"dataset": spec.dataset, "dataset_type": spec.group, "messages": messages},
        }


def _alpaca_code_records(spec: DatasetSpec, rows: Iterable[dict], limit: int | None) -> Iterator[dict]:
    emitted = 0
    for row in rows:
        instruction = str(row.get("instruction") or row.get("prompt") or "").strip()
        response = str(row.get("output") or row.get("response") or row.get("completion") or "").strip()
        context = str(row.get("input") or row.get("context") or "").strip()
        if not instruction or not response:
            continue
        user = instruction if not context else f"{instruction}\n\nInput:\n{context}"
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": response}]
        split_key = hashlib.sha256((user + "\n" + response).encode("utf-8")).hexdigest()
        metadata = {
            "dataset": spec.dataset,
            "dataset_type": spec.group,
            "messages": messages,
            "split_key": split_key,
        }
        yield {
            "text": f"Programming task:\n{user}\n\nSolution:\n{response}",
            "kind": "code",
            "language": "en",
            "category": "coding-pretrain",
            "source": f"https://huggingface.co/datasets/{spec.dataset}",
            "metadata": metadata,
        }
        yield {
            "text": json.dumps(messages, ensure_ascii=False),
            "kind": "instruction",
            "language": "en",
            "category": "coding-instruction",
            "source": f"https://huggingface.co/datasets/{spec.dataset}",
            "metadata": metadata,
        }
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def _gsm8k_records(spec: DatasetSpec, rows: Iterable[dict], limit: int | None) -> Iterator[dict]:
    emitted = 0
    for row in rows:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        split_key = hashlib.sha256((question + "\n" + answer).encode("utf-8")).hexdigest()
        metadata = {
            "dataset": spec.dataset,
            "dataset_type": spec.group,
            "messages": messages,
            "split_key": split_key,
        }
        yield {
            "text": f"Math problem:\n{question}\n\nWorked solution:\n{answer}",
            "kind": "math",
            "language": "en",
            "category": "math-pretrain",
            "source": f"https://huggingface.co/datasets/{spec.dataset}",
            "metadata": metadata,
        }
        yield {
            "text": json.dumps(messages, ensure_ascii=False),
            "kind": "instruction",
            "language": "en",
            "category": "math-instruction",
            "source": f"https://huggingface.co/datasets/{spec.dataset}",
            "metadata": metadata,
        }
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def records(spec: DatasetSpec, rows: Iterable[dict], limit: int | None) -> Iterator[dict]:
    if spec.transform in {"wikipedia", "fineweb"}:
        yield from _text_records(spec, rows, limit)
    elif spec.transform == "dolly":
        yield from _dolly_records(spec, rows, limit)
    elif spec.transform == "oasst1":
        yield from _oasst_records(spec, rows, limit)
    elif spec.transform == "alpaca-code":
        yield from _alpaca_code_records(spec, rows, limit)
    elif spec.transform == "gsm8k":
        yield from _gsm8k_records(spec, rows, limit)
    else:
        raise ValueError(f"Unsupported transform: {spec.transform}")
