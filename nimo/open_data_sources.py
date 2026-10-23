"""Registry and transforms for Nimo's optional openly licensed training corpora."""
from __future__ import annotations

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
    streaming: bool = True
    default: bool = True
    terms_note: str = ""


DATASETS = {
    "wikipedia-en": DatasetSpec("wikipedia-en", "wikimedia/wikipedia", "20231101.en", "train", "CC-BY-SA-3.0 AND GFDL", "pretrain", "en", "wikipedia"),
    "wikipedia-ml": DatasetSpec("wikipedia-ml", "wikimedia/wikipedia", "20231101.ml", "train", "CC-BY-SA-3.0 AND GFDL", "pretrain", "ml", "wikipedia"),
    "oasst1": DatasetSpec("oasst1", "OpenAssistant/oasst1", None, "train", "Apache-2.0", "instruction", "multi", "oasst1", streaming=False),
    "dolly-15k": DatasetSpec("dolly-15k", "databricks/databricks-dolly-15k", None, "train", "CC-BY-SA-3.0", "instruction", "en", "dolly"),
    "fineweb-edu": DatasetSpec(
        "fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "ODC-BY-1.0", "pretrain", "en", "fineweb",
        default=False, terms_note="Dataset use is also subject to Common Crawl Terms of Use.",
    ),
}

STARTER_BUNDLE = ("wikipedia-en", "wikipedia-ml", "oasst1", "dolly-15k")
PRESETS = {
    "quick": {"wikipedia-en": 1000, "wikipedia-ml": 1000, "oasst1": 2000, "dolly-15k": 2000},
    "balanced": {"wikipedia-en": 5000, "wikipedia-ml": 5000, "oasst1": 10000, "dolly-15k": 10000},
    "large": {"wikipedia-en": 25000, "wikipedia-ml": 25000, "oasst1": 40000, "dolly-15k": 15000},
}


def catalog() -> list[dict]:
    return [asdict(DATASETS[key]) for key in DATASETS]


def load_source(spec: DatasetSpec):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('Install open-data support with: pip install -e ".[open-data]"') from exc
    kwargs = {"split": spec.split, "streaming": spec.streaming}
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
                "text": chunk, "kind": "raw", "language": spec.language,
                "category": "wikipedia" if spec.transform == "wikipedia" else "educational-web",
                "source": str(row.get("url") or f"https://huggingface.co/datasets/{spec.dataset}"),
                "metadata": {"dataset": spec.dataset, "config": spec.config, "title": str(row.get("title") or "")[:500], "chunk": index},
            }
            emitted += 1


def _dolly_records(spec: DatasetSpec, rows: Iterable[dict], limit: int | None) -> Iterator[dict]:
    emitted = 0
    for row in rows:
        instruction, response = str(row.get("instruction") or "").strip(), str(row.get("response") or "").strip()
        context = str(row.get("context") or "").strip()
        if not instruction or not response:
            continue
        user = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": response}]
        yield {
            "text": json.dumps(messages, ensure_ascii=False), "kind": "instruction", "language": "en",
            "category": str(row.get("category") or "instruction")[:120],
            "source": f"https://huggingface.co/datasets/{spec.dataset}",
            "metadata": {"dataset": spec.dataset, "messages": messages},
        }
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def oasst_paths(rows: Iterable[dict]) -> list[list[dict]]:
    clean = []
    for raw in rows:
        row = dict(raw)
        ident, text, role = str(row.get("message_id") or ""), str(row.get("text") or "").strip(), str(row.get("role") or "").lower()
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
            seen.add(current["message_id"]); chain.append(current)
            parent = current.get("parent_id"); current = by_id.get(str(parent)) if parent else None
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
            "text": json.dumps(messages, ensure_ascii=False), "kind": "conversation",
            "language": detect_language("\n".join(m["content"] for m in messages)), "category": "assistant-conversation",
            "source": f"https://huggingface.co/datasets/{spec.dataset}",
            "metadata": {"dataset": spec.dataset, "messages": messages},
        }


def records(spec: DatasetSpec, rows: Iterable[dict], limit: int | None) -> Iterator[dict]:
    if spec.transform in {"wikipedia", "fineweb"}:
        yield from _text_records(spec, rows, limit)
    elif spec.transform == "dolly":
        yield from _dolly_records(spec, rows, limit)
    elif spec.transform == "oasst1":
        yield from _oasst_records(spec, rows, limit)
    else:
        raise ValueError(f"Unsupported transform: {spec.transform}")
