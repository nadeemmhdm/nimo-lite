# Dataset Guide

Nimo supports two dataset layers: the original reviewed sample manager and the scalable native corpus store used for tokenizer training, pretraining, instruction tuning, RAG, code, and math.

## Supported imports

TXT, Markdown, JSON, JSONL, Python, JavaScript, TypeScript, HTML, CSS, SQL, Bash, and bounded ZIP archives.

JSON may use:
- `{ "text": "...", "kind": "raw" }`
- `{ "question": "...", "answer": "..." }`
- `{ "messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}] }`

Track source and license/permission for every item.

## Kinds

`raw`, `knowledge`, `code`, `math`, `instruction`, and `conversation`.

## Metadata

Each document stores source, license, detected/declared language, category, status, SHA-256 checksum, SimHash, stable split, version, timestamps, and optional structured metadata.

## Deduplication and splits

Exact duplicate text is rejected. A conservative 64-bit SimHash comparison catches very near duplicates among recent same-kind rows. Splits derive deterministically from the checksum (90/5/5 train/validation/test) and are stored explicitly.

Do not copy validation/test material into training under a new filename. Dataset hashes are embedded in native checkpoints and resume rejects changed splits.

## Languages and formatting

UTF-8 is required. Malayalam is preserved. Manglish remains ordinary Latin-script text. Code imports preserve indentation/newlines. Mathematical symbols are preserved by the byte-fallback tokenizer.

## Scale

The native corpus defaults to a configurable 250,000-document ceiling (`NIMO_MAX_DOCUMENTS`) instead of the legacy 5,000-sample prototype ceiling. Import requests and archive expansion remain bounded for resource safety.
