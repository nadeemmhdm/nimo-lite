# Dataset Guide

Nimo supports two dataset layers: the original reviewed sample manager and the scalable native corpus store used for tokenizer training, pretraining, instruction tuning, RAG, code, and math.

## Open dataset downloader

Large training datasets are **not bundled in the GitHub repository**. Standard Nimo bootstrap installs open-data support. For a manual editable install, use:

```bash
python -m pip install -e '.[open-data]'
```

Then download the default quick mix with one command:

```bash
nimo-data bootstrap
```

Unless `NIMO_DATA` or `NIMO_HOME` is set, Nimo stores the dataset cache, imported corpus, provenance, and manifest under `~/.nimo-lite/data`. Re-running the command reuses the Hugging Face cache and Nimo's SHA-256 exact-deduplication, so already imported content is not inserted twice.

### Four selectable dataset types

| Type | Default sources | Main use |
|---|---|---|
| `general` | English Wikipedia, OpenAssistant OASST1, Databricks Dolly 15K | general text/knowledge + assistant instruction data |
| `malayalam` | Malayalam Wikipedia | Malayalam native-script language and knowledge |
| `code` | CodeAlpaca | programming/code patterns |
| `math` | GSM8K | arithmetic word problems and worked reasoning |

Interactive selection:

```bash
nimo-data select
```

Non-interactive selection:

```bash
nimo-data bootstrap --types general,malayalam
nimo-data bootstrap --types code,math
nimo-data bootstrap --types all
```

Profiles control how many source examples are imported, not model size:

```bash
nimo-data bootstrap --preset quick
nimo-data bootstrap --preset balanced
nimo-data bootstrap --preset large
```

Inspect the catalog or current local state:

```bash
nimo-data list
nimo-data status
```

FineWeb-Edu remains explicit opt-in because it is much larger and its use is also subject to Common Crawl terms:

```bash
nimo-data bootstrap --include-fineweb
```

The older `nimo-open-data` command remains an alias for compatibility.

## Local storage

The downloader keeps remote dataset cache files under:

```text
~/.nimo-lite/data/datasets-cache/
```

Nimo's imported/processed corpus and provenance remain under the selected Nimo data root. `~/.nimo-lite/data/open-data/manifest.json` records the last selected types, preset, cache path, dataset keys, licenses, import counts, and corpus statistics.

The repository itself does not receive downloaded dataset payloads. For manual development, `data/` is gitignored as an additional safeguard.

## Supported manual imports

TXT, Markdown, JSON, JSONL, Python, JavaScript, TypeScript, HTML, CSS, SQL, Bash, and bounded ZIP archives.

JSON may use:
- `{ "text": "...", "kind": "raw" }`
- `{ "question": "...", "answer": "..." }`
- `{ "messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}] }`

Track source and license/permission for every item.

## Kinds

`raw`, `knowledge`, `code`, `math`, `instruction`, and `conversation`.

## Metadata

Each document stores source, license, detected/declared language, category, status, SHA-256 checksum, SimHash, stable split, version, timestamps, and optional structured metadata. Open-data imports additionally record the dataset key and selected dataset type.

## Deduplication and splits

Exact duplicate text is rejected. A conservative 64-bit SimHash comparison catches very near duplicates among recent same-kind rows. Splits derive deterministically from the checksum (90/5/5 train/validation/test) and are stored explicitly.

For code/math sources that create both pretraining and instruction-tuning views of the same example, Nimo uses one shared split key so paired views cannot land in different train/validation/test partitions.

Do not copy validation/test material into training under a new filename. Dataset hashes are embedded in native checkpoints and resume rejects changed splits.

## Languages and formatting

UTF-8 is required. Malayalam is preserved. Manglish remains ordinary Latin-script text. Code imports preserve indentation/newlines. Mathematical symbols are preserved by the byte-fallback tokenizer.

## Scale

The native corpus defaults to a configurable 250,000-document ceiling (`NIMO_MAX_DOCUMENTS`) instead of the legacy 5,000-sample prototype ceiling. Import requests and archive expansion remain bounded for resource safety.

The `large` open-data preset can approach that corpus ceiling. Increase `NIMO_MAX_DOCUMENTS` deliberately if you need a larger local corpus and have sufficient disk/RAM.

## Licensing

Open datasets retain their own licenses and terms; Nimo's MIT source license does not relicense downloaded data. Review [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and each upstream dataset card before redistribution or model release.
