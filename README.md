# Nimo Lite

**An original, locally trainable generative AI stack.**

Nimo Lite is a local-first AI workbench built around Nimo's own tokenizer, decoder-only Transformer, training engine, memory, retrieval, and chat runtime. It does **not** replace Nimo with OpenAI, Gemini, Claude, Ollama, Llama, Qwen, Mistral, Hugging Face pretrained weights, or another pretrained LLM.

The project version remains **0.2.0**. Model architecture/checkpoint schema versions are separate from the application version.

## What Nimo is now

Normal chat follows this path:

```text
User
  -> persistent conversation history
  -> relevant local memory
  -> BM25 local knowledge retrieval
  -> optional explicitly-enabled web evidence
  -> token-budgeted chat prompt
  -> NimoLM v2
  -> temperature/top-k/top-p sampling
  -> streamed response
```

`NimoIntent` is retained for optional classification/routing experiments. It is **not** the normal chat-answer engine and fixed `RESPONSES` are not used for personal-assistant replies.

## Native language stack

- **Tokenizer:** locally trained multilingual byte-subword tokenizer with full byte fallback and explicit `PAD/BOS/EOS/SYSTEM/USER/ASSISTANT/TOOL/MEMORY` tokens.
- **NimoLM v2:** decoder-only causal Transformer implemented from PyTorch primitives with RMSNorm, RoPE, SwiGLU, causal SDPA attention, tied embeddings, dropout control, proper random initialization, and inference KV cache.
- **Profiles:** Tiny (~13M with 8k vocab), Small (~32M), Medium (~88M). Parameter count changes with tokenizer vocabulary.
- **Context:** 512–2048 tokens by hardware profile, configurable up to 4096 in the current runtime.
- **Training:** base-model next-token pretraining and assistant-only-loss instruction tuning are separate operations.
- **Inference:** precise/balanced/creative sampling presets, Fast/Balanced/Deep response budgets, streaming, stop, seed/deterministic mode, penalties, and optional FP16/BF16/INT8 inference.
- **Languages:** tokenizer/training are UTF-8 native and designed for English, Malayalam, Manglish, code, mathematics, punctuation, and symbols.
- **Memory/RAG:** long-term memory and retrieved knowledge stay in SQLite and are injected at inference time; they do not silently modify model weights.

## One-command install and run

Python 3.10+ is required. The bootstrap checks out the official repository (using Git when available, otherwise a bounded GitHub ZIP), creates an isolated virtual environment, installs requirements, and launches Nimo on `127.0.0.1`.

**Linux/macOS**
```bash
python3 -c "import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/nadeemmhdm/nimo-lite/main/bootstrap.py').read())"
```

**Windows PowerShell**
```powershell
py -c "import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/nadeemmhdm/nimo-lite/main/bootstrap.py').read())"
```

Default install data lives under `~/.nimo-lite`. Set `NIMO_HOME` to use another location.

### Manual install

```bash
git clone https://github.com/nadeemmhdm/nimo-lite.git
cd nimo-lite
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e '.[test]'
nimo-run --data ./data
```

Open `http://127.0.0.1:8080`.

## Updates

Core Nimo remains offline by default. Release checks are opt-in:

```bash
nimo-update --check
nimo-update
```

To check/install a newer official release before launch:

```bash
nimo-run --auto-update
```

The updater accepts only the official GitHub repository, bounds downloads/extraction, blocks archive traversal, and never runs model-generated shell commands.

## Train a real Nimo model

The recommended sequence is:

1. **Add/import corpus data.** TXT, Markdown, JSON, JSONL, Python, JavaScript, TypeScript, HTML, CSS, SQL, Bash, and bounded ZIP imports are supported. Every item tracks source, license, language, category, checksum, review status, split, and version.
2. **Review and approve data.** Exact and conservative near-duplicate checks run before insertion. Train/validation/test assignment is stable so held-out data is not silently moved into training.
3. **Train the tokenizer.** Start with 8,000 tokens for Tiny/Small or 16,000 for larger corpora. The tokenizer is created only from local approved text and saved at `data/tokenizer/tokenizer.json`.
4. **Pretrain Base Model.** Next-token prediction learns language/code/math patterns from raw approved corpus text.
5. **Instruction Tune Chat Model.** Structured multi-turn data uses loss primarily on assistant content; system/user/tool text is context, not answer targets.
6. **Evaluate and activate.** Validation loss selects versions; test data is measured afterward. Loss reduction alone is not treated as proof of intelligence.

The bundled starter material is a **small demonstration corpus**, not enough to create a generally capable model. Useful models require much larger, diverse, correctly licensed data and substantially more compute.

## Chat UX

The local UI supports streamed generation, animated thinking state, stop generation, regenerate, edit/resend, copy, delete, persistent chats, rename/delete chat, model selection, generation presets, context usage, source/memory indicators, and safe Markdown rendering for headings, emphasis, lists, code, tables, and links.

UI controls use local SVG icons rather than decorative emoji. Nimo itself is instructed to use emoji only when it is naturally relevant to a user's message—not as decoration on every response.

## Data and memory

The data root contains:

```text
data/
  raw/
  cleaned/
  tokenizer/
  pretrain/
  instruction/
  validation/
  test/
  chats/
  models/
  nimo.sqlite
```

Automatic memory is conservative and inspectable. It targets low-risk preferences, project/workflow facts, names, tools, and goals. Secrets and sensitive categories are filtered from automatic memory. Users can list and delete stored memories.

## Hardware profiles

Nimo detects available RAM and accelerator support:

| Profile | Typical RAM | Suggested model | Context |
|---|---:|---|---:|
| Low Memory | ~6–10 GB | Tiny | 512 |
| Standard | ~10–24 GB | Small | 1024 |
| Performance | 24+ GB / suitable GPU | Medium | 2048 |

CPU, CUDA, and Apple Metal/MPS are supported targets. Training checks memory before starting and keeps pause/save/resume/stop checkpoints.

## Security and privacy

- loopback-only server by default
- local session token plus Host/Origin checks
- no telemetry or hidden cloud inference
- no automatic data upload
- path confinement and bounded requests
- ZIP traversal and expanded-size protection
- `torch.load(..., weights_only=True)` for model loading
- architecture/tokenizer bounds before checkpoint construction
- controlled web reader with private-network/SSRF blocking
- no shell/tool execution exposed to the language model

Optional online search must be enabled by the user. Retrieved web text is treated as untrusted data, not instructions.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Installation](docs/INSTALLATION.md)
- [Training guide](docs/TRAINING_GUIDE.md)
- [Dataset guide](docs/DATASET_GUIDE.md)
- [Tokenizer guide](docs/TOKENIZER_GUIDE.md)
- [Model card](docs/MODEL_CARD.md)
- [Evaluation](docs/EVALUATION.md)
- [Security](SECURITY.md)
- [Roadmap](ROADMAP.md)

## Development

```bash
python -m compileall -q nimo scripts
node --check nimo/static/runtime-enhancements.js
python -m pytest -q
```

Nimo Lite is an experimental research/workbench project. A technically correct architecture and a successful training run do not imply ChatGPT-scale capability; model quality depends on data, training budget, evaluation quality, and hardware.
