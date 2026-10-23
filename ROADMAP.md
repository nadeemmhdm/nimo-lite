# Roadmap

Nimo Lite's direction is an honest, original, locally trainable small-language-model stack.

## Implemented foundation

- native multilingual byte-subword tokenizer training
- decoder-only NimoLM v2 with RMSNorm, RoPE, SwiGLU and KV cache
- Tiny/Small/Medium model profiles
- separate base pretraining and instruction tuning
- assistant-only SFT loss masking
- persistent multi-turn chats and token-budgeted context
- inspectable local memory
- BM25 local RAG
- streamed sampling with generation presets
- CPU/CUDA/MPS training targets and mixed-precision hooks
- local checkpoint/resume state and tokenizer/data hashes
- scalable corpus imports with exact/near deduplication
- optional controlled web evidence
- release updater/bootstrap tooling

## Next engineering priorities

1. Improve tokenizer training speed/quality on multi-GB corpora (incremental BPE/unigram experiments).
2. Add packed-sequence training to reduce padding waste.
3. Add gradient checkpointing and memory-efficient attention paths for longer context.
4. Add stronger multilingual/coding/math held-out benchmark suites.
5. Add Nimo-trained embedding/retrieval model as an optional replacement for lexical-only RAG.
6. Add safe INT4 inference for Nimo checkpoints after correctness/performance validation.
7. Add model export format with publisher signatures/authentication.
8. Add extractive/local learned conversation summarization quality gates.
9. Add richer dataset quality dashboards and contamination reports.
10. Benchmark Windows, macOS/MPS, Linux CPU and multiple CUDA tiers.

## Non-goals

- hiding third-party pretrained LLM weights behind a Nimo name
- claiming frontier-model capability from small internal tests
- automatic shell/computer control from generated text
- telemetry or hidden cloud inference
- silently training on user conversations, memory, or retrieved web pages
