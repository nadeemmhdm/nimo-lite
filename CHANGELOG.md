# Changelog

Project package version remains **0.2.0**. This file tracks unreleased development work separately from release numbering.

## Unreleased

### Native generative stack
- Added Nimo's locally trained multilingual byte-subword tokenizer with special chat/tool/memory tokens and byte fallback.
- Added modern NimoLM v2 decoder architecture with RMSNorm, RoPE, SwiGLU, causal SDPA, tied embeddings, KV-cache generation, modern sampling controls, and Tiny/Small/Medium presets.
- Added separate base pretraining and assistant-masked instruction tuning with gradient accumulation, warmup/cosine scheduling, mixed-precision support, best-validation checkpoints, dataset hashes, tokens/second and history metrics.
- Added persistent multi-turn chats, edit/delete/rename, context budgeting, memory injection and BM25 RAG.
- Removed fixed-response intent routing from normal assistant generation; NimoIntent remains a compatibility/optional routing classifier.
- Added scalable corpus storage/imports for text, Markdown, JSON/JSONL, common code files, math/knowledge, conversations and bounded ZIP archives.
- Added streaming chat UX, stop/regenerate/edit/copy/delete actions, model/generation selectors, context/source/memory indicators, safe Markdown rendering and animated local SVG thinking state.
- Added FP32/FP16/BF16/INT8 inference selection for Nimo's own checkpoints.
- Added one-command bootstrap plus opt-in official GitHub release updater.
- Expanded security checks around archive traversal, tokenizer/checkpoint validation and controlled web retrieval.
- Added native tokenizer/model/training/retrieval/conversation tests.

### Compatibility
- Legacy v1/v2 byte language checkpoints remain loadable.
- Legacy byte text training remains available as a compatibility task.
- Image classification, intent classification, knowledge base, model manager, review flows, themes, media preparation and exports remain present.
