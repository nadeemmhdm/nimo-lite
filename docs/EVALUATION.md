# Evaluation

Nimo separates optimization metrics from capability claims.

## Language-model metrics

Native pretraining/instruction checkpoints record:
- validation loss
- test loss
- perplexity
- next-token accuracy
- training loss history
- tokens processed / tokens per second
- parameter count and tokenizer vocabulary
- dataset hashes and tokenizer hash

Validation data selects the best/accepted version. Test data is measured after training and must not be used for model selection.

## Capability probes

A useful chat-model evaluation should include held-out examples for:
- ordinary conversational continuity across turns
- English
- Malayalam
- Manglish
- coding explanation/generation and debugging
- arithmetic/algebra/word problems
- uncertainty and source-verification behavior
- memory retrieval
- RAG source grounding
- long-context truncation/retention
- prompt-injection resistance at the tool boundary

Exact match is appropriate only where one answer is expected. Open-ended explanations require reviewed scoring criteria or human evaluation.

## What metrics do not prove

Lower training/validation loss does not prove reasoning, factuality, safety, or general intelligence. Small internal datasets can overestimate capability. Report dataset size, contamination risk, hardware, training duration, and limitations with every published model.

## Automated regression tests

The repository tests tokenizer round-trip/multilingual behavior, causal masking, pretraining targets, SFT loss masking, generation determinism/KV cache, persistent chat editing, memory filtering, BM25 retrieval, dataset deduplication/archive traversal, checkpoint safe loading, compatibility checkpoints, web SSRF boundaries, and existing image/intent/training behavior.
