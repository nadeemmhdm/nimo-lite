# Nimo Lite Starter v0.2.0

## Purpose

Two small original models trained from random initialization for local experimentation. Recommended assistant mode uses a trained intent classifier and explicit authored responses, with separate local-knowledge retrieval. The text generator is an experimental option and can produce nonsense. Neither model is a generally capable personal assistant.

## Models and training

| Component | Architecture | Objective | Training |
|---|---|---|---|
| NLP intent routing | 2048 hashed word/character features → 11-class linear head, 22,539 parameters | Cross-entropy classification | AdamW, learning rate 0.03, 60 epochs |
| Original language model | 2-layer causal Transformer, width 64, 4 heads, context 256 bytes, 133,120 parameters | Answer-only next-byte cross-entropy; prompt targets masked | AdamW, learning rate 0.003, 80 epochs |

Both use seed 42, norm clipping 1, batch size 4, and CPU execution. No pretrained embeddings, tokenizer, model weights, or remote training service were used. The tokenizer is UTF-8 bytes plus three special tokens. This small release run completed in approximately 44.3 seconds including both training loops and generation probes on the recorded Linux CPU environment; this is not a portable performance guarantee.

## Data and provenance

`nimo/resources/assistant-basics.json` contains **140 original CC0-1.0 examples** authored for this project: 132 utterances across 11 basic intents, plus eight short math, code, and Malayalam/Manglish examples. This is a narrow educational dataset, not a large factual corpus. No private workspace notes, uploaded user images, scraped pages, proprietary datasets, or user feedback were used.

Intent labels cover greetings, identity, help, privacy, teaching, knowledge, machine learning, NLP, thanks, uncertainty, and requests outside scope. The assistant response text is an explicit application policy in `nimo/nlp.py`; the classifier selects an allowed response. It does not generate that response. Low-confidence predictions fall back to an uncertainty message. A confidence score is uncalibrated and can be misleading.

The exact source checksum, split IDs, metrics, hardware observation, and generation probes are in `docs/starter-training-results.json`. Each intent has 10 training, one validation, and one test example. Text training uses 112 train, 14 validation, and 14 test examples. Equal normalized questions stay in the same split. Similar paraphrases may still cross splits; these are internal authored holdouts, not an independent external benchmark.

## Actual results

| Measurement | Intent classifier | Language model |
|---|---:|---:|
| Validation accuracy | 81.82% (9/11 requests) | 57.87% (byte targets) |
| Held-out test accuracy | 63.64% (7/11 requests) | 47.09% (1,444 byte targets) |
| Test cross-entropy | 1.03281 | 4.48575 |
| Test macro F1 | 0.53030 | Not applicable |
| Test perplexity | Not applicable | 88.74340 |

Activation used validation criteria, not test results. These acceptance criteria are minimal pipeline gates, not a quality or safety certification. The high training familiarity of sample probes is not generalization evidence. The generator reproduced some familiar ML/NLP answers and produced corrupted text for other prompts. The full unedited probe outputs are retained in the report.

## Behavior and limits

Assistant mode retrieves local notes with sources when they match. Otherwise it routes a request through the active intent model to one of the allowed authored responses. It has no shell, arbitrary filesystem, banking, booking, or network-execution tools. Knowledge storage, supervised classification, and language generation are displayed separately in the interface.

This release does not demonstrate fluent multilingual chat, general reasoning, reliable code generation, visual question answering, OCR, object detection, or universal prompt-injection resistance. The 11-request intent test is very small; expect misclassification. Important answers require verification. Do not use the models for medical, financial, legal, or safety-critical decisions.

## Installation and redistribution

The release ZIP includes model checkpoints, SHA-256 hashes, the original public data, and an immutable public evaluation set. It includes no personal data and imports no new training examples automatically. From the project environment:

```bash
python -m nimo.install nimo-lite-starter-v0.2.0.zip --data ./data
nimo-lite --data ./data
```

Install only a bundle from a source you trust. Checksums detect file changes; they do not authenticate a publisher. Local model installation uses a bounded archive reader and PyTorch `weights_only=True`; it is not an OS sandbox. The original starter data and these experimental weights are released under CC0-1.0, with no warranty. Application source remains MIT licensed.
