# Roadmap and known limitations

## Current foundation

The first version trains tiny text and image-classification models locally. The default text chat mode retrieves saved notes verbatim with sources. It does not synthesize verified answers. Images can be manually annotated with multiple labels and boxes; only single-label classification has a training architecture.

## Next priorities

- Dedicated immutable evaluation datasets and persistent split assignment across dataset growth.
- Better multilingual sequence construction, held-out instruction benchmarks, and longer context within measured memory budgets.
- OCR architecture, detector training, captioning, and visual question answering, each independently evaluated.
- Validation of code examples inside an OS-enforced sandbox; currently code is stored and trained as text only.
- Multi-source search and explicit document approval with more complete provenance.
- Better poisoning review, near-duplicate methods, conflict resolution, and optional malware scanning.
- Hard OS resource quotas, per-worker restricted credentials, export streaming, encrypted backup, and data retention controls.
- Calibrated classifier confidence, independent external tests, and task-specific regression gates.
- Real-device Windows/macOS/CUDA/Metal and minimum-6-GB benchmarks.
- Native video decode limits, frame-group split protection by source clip, audio transcription, and temporal models.

## Explicitly not demonstrated

General reasoning, reliable mathematics or coding, Malayalam/Manglish conversational ability, object detection, OCR, screenshot understanding, visual question answering, video understanding, speech transcription, ChatGPT-equivalent quality, and universal jailbreak resistance. No corpus large enough to establish those capabilities is supplied. A UTF-8 tokenizer supports storing language examples but does not confer language proficiency.

The application is a small-data local prototype. It is not designed for millions of examples, adversarial multiuser hosting, or certified handling of regulated data. There is no llama.cpp/GGUF integration; the interface is inspired by compact local chat tools while the models run in PyTorch.
