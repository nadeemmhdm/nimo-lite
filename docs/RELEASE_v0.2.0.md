# Nimo Lite v0.2.0

A local assistant starter with original trained models and a clearer learning workflow.

## What's included

- Dark and light themes with system-aware defaults and local preference persistence.
- A redesigned chat composer, quick prompts, response-source labels, mobile navigation, accessible form labels, and visible keyboard focus.
- Guided prepare → review → train → evaluate workflow, searchable/paginated datasets, explicitly gated batch review, live progress, and compact model comparisons.
- 140 original CC0 educational examples covering basic assistant behavior, ML/NLP, privacy, uncertainty, and small math/code/language examples.
- A randomly initialized NLP intent classifier trained for 60 epochs and a separate original causal byte language model trained for 80 epochs.
- Answer-focused language training, optional 256-byte context, checksum-aware previous-training overlap checks, and bounded local installation of release models with immutable public reevaluation samples.

## Measured results and limits

The intent model correctly classified **7 of 11 held-out requests** (macro F1 0.5303). The language model reached **47.09% held-out byte accuracy**, with perplexity 88.74. These small authored internal tests do not establish general assistant quality, fluency, or reasoning. The generator still produces corrupted text on some prompts. Recommended assistant mode uses sourced notes and clearly identified intent routing to authored responses.

**26 Python tests pass**, including real training, checkpoints, intent behavior, bundle integrity, review gates, and security regression checks. Browser checks pass for both themes, theme persistence, mobile layout, knowledge/intent chat, image approval, batch review, export, and installed-model evaluation. Only Linux CPU execution is verified.

## Install the starter weights

1. Download the source and `nimo-lite-starter-v0.2.0.zip`.
2. Set up the Python environment using README.md.
3. Check the download against `SHA256SUMS`.
4. Run `python -m nimo.install nimo-lite-starter-v0.2.0.zip --data ./data`.
5. Start `nimo-lite --data ./data` and open http://127.0.0.1:8080.

The release bundle contains only original public training fixtures and their model artifacts. It includes no personal workspace notes, images, credentials, or user feedback. Code is MIT licensed; original starter data and starter weights are CC0-1.0. No external pretrained weights are included.

The application remains experimental, single-user, and loopback-only. There is no OS sandbox, storage encryption, hard process memory quota, complete poisoning detector, or completed standardized Codex Security scan. See SECURITY.md and docs/STARTER_MODEL_CARD.md.
