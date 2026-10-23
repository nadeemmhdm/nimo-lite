# Changelog

All notable project changes are documented here. Versions follow semantic versioning once the public API stabilizes.

## 0.2.0 — Assistant starter and interface redesign

### Added

- Dark/light themes, local preference persistence, responsive mobile navigation, keyboard-friendly chat, and explicit answer-source labels.
- Guided training steps, original starter-data loading, paginated/filterable review, guarded bulk approval, and clear experiment metrics.
- Original trained 11-class NLP intent model with bounded authored assistant responses.
- Real 80-epoch language-model starter experiment and 60-epoch intent experiment on 140 original CC0 examples, with full provenance and unedited generation probes.
- Bounded offline model-bundle installation, checksum verification, separate immutable public reevaluation, and a detailed model card.

### Changed

- New language runs use answer-only targets and optional 256-byte context. Legacy checkpoint objectives remain resumable.
- Larger dataset splits reserve approximately 10% each for validation/test within leakage groups.
- Prior-training leakage checks include checksums when sample IDs change across reimport.
- Knowledge retrieval ignores common English stop words to reduce irrelevant matches.

### Verified

- 26 Python tests and expanded dark/light/mobile browser checks. Linux CPU only; internal tiny-data results are not general capability evidence.

## 0.1.0 — Initial experimental source

### Added

- Original byte tokenizer, tiny causal Transformer, and single-label image CNN.
- Local dashboard for chat, example review, image annotation, datasets, models, knowledge, security, and resources.
- Real training jobs, loss and held-out metrics, checkpoint save/pause/resume/stop, validation-based activation, and rollback.
- Source/license records, checksums, duplicate handling, image near-duplicate grouping, versioned annotations, and dataset audit trail.
- Local feedback and knowledge storage; optional fixed-provider Wikipedia search.
- COCO annotation export and optional video-frame preparation.
- Local request boundaries, documented limitations, tests, and original synthetic examples.

### Release status

Experimental source release candidate. No hosted GitHub release, production certification, cross-platform certification, or pretrained assistant capability is implied by this entry. See docs/EVALUATION.md for actual verification results.
