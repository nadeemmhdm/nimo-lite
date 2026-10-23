# Verified evaluation record

## v0.2.0 verification

**26 automated Python tests pass** on the Linux CPU environment. The expanded suite covers NLP intent training, answer-target masking, original math/code fixtures, review-gated starter import, public model-bundle installation, checksum tampering rejection, immutable bundled reevaluation, source-aware assistant replies, architecture bounds, and prior-training checksum overlap after reimport. The original training/media/security checks remain included. A built v0.2.0 wheel was inspected for required resources and exclusion of local datasets/tokens/weights, then imported from an isolated installation target; app startup, starter-data access, and local theme assets passed. Explicit package discovery prevents accidental inclusion of workspace artifact directories. One upstream Starlette TestClient deprecation warning remains.

Playwright/Chromium checks pass for desktop dark/light themes, theme persistence after reload, quick prompts, trained assistant replies, sourced knowledge chat, original image upload and explicit label approval, review-gated batch approval, dataset export, installed-model evaluation, all main routes, and 390px mobile navigation without horizontal overflow. No JavaScript page errors were observed. Screenshots are in `docs/screenshots/`.

The v0.2.0 starter experiment used 140 original authored CC0 examples and random model initialization. The **intent classifier scored 7/11 on the internal test set**, macro F1 0.53030. The **language model scored 47.09% byte accuracy** on 1,444 held-out byte targets, loss 4.48575 and perplexity 88.74340. Intent validation was 9/11; text validation byte accuracy was 57.87%. These are small internal educational holdouts, not reliable external generalization estimates. The original generator still emits malformed text on some prompts; the full unedited probes are preserved.

See [STARTER_MODEL_CARD.md](STARTER_MODEL_CARD.md) and [starter-training-results.json](starter-training-results.json) for source hashes, architecture, optimizer, epochs, splits, acceptance decisions, hardware observation, and exact probe outputs. This run reported approximately 371.6 MiB RSS after training, not peak memory. Both training loops and generation probes completed in 44.3 seconds in this environment; this is not a portable hardware guarantee.

The recommended assistant uses learned intent routing and authored responses, not a fabricated claim that the tiny generator has become generally capable. Knowledge retrieval and weight training remain separate. Only the original public fixtures were used for release weights; no private workspace data was included.

## Historical v0.1.0 smoke experiment

The following record describes the original execution baseline and remains for comparison. It uses the earlier full-sequence objective and default 128-byte model, so its losses are not directly comparable with v0.2.0 answer-only training.

## Environment and method

Validation performed on Linux x86_64, Python 3.12.14, CPU PyTorch 2.14.1+cpu, three visible CPU cores, and an 8 GiB cgroup memory ceiling. The host exposes more memory; Nimo honors the cgroup ceiling. These are not results from a 6 GB physical computer. No CUDA, Metal, Windows, or macOS claim is made.

Run the reproducible pipeline experiment with:

```bash
python scripts/smoke_experiment.py
```

Its complete machine-readable record is [smoke-results.json](smoke-results.json). Models use 124,928 parameters for text and 38,050 for the two-class CNN. All weights begin randomly, seed 42. No pretrained models or external training corpora were used.

## Synthetic image experiment

Twelve original seeded color/noise images, six per class; eight train, two validation, two test. Six training epochs. Validation cross-entropy decreased from **0.69135** to **0.00002289**. Test accuracy, macro precision, macro recall, and macro F1 were **1.0 on two test images**.

This trivially separable red/green fixture demonstrates connected labels, optimization, held-out execution, metrics, and checkpoint activation. Two synthetic test images cannot establish useful image recognition, robustness, calibrated confidence, or generalization to photographs. Near-duplicate grouping uses an approximate average hash, not a complete leakage detector.

## Synthetic text experiment

Eight short original counting-format examples; six train, one validation, one test. Two epochs. Validation byte cross-entropy decreased from **5.65514** to **3.45099**. Held-out test byte loss was **3.48483**, byte accuracy **0.48889**, and perplexity **32.61675** on 45 next-byte targets.

These measurements do not establish counting accuracy, reasoning, coding, factuality, Malayalam fluency, or conversation ability. Closely templated examples are intentionally a pipeline smoke test. No useful general assistant has been trained by this experiment.

## Functional verification

The automated Python suite contains **17 passing tests**, covering:

- UTF-8 round trips for English, Malayalam, Manglish, mathematics, and code; causality and random initialization.
- Real classifier and language-model training, held-out splits, checkpoint safe loading, pause/stop, and resumed weight updates.
- Actual subprocess training, evaluation endpoint, accepted-version activation, export, and unverified classifier suggestions.
- Image import, original/processed separation, corrupt-image rejection, duplicate detection, labels, normalized boxes, and COCO export.
- Dataset review, contradiction detection, near-duplicate conflict rejection, category renaming, and advisory sensitive/injection review.
- Offline local retrieval without attempted network connection, stored feedback remaining separate from training, and note deletion.
- Host/Origin/token checks, path confinement, memory preflight, and fixed-provider search source attribution.
- Generated local video decoding and timestamped pending frame samples through optional OpenCV 5.0.0.93.

The test runtime includes a Starlette TestClient deprecation warning from installed dependencies. It does not affect the observed test outcomes. Python compilation and JavaScript syntax checking also passed.

## Browser and internet verification

A Playwright/Chromium browser check exercises desktop/mobile navigation, sourced knowledge entry/retrieval, image upload, explicit annotation verification and approval, dataset export download, and all main page routes. The reproducible optional script is `scripts/browser-smoke.cjs`; it requires Playwright/Chromium and a running local server. The screenshot is [screenshots/workbench.png](screenshots/workbench.png).

A live request through the application's explicit online-search flow returned HTTP 200 and five Wikipedia results for the nonprivate query `machine learning`; online mode was disabled afterward. Automated attribution tests also mock the provider to verify fixed-origin requests, retrieval dates, untrusted labels, and no automatic memory/training import. Cross-source factual verification is not implemented.

## Performance limits

The smoke process reported approximately **365 MiB RSS** after the tiny experiments. This is a single observation, not peak memory, an OS memory reservation, or a minimum-RAM guarantee. Recorded training durations exclude dependency installation, worker startup, and some initialization; they are not end-to-end latency benchmarks. Inference latency is measured per actual chat request in the UI. No broad performance or accuracy benchmark has been performed.

A dedicated frozen external benchmark, meaningful task datasets, robustness evaluation, poisoning evaluation, and real-device low-memory/cross-platform measurements remain future work.
