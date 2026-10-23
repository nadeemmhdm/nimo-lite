# Verified evaluation record

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
