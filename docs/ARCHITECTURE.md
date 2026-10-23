# Architecture

## Components and trust boundaries

The local browser communicates with one loopback FastAPI process. SQLite in WAL mode stores samples, knowledge, feedback, jobs, model metadata, and audit events. Image bytes, processed PNGs, local model files, and logs are stored under the chosen data directory. The default is `./data` relative to the launch directory.

The server launches one fixed Python module in a child process for training. Generated text and dataset content never enter command arguments, a shell, or Python evaluation. No model tools are exposed. The worker shares the user's OS privileges; it is not a native sandbox. Optional video decoding should only process trusted local files.

```mermaid
flowchart LR
  U[Local browser] --> A[Loopback API]
  A --> S[(SQLite: review and metadata)]
  A --> I[Original and processed images]
  A --> K[Lexical knowledge retrieval]
  A --> W[Single training worker]
  W --> T[Original byte Transformer]
  W --> V[Original classifier CNN]
  W --> C[Atomic local checkpoints]
  A -->|Explicit query, online enabled| E[Fixed Wikipedia API]
```

## Original models

**NimoLM**: a UTF-8 byte tokenizer with 259 tokens (256 byte values plus PAD, BOS, EOS), 64-dimensional token/position embeddings, two causal Transformer encoder blocks, four attention heads, 192-dimensional feed-forward layers, LayerNorm, and a vocabulary projection. Causal masks prevent future-token attention. All weights use random framework initialization. New training uses answer-only next-byte targets: prompt targets are masked, questions are truncated to reserve at least half the context for answers, and context can be 128 or 256 bytes. Existing v0.1 checkpoints retain their full-sequence training objective on resume. No pretrained tokenizer, embeddings, or weights are used.

**NimoVision**: two convolution layers (3→16→32), ReLU, spatial pooling, a 512→64 projection, and a class head. Input is RGB, resized bilinearly to 32×32 and scaled to [0,1]. This architecture is for single-label classification, not boxes, captions, OCR, or vision-language reasoning.

**NimoIntent**: stable Unicode word/character feature hashing into 2048 dimensions, L2 normalization, and a randomly initialized 11-class linear layer. It trains supervised request classification independently of the language model. Classifier outputs map through an explicit response allowlist and never select executable tools. Confidence and margin thresholds trigger fallback; scores are uncalibrated.

Training uses AdamW with learning rate 0.003 (0.03 for the intent classifier), clipped gradient norm 1, deterministic per-epoch sample order, and seed 42. CUDA and Metal detection are implemented; CPU is the tested execution path. Checkpoint files include optimizer and RNG states plus a version/checksum snapshot.

## Splits and evaluation

Exact duplicate imports are rejected using canonical decoded-image or normalized text-pair checksums. Image average-hash distance at most four creates transitive near-duplicate groups; normalized equal text questions share groups. Groups are partitioned deterministically by class. Each bucket reserves approximately 10% of its groups for test and 10% for validation (at least one group each), assigning the rest to training. Intent buckets are grouped by label; text generation uses one overall text bucket. This deliberately conservative small-data split is not a configurable production split strategy.

A near-duplicate image group with inconsistent labels fails validation. Every image category must exist independently in every split. A tiny validation/test set is sufficient only to verify execution, not establish generalization. The approximate hash can miss related images or incorrectly group unrelated images. Video frames from the same clip are not yet guaranteed to share a split, so do not use frame datasets as evidence of independent video generalization.

New runs begin from random weights unless a local checkpoint is explicitly selected. The worker compares validation loss against random initialization or a compatible active version evaluated on the same validation data. Image accuracy must also reach at least 0.5 and not fall below the reference. Test data is measured after training, not used for activation. Cross-task capability regression gates are not implemented.

## Resource behavior

RAM detection includes Linux cgroup limits. Profiles choose batches 2/4/8/16 for memory tiers 6/8/16/32 GB; all use fixed small architectures, 128- or 256-byte text contexts, 32×32 inputs, and at most two CPU threads. Training requires 1.5 GiB available memory and 0.2 GiB disk before launch; the worker checks for less than 0.5 GiB available at batch boundaries. VRAM below 1 GiB causes CPU fallback. This does not reserve memory or enforce an OS quota. A single forward/backward step cannot be interrupted safely until it completes.

Imports are bounded to 12 MiB per request and 16 MP per image; datasets are limited to 5000 samples. Image tensors are loaded on demand by batch. The API still reads sample metadata and assembles ZIP exports in memory; keep datasets small. Dataset mutation is blocked during queued, running, or paused jobs. Separate inference requests load only the selected local model and are serialized with a lock.

## Model distribution

The release builder trains only original data in a separate ignored artifact directory. A bounded ZIP contains accepted text/intent checkpoints, source hashes, the public data, and public evaluation rows. The local installer checks expanded size, supported member names, checkpoint/evaluation checksums, model task, and architecture bounds before registration. It retains old model versions and stores public test rows separately rather than approving them for future training. Reevaluation verifies the stored evaluation checksum and uses the original held-out split. Checksum integrity is not publisher authentication; the user must trust the source.

## Interface state

The static local interface uses semantic forms, explicit labels, keyboard focus styles, and a responsive navigation drawer. Dark/light preference is stored in browser localStorage and never sent to the server. Source labels distinguish authored intent replies, retrieved knowledge, and experimental generation. Bulk review requires explicit acknowledgment and preserves per-example flagged-data checks. Dataset filtering and pagination only change the view; they never silently approve examples.
