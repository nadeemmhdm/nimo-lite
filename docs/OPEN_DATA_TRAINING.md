# Open-data training

Nimo can train its native tokenizer and NimoLM checkpoints from openly available datasets without using a pretrained external model or cloud inference service. Dataset download is an explicit action; no third-party corpus is bundled in this repository.

## Starter bundle

| Dataset | Use | Built-in subset/config | License recorded by Nimo |
| --- | --- | --- | --- |
| Wikimedia Wikipedia | Pretraining | `20231101.en` | CC BY-SA 3.0 + GFDL |
| Wikimedia Wikipedia | Pretraining | `20231101.ml` | CC BY-SA 3.0 + GFDL |
| OpenAssistant OASST1 | Instruction tuning | `train` | Apache-2.0 |
| Databricks Dolly 15k | Instruction tuning | `train` | CC BY-SA 3.0 |

FineWeb-Edu is available only through `--include-fineweb`. Its dataset card lists ODC-By 1.0 and also states that use is subject to Common Crawl Terms of Use, so Nimo does not include it in the default starter bundle.

Upstream references:

- https://huggingface.co/datasets/wikimedia/wikipedia
- https://huggingface.co/datasets/OpenAssistant/oasst1
- https://huggingface.co/datasets/databricks/databricks-dolly-15k
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- https://wikimediafoundation.org/what-we-do/wikimedia-projects/wikipedia/

Licenses and upstream terms can change. Before redistributing datasets or trained checkpoints, review the current upstream terms and attribution/share-alike requirements. Nimo records dataset identity, source URL, and license metadata in the corpus database, but this documentation is not legal advice and does not make a claim about how a dataset license applies to model weights in every jurisdiction.

## Install the optional dataset loader

```bash
pip install -e ".[open-data]"
```

The `datasets` dependency is optional so normal local chat/install remains lightweight.

## Inspect the catalog

```bash
nimo-open-data list
```

## Prepare data only

```bash
nimo-open-data prepare --data data --preset quick
```

Available presets:

- `quick`: 1,000 English Wikipedia chunks, 1,000 Malayalam Wikipedia chunks, up to 2,000 OASST1 conversations and 2,000 Dolly instructions.
- `balanced`: 5,000 English + 5,000 Malayalam Wikipedia chunks, up to 10,000 OASST1 conversations and 10,000 Dolly instructions.
- `large`: 25,000 English + 25,000 Malayalam Wikipedia chunks, up to 40,000 OASST1 conversations and 15,000 Dolly instructions.

Imports use exact SHA-256 deduplication and Nimo's stable checksum-based 90/5/5 train/validation/test split. Built-in curated imports skip the expensive recent-row SimHash scan so tens of thousands of rows can be ingested efficiently. User-supplied corpus imports keep the existing conservative near-duplicate checks.

To review imported data manually before training:

```bash
nimo-open-data prepare --data data --preset quick --pending
```

## Train end to end

The following command performs, in order: dataset preparation, local tokenizer training, NimoLM pretraining, and instruction tuning from the resulting pretraining checkpoint.

```bash
nimo-open-data train --data data --preset quick
```

For an 8 GB-class machine, start with a tiny model and a shorter context:

```bash
nimo-open-data train --data data --preset quick --profile tiny --context 512
```

To prepare once and train later without re-downloading/importing:

```bash
nimo-open-data prepare --data data --preset balanced
nimo-open-data train --data data --preset balanced --skip-prepare --profile tiny --context 512
```

FineWeb-Edu remains opt-in:

```bash
nimo-open-data train --data data --preset balanced --include-fineweb
```

Training is synchronous in this CLI. Checkpoints, optimizer state, dataset hashes, tokenizer checksum, validation/test metrics, and job progress are stored by the existing Nimo training engine. A `quick` run validates the workflow; it should not be interpreted as producing ChatGPT-scale capability. Stronger quality still requires substantially more high-quality data, compute, evaluation, and iterative training.

## What is transformed

Wikipedia rows become pretraining text documents with title/source metadata. Dolly rows become user/assistant instruction pairs; context is preserved inside the user message. OASST1 is reconstructed into non-overlapping conversation paths ending in a preferred assistant response where possible, instead of training on disconnected messages. Malayalam text is preserved as UTF-8 and handled by Nimo's byte-fallback tokenizer.

The open-data tokenizer path reads a deterministic checksum-ordered corpus stream and retains Nimo's existing tokenizer byte budget, avoiding the normal UI listing limit without loading the entire corpus into memory.
