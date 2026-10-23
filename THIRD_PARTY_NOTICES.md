# Third-party notices

Nimo Lite's own source is MIT licensed. Its architectures use standard published Transformer and convolutional-network concepts implemented through PyTorch. No external model weights, scraped corpora, or third-party model source snapshots are bundled.

Dependencies retain their own licenses. Examples include PyTorch (BSD-style), FastAPI and Uvicorn (BSD-family), Pillow (HPND), NumPy (BSD), psutil (BSD), python-multipart (Apache 2.0), httpx (BSD), pytest (MIT), optional Hugging Face `datasets` (Apache 2.0), and optional OpenCV (Apache 2.0 for recent releases, with bundled component notices). Consult each installed distribution's license and bundled notices before redistribution; this list is an orientation, not a legal compliance audit.

The original synthetic examples, generated test fixtures, and original v0.2.0 starter weights are offered under CC0-1.0. User-provided datasets and model artifacts are not automatically covered by the source code license. Their source, licensing, privacy, and consent obligations remain with the importer. The application records declarations and cannot verify legal rights.

## Optional open-data downloader

Nimo does not bundle these datasets. When a user explicitly runs `nimo-data`, the optional downloader can fetch and cache selected upstream datasets on that user's device.

- Wikimedia Wikipedia English and Malayalam snapshots: recorded as `CC-BY-SA-3.0 AND GFDL` by Nimo's dataset catalog. Attribution/share-alike obligations remain applicable.
- OpenAssistant OASST1: Apache-2.0.
- Databricks Dolly 15K: CC-BY-SA-3.0.
- CodeAlpaca (`thisisanshgupta/CodeAlpaca`): Apache-2.0 according to its Hugging Face dataset card.
- OpenAI GSM8K: MIT according to its Hugging Face dataset card.
- FineWeb-Edu: ODC-BY-1.0 and explicit opt-in; use is also subject to Common Crawl Terms of Use.

Dataset cards, terms, licenses, and upstream contents can change. Before redistributing a trained model or dataset derivative, re-check the exact upstream revision and obligations that applied to the data you used. Nimo stores dataset keys, declared licenses, and provenance metadata to make that review easier.

Optional search uses Wikipedia's API and links to source articles. Respect Wikimedia API usage policy and each source's attribution/license requirements. Search results are not bundled training data.
