# Installation and operation

## Requirements

Python 3.10 or newer, approximately 1 GB free for the Python environment and CPU PyTorch wheel, plus space for images/checkpoints. At least 6 GB system RAM is the intended baseline, with 1.5 GiB currently available before training. Large GPU wheels can consume substantially more disk. A compatible PyTorch runtime is required for model inference/training.

Linux setup is in README.md. For GPU installation use the official PyTorch package selector and install the version matching your hardware before installing this project. No CUDA or Metal validation was performed in this environment. Optional video preparation:

```bash
python -m pip install -e '.[video]'
```

After dependencies are installed, disconnect the network and run:

```bash
nimo-lite --data ./data --port 8080
```

Only one server may own a data directory. Use `127.0.0.1` or `localhost`; external hostnames are deliberately rejected. The app never downloads weights at startup. First launch has no capable model; knowledge retrieval works as soon as notes are added. Model chat requires an accepted locally trained text checkpoint.

## Windows / macOS target instructions

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch
python -m pip install -e ".[test]"
nimo-lite --data .\data
```

macOS: follow Linux commands with a compatible macOS PyTorch wheel. Apple Silicon may select Metal inside training. These instructions have not been verified on those operating systems; no installation certification is claimed.

## Backup, export, and deletion

Use Export Dataset to create a portable ZIP containing original and processed images, a manifest, text examples, and COCO boxes. Export Model downloads a local checkpoint. To back up everything, stop the server and copy the complete data directory into protected storage. Restore a full backup only while the server is stopped. The application does not unpack arbitrary archive imports.

Use per-sample and per-note Delete controls for routine removal. To reset all local state, stop the application and move the data directory aside before launching with a fresh directory. Existing checkpoints may retain learned information after sample deletion. SQL deletion is not forensic secure erasure.

## Trained starter installation

Download the starter ZIP from the repository release and verify SHA256SUMS. Install only a bundle you trust:

```bash
python -m nimo.install nimo-lite-starter-v0.2.0.zip --data ./data
```

The local CLI accepts only fixed supported checkpoint member names, bounded expanded sizes, matching checksums, and safe-loadable models within architecture limits. It imports no training examples automatically. Public evaluation examples are stored separately for reevaluation. Installation changes active model versions and retains older versions for rollback. Stop training before installation.

## Repository and releases

The existing repository is `nadeemmhdm/nimo-lite`. Its private visibility is preserved. GitHub access is required for cloning and downloading release assets while private. Source licensing is MIT; repository visibility is a separate setting.

The earlier `nemo-lite` name was not an accessible repository. Creation of a new repository is restricted by the connection, but the existing `nimo-lite` repository grants push access. Release availability is shown on its GitHub Releases page.
