# Installation

## One-command bootstrap

Linux/macOS:
```bash
python3 -c "import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/nadeemmhdm/nimo-lite/main/bootstrap.py').read())"
```

Windows PowerShell:
```powershell
py -c "import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/nadeemmhdm/nimo-lite/main/bootstrap.py').read())"
```

The bootstrap uses Git if available, otherwise downloads a bounded official GitHub ZIP; creates `~/.nimo-lite/venv`; installs the package and dependencies; creates `~/.nimo-lite/data`; and starts the loopback server.

Override the installation root with `NIMO_HOME`.

## Manual

```bash
git clone https://github.com/nadeemmhdm/nimo-lite.git
cd nimo-lite
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
nimo-run --data ./data
```

Windows activation: `.venv\Scripts\Activate.ps1`.

## Updates

```bash
nimo-update --check
nimo-update
nimo-run --auto-update
```

Update checks require internet and are opt-in. Normal inference/training does not.

## Hardware

Python 3.10+ and PyTorch are required. CPU is supported. CUDA and Apple Metal/MPS are detected automatically. Start with the Tiny profile on low-memory machines.

The server intentionally binds only to `127.0.0.1`; do not expose it as a public service.
