"""Nimo Lite: experimental local training, never pretrained by default."""
__version__ = "0.2.0"

# Keep training forward() untouched while installing the dedicated local inference path.
# Import-time patching is intentionally local to Nimo's own architecture and downloads
# no model weights or external runtime.
try:
    from .fast_inference import install_fast_inference
    install_fast_inference()
except Exception:
    # A missing optional acceleration capability must never prevent training/import.
    pass
