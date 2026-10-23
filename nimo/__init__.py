"""Nimo Lite: experimental local training, never pretrained by default."""
__version__ = "0.2.0"

# Keep training forward() untouched while installing local inference optimizations.
# These hooks operate only on Nimo's own locally trained model and download nothing.
try:
    from .fast_inference import install_fast_inference
    from .inference_policy import install_inference_policy

    install_fast_inference()
    install_inference_policy()
except Exception:
    # Optional acceleration must never prevent training/import on unsupported hardware.
    pass
