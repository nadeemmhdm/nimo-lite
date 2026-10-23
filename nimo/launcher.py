"""Convenient launcher with optional release-based self-update."""
from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Launch Nimo Lite")
    parser.add_argument("--data", default=os.environ.get("NIMO_DATA", "data"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--auto-update",
        action="store_true",
        default=os.environ.get("NIMO_AUTO_UPDATE") == "1",
        help="Check/install a newer official GitHub release before launch",
    )
    parser.add_argument(
        "--compile",
        choices=("auto", "on", "off"),
        default=os.environ.get("NIMO_COMPILE", "auto"),
        help="torch.compile policy for Nimo inference. Auto enables only after a measured backend win.",
    )
    parser.add_argument(
        "--inference-threads",
        type=int,
        default=int(os.environ["NIMO_INFERENCE_THREADS"]) if os.environ.get("NIMO_INFERENCE_THREADS", "").isdigit() else None,
        help="Override CPU inference threads. Without this Nimo runs a small local tuning probe.",
    )
    parser.add_argument(
        "--warmup",
        choices=("auto", "on", "off"),
        default="auto",
        help="Small invisible model warmup after load.",
    )
    args = parser.parse_args()

    os.environ["NIMO_COMPILE"] = args.compile
    if args.inference_threads is not None:
        os.environ["NIMO_INFERENCE_THREADS"] = str(max(1, args.inference_threads))
    if args.warmup != "auto":
        os.environ["NIMO_WARMUP"] = "1" if args.warmup == "on" else "0"

    if args.auto_update:
        try:
            from .updater import install_update

            result = install_update()
            if result.get("installed"):
                print("Nimo update installed. Restart manually if package files are locked.")
        except Exception as exc:
            print("Update check skipped:", exc)

    import uvicorn
    from .runtime import create_app

    uvicorn.run(create_app(args.data), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
