"""Convenient launcher with optional release-based self-update."""
from __future__ import annotations
import argparse,os

def main():
    parser=argparse.ArgumentParser(description="Launch Nimo Lite");parser.add_argument("--data",default=os.environ.get("NIMO_DATA","data"));parser.add_argument("--port",type=int,default=8080);parser.add_argument("--auto-update",action="store_true",default=os.environ.get("NIMO_AUTO_UPDATE")=="1",help="Check/install a newer official GitHub release before launch");args=parser.parse_args()
    if args.auto_update:
        try:
            from .updater import install_update
            result=install_update()
            if result.get("installed"):print("Nimo update installed. Restart manually if package files are locked.")
        except Exception as exc:print("Update check skipped:",exc)
    import uvicorn
    from .runtime import create_app
    uvicorn.run(create_app(args.data),host="127.0.0.1",port=args.port,workers=1)
if __name__=="__main__":main()
