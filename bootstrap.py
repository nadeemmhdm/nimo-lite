"""One-command Nimo Lite installer/launcher using only the Python standard library."""
from __future__ import annotations
import io,os,shutil,subprocess,sys,urllib.request,venv,zipfile
from pathlib import Path
REPO_URL="https://github.com/nadeemmhdm/nimo-lite.git";ZIP_URL="https://github.com/nadeemmhdm/nimo-lite/archive/refs/heads/main.zip";MAX_ZIP=128*1024*1024

def run(cmd,cwd=None):print("[Nimo]"," ".join(map(str,cmd)));subprocess.check_call(list(map(str,cmd)),cwd=cwd)
def safe_extract(raw,target):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        total=0
        for info in archive.infolist():
            p=Path(info.filename)
            if p.is_absolute() or ".." in p.parts:raise RuntimeError("Unsafe repository archive")
            total+=info.file_size
            if total>MAX_ZIP:raise RuntimeError("Repository archive exceeds safety limit")
        archive.extractall(target)
def obtain_source(target):
    git=shutil.which("git")
    if (target/".git").exists() and git:
        try:
            remote=subprocess.check_output([git,"-C",str(target),"remote","get-url","origin"],text=True).strip();dirty=subprocess.check_output([git,"-C",str(target),"status","--porcelain"],text=True).strip()
            if remote.rstrip("/") in {REPO_URL.rstrip("/"),"https://github.com/nadeemmhdm/nimo-lite"} and not dirty:run([git,"-C",target,"pull","--ff-only","origin","main"]);return
        except Exception:pass
    if target.exists() and any(target.iterdir()):
        backup=target.with_name(target.name+".previous")
        if backup.exists():shutil.rmtree(backup)
        target.rename(backup)
    target.parent.mkdir(parents=True,exist_ok=True)
    if git:run([git,"clone","--depth","1",REPO_URL,target]);return
    req=urllib.request.Request(ZIP_URL,headers={"User-Agent":"NimoLite-Bootstrap/0.2.0"})
    with urllib.request.urlopen(req,timeout=30) as response:raw=response.read(MAX_ZIP+1)
    if len(raw)>MAX_ZIP:raise RuntimeError("Repository download too large")
    temp=target.parent/"nimo-download"
    if temp.exists():shutil.rmtree(temp)
    temp.mkdir();safe_extract(raw,temp);roots=[p for p in temp.iterdir() if p.is_dir()]
    if len(roots)!=1:raise RuntimeError("Unexpected repository archive")
    roots[0].rename(target);shutil.rmtree(temp,ignore_errors=True)
def main():
    root=Path(os.environ.get("NIMO_HOME",Path.home()/".nimo-lite")).expanduser().resolve();source=root/"app";env=root/"venv";data=root/"data";obtain_source(source)
    if not env.exists():print("[Nimo] Creating isolated Python environment");venv.EnvBuilder(with_pip=True).create(env)
    python=env/("Scripts/python.exe" if os.name=="nt" else "bin/python");run([python,"-m","pip","install","--upgrade","pip"]);run([python,"-m","pip","install","-e",source]);data.mkdir(parents=True,exist_ok=True)
    print("[Nimo] Starting local assistant at http://127.0.0.1:8080");os.chdir(source);raise SystemExit(subprocess.call([str(python),"-m","nimo.runtime","--data",str(data),"--port","8080"]))
if __name__=="__main__":main()
