import os
import platform
import shutil
import psutil


def hardware(root='.'):
    mem=psutil.virtual_memory();limits=[]
    for path in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            raw=open(path).read().strip()
            if raw!="max":
                value=int(raw)
                if value>0:limits.append(value)
        except (OSError,ValueError):
            pass
    total=min([mem.total]+limits);available=min(mem.available,total)
    try:
        used=int(open('/sys/fs/cgroup/memory.current').read());available=min(available,max(0,total-used))
    except (OSError,ValueError):pass
    gib=total/2**30
    tier="low" if gib<10 else "standard" if gib<24 else "performance"
    defaults={
        "low":{"model_profile":"tiny","context":512,"batch_size":1,"gradient_accumulation":8,"tokenizer_vocab":8000},
        "standard":{"model_profile":"small","context":1024,"batch_size":2,"gradient_accumulation":4,"tokenizer_vocab":8000},
        "performance":{"model_profile":"medium","context":2048,"batch_size":4,"gradient_accumulation":2,"tokenizer_vocab":16000},
    }[tier]
    return dict(os=platform.platform(),cpu=platform.processor() or platform.machine(),cores=os.cpu_count(),
        ram_gb=round(gib,2),available_gb=round(available/2**30,2),disk_free_gb=round(shutil.disk_usage(root).free/2**30,2),
        tier=tier,profile=tier,threads=max(1,min(4,(os.cpu_count() or 2)//2)),image_size=32,
        process_ram_mb=round(psutil.Process().memory_info().rss/2**20,1),cpu_percent=psutil.cpu_percent(),
        **defaults)


def accelerator():
    import torch
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        return dict(device='cuda',name=torch.cuda.get_device_name(),vram_gb=round(total/2**30,2),free_vram_gb=round(free/2**30,2),
                    bf16=bool(torch.cuda.is_bf16_supported()))
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
        return dict(device='mps',name='Apple Metal',vram_gb=None,free_vram_gb=None,bf16=False)
    return dict(device='cpu',name='CPU',vram_gb=0,free_vram_gb=0,bf16=False)


def recommended_profile(root='.'):
    hw=hardware(root);acc=accelerator()
    profile=hw["model_profile"]
    if acc["device"]=="cpu" and profile=="medium":profile="small"
    return {"hardware_profile":hw["profile"],"model_profile":profile,"context":hw["context"],
            "batch_size":hw["batch_size"],"gradient_accumulation":hw["gradient_accumulation"],
            "tokenizer_vocab":hw["tokenizer_vocab"],"accelerator":acc}
