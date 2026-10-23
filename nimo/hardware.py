import os
import platform
import shutil
import psutil


def hardware(root='.'):
    mem = psutil.virtual_memory()
    # Linux containers can expose host memory; honor cgroup ceilings.
    limits = []
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            value = int(open(path).read().strip())
            if value > 0:
                limits.append(value)
        except (OSError, ValueError):
            pass
    total = min([mem.total] + limits)
    available = min(mem.available, total)
    try:
        used = int(open('/sys/fs/cgroup/memory.current').read())
        available = min(available, max(0, total-used))
    except (OSError, ValueError):
        pass
    gib = total / 2**30
    tier = 6 if gib < 8 else 8 if gib < 12 else 16 if gib < 32 else 32
    return dict(os=platform.platform(), cpu=platform.processor() or platform.machine(),
                cores=os.cpu_count(), ram_gb=round(gib, 2), available_gb=round(available/2**30, 2),
                disk_free_gb=round(shutil.disk_usage(root).free/2**30, 2), tier=tier,
                batch_size={6:2, 8:4, 16:8, 32:16}[tier], context=128,
                image_size=32, threads=max(1, min(2, (os.cpu_count() or 2)//2)),
                process_ram_mb=round(psutil.Process().memory_info().rss/2**20, 1),
                cpu_percent=psutil.cpu_percent())


def accelerator():
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return dict(device='cuda', name=torch.cuda.get_device_name(), vram_gb=total/2**30,
                    free_vram_gb=free/2**30)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return dict(device='mps', name='Apple Metal', vram_gb=None)
    return dict(device='cpu', name='CPU', vram_gb=0)
