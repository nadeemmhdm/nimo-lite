"""Optional sequential video sampling; no video understanding or audio transcription claim."""
import io
import math
from PIL import Image
from .store import uid


def extract(store, raw, source, license):
    if not source.strip() or not license.strip(): raise ValueError('Source and usage declaration required')
    try: import cv2
    except ImportError as exc: raise ValueError('Install nimo-lite[video] to extract frames') from exc
    path=store.root/'video'/f'{uid()}.video'
    path.write_bytes(raw)
    cap=cv2.VideoCapture(str(path)); results=[]
    try:
        if not cap.isOpened(): raise ValueError('Unsupported or corrupt video')
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps<=0: raise ValueError('Video has invalid frame rate')
        # At most 30 seconds, one frame per second, sequential decoding.
        for second in range(30):
            cap.set(cv2.CAP_PROP_POS_MSEC,second*1000)
            ok,frame=cap.read()
            if not ok: break
            if frame.shape[0]*frame.shape[1]>16_000_000: raise ValueError('Frame exceeds 16 MP')
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); image.thumbnail((1024,1024))
            out=io.BytesIO(); image.save(out,format='PNG')
            sample=store.add_image(out.getvalue(),f'{source} @ {second}s',license)
            results.append(dict(sample=sample,timestamp_s=second))
        if not results: raise ValueError('No frames could be decoded')
    finally:
        cap.release(); path.unlink(missing_ok=True)
    return dict(frames=results,note='Frames pending annotation; no automatic captions or transcription')
