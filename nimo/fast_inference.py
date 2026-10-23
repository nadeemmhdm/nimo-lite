"""Fast local inference path for NimoLMV2.

Training keeps using NimoLMV2.forward(). This module installs an inference-only path with
preallocated per-layer KV buffers, prompt-prefix reuse, optimized SDPA, low-overhead
sampling, CPU thread tuning, tokenizer reuse, optional benchmark-gated torch.compile,
and measured runtime metrics. It never downloads weights.
"""
from __future__ import annotations

import codecs
import os
import threading
import time
import weakref
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

_INSTALLED = False
_ENGINES: "weakref.WeakKeyDictionary[object, FastInferenceEngine]" = weakref.WeakKeyDictionary()
_ENGINES_LOCK = threading.RLock()
_THREAD_TUNED = False
_THREAD_INFO = {"selected": None, "candidates": {}}
_TOKENIZER_CACHE: dict[str, object] = {}
_TOKENIZER_CACHE_LOCK = threading.RLock()
_COMPILE_MODES = {"auto", "on", "off"}


def _device_name(model) -> str:
    try:
        device = next(model.parameters()).device.type
    except StopIteration:
        return "cpu"
    return str(device)


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _compile_mode() -> str:
    requested = os.environ.get("NIMO_COMPILE", "auto").strip().lower() or "auto"
    return requested if requested in _COMPILE_MODES else "auto"


def tune_cpu_threads(force: bool = False) -> dict:
    """Small one-time local matmul benchmark; users can override with NIMO_INFERENCE_THREADS."""
    global _THREAD_TUNED, _THREAD_INFO
    if _THREAD_TUNED and not force:
        return dict(_THREAD_INFO)
    override = os.environ.get("NIMO_INFERENCE_THREADS", "").strip()
    if override:
        chosen = max(1, int(override))
        torch.set_num_threads(chosen)
        _THREAD_INFO = {"selected": chosen, "candidates": {}, "source": "NIMO_INFERENCE_THREADS"}
        _THREAD_TUNED = True
        return dict(_THREAD_INFO)
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except Exception:
        physical = os.cpu_count() or 1
    candidates = sorted({1, 2, 4, max(1, min(int(physical), 8))})
    candidates = [x for x in candidates if x <= max(1, os.cpu_count() or x)]
    torch.manual_seed(123)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    timings = {}
    previous = torch.get_num_threads()
    try:
        for threads in candidates:
            torch.set_num_threads(threads)
            _ = a @ b
            samples = []
            for _ in range(2):
                start = time.perf_counter()
                _ = a @ b
                samples.append(time.perf_counter() - start)
            timings[threads] = min(samples)
        chosen = min(timings, key=timings.get) if timings else previous
        torch.set_num_threads(chosen)
    except RuntimeError:
        chosen = previous
        try:
            torch.set_num_threads(previous)
        except RuntimeError:
            pass
    _THREAD_INFO = {
        "selected": int(chosen),
        "candidates": {str(k): v for k, v in timings.items()},
        "source": "microbenchmark",
    }
    _THREAD_TUNED = True
    return dict(_THREAD_INFO)


@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


class KVCache:
    """Batch-1 preallocated cache. No growing torch.cat is performed during decode."""

    def __init__(self, model):
        self.context = int(model.context)
        self.layers: list[LayerKV] = []
        device = next(model.parameters()).device
        dtype = model.token.weight.dtype
        for block in model.blocks:
            heads = int(block.attn.heads)
            head_dim = int(block.attn.head_dim)
            shape = (1, heads, self.context, head_dim)
            self.layers.append(
                LayerKV(
                    torch.empty(shape, device=device, dtype=dtype),
                    torch.empty(shape, device=device, dtype=dtype),
                )
            )
        self.length = 0

    @property
    def bytes(self) -> int:
        return sum(
            layer.key.numel() * layer.key.element_size()
            + layer.value.numel() * layer.value.element_size()
            for layer in self.layers
        )

    def reset(self) -> None:
        self.length = 0

    def truncate(self, length: int) -> None:
        if not 0 <= int(length) <= self.length:
            raise ValueError("Invalid KV cache truncation")
        self.length = int(length)


@dataclass
class DecodeSession:
    cache: KVCache
    tokens: list[int] = field(default_factory=list)
    logits: torch.Tensor | None = None

    def reset(self) -> None:
        self.cache.reset()
        self.tokens.clear()
        self.logits = None


def _common_prefix(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


class FastInferenceEngine:
    """Dedicated prefill/decode engine for one local Nimo model instance."""

    def __init__(self, model):
        # Keep only a weak reference so clearing the runtime ModelCache can actually
        # unload a model. WeakKeyDictionary values must not strongly retain their key.
        self._model_ref = weakref.ref(model)
        model.eval()
        self.device = _device_name(model)
        if self.device == "cpu":
            tune_cpu_threads()
        self.session = DecodeSession(KVCache(model))
        self.lock = threading.RLock()
        self.last_metrics = {}
        self.compile_mode = _compile_mode()
        self.compile_info = {
            "requested": self.compile_mode,
            "selected": "off",
            "reason": "not evaluated yet",
        }
        self._compile_checked = False
        self._compiled_forward = None

    @property
    def model(self):
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Nimo model was unloaded")
        return model

    def reset(self) -> None:
        with self.lock:
            self.session.reset()

    def _forward_eager(self, token_ids: list[int] | torch.Tensor, start_pos: int) -> torch.Tensor:
        """Inference-only forward writing K/V directly into preallocated cache buffers."""
        from .models import _rope

        model = self.model
        if isinstance(token_ids, torch.Tensor):
            tokens = token_ids.to(device=next(model.parameters()).device, dtype=torch.long)
            if tokens.ndim == 1:
                tokens = tokens.unsqueeze(0)
        else:
            tokens = torch.tensor(
                [token_ids], device=next(model.parameters()).device, dtype=torch.long
            )
        batch, length = tokens.shape
        if batch != 1:
            raise ValueError("Fast inference path is optimized for batch size 1")
        end = start_pos + length
        if end > model.context:
            raise ValueError("Input exceeds model context")

        positions = torch.arange(start_pos, end, device=tokens.device)
        x = model.token(tokens)
        for index, block in enumerate(model.blocks):
            attn = block.attn
            qkv = attn.qkv(block.norm1(x)).view(
                1, length, 3, attn.heads, attn.head_dim
            )
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            q, k = _rope(q, k, positions, attn.rope_base)
            layer = self.session.cache.layers[index]
            layer.key[:, :, start_pos:end, :].copy_(k)
            layer.value[:, :, start_pos:end, :].copy_(v)
            keys = layer.key[:, :, :end, :]
            values = layer.value[:, :, :end, :]

            if start_pos == 0:
                y = F.scaled_dot_product_attention(
                    q, keys, values, dropout_p=0.0, is_causal=length > 1
                )
            elif length == 1:
                y = F.scaled_dot_product_attention(
                    q, keys, values, dropout_p=0.0, is_causal=False
                )
            else:
                # Prefix extension needs an offset-causal mask. Single-token decode
                # never allocates a square causal mask.
                key_pos = torch.arange(end, device=tokens.device)
                query_pos = torch.arange(start_pos, end, device=tokens.device)
                allowed = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
                y = F.scaled_dot_product_attention(
                    q, keys, values, attn_mask=allowed, dropout_p=0.0, is_causal=False
                )
            y = y.transpose(1, 2).contiguous().view(1, length, -1)
            x = x + attn.out(y)
            x = x + block.ff(block.norm2(x))

        self.session.cache.length = end
        return model.head(model.norm(x))

    def _compile_candidate(self):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile unavailable")
        return torch.compile(
            self._forward_eager,
            dynamic=True,
            fullgraph=False,
            mode="reduce-overhead",
        )

    def _ensure_compile(self) -> None:
        """Resolve Auto/On/Off once per loaded model without blindly enabling compile."""
        if self._compile_checked:
            return
        self._compile_checked = True
        requested = self.compile_mode
        if requested == "off":
            self.compile_info = {"requested": requested, "selected": "off", "reason": "explicitly disabled"}
            return

        if requested == "auto" and self.device != "cuda":
            # For Nimo's small batch-1 CPU/MPS path, compile startup can dominate a chat
            # session. Never auto-enable without a measured backend-specific win.
            self.compile_info = {
                "requested": requested,
                "selected": "off",
                "reason": f"Auto compile benchmark is disabled on {self.device}; eager path retained",
            }
            return

        try:
            compiled = self._compile_candidate()
        except Exception as exc:
            self.compile_info = {
                "requested": requested,
                "selected": "off",
                "reason": "compile unavailable",
                "error": type(exc).__name__,
            }
            return

        if requested == "on":
            self._compiled_forward = compiled
            self.compile_info = {
                "requested": requested,
                "selected": "on",
                "reason": "explicit opt-in",
            }
            return

        # Auto is intentionally benchmark-gated and currently only considered on CUDA.
        model = self.model
        vocab = max(9, int(getattr(model, "vocab_size", 512)))
        probe = [min(vocab - 1, 8 + index) for index in range(4)]
        try:
            eager_samples = []
            for _ in range(2):
                self.session.reset()
                start = time.perf_counter()
                self._forward_eager(probe, 0)
                _sync(self.device)
                eager_samples.append(time.perf_counter() - start)

            self.session.reset()
            compile_start = time.perf_counter()
            compiled(probe, 0)
            _sync(self.device)
            compile_first_s = time.perf_counter() - compile_start

            compiled_samples = []
            for _ in range(2):
                self.session.reset()
                start = time.perf_counter()
                compiled(probe, 0)
                _sync(self.device)
                compiled_samples.append(time.perf_counter() - start)

            eager_s = min(eager_samples)
            compiled_s = min(compiled_samples)
            speedup = eager_s / max(compiled_s, 1e-9)
            use_compiled = speedup >= 1.08
            self._compiled_forward = compiled if use_compiled else None
            self.compile_info = {
                "requested": "auto",
                "selected": "on" if use_compiled else "off",
                "eager_probe_s": eager_s,
                "compiled_probe_s": compiled_s,
                "steady_speedup": speedup,
                "compile_first_call_s": compile_first_s,
                "minimum_speedup": 1.08,
            }
        except Exception as exc:
            self._compiled_forward = None
            self.compile_info = {
                "requested": "auto",
                "selected": "off",
                "reason": "compile benchmark fallback",
                "error": type(exc).__name__,
            }
        finally:
            self.session.reset()

    def _run_forward(self, token_ids: list[int] | torch.Tensor, start_pos: int) -> torch.Tensor:
        self._ensure_compile()
        if self._compiled_forward is None:
            return self._forward_eager(token_ids, start_pos)
        try:
            return self._compiled_forward(token_ids, start_pos)
        except Exception as exc:
            # A backend compile failure must never take local chat down. Rebuild the
            # exact eager prefix before retrying so cache correctness is preserved.
            prefix = list(self.session.tokens[:start_pos])
            self._compiled_forward = None
            self.compile_info = {
                **self.compile_info,
                "selected": "off",
                "reason": "compiled execution fallback",
                "error": type(exc).__name__,
            }
            self.session.reset()
            if prefix:
                prefix_logits = self._forward_eager(prefix, 0)
                self.session.tokens.extend(prefix)
                self.session.logits = prefix_logits[:, -1, :]
            return self._forward_eager(token_ids, start_pos)

    def prefill(self, prompt_ids) -> tuple[torch.Tensor, dict]:
        """Process only the prompt suffix not already represented by the exact cache prefix."""
        ids = [int(value) for value in prompt_ids]
        if not ids:
            raise ValueError("Prompt must contain at least one token")
        if len(ids) > self.model.context:
            ids = ids[-self.model.context:]

        old_len = len(self.session.tokens)
        common = _common_prefix(self.session.tokens, ids)
        if common == 0:
            self.session.reset()
        elif common < old_len:
            if common == len(ids):
                replay = max(0, common - 1)
                self.session.cache.truncate(replay)
                self.session.tokens = ids[:replay]
                self.session.logits = None
                common = replay
            else:
                self.session.cache.truncate(common)
                self.session.tokens = self.session.tokens[:common]
                self.session.logits = None

        suffix = ids[common:]
        started = time.perf_counter()
        if suffix:
            logits = self._run_forward(suffix, common)
            self.session.tokens.extend(suffix)
            self.session.logits = logits[:, -1, :]
        elif self.session.logits is None:
            replay = ids[-1:]
            prefix = ids[:-1]
            self.session.reset()
            if prefix:
                self._run_forward(prefix, 0)
                self.session.tokens.extend(prefix)
            logits = self._run_forward(replay, len(prefix))
            self.session.tokens.extend(replay)
            self.session.logits = logits[:, -1, :]
        _sync(self.device)
        elapsed = time.perf_counter() - started
        return self.session.logits, {
            "prefill_s": elapsed,
            "reused_prefix_tokens": common,
            "processed_prompt_tokens": len(suffix),
            "prompt_tokens": len(ids),
        }

    def decode(self, token: int) -> torch.Tensor:
        """Decode one token without rebuilding or concatenating the prior K/V state."""
        position = self.session.cache.length
        logits = self._run_forward([int(token)], position)
        self.session.tokens.append(int(token))
        self.session.logits = logits[:, -1, :]
        return self.session.logits

    @staticmethod
    def _sample(
        scores,
        counts,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        generator,
    ):
        scores = scores.float().clone()
        if counts:
            ids = [token for token in counts if 0 <= token < scores.numel()]
            if ids:
                index = torch.tensor(ids, device=scores.device, dtype=torch.long)
                values = scores.index_select(0, index)
                if repetition_penalty and repetition_penalty != 1.0:
                    values = torch.where(
                        values >= 0,
                        values / repetition_penalty,
                        values * repetition_penalty,
                    )
                if frequency_penalty or presence_penalty:
                    frequencies = torch.tensor(
                        [counts[token] for token in ids],
                        device=scores.device,
                        dtype=values.dtype,
                    )
                    values = values - float(frequency_penalty) * frequencies - float(presence_penalty)
                scores.index_copy_(0, index, values)

        if temperature is None or temperature <= 0:
            return int(scores.argmax())
        temperature = max(float(temperature), 1e-5)

        if top_k:
            k = min(int(top_k), scores.numel())
            values, indices = torch.topk(scores, k)
            probs = (values / temperature).softmax(-1)
            if top_p and 0 < float(top_p) < 1:
                cumulative = probs.cumsum(-1)
                keep = cumulative - probs <= float(top_p)
                probs = probs * keep
                probs = probs / probs.sum().clamp_min(1e-12)
            selected = torch.multinomial(probs, 1, generator=generator)
            return int(indices[selected])

        probs = (scores / temperature).softmax(-1)
        if top_p and 0 < float(top_p) < 1:
            sorted_probs, sorted_ids = probs.sort(descending=True)
            keep = sorted_probs.cumsum(-1) - sorted_probs <= float(top_p)
            sorted_probs = sorted_probs * keep
            sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
            return int(sorted_ids[torch.multinomial(sorted_probs, 1, generator=generator)])
        return int(torch.multinomial(probs, 1, generator=generator))

    def generate_ids(
        self,
        prompt_ids,
        tokenizer,
        limit=256,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop_sequences=None,
        seed=None,
        deterministic=False,
    ):
        with self.lock, torch.inference_mode():
            self.model.eval()
            overall = time.perf_counter()
            logits, prefill = self.prefill(prompt_ids)
            generator = None
            if seed is not None:
                generator = torch.Generator(device=logits.device)
                generator.manual_seed(int(seed))
            if deterministic:
                temperature = 0.0
            stop_ids = [tokenizer.encode(sequence) for sequence in (stop_sequences or []) if sequence]
            eos = tokenizer.special_id("EOS")
            pad = tokenizer.special_id("PAD")
            produced: list[int] = []
            counts = {}
            max_steps = min(int(limit), max(0, self.model.context - len(self.session.tokens)))
            first_token_s = None
            decode_start = None
            decoded_steps = 0

            for _ in range(max_steps):
                pred = self._sample(
                    logits[0] if logits.ndim == 2 else logits,
                    counts,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    frequency_penalty,
                    presence_penalty,
                    generator,
                )
                if first_token_s is None:
                    _sync(self.device)
                    first_token_s = time.perf_counter() - overall
                    decode_start = time.perf_counter()
                if pred in (eos, pad):
                    break
                produced.append(pred)
                counts[pred] = counts.get(pred, 0) + 1
                if any(stop and produced[-len(stop):] == stop for stop in stop_ids):
                    break

                # Yield before the next decode so visible first-token latency is real.
                yield pred
                logits = self.decode(pred)
                decoded_steps += 1

            _sync(self.device)
            total = time.perf_counter() - overall
            decode_elapsed = time.perf_counter() - decode_start if decode_start is not None else 0.0
            self.last_metrics = {
                **prefill,
                "first_token_s": first_token_s,
                "generation_s": total,
                "decode_tokens": decoded_steps,
                "decode_tokens_per_s": decoded_steps / decode_elapsed if decode_elapsed > 0 else 0.0,
                "kv_cache_mb": self.session.cache.bytes / (1024 * 1024),
                "device": self.device,
                "torch_threads": torch.get_num_threads(),
                "thread_tuning": dict(_THREAD_INFO),
                "compile": dict(self.compile_info),
            }


def get_engine(model) -> FastInferenceEngine:
    with _ENGINES_LOCK:
        engine = _ENGINES.get(model)
        if engine is None:
            engine = FastInferenceEngine(model)
            _ENGINES[model] = engine
        return engine


def reset_inference_cache(model=None) -> None:
    with _ENGINES_LOCK:
        if model is not None:
            engine = _ENGINES.get(model)
            if engine:
                engine.reset()
            return
        for engine in list(_ENGINES.values()):
            engine.reset()


def inference_metrics(model) -> dict:
    return dict(get_engine(model).last_metrics)


def install_fast_inference() -> None:
    """Install local fast-path hooks without changing the training forward method."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .models import NimoLMV2
    from .tokenizer import NimoTokenizer

    original_from_artifact = NimoTokenizer.from_artifact.__func__

    @classmethod
    def cached_from_artifact(cls, artifact):
        key = str(artifact.get("sha256") or "") if isinstance(artifact, dict) else ""
        if key:
            with _TOKENIZER_CACHE_LOCK:
                cached = _TOKENIZER_CACHE.get(key)
                if cached is not None:
                    return cached
        tokenizer = original_from_artifact(cls, artifact)
        if key:
            with _TOKENIZER_CACHE_LOCK:
                if len(_TOKENIZER_CACHE) >= 8:
                    _TOKENIZER_CACHE.pop(next(iter(_TOKENIZER_CACHE)))
                _TOKENIZER_CACHE[key] = tokenizer
        return tokenizer

    def fast_generate_ids(self, prompt_ids, tokenizer, **kwargs):
        yield from get_engine(self).generate_ids(prompt_ids, tokenizer, **kwargs)

    def fast_generate(self, prompt_ids, tokenizer, **kwargs):
        return tokenizer.decode(list(fast_generate_ids(self, prompt_ids, tokenizer, **kwargs)))

    def fast_stream_generate(self, prompt_ids, tokenizer, **kwargs):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        for token in fast_generate_ids(self, prompt_ids, tokenizer, **kwargs):
            piece = tokenizer.tokens[token] if token >= 8 and token < len(tokenizer.tokens) else b""
            text = decoder.decode(piece, final=False)
            if text:
                yield text
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail

    NimoTokenizer.from_artifact = cached_from_artifact
    NimoLMV2.generate_ids = fast_generate_ids
    NimoLMV2.generate = fast_generate
    NimoLMV2.stream_generate = fast_stream_generate
    _INSTALLED = True
