"""Nimo native tokenizer: trainable local byte-subword vocabulary.

The tokenizer is trained only from local text. It starts with a complete byte fallback
alphabet, then learns frequent variable-length byte pieces. This preserves lossless
round-trips for English, Malayalam, Manglish, code, mathematics and arbitrary UTF-8.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ("PAD","BOS","EOS","SYSTEM","USER","ASSISTANT","TOOL","MEMORY")
SPECIAL_TEXT = {name: f"<|{name.lower()}|>" for name in SPECIAL_TOKENS}
FORMAT_VERSION = 1
BYTE_OFFSET = len(SPECIAL_TOKENS)
BASE_VOCAB = BYTE_OFFSET + 256


def _pieces(text: str):
    # Unicode-aware: \w keeps Malayalam letters; whitespace/punctuation remain learnable.
    return re.findall(r"\s+|[\w]+|[^\w\s]+", text, flags=re.UNICODE)


class NimoTokenizer:
    def __init__(self, tokens: list[bytes], metadata: dict | None = None):
        if len(tokens) < BASE_VOCAB:
            raise ValueError("Tokenizer vocabulary is missing byte fallback tokens")
        self.tokens = list(tokens)
        self.metadata = dict(metadata or {})
        self._id_by_bytes = {piece: i for i, piece in enumerate(self.tokens) if i >= BYTE_OFFSET}
        self._max_piece = max((len(x) for x in self.tokens[BASE_VOCAB:]), default=1)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def special_id(self, name: str) -> int:
        try:
            return SPECIAL_TOKENS.index(name.upper())
        except ValueError as exc:
            raise ValueError(f"Unknown special token: {name}") from exc

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        raw = text.encode("utf-8")
        ids = [self.special_id("BOS")] if add_bos else []
        i = 0
        # Longest learned byte piece, then guaranteed single-byte fallback.
        while i < len(raw):
            matched = None
            upper = min(len(raw), i + self._max_piece)
            for end in range(upper, i + 1, -1):
                candidate = raw[i:end]
                token_id = self._id_by_bytes.get(candidate)
                if token_id is not None:
                    matched = (token_id, end)
                    break
            if matched is None:
                ids.append(BYTE_OFFSET + raw[i])
                i += 1
            else:
                ids.append(matched[0])
                i = matched[1]
        if add_eos:
            ids.append(self.special_id("EOS"))
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        out = bytearray()
        visible = []
        for value in ids:
            value = int(value)
            if value < BYTE_OFFSET:
                if not skip_special:
                    visible.append(SPECIAL_TEXT[SPECIAL_TOKENS[value]])
                continue
            if 0 <= value < len(self.tokens):
                out.extend(self.tokens[value])
        text = out.decode("utf-8", errors="replace")
        return ("".join(visible) + text) if visible else text

    def encode_chat(self, messages: list[dict], add_assistant_prompt: bool = True) -> list[int]:
        ids = [self.special_id("BOS")]
        role_map = {"system":"SYSTEM","user":"USER","assistant":"ASSISTANT","tool":"TOOL","memory":"MEMORY"}
        for message in messages:
            role = role_map.get(str(message.get("role","")).lower())
            if not role:
                continue
            ids.append(self.special_id(role))
            ids.extend(self.encode(str(message.get("content",""))))
            ids.append(self.special_id("EOS"))
        if add_assistant_prompt:
            ids.append(self.special_id("ASSISTANT"))
        return ids

    def artifact(self) -> dict:
        encoded = []
        for index, piece in enumerate(self.tokens):
            if index < BYTE_OFFSET:
                encoded.append({"special": SPECIAL_TOKENS[index]})
            else:
                encoded.append({"bytes": base64.b64encode(piece).decode("ascii")})
        payload = {
            "format":"nimo-tokenizer",
            "version":FORMAT_VERSION,
            "method":"byte-subword-frequency-v1",
            "special_tokens": list(SPECIAL_TOKENS),
            "tokens": encoded,
            "metadata": self.metadata,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",",":")).encode()
        payload["sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.artifact(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_artifact(cls, artifact: dict):
        if artifact.get("format") != "nimo-tokenizer" or artifact.get("version") != FORMAT_VERSION:
            raise ValueError("Unsupported tokenizer artifact")
        tokens = []
        items = artifact.get("tokens")
        if not isinstance(items, list) or len(items) < BASE_VOCAB:
            raise ValueError("Invalid tokenizer vocabulary")
        if len(items) > 65536:
            raise ValueError("Tokenizer vocabulary exceeds safe limit")
        for index, item in enumerate(items):
            if index < BYTE_OFFSET:
                if item.get("special") != SPECIAL_TOKENS[index]:
                    raise ValueError("Tokenizer special-token order mismatch")
                tokens.append(b"")
            else:
                try:
                    piece = base64.b64decode(item["bytes"], validate=True)
                except Exception as exc:
                    raise ValueError("Invalid tokenizer byte token") from exc
                if not 1 <= len(piece) <= 32:
                    raise ValueError("Tokenizer piece length exceeds safe limit")
                tokens.append(piece)
        return cls(tokens, artifact.get("metadata") or {})

    @classmethod
    def load(cls, path):
        raw = Path(path).read_bytes()
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("Tokenizer artifact is too large")
        artifact = json.loads(raw)
        tokenizer = cls.from_artifact(artifact)
        claimed = artifact.get("sha256")
        if claimed:
            check = dict(artifact)
            check.pop("sha256", None)
            digest = hashlib.sha256(json.dumps(check, sort_keys=True, separators=(",",":")).encode()).hexdigest()
            if digest != claimed:
                raise ValueError("Tokenizer checksum mismatch")
        return tokenizer


def train_tokenizer(texts, vocab_size: int = 8000, max_corpus_bytes: int = 16 * 1024 * 1024,
                    max_piece_bytes: int = 16, min_frequency: int = 2) -> NimoTokenizer:
    if not 512 <= int(vocab_size) <= 32768:
        raise ValueError("Vocabulary size must be between 512 and 32768")
    vocab_size = int(vocab_size)
    max_piece_bytes = max(2, min(int(max_piece_bytes), 32))
    counts = Counter()
    total = 0
    docs = 0
    # Count substrings inside Unicode-aware lexical pieces. This is deterministic and
    # much faster than naive corpus-wide iterative BPE for large local datasets.
    for text in texts:
        if not isinstance(text, str) or not text:
            continue
        data = text.encode("utf-8")
        if total >= max_corpus_bytes:
            break
        if total + len(data) > max_corpus_bytes:
            data = data[:max_corpus_bytes-total]
            text = data.decode("utf-8", errors="ignore")
            data = text.encode("utf-8")
        total += len(data); docs += 1
        for piece_text in _pieces(text):
            piece = piece_text.encode("utf-8")
            if len(piece) < 2:
                continue
            # Cap a single lexical piece so pathological strings cannot explode n-gram work.
            if len(piece) > 256:
                piece = piece[:256]
            upper = min(max_piece_bytes, len(piece))
            for n in range(2, upper + 1):
                for i in range(0, len(piece) - n + 1):
                    candidate = piece[i:i+n]
                    # UTF-8 fragments are allowed internally because decoding is assembled
                    # after concatenation and byte fallback guarantees reversibility.
                    counts[candidate] += 1
        # Prevent unbounded Counter growth on highly diverse corpora.
        if len(counts) > max(vocab_size * 80, 250_000):
            counts = Counter(dict(counts.most_common(max(vocab_size * 40, 125_000))))

    base = [b""] * BYTE_OFFSET + [bytes([i]) for i in range(256)]
    wanted = vocab_size - len(base)
    ranked = [
        (freq * (len(piece)-1), freq, len(piece), piece)
        for piece, freq in counts.items()
        if freq >= min_frequency and len(piece) > 1
    ]
    ranked.sort(reverse=True)
    learned = []
    seen = set(base[BYTE_OFFSET:])
    for _, _, _, piece in ranked:
        if piece in seen:
            continue
        learned.append(piece); seen.add(piece)
        if len(learned) >= wanted:
            break
    metadata = {
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": len(base) + len(learned),
        "training_documents": docs,
        "training_bytes": total,
        "max_piece_bytes": max_piece_bytes,
        "languages":"UTF-8 multilingual; English/Malayalam/Manglish/code/math supported",
    }
    return NimoTokenizer(base + learned, metadata)
