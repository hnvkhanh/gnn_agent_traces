"""Frozen sentence embedding of the user prompt, the extra component of the *+prompt guards.

    vecs = prompt_vectors(texts)        # {text: float32 tensor [DIM]}, L2-normalised, cached on disk

Encoder: MODEL (sentence-transformers/all-MiniLM-L6-v2) through `transformers` AutoTokenizer / AutoModel, mean pooling
over the attention mask, then L2 normalisation, under torch.no_grad (never trained). It downloads once from Hugging
Face; nothing from the traces is sent anywhere. Vectors are cached in data/cache/prompt_emb_<model>.pt keyed by the
sha1 of the text, so each distinct prompt is embedded once and worker processes only read the file.

PROMPT_ENCODER=hash (environment) swaps in a deterministic offline stand-in (a token-hash vector, no download and no
cache file), used by the tests.
"""
import hashlib
import os
import re
import zlib

import numpy as np
import torch

from .cache import ROOT

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
CACHE_DIR = os.path.join(ROOT, "data", "cache")


def _key(text):
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def encoder_name():
    return os.environ.get("PROMPT_ENCODER") or MODEL


def cache_path(model=None):
    model = model or encoder_name()
    return os.path.join(CACHE_DIR, "prompt_emb_%s.pt" % re.sub(r"[^A-Za-z0-9._-]+", "_", model))


def _hash_encode(texts):
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in re.findall(r"[a-z0-9_]+", (t or "").lower()):
            h = zlib.crc32(tok.encode())
            out[i, h % DIM] += 1.0 if (h >> 16) & 1 else -1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return torch.tensor(out / np.where(n == 0, 1, n))


def _hf_encode(texts, model, device=None, batch=64):
    from transformers import AutoModel, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModel.from_pretrained(model).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            h = m(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(v, dim=1).float().cpu())
    return torch.cat(out) if out else torch.zeros((0, DIM))


def encode(texts, model=None):
    """[n, DIM] tensor for a list of texts (no cache)."""
    model = model or encoder_name()
    return _hash_encode(texts) if model == "hash" else _hf_encode(texts, model)


def prompt_vectors(texts, model=None):
    """{text: tensor [DIM]} for every distinct text; reads and extends the on-disk cache (not for 'hash')."""
    model = model or encoder_name()
    uniq = sorted(set(texts))
    if model == "hash":
        return dict(zip(uniq, encode(uniq, model)))
    path = cache_path(model)
    cache = torch.load(path) if os.path.exists(path) else {}
    todo = [t for t in uniq if _key(t) not in cache]
    if todo:
        for t, v in zip(todo, encode(todo, model)):
            cache[_key(t)] = v.clone()
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(cache, path + ".tmp")
        os.replace(path + ".tmp", path)
    return {t: cache[_key(t)] for t in uniq}
