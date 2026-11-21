"""Frozen sentence embeddings of action and observation node text: the content features of the *+content guards.

    texts = node_texts(tr, ablate)       # (action texts, observation texts): exactly what build.py hashes
    V = node_vectors(list_of_texts)      # float32 [n, DIM], cached on disk

Source. The content features follow Zavrak, "Content-Aware Attack Detection in LLM Agent Tool-Call Traffic: An
Empirical Study of Features, Architectures, and Evaluation Protocols", arXiv:2605.11053 (2026), Sec. 4.2: a frozen
all-MiniLM-L6-v2 embedding (384-d) of a tool call's serialized arguments and one of its tool response. That paper puts
both on one node per tool call and classifies whole sessions. Here the action node gets the embedding of its call text
and the observation node that of its result, so the causal graph of docs/GRAPH.md is unchanged: action i still only
sees observations j < i.

Texts. node_texts applies the text ablations of src/graph/files.py (`attacker_values`: attacker constants masked;
`injected`: injected text removed from observations) exactly as the hashed features do, so an ablation cannot leak
through the embedding.

Difference from the paper (mode). The paper truncates each input to 512 characters. In data/tracegen_all, 60% of the
injected observations start the injection after character 512 (median 726, 90th percentile ~2,000) and 1% of
observations exceed 30k characters, so truncation would hide most injections. NODE_EMBED_MODE:
  chunk     (default) the whole text, as windows of WINDOW tokens ([CLS] + WINDOW-2 tokens + [SEP], consecutive windows
            overlap by OVERLAP tokens); each window is mean-pooled over its tokens and L2-normalised, and the node
            feature is [mean over windows ; max over windows], so one injected window is not averaged away
  trunc512  the paper's setting: the first 512 characters, one window; the feature is [v ; v] (same DIM)

Encoder: MODEL through `transformers` under torch.no_grad (never trained); downloads once from Hugging Face, nothing
from the traces is sent anywhere. Cache: data/cache/node_emb_<model>_<mode>.pt, {keys: [sha1 of text], vecs: float16
[n, DIM]}; worker processes only read it once the runner has warmed it. PROMPT_ENCODER=hash (as in
src/prompt_embed.py) swaps in the offline token-hash stand-in, with no download and no cache file (used by the tests).
"""
import json
import os
import re

import torch

from .prompt_embed import CACHE_DIR, MODEL, _hash_encode, _key, encoder_name
from .prompt_embed import DIM as BASE_DIM

DIM = 2 * BASE_DIM
WINDOW = 256          # all-MiniLM-L6-v2's max_seq_length, special tokens included
OVERLAP = 32
TRUNC_CHARS = 512
MODES = ("chunk", "trunc512")
TEXT_ABLATIONS = ("injected", "attacker_values")   # ablations that change node text (the rest only drop edges)


def mode():
    m = os.environ.get("NODE_EMBED_MODE") or "chunk"
    assert m in MODES, "NODE_EMBED_MODE must be one of %s, not %r" % (MODES, m)
    return m


def config():
    """What decides the vectors (enters the run fingerprint)."""
    return dict(model=encoder_name(), mode=mode(), window=WINDOW, overlap=OVERLAP, trunc_chars=TRUNC_CHARS)


def node_texts(tr, ablate=()):
    """(action texts, observation texts) of a trace, as src/graph/build.py and files.py hash them under `ablate`."""
    ab = set(ablate)
    if "attacker_values" in ab:
        from .graph.files import mask_attacker_values
        tr = mask_attacker_values(tr)
    inj = tr.get("injected", "") if "injected" in ab else ""
    strip = (lambda t: t.replace(inj, "")) if inj else (lambda t: t)
    ev = tr["events"]
    return ([e["tool"] + " " + json.dumps(e["args"], sort_keys=True) for e in ev],
            [strip(e["observation"]) for e in ev])


def windows(ids, size, overlap=0):
    """Token windows of at most `size` that cover every id; consecutive windows share `overlap` ids."""
    if len(ids) <= size:
        return [ids]
    step = size - overlap
    out = []
    for s in range(0, len(ids), step):
        out.append(ids[s:s + size])
        if s + size >= len(ids):
            break
    return out


def pool(owner, vecs, n):
    """[mean ; max] of the window vectors of each of n texts (owner[k] = text of window k)."""
    owner = torch.as_tensor(owner, dtype=torch.long)
    cnt = torch.zeros(n).index_add_(0, owner, torch.ones(len(owner))).clamp(min=1)
    mean = torch.zeros(n, vecs.shape[1]).index_add_(0, owner, vecs) / cnt[:, None]
    mx = torch.full((n, vecs.shape[1]), -float("inf")).scatter_reduce(
        0, owner[:, None].expand_as(vecs), vecs, "amax", include_self=True)
    return torch.cat([mean, mx], 1)


def _hf_encode(texts, model, md, device=None, batch=128):
    from transformers import AutoModel, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model)
    tok.model_max_length = 10 ** 9   # we window ourselves: no "sequence too long" warning
    m = AutoModel.from_pretrained(model).to(device).eval()
    if md == "trunc512":
        texts = [t[:TRUNC_CHARS] for t in texts]
    ids = tok(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    inner = WINDOW - 2
    wins = [[x[:inner]] if md == "trunc512" else windows(x, inner, OVERLAP) for x in ids]
    flat = [(i, w) for i, ws in enumerate(wins) for w in ws]
    order = sorted(range(len(flat)), key=lambda k: len(flat[k][1]))   # length-sorted batches: little padding
    vec = torch.zeros(len(flat), BASE_DIM)
    with torch.no_grad():
        for b in range(0, len(order), batch):
            ks = order[b:b + batch]
            seqs = [[tok.cls_token_id] + flat[k][1] + [tok.sep_token_id] for k in ks]
            L = max(map(len, seqs))
            inp = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for r, s in enumerate(seqs):
                inp[r, :len(s)], att[r, :len(s)] = torch.tensor(s), 1
            inp, att = inp.to(device), att.to(device)
            h = m(input_ids=inp, attention_mask=att).last_hidden_state
            mask = att.unsqueeze(-1).to(h.dtype)
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            vec[ks] = torch.nn.functional.normalize(v, dim=1).float().cpu()
    return pool([i for i, _ in flat], vec, len(texts))


def encode(texts, model=None, md=None):
    """[n, DIM] tensor for a list of texts (no cache)."""
    model, md = model or encoder_name(), md or mode()
    if not texts:
        return torch.zeros((0, DIM))
    if model == "hash":
        v = _hash_encode(texts)
        return torch.cat([v, v], 1)
    return _hf_encode(texts, model, md)


def cache_path(model=None, md=None):
    model, md = model or encoder_name(), md or mode()
    return os.path.join(CACHE_DIR, "node_emb_%s_%s.pt" % (re.sub(r"[^A-Za-z0-9._-]+", "_", model), md))


def node_vectors(texts, model=None, md=None):
    """float32 [len(texts), DIM] in the order given; reads and extends the on-disk cache (not for 'hash')."""
    model, md = model or encoder_name(), md or mode()
    if model == "hash":
        uniq = sorted(set(texts))
        tab = dict(zip(uniq, encode(uniq, model, md)))
        return torch.stack([tab[t] for t in texts]) if texts else torch.zeros((0, DIM))
    path = cache_path(model, md)
    keys, vecs = [], torch.zeros((0, DIM), dtype=torch.float16)
    if os.path.exists(path):
        c = torch.load(path)
        keys, vecs = c["keys"], c["vecs"]
    row = {k: i for i, k in enumerate(keys)}
    todo = sorted({t for t in texts if _key(t) not in row})
    if todo:
        new = encode(todo, model, md).to(torch.float16)
        for t in todo:
            row[_key(t)] = len(keys)
            keys.append(_key(t))
        vecs = torch.cat([vecs, new])
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = "%s.%d.tmp" % (path, os.getpid())
        torch.save(dict(keys=keys, vecs=vecs, model=model, mode=md), tmp)
        os.replace(tmp, path)
    if not texts:
        return torch.zeros((0, DIM))
    return vecs[torch.tensor([row[_key(t)] for t in texts])].float()


__all__ = ["DIM", "MODEL", "config", "encode", "mode", "node_texts", "node_vectors", "pool", "windows"]
