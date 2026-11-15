"""Stages 1-2: extract original prompts, then ask the generator for N paraphrases per prompt in one call."""
import json
import logging
import re

from tracegen import config as C
from tracegen import io, ollama
from tracegen.dojo import load_suite

log = logging.getLogger("tracegen")

TEMPLATE = """Rewrite the following request in {N} different ways.
Keep exactly the same intent and every specific detail
(names, amounts, dates, times, file names, email addresses, IBANs, URLs, numbers) unchanged.
Do not add or remove any requirement. Vary wording, tone, and sentence structure
(casual, formal, terse, detailed). Return only a JSON list of strings.
Request: "{prompt}\""""


# ---------------------------------------------------------------- stage 1
def extract(cfg, suites, limit=None):
    out = C.out_path(cfg, "originals.jsonl")
    done = io.keys(out, lambda r: (r["suite"], r["task_id"]))
    n = 0
    for s in suites:
        suite = load_suite(cfg["benchmark_version"], s)
        for i, (tid, task) in enumerate(sorted(suite.user_tasks.items(), key=lambda kv: _task_num(kv[0]))):
            if limit is not None and i >= limit:
                break
            if (s, tid) in done:
                continue
            io.append_jsonl(out, {"suite": s, "task_id": tid, "prompt": task.PROMPT})
            n += 1
    log.info("extract: %d new originals -> %s", n, out)
    return originals(cfg, suites, limit)


def _task_num(tid):
    m = re.search(r"(\d+)$", tid)
    return int(m.group(1)) if m else 0


def originals(cfg, suites, limit=None):
    rows = [r for r in io.read_jsonl(C.out_path(cfg, "originals.jsonl")) if r["suite"] in suites]
    if limit is None:
        return rows
    out, per = [], {}
    for r in sorted(rows, key=lambda r: (r["suite"], _task_num(r["task_id"]))):
        if per.get(r["suite"], 0) < limit:
            per[r["suite"]] = per.get(r["suite"], 0) + 1
            out.append(r)
    return out


# ---------------------------------------------------------------- stage 2
def parse_paraphrases(text):
    """Parse a JSON list of strings out of a model reply. Accepts a bare list, a code fence, a dict whose single
    list value holds the strings, or a list embedded in prose. Raises ValueError when nothing valid is found."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.I).strip()
    cands = [t]
    a, b = t.find("["), t.rfind("]")
    if 0 <= a < b:
        cands.append(t[a:b + 1])
    for c in cands:
        try:
            v = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict):
            lists = [x for x in v.values() if isinstance(x, list)]
            v = lists[0] if len(lists) == 1 else None
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v
    raise ValueError("no JSON list of strings in reply: %r" % (text or "")[:200])


def clean(original, paras, n):
    """Strip, drop empties, exact duplicates and paraphrases identical to the original. Keep at most n."""
    seen, out = {original.strip()}, []
    for p in paras:
        p = p.strip().strip('"').strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out[:n]


def generate_one(chat, original, n, seed, max_attempts):
    """chat(messages, seed) -> text. Returns (paraphrases, attempts, used_seed, error)."""
    msg = [{"role": "user", "content": TEMPLATE.format(N=n, prompt=original)}]
    err = None
    for attempt in range(max_attempts):
        s = seed + attempt   # a new seed per retry, otherwise the same invalid reply comes back
        try:
            return parse_paraphrases(chat(msg, s)), attempt + 1, s, None
        except ValueError as e:
            err = str(e)
            log.debug("invalid JSON (attempt %d): %s", attempt + 1, e)
    return [], max_attempts, None, err


def seed_for(cfg, r):
    return io.stable_int(r["suite"], r["task_id"], cfg["generate"]["base_seed"])


def generate(cfg, rows, on_candidates=None, url=None, model=None):
    """Append candidates.jsonl and generate_log.jsonl. Resumable per original. on_candidates(list) streams them."""
    g = cfg["generate"]
    url, model = url or C.generator_url(cfg), model or C.generator_model(cfg)
    out, logp = C.out_path(cfg, "candidates.jsonl"), C.out_path(cfg, "generate_log.jsonl")
    done = io.keys(logp, lambda r: (r["suite"], r["task_id"]))
    todo = [r for r in rows if (r["suite"], r["task_id"]) not in done]
    if not todo:
        log.info("generate: nothing to do")
        return
    tag = ollama.model_info(url, model)["tag"]

    def chat(messages, seed):
        return ollama.chat(url, model, messages, temperature=g["temperature"], seed=seed,
                           num_predict=g["num_predict"], timeout=600)[0]

    for k, r in enumerate(todo, 1):
        seed = seed_for(cfg, r)
        raw, attempts, used, err = generate_one(chat, r["prompt"], g["n"], seed, g["max_attempts"])
        paras = clean(r["prompt"], raw, g["n"])
        cands = [{"variant_id": "%s/%s/p%d" % (r["suite"], r["task_id"], i), "suite": r["suite"],
                  "task_id": r["task_id"], "original": r["prompt"], "paraphrase": p, "gen_model": model,
                  "gen_model_tag": tag, "seed": used, "temperature": g["temperature"]} for i, p in enumerate(paras)]
        for c in cands:
            io.append_jsonl(out, c)
        io.append_jsonl(logp, {"suite": r["suite"], "task_id": r["task_id"], "n_raw": len(raw), "n_kept": len(cands),
                               "attempts": attempts, "seed": used, "error": err})
        log.info("generate [%d/%d] %s/%s: %d paraphrases (%d attempts)%s", k, len(todo), r["suite"], r["task_id"],
                 len(cands), attempts, " ERROR " + err if err else "")
        if on_candidates is not None:
            on_candidates(cands)
