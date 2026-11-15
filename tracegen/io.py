"""Append-only JSONL helpers shared by every stage (thread-safe appends, resume keys)."""
import hashlib
import json
import threading
from pathlib import Path

_LOCK = threading.Lock()


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a torn last line from a killed run: ignore it, it gets redone
                continue
    return out


def append_jsonl(path, rec):
    line = json.dumps(rec, ensure_ascii=False, default=str)
    with _LOCK:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()


def write_jsonl(path, recs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def keys(path, fn):
    return {fn(r) for r in read_jsonl(path)}


def stable_int(*parts, mod=2 ** 31):
    h = hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()
    return int(h, 16) % mod


def trace_key(r):
    return (r["mode"], r["suite"], r["task_id"], r["variant_id"], r["agent_model"], r["seed"],
            r.get("injection_task_id"), r.get("attack_name"))
