"""Build an interactive, self-contained HTML dashboard comparing the traces of each agent model.

Usage (stdlib only, nothing to install):
    python scripts/build_dashboard.py                      # scans data/tracegen*/traces.jsonl
    python scripts/build_dashboard.py data/tracegen data/tracegen_qwen38 --out results/dashboard.html

Open the resulting HTML in a browser (Plotly is loaded from a CDN, so you need internet access once).
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clip(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + "…"


def err_type(e):
    if not e:
        return None
    m = re.match(r"\s*([A-Za-z_.]+(?:Error|Exception|Timeout)[A-Za-z_]*)", str(e))
    return m.group(1) if m else clip(e, 40)


def load_run(path, run):
    out = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            calls = []
            for c in r.get("tool_calls") or []:
                calls.append([
                    c.get("name"),
                    clip(json.dumps(c.get("args"), ensure_ascii=False), 200),
                    clip(c.get("result"), 300),
                    clip(c.get("error"), 120) if c.get("error") else None,
                ])
            inj = r.get("injections") or {}
            out.append({
                "id": r.get("trace_id"),
                "run": run,
                "model": r.get("agent_model"),
                "mode": r.get("mode"),
                "suite": r.get("suite"),
                "task": r.get("task_id"),
                "inj": r.get("injection_task_id"),
                "atk": r.get("attack_name"),
                "orig": bool(r.get("is_original")),
                "util": bool(r.get("utility")),
                "sec": r.get("security"),  # True = injection goal achieved (attack succeeded)
                "steps": r.get("n_steps") or 0,
                "tin": r.get("tokens_in") or 0,
                "tout": r.get("tokens_out") or 0,
                "lat": r.get("latency_s") or 0,
                "err": err_type(r.get("error")),
                "errmsg": clip(r.get("error"), 300) if r.get("error") else None,
                "gt": bool(r.get("matches_ground_truth_sequence")),
                "tools": [c[0] for c in calls],
                "prompt": clip(r.get("prompt"), 400),
                "ans": clip(r.get("final_answer"), 600),
                "calls": calls,
                "injtxt": clip(" | ".join(str(v) for v in inj.values()), 500) if inj else None,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="run directories containing traces.jsonl (default: data/tracegen*)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "dashboard.html"))
    a = ap.parse_args()

    dirs = a.dirs or sorted(d for d in glob.glob(os.path.join(ROOT, "data", "tracegen*")) if os.path.isdir(d))
    rows = []
    for d in dirs:
        p = os.path.join(d, "traces.jsonl")
        if not os.path.exists(p):
            print(f"skip {d}: no traces.jsonl", file=sys.stderr)
            continue
        run = os.path.basename(os.path.normpath(d))
        got = load_run(p, run)
        print(f"{run}: {len(got)} traces")
        rows += got
    if not rows:
        sys.exit("no traces found")

    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_template.html"), encoding="utf8").read()
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("__DATA__", data)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf8") as f:
        f.write(html)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
