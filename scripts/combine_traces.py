"""Combine tracegen traces of several agent LLMs into one dataset, keeping only runs without an error.

    python scripts/combine_traces.py                                  # every data/tracegen*/ dir -> data/tracegen_all
    python scripts/combine_traces.py --sources data/tracegen data/tracegen_qwen38 --out data/tracegen_all --force

The output has the tracegen layout (traces.jsonl, splits/{train,val,test}.jsonl, and originals / verified variants when
all sources share them), so it loads like any tracegen dir:
    python scripts/run_tracegen_gnn.py --tracegen data/tracegen_all
A run counts as errored when its `error` field is non-empty (e.g. an endpoint 500). Runs that occur in more than one
source (same tracegen.io.trace_key or trace_id) are kept once, from the first source. Splits are task-level: the union
of the sources' split files (they must agree), else tracegen's seeded assign_splits with config.yaml's build settings.
manifest.json records the per-source / per-agent counts.
"""
import argparse
import collections
import datetime
import filecmp
import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tracegen import io  # noqa: E402
from tracegen.stages.build import assign_splits  # noqa: E402

SPLITS = ("train", "val", "test")
SHARED = ("originals.jsonl", "verified_variants.jsonl")


def default_sources(out):
    out = os.path.normcase(os.path.abspath(out))
    return [d for d in sorted(glob.glob(os.path.join(ROOT, "data", "tracegen*")))
            if os.path.isfile(os.path.join(d, "traces.jsonl")) and os.path.normcase(os.path.abspath(d)) != out]


def source_splits(src):
    out = {}
    for name in SPLITS:
        for r in io.read_jsonl(os.path.join(src, "splits", "%s.jsonl" % name)):
            out[(r["suite"], r["task_id"])] = name
    return out


def split_settings():
    try:
        import yaml
        b = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))["build"]
        return b["split_ratios"], b["split_seed"]
    except (ImportError, OSError, KeyError, TypeError):
        return [0.7, 0.15, 0.15], 0


def combine(sources):
    kept, seen_key, seen_id = [], set(), set()
    counts = collections.defaultdict(lambda: collections.Counter())  # (source, agent) -> read/errored/duplicate/kept
    split_of, conflicts = {}, []
    for src in sources:
        name = os.path.relpath(src, ROOT)
        for k, sp in source_splits(src).items():
            if split_of.setdefault(k, sp) != sp:
                conflicts.append((k, split_of[k], sp, name))
        for r in io.read_jsonl(os.path.join(src, "traces.jsonl")):
            c = counts[(name, r.get("agent_model"))]
            c["read"] += 1
            if r.get("error"):
                c["errored"] += 1
                continue
            key = io.trace_key(r)
            if key in seen_key or r.get("trace_id") in seen_id:
                c["duplicate"] += 1
                continue
            seen_key.add(key)
            seen_id.add(r.get("trace_id"))
            c["kept"] += 1
            kept.append(r)
    if conflicts:
        raise SystemExit("split files disagree for %d tasks, e.g. %s" % (len(conflicts), conflicts[:3]))
    missing = sorted({(r["suite"], r["task_id"]) for r in kept} - set(split_of))
    if missing:
        ratios, seed = split_settings()
        fresh = assign_splits(list(split_of) + missing, ratios, seed)
        bad = [k for k in split_of if fresh[k] != split_of[k]]
        if bad:
            raise SystemExit("source split files disagree with assign_splits for %s" % bad[:3])
        split_of.update({k: fresh[k] for k in missing})
    for r in kept:
        r["split"] = split_of[(r["suite"], r["task_id"])]
    return kept, counts, len(missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", help="tracegen dirs (default: every data/tracegen*/ with a traces.jsonl)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "tracegen_all"))
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    a = ap.parse_args()

    sources = [os.path.abspath(s) for s in a.sources] if a.sources else default_sources(a.out)
    if not sources:
        raise SystemExit("no source dirs with a traces.jsonl")
    for s in sources:
        if not os.path.isfile(os.path.join(s, "traces.jsonl")):
            raise SystemExit("%s has no traces.jsonl" % s)
    if os.path.isdir(a.out) and os.listdir(a.out):
        if not a.force:
            raise SystemExit("%s is not empty; pass --force to overwrite" % a.out)
        shutil.rmtree(a.out)

    kept, counts, n_assigned = combine(sources)
    io.write_jsonl(os.path.join(a.out, "traces.jsonl"), kept)
    for sp in SPLITS:
        io.write_jsonl(os.path.join(a.out, "splits", "%s.jsonl" % sp), [r for r in kept if r["split"] == sp])

    copied = []
    for f in SHARED:
        paths = [os.path.join(s, f) for s in sources]
        if all(os.path.isfile(p) for p in paths) and all(filecmp.cmp(paths[0], p, shallow=False) for p in paths[1:]):
            shutil.copyfile(paths[0], os.path.join(a.out, f))
            copied.append(f)
        else:
            print("warning: %s missing or different across sources; not copied" % f)

    by_split = collections.Counter(r["split"] for r in kept)
    by_agent_mode = collections.Counter((r["agent_model"], r["mode"]) for r in kept)
    manifest = dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        sources=[os.path.relpath(s, ROOT) for s in sources],
        filter="error field empty",
        traces=len(kept),
        splits={sp: by_split.get(sp, 0) for sp in SPLITS},
        tasks_split_by_assign_splits=n_assigned,
        copied=copied,
        counts=[dict(source=s, agent=m, **{k: c[k] for k in ("read", "errored", "duplicate", "kept")}) for (s, m), c in sorted(counts.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))],
        by_agent={m: {mode: n for (mm, mode), n in by_agent_mode.items() if mm == m}
                  for m in sorted({m for m, _ in by_agent_mode})},
    )
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("%-26s %-14s %6s %7s %5s %6s" % ("source", "agent", "read", "errored", "dup", "kept"))
    for row in manifest["counts"]:
        print("%-26s %-14s %6d %7d %5d %6d" % (row["source"], row["agent"], row["read"], row["errored"],
                                              row["duplicate"], row["kept"]))
    print("total kept %d -> %s  splits %s" % (len(kept), os.path.relpath(a.out, ROOT), manifest["splits"]))


if __name__ == "__main__":
    main()
