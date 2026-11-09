"""Reuse experiment results whose inputs have not changed.

A result cell (e.g. 'inbench/asb', 'transfer/asb->injecagent', 'ablate/memory/asb') is a list of runs, one per seed.
Its fingerprint is a sha256 over
  * the training code that decides the numbers (src/train.py, src/eval.py, src/model/gnn.py, src/sandbox/policy.py,
    line endings normalised),
  * seeds, epochs and the evaluate() options of the cell,
  * for every seed, the exact train and test inputs: trace ids, every graph tensor (x, edge_index, y) and meta.
Hashing the graphs themselves means a code change in graph building or the sandbox only invalidates the cells whose
inputs it really changes. A cell is reused when a stored result has the same fingerprint; otherwise it is computed.

Results written before fingerprints existed ('legacy', e.g. experiments_all.json) are only accepted with
adopt_legacy=True, and only if the results file is newer than every trace file the cell reads and every training-code
file. Their current fingerprint is then recorded so later runs check them exactly.
"""
import glob
import hashlib
import json
import os
import re
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CODE = ["src/train.py", "src/eval.py", "src/model/gnn.py", "src/sandbox/policy.py"]
_DIGEST = {}


def code_hash():
    h = hashlib.sha256()
    for p in TRAIN_CODE:
        h.update(p.encode())
        h.update(open(os.path.join(ROOT, p), "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def graph_digest(g):
    key = id(g)
    if key not in _DIGEST:
        h = hashlib.sha256()
        for nt in sorted(g.node_types):
            h.update(nt.encode())
            h.update(g[nt].x.numpy().tobytes())
            if "y" in g[nt]:
                h.update(g[nt].y.numpy().tobytes())
        for et in sorted(g.edge_types):
            h.update(repr(et).encode())
            h.update(g[et].edge_index.numpy().tobytes())
        _DIGEST[key] = (g, h.digest())  # keep g alive so id() is not reused
    return _DIGEST[key][1]


def data_hash(parts):
    """parts: per seed (train_graphs, train_metas, test_graphs, test_metas)."""
    h = hashlib.sha256()
    for tg, tm, eg, em in parts:
        for side_g, side_m in ((tg, tm), (eg, em)):
            h.update(b"|side|%d" % len(side_g))
            for g in side_g:
                h.update(graph_digest(g))
            h.update(json.dumps(side_m, sort_keys=True, default=str).encode())
    return h.hexdigest()


def benches_of(key):
    """Benchmarks whose traces a cell reads, from its key."""
    kind, _, rest = key.partition("/")
    if kind == "inbench":
        return [rest]
    if kind == "ablate":
        return [rest.split("/")[-1]]
    if kind == "transfer":
        src, _, tgt = rest.partition("->")
        return [b.split(":")[0] for b in src.split("+") + [tgt]]
    if kind == "shift":
        return ["asb"]
    return []


class ResultCache:
    def __init__(self, outdir, only, seeds, epochs, model, reuse=True, adopt_legacy=False, traces="traces/%s__*.jsonl"):
        self.outdir, self.only, self.model = outdir, only, model
        self.reuse, self.adopt_legacy, self.traces = reuse, adopt_legacy, traces
        self.base = dict(code=code_hash(), seeds=list(seeds), epochs=epochs)
        self.cands = {}  # key -> [(runs, fingerprint or None, source path, stored provenance)]
        self.status = {}  # key -> {fp, status, source, time}
        if reuse:
            for d in (outdir, os.path.join(outdir, "legacy")):
                for p in sorted(glob.glob(os.path.join(d, "experiments_*.json"))):
                    fps = {}
                    fpp = os.path.join(d, "fingerprints_%s.json" % os.path.basename(p)[len("experiments_"):-len(".json")])
                    if os.path.exists(fpp):
                        fps = json.load(open(fpp, encoding="utf-8"))
                    for k, runs in json.load(open(p, encoding="utf-8")).items():
                        rec = fps.get(k, {})
                        self.cands.setdefault(k, []).append((runs, rec.get("fp"), p, rec))

    def _legacy_ok(self, key, path):
        t = os.path.getmtime(path)
        files = [os.path.join(ROOT, f) for f in TRAIN_CODE]
        slug = self.model.replace(":", "_")
        for b in benches_of(key):
            files += glob.glob(os.path.join(ROOT, "traces", "%s__%s.jsonl" % (b, slug)))
        return all(os.path.getmtime(f) < t for f in files)

    def get(self, key, parts, compute, **options):
        """Return the runs of `key`: stored ones if their fingerprint matches, else compute()."""
        fp = hashlib.sha256(json.dumps(dict(self.base, key=key, options=options, data=data_hash(parts)),
                                       sort_keys=True, default=str).encode()).hexdigest()
        for runs, old_fp, src, rec in self.cands.get(key, []):
            if old_fp == fp:  # keep where the numbers originally came from
                self.status[key] = dict(fp=fp, status="reused", source=os.path.relpath(src, ROOT),
                                        origin=rec.get("origin", "unknown"), time=rec.get("time"), seconds=rec.get("seconds"))
                print("[cache] reused   %s" % key, flush=True)
                return runs
        for runs, old_fp, src, rec in self.cands.get(key, []):
            if old_fp is None and self.adopt_legacy and self._legacy_ok(key, src):
                self.status[key] = dict(fp=fp, status="adopted", source=os.path.relpath(src, ROOT), origin="legacy",
                                        time=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(src))))
                print("[cache] adopted  %s (legacy %s)" % (key, os.path.relpath(src, ROOT)), flush=True)
                return runs
        t0 = time.time()
        runs = compute()
        self.status[key] = dict(fp=fp, status="computed", origin="trained", seconds=round(time.time() - t0, 1),
                                time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        print("[cache] computed %s (%.0fs)" % (key, time.time() - t0), flush=True)
        return runs

    def save(self):
        os.makedirs(self.outdir, exist_ok=True)
        with open(os.path.join(self.outdir, "fingerprints_%s.json" % self.only), "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.status, f, indent=1, sort_keys=True)
        n = {s: sum(v["status"] == s for v in self.status.values()) for s in ("reused", "adopted", "computed")}
        print("[cache] %s: %d reused, %d adopted, %d computed" % (self.only, n["reused"], n["adopted"], n["computed"]),
              flush=True)


def common_benches(benches, llms):
    """Benchmarks that have trace files for every one of `llms` (the cross-LLM section compares like with like)."""
    return [b for b in benches
            if all(glob.glob(os.path.join(ROOT, "traces", "%s__%s.jsonl" % (b, re.sub(":", "_", m)))) for m in llms)]
