"""Bring stored traces to the final 2-label schema: every action is "approve" or "reject".

  * calls to tools that do not exist (old "escalate"/unknown-tool events) are removed and counted in n_invalid
  * any other non-"reject" action (old "escalate" or null) becomes "approve"; "reject" is never changed
  * step numbers are renumbered; traces left with no actions are dropped
Run only when no collector is appending to the files.
    python scripts/finalize_labels.py "traces/*.jsonl"
"""
import glob
import json
import sys

tot = dict(files=0, traces=0, dropped_traces=0, removed_events=0, relabeled=0)
for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "traces/*.jsonl")):
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        tr = json.loads(line)
        names = {t["name"] for t in tr["tools"]}
        kept = [e for e in tr["events"] if e["tool"] in names]
        tr["n_invalid"] = tr.get("n_invalid", 0) + len(tr["events"]) - len(kept)
        tot["removed_events"] += len(tr["events"]) - len(kept)
        for i, e in enumerate(kept):
            e["step"] = i
            if e["label"] not in ("approve", "reject"):
                e["label"] = "approve"
                tot["relabeled"] += 1
        tr["events"] = kept
        if not kept:
            tot["dropped_traces"] += 1
            continue
        rows.append(json.dumps(tr))
        tot["traces"] += 1
    open(path, "w").write("\n".join(rows) + "\n")
    tot["files"] += 1
    print(path, len(rows))
print(tot)
