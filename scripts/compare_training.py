"""Compare training settings (results dirs of scripts/run_tracegen_gnn.py) on seed stability and score.

    python scripts/compare_training.py results/tracegen_all_repo20 results/tracegen_all_stable --out cmp.json

For every guard and pseudo-guard ('<guard>@<rule>', from --extra) found in a dir, per seed:
  cv      pooled cross-validation AUPRC and recall at 1% FPR (src/tracegen_gnn.py cv_reports)
  test    fixed-split test AUPRC (the 'main' cell)
and the seed spread of each: sd over seeds, plus the per-cell seed sd averaged over CV cells (how much one cell's
score moves with the seed alone)."""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tracegen", default=os.path.join(ROOT, "data", "tracegen_all"))
    ap.add_argument("--agentdojo", default=os.path.join(ROOT, "data", "agentdojo"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=0, help="bootstrap resamples for the pooled CV interval (0: none)")
    ap.add_argument("--out", default=None, help="write the table as JSON")
    a = ap.parse_args()
    from src.tracegen_gnn import Data, cells, cv_reports
    data = Data(a.tracegen, a.agentdojo)
    cvk = cells(data, ["cv"], a.folds)
    rows = []
    for d in a.dirs:
        store = json.load(open(os.path.join(d, "experiments.json"), encoding="utf-8"))
        seeds = sorted({int(s) for k in cvk + ["main"] for s in store.get(k, {})})
        pools = [store[k][str(s)]["run"].get("_pool") for k in cvk for s in seeds if str(s) in store.get(k, {})]
        names = sorted(set.intersection(*[set(p["scores"]) for p in pools])) if pools else []
        tr = next((store[k][s]["run"].get("_train") for k in store for s in store[k]), None)
        for g in names:
            cv = cv_reports(data, store, seeds, a.folds, n_boot=a.n_boot, guards=[g])
            cv_ap = [r["test"][g]["auprc"] for r in cv]
            cv_rf = [r["test"][g]["recall_at_fpr"] for r in cv]
            test = [store["main"][str(s)]["run"]["test"][g]["auprc"] for s in seeds
                    if str(s) in store.get("main", {}) and g in store["main"][str(s)]["run"]["test"]]
            cell_sd = [np.std([store[k][str(s)]["run"]["test"][g]["auprc"] for s in seeds if str(s) in store[k]])
                       for k in cvk if k in store and len(store[k]) > 1]
            rows.append(dict(dir=os.path.basename(os.path.normpath(d)), train=tr, guard=g, seeds=len(cv),
                             cv_auprc=cv_ap, cv_recall_at_fpr=cv_rf, test_auprc=test,
                             cv_cell_seed_sd=float(np.mean(cell_sd)) if cell_sd else None,
                             n_reject_cv=cv[0]["test"][g]["n_reject"] if cv else None))

    def pm(v):
        v = [x for x in v if x is not None]
        return "%.3f ± %.3f" % (np.mean(v), np.std(v)) if v else "–"

    print("%-24s %-28s %5s  %-15s %-15s %-15s %s" % ("dir", "guard", "seeds", "CV AUPRC", "CV R@1%FPR",
                                                     "test AUPRC", "cell seed sd"))
    for r in rows:
        print("%-24s %-28s %5d  %-15s %-15s %-15s %s" % (
            r["dir"], r["guard"], r["seeds"], pm(r["cv_auprc"]), pm(r["cv_recall_at_fpr"]), pm(r["test_auprc"]),
            "–" if r["cv_cell_seed_sd"] is None else "%.3f" % r["cv_cell_seed_sd"]))
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
