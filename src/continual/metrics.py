"""Continual-learning metrics from a cell's matrix R (rows: init, step 0..T-1; columns: benchmarks in stream order).

    AP   average performance after the last step          mean_j R[T-1][j]
    AF   average forgetting                              mean_{j<T-1} max_{j<=i<T-1} R[i][j] - R[T-1][j]
    BWT  backward transfer                               mean_{j<T-1} R[T-1][j] - R[j][j]
    FWT  forward transfer (zero-shot vs untrained init)  mean_{j>=1} R[j-1][j] - R_init[j]
For ASR (lower is better) 'asr_forgetting' = mean_{j<T-1} ASR[T-1][j] - ASR[j][j]: how many more attacks on earlier
benchmarks get through once the guard has moved on.
"""
import math

import numpy as np


def _get(cell, part, key):
    if key.startswith("asr:"):
        return cell[part]["asr"][key[4:]]
    if key.startswith("recall:"):
        return cell[part]["recall"][key[7:]]
    return cell[part][key]


def matrix(rec, key="macro_f1", part="all"):
    """(init row, T x T matrix) in stream order."""
    order = rec["config"]["order"]
    R = rec["R"]
    init = [_get(R["init"][b], part, key) for b in order]
    mat = [[_get(R[str(i)][b], part, key) for b in order] for i in range(len(order))]
    return init, mat


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def summary(init, mat):
    T = len(mat)
    return dict(
        AP=_mean(mat[T - 1]),
        AF=_mean([max(mat[i][j] for i in range(j, T - 1)) - mat[T - 1][j] for j in range(T - 1)]),
        BWT=_mean([mat[T - 1][j] - mat[j][j] for j in range(T - 1)]),
        FWT=_mean([mat[j - 1][j] - init[j] for j in range(1, T)]),
        diag=_mean([mat[j][j] for j in range(T)]))


def cell_metrics(rec):
    f1 = summary(*matrix(rec))
    real = summary(*matrix(rec, part="real"))
    rej = summary(*matrix(rec, "recall:reject"))
    asr = summary(*matrix(rec, "asr:asr_defended"))
    _, ben = matrix(rec, "asr:benign_block")
    _, fpr = matrix(rec, "asr:approve_fpr")
    return dict(AP=f1["AP"], AF=f1["AF"], BWT=f1["BWT"], FWT=f1["FWT"], diag=f1["diag"],
                AP_real=real["AP"], AF_real=real["AF"], reject_recall=rej["AP"], AF_reject=rej["AF"],
                asr_final=asr["AP"], asr_forgetting=asr["BWT"], asr_diag=asr["diag"],
                benign_block=_mean(ben[-1]), approve_fpr=_mean(fpr[-1]), seconds=rec.get("seconds", float("nan")))


def aggregate(values):
    """mean, std (population, as in the main suite) and n of a list, NaN-safe."""
    v = np.array([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    if not len(v):
        return dict(mean=float("nan"), std=float("nan"), n=0)
    return dict(mean=float(v.mean()), std=float(v.std()), n=int(len(v)))
