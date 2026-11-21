"""Score-based metrics for a rare positive class (reject), with bootstrap intervals over traces.

    rank_report(y, s, lens)   y: 0/1 per action, s: reject score per action (higher = more reject), lens: actions per trace

Macro-F1 at one threshold moves by several points when a single reject action flips, and is capped at 0.5 on a set with
no reject. These metrics use the whole ranking instead:
  auprc           average precision of the reject class; chance level is the reject share (`prevalence`)
  recall_at_fpr   share of reject actions caught at the strictest threshold that rejects at most FPR of the approve
                  actions (ties are taken whole, so a constant score gives 0)
Both are None when the set has no reject or no approve action. `*_ci` is the 95% percentile interval over traces
resampled with replacement (resamples without a reject action are skipped).
"""
import numpy as np

FPR = 0.01


def average_precision(y, s):
    y, s = np.asarray(y, dtype=bool), np.asarray(s, dtype=float)
    if not y.any():
        return None
    o = np.argsort(-s, kind="stable")
    y, s = y[o], s[o]
    last = np.r_[s[1:] != s[:-1], True]            # last index of each group of tied scores
    tp, n = np.cumsum(y)[last], (np.arange(len(y)) + 1)[last]
    return float(np.sum(np.diff(np.r_[0, tp]) / y.sum() * tp / n))


def recall_at_fpr(y, s, fpr=FPR):
    y, s = np.asarray(y, dtype=bool), np.asarray(s, dtype=float)
    if not y.any() or y.all():
        return None
    neg = np.sort(s[~y])[::-1]
    k = int(np.floor(fpr * len(neg)))              # approve actions that may be rejected
    thr = neg[k] if k < len(neg) else -np.inf      # reject only scores strictly above the (k+1)-th approve score
    return float((s[y] > thr).mean())


def threshold_for_f1(y, s):
    """Threshold t (predict reject when s >= t) with the best reject F1; None if y has no reject. Ties: the highest t."""
    y, s = np.asarray(y, dtype=bool), np.asarray(s, dtype=float)
    if not y.any():
        return None
    best, bt = -1.0, None
    for t in np.unique(s)[::-1]:
        p = s >= t
        tp = float((p & y).sum())
        f1 = 2 * tp / (p.sum() + y.sum())
        if f1 > best:
            best, bt = f1, float(t)
    return bt


def paired_delta(y, sa, sb, lens, n_boot=200, seed=0, fpr=FPR):
    """metric(sa) - metric(sb) on the same actions (two guards scoring the same traces), for AUPRC and recall at FPR.
    The interval is a paired bootstrap: each resample of traces scores both guards. `p_le0` is the share of resamples
    where the AUPRC difference is <= 0 (small = guard a is better). Values are None without a reject action."""
    y, sa, sb = np.asarray(y, dtype=int), np.asarray(sa, dtype=float), np.asarray(sb, dtype=float)
    sub = lambda f, *a: None if any(v is None for v in a) else f(*a)
    d = lambda m, yy, I=None: sub(lambda p, q: p - q, m(yy, sa if I is None else sa[I]), m(yy, sb if I is None else sb[I]))
    out = dict(d_auprc=d(average_precision, y), d_recall_at_fpr=d(lambda yy, s: recall_at_fpr(yy, s, fpr), y),
               d_auprc_ci=None, d_recall_at_fpr_ci=None, p_le0=None)
    if out["d_auprc"] is None or not n_boot or y.all():
        return out
    ends = np.cumsum(lens)
    idx = [np.arange(e - n, e) for e, n in zip(ends, lens)]
    rng, ap, rc = np.random.default_rng(seed), [], []
    for _ in range(n_boot):
        I = np.concatenate([idx[t] for t in rng.integers(0, len(idx), len(idx))])
        yy = y[I]
        if yy.any() and not yy.all():
            ap.append(d(average_precision, yy, I))
            rc.append(d(lambda q, s: recall_at_fpr(q, s, fpr), yy, I))
    if ap:
        out["d_auprc_ci"] = [float(v) for v in np.percentile(ap, [2.5, 97.5])]
        out["d_recall_at_fpr_ci"] = [float(v) for v in np.percentile(rc, [2.5, 97.5])]
        out["p_le0"] = float(np.mean(np.asarray(ap) <= 0))
    return out


def rank_report(y, s, lens, n_boot=200, seed=0, fpr=FPR):
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    out = dict(auprc=average_precision(y, s), recall_at_fpr=recall_at_fpr(y, s, fpr), fpr=fpr,
               prevalence=float(y.mean()) if len(y) else None, auprc_ci=None, recall_at_fpr_ci=None)
    if out["auprc"] is None or out["recall_at_fpr"] is None or not n_boot:
        return out
    ends = np.cumsum(lens)
    idx = [np.arange(e - n, e) for e, n in zip(ends, lens)]
    rng, ap, rc = np.random.default_rng(seed), [], []
    for _ in range(n_boot):
        I = np.concatenate([idx[t] for t in rng.integers(0, len(idx), len(idx))])
        yy, ss = y[I], s[I]
        if yy.any() and not yy.all():
            ap.append(average_precision(yy, ss))
            rc.append(recall_at_fpr(yy, ss, fpr))
    if ap:
        out["auprc_ci"] = [float(v) for v in np.percentile(ap, [2.5, 97.5])]
        out["recall_at_fpr_ci"] = [float(v) for v in np.percentile(rc, [2.5, 97.5])]
    return out
