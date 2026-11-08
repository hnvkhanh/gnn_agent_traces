import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from .sandbox.policy import LABELS


def report(y, p, names=LABELS):
    ids = list(range(len(names)))
    return dict(
        macro_f1=float(f1_score(y, p, average="macro", labels=ids, zero_division=0)),
        acc=float((np.array(y) == np.array(p)).mean()),
        recall=dict(zip(names, recall_score(y, p, labels=ids, average=None, zero_division=0).round(3).tolist())),
        confusion=confusion_matrix(y, p, labels=ids).tolist())


def tool_lookup_baseline(train_meta, train_y, test_meta):
    """Predict by tool name: majority train label per tool; unseen tool -> global majority."""
    from collections import Counter, defaultdict
    per, glob = defaultdict(Counter), Counter()
    k = 0
    for m in train_meta:
        for t in m["tools"]:
            if train_y[k] >= 0:  # -1 = masked (e.g. escalate in binary mode)
                per[t][train_y[k]] += 1
                glob[train_y[k]] += 1
            k += 1
    gm = glob.most_common(1)[0][0]
    return [per[t].most_common(1)[0][0] if t in per else gm for m in test_meta for t in m["tools"]]
