import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import unsure_continual as U  # noqa: E402


def metas(n_groups=20, per=3):
    m = []
    for g in range(n_groups):
        for k in range(per):
            m.append({"group": "g%d" % g, "attack_id": ("atk%d" % (g % 5)) if k else None})
    return m


def test_val_split_is_disjoint_from_fit_and_leaves_test_alone():
    ms = metas()
    tr = list(range(0, 40))
    test = list(range(40, len(ms)))
    fit, val = U.val_split(tr, ms, 0)
    assert val and fit and not set(fit) & set(val)
    assert not {ms[i]["group"] for i in fit} & {ms[i]["group"] for i in val}
    assert not ({ms[i]["attack_id"] for i in fit} & {ms[i]["attack_id"] for i in val}) - {None}
    assert not set(fit + val) & set(test)


def test_folds_have_disjoint_groups_and_views_never_include_queried_heldout():
    ms = metas()
    te = list(range(len(ms)))
    a, b = U.fold_split(te, ms, 1)
    assert not {ms[i]["group"] for i in a} & {ms[i]["group"] for i in b} and sorted(a + b) == te
    v = U.views_for("heldout", b, a[:10], ms)
    assert not set(v["all"]) & set(a)                      # scored fold never contains queried traces
    assert set(v["attack_shared"]) | set(v["attack_novel"]) == {i for i in b if ms[i]["attack_id"]}
    assert not set(v["attack_shared"]) & set(v["attack_novel"])
    f = U.views_for("full", te, a[:10], ms)
    assert not set(f["nonqueried"]) & set(a[:10]) and len(f["nonqueried"]) == len(te) - 10


def test_query_selection_uses_confidence_not_labels():
    mc = {0: .5, 1: .95, 2: .7, 3: .99, 4: .89}
    pool = list(mc)
    assert U.select_queries("targeted", pool, mc, .9, 0, random.Random(0)) == [0, 2, 4]
    assert U.select_queries("targeted", pool, mc, .9, 0, random.Random(0), budget=.5) == [0, 2]
    assert U.select_queries("confident", pool, mc, .9, 2, random.Random(0)) == [3, 1]
    assert len(U.select_queries("random", pool, mc, .9, 3, random.Random(0))) == 3
    assert U.select_queries("oracle", pool, mc, .9, 0, random.Random(0)) == pool
    # more permissive threshold never queries fewer
    assert len(U.select_queries("targeted", pool, mc, .97, 0, random.Random(0))) >= 3


def test_temperature_and_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    flip = rng.random(2000) < .2
    z = np.where((y ^ flip)[:, None] == np.array([0, 1]), 20., -20.)
    T = U.fit_temperature(z, y)
    assert T > 5
    m1, m2 = U.metrics(z, y, 1., .9), U.metrics(z, y, T, .9)
    assert m2["ece"] < m1["ece"] / 3 and m1["accuracy"] == m2["accuracy"]      # argmax invariant to T
    assert m2["unsure"] > m1["unsure"] and m2["cw_fn"] + m2["cw_fp"] < m1["cw_fn"] + m1["cw_fp"]
    assert U.fit_temperature(z, np.zeros(2000, int)) == 1.0


def test_fingerprint_changes_with_args(tmp_path):
    t = tmp_path / "x.jsonl"
    t.write_text("{}\n")
    a = {"epochs": 1}
    assert U.fingerprint(a, [str(t)]) == U.fingerprint(a, [str(t)])
    assert U.fingerprint(a, [str(t)]) != U.fingerprint({"epochs": 2}, [str(t)])


def test_report_builds_from_synthetic_cells():
    m = {k: .5 for k in U.METRICS}
    m["n"] = 10
    recs = [{"scenario": s, "direction": "d", "variant": v, "round": r, "view": w, "queried_traces": r, "queried_actions": [1, 1],
             "T": 3., "T0": 2., "before": m, "after": m} for s, ws in U.VIEWS.items() for v in U.variant_names(["finetune", "replay"], "replay")
            for r in ((1,) if v.endswith("/oracle") else (1, 2)) for w in ws]
    cells = [{"model": "gnn", "seed": 0, "T0": 2., "n_fit": 1, "n_val": 1, "n_test": 1, "folds": [1, 1], "records": recs}]
    args = argparse.Namespace(bench="asb", split="task", threshold=.9, update_epochs=1, update_lr=1e-3, buffer=50)
    js, md, page = U.build(cells, args)
    assert "heldout / all" in md and "<svg" in page and "gnn" in js["aggregate"].__str__()


def test_variants_and_fingerprint_follow_strategies(tmp_path):
    assert U.variant_names(["finetune", "ergnn"], "replay") == ["finetune/targeted", "ergnn/targeted", "replay/random", "replay/confident", "replay/oracle"]
    h = tmp_path / "h.json"
    h.write_text('{"twp": {"hp": {"lam_l": 100.0, "beta": 0.001}}}')
    assert U.hp_for("twp", str(h)) == {"lam_l": 100.0, "beta": 0.001}
    assert U.hp_for("ewc", str(h)) is None and U.hp_for("replay", str(h)) == {}


def test_repo_strategies_are_the_ones_used():
    import pytest
    pytest.importorskip("torch_geometric")
    from src.continual.strategies import STRATEGIES
    assert set(U.DEFAULT_STRATEGIES + U.TUNED_NEEDED) <= set(STRATEGIES)
