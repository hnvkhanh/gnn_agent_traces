"""Tests of the continual graph learning experiments (src/continual, scripts/run_continual.py).

Leakage controls C1-C6 run on the stored qwen2.5:1.5b traces; training tests run on CPU with tiny epochs.
"""
import copy
import json
import os
import subprocess
import sys

import pytest
import torch

import src.train as T
from src.continual import metrics as M
from src.continual import report as REP
from src.continual import run as RUN
from src.continual import stream as S
from src.continual.strategies import STRATEGIES, TUNED, batches, flat_grad, make
from src.model.gnn import HeteroGNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def exps():
    return S.load_stream()


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(T, "DEVICE", torch.device("cpu"))  # deterministic, and keeps the GPUs free


def tiny(exps, n_tr=16, n_te=8, seed=0):
    """A small slice of the real seed-0 split: (experiences, split) with the same train/test separation."""
    sp = S.splits(exps, seed)
    out, tsp = {}, {}
    for b, e in exps.items():
        tr, te = sp[b][0][:n_tr], sp[b][1][:n_te]
        idx = tr + te
        out[b] = S.Experience(b, [e.graphs[i] for i in idx], [e.metas[i] for i in idx], [e.ids[i] for i in idx],
                              [e.tasks[i] for i in idx])
        tsp[b] = (list(range(len(tr))), list(range(len(tr), len(idx))))
    return out, tsp


# ---------------------------------------------------------------- C6: the right traces, and only those
def test_stream_is_qwen_1_5b_without_agentpoison(exps):
    assert list(exps) == ["asb", "injecagent", "agentdojo"] and "agentpoison" not in S.BENCHES
    for b, e in exps.items():
        assert len(e) > 100
        assert {m["model"] for m in e.metas} == {"qwen2.5:1.5b"}
        assert {m["benchmark"] for m in e.metas} == {b}
        assert len(set(e.ids)) == len(e.ids)


def test_loader_matches_train_load(exps):
    g, m = T.load("traces/agentdojo__*.jsonl", models=["qwen2.5:1.5b"])
    e = exps["agentdojo"]
    assert m == e.metas and len(g) == len(e.graphs)
    assert all(torch.equal(a["action"].x, b["action"].x) for a, b in zip(g, e.graphs))


# ---------------------------------------------------------------- C1-C3: splits on the stored traces
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stream_split_has_no_leakage(exps, seed):
    sp = S.splits(exps, seed)
    assert S.audit(exps, sp) == dict(traces=0, groups=0, attacks=0, task_text=0, attacker_tokens=0)
    S.check(exps, sp)
    for b, (tr, te) in sp.items():
        assert tr and te and not set(tr) & set(te)
        assert sum(exps[b].metas[i]["attack"] for i in te) > 0
        assert sum(exps[b].metas[i]["attack"] for i in tr) > 0
    # the same test sets as the main suite's inbench split
    for b, e in exps.items():
        assert sp[b] == T.split(e.metas, "task", None, seed)


def test_benchmarks_share_no_groups_or_attacks(exps):
    for a in S.BENCHES:
        for b in S.BENCHES:
            if a < b:
                assert not {m["group"] for m in exps[a].metas} & {m["group"] for m in exps[b].metas}
                assert not ({m["attack_id"] for m in exps[a].metas} & {m["attack_id"] for m in exps[b].metas}) - {None}


def test_check_raises_on_a_leaky_split(exps):
    sp = S.splits(exps, 0)
    tr, te = sp["asb"]
    leaky = dict(sp, asb=(tr + te[:1], te))
    with pytest.raises(AssertionError):
        S.check(exps, leaky)


# ---------------------------------------------------------------- C5: tuning never sees test
def test_tune_split_is_inside_training_and_disjoint(exps):
    tsp, test = S.tune_splits(exps, 0), S.splits(exps, 0)
    zero = dict(traces=0, groups=0, attacks=0, task_text=0, attacker_tokens=0)
    assert S.audit(exps, tsp) == zero  # tune-train vs validation
    assert S.overlaps(exps, S.side(exps, tsp, 1), S.side(exps, test, 1)) == zero  # validation vs test
    assert S.overlaps(exps, S.side(exps, tsp, 0), S.side(exps, test, 1)) == zero  # tune-train vs test
    for b in exps:
        assert set(tsp[b][0]) | set(tsp[b][1]) <= set(test[b][0])
        assert tsp[b][1]


# ---------------------------------------------------------------- strategies
HP = {"ewc": {"lam": 100.0}, "mas": {"lam": 1.0}, "lwf": {"alpha": 1.0}, "twp": {"lam_l": 100.0, "beta": 1e-3}}


def test_every_strategy_runs_and_scores_every_benchmark(exps):
    ex, sp = tiny(exps)
    for s in STRATEGIES:
        rec = RUN.run_cell(ex, sp, S.CANONICAL, s, HP.get(s, {}), 6, 0, 1)
        assert set(rec["R"]) == {"init", "0", "1", "2"}
        for row in rec["R"].values():
            assert set(row) == set(S.BENCHES)
            for cell in row.values():
                assert 0.0 <= cell["all"]["macro_f1"] <= 1.0 and "asr" in cell["all"] and "macro_f1" in cell["real"]
        m = M.cell_metrics(rec)
        assert all(k in m for k in ("AP", "AF", "BWT", "FWT", "asr_forgetting"))


def test_memories_hold_only_training_graphs(exps):
    ex, sp = tiny(exps)
    train = {(b, ex[b].ids[i]) for b in ex for i in sp[b][0]}
    test = {(b, ex[b].ids[i]) for b in ex for i in sp[b][1]}
    for s in ("agem", "replay", "ergnn"):
        rec = RUN.run_cell(ex, sp, S.CANONICAL, s, {}, 6, 0, 1)
        for k, st in enumerate(rec["steps"]):
            mem = {tuple(x) for x in st["memory"]}
            assert mem and len(st["memory"]) <= 6
            assert mem <= train and not mem & test
            assert {b for b, _ in mem} <= set(S.CANONICAL[: k + 1])  # only finished experiences
        if s == "ergnn":  # memory shared by every seen experience
            assert {b for b, _ in rec["steps"][-1]["memory"]} == set(S.BENCHES)


def test_joint_trains_on_exactly_the_union_so_far(exps):
    ex, sp = tiny(exps)
    rec = RUN.run_cell(ex, sp, S.CANONICAL, "joint", {}, 0, 0, 1)
    sizes = [len(sp[b][0]) for b in S.CANONICAL]
    assert [st["trained_graphs"] for st in rec["steps"]] == [sum(sizes[: k + 1]) for k in range(3)]
    strat = make("joint", {}, 0, 0, lambda: HeteroGNN(8))
    cur = [ex["agentdojo"].graphs[i] for i in sp["agentdojo"][0]]
    past = [[ex[b].graphs[i] for i in sp[b][0]] for b in ("asb", "injecagent")]
    _, graphs = strat.start(2, HeteroGNN(8), cur, past)
    assert [id(g) for g in graphs] == [id(g) for p in past for g in p] + [id(g) for g in cur]


@pytest.mark.parametrize("name", ["ewc", "mas", "twp"])
def test_penalty_is_zero_at_the_anchor_and_grows_away_from_it(exps, name):
    ex, sp = tiny(exps)
    torch.manual_seed(0)
    model = HeteroGNN(64)
    graphs = [ex["asb"].graphs[i] for i in sp["asb"][0]]
    strat = make(name, HP[name], 0, 0, None)
    strat.end(0, model, graphs, [("asb", ex["asb"].ids[i]) for i in sp["asb"][0]], torch.ones(2))
    assert float(strat.penalty(model)) == 0.0
    assert any(float(i.sum()) > 0 for i in strat.terms[0][0])
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    assert float(strat.penalty(model)) > 0.0


def test_agem_gradient_never_conflicts_with_memory(exps):
    ex, sp = tiny(exps)
    torch.manual_seed(0)
    model = HeteroGNN(64).eval()  # no dropout, so the reference gradient can be recomputed exactly
    strat = make("agem", {}, 4, 0, None)
    asb = [ex["asb"].graphs[i] for i in sp["asb"][0]]
    strat.end(0, model, asb, [("asb", ex["asb"].ids[i]) for i in sp["asb"][0]], torch.ones(2))
    w = torch.ones(2)
    mem = [x[2] for x in strat.mem]
    model.zero_grad()
    b = next(batches(mem))
    torch.nn.functional.cross_entropy(model(b), b["action"].y, weight=w).backward()
    ref = flat_grad(model)
    for bench in ("injecagent", "agentdojo"):
        cur = next(batches([ex[bench].graphs[i] for i in sp[bench][0]], 8))
        model.zero_grad()
        torch.nn.functional.cross_entropy(model(cur), cur["action"].y, weight=w).backward()
        strat.after_backward(model, w)
        g = flat_grad(model)
        assert float(torch.dot(g, ref)) >= -1e-4 * float(g.norm() * ref.norm())


def test_lwf_distillation_is_zero_against_itself(exps):
    ex, sp = tiny(exps)
    model = HeteroGNN(64).eval()
    strat = make("lwf", {"alpha": 1.0}, 0, 0, None)
    strat.start(1, model, [], [])
    b = next(batches([ex["asb"].graphs[i] for i in sp["asb"][0]], 8))
    assert abs(float(strat.distill(model, b, model(b)))) < 1e-6


def test_one_experience_finetune_equals_train_evaluate(exps):
    """The CL engine reproduces the main suite's training loop exactly (same seed, same RNG stream)."""
    e = exps["agentdojo"]
    sp = S.splits({"agentdojo": e}, 0)
    rec = RUN.run_cell({"agentdojo": e}, sp, ["agentdojo"], "finetune", {}, 0, 0, 2)
    tr, te = sp["agentdojo"]
    res, _ = T.evaluate([e.graphs[i] for i in tr], [e.metas[i] for i in tr], [e.graphs[i] for i in te],
                        [e.metas[i] for i in te], epochs=2, seed=0, models=("gnn",))
    got = rec["R"]["0"]["agentdojo"]["all"]
    assert got["confusion"] == res["gnn"]["confusion"] and got["macro_f1"] == res["gnn"]["macro_f1"]
    assert got["asr"] == res["gnn"]["asr"]


# ---------------------------------------------------------------- metrics and tuning
def _rec(f1, init, asr, order=("a", "b", "c")):
    def cell(v, a):
        return dict(all=dict(macro_f1=v, recall=dict(approve=v, reject=v),
                             asr=dict(asr_defended=a, benign_block=0.0, approve_fpr=0.0)), real=dict(macro_f1=v))
    R = {"init": {b: cell(init[j], 0.0) for j, b in enumerate(order)}}
    for i, row in enumerate(f1):
        R[str(i)] = {b: cell(row[j], asr[i][j]) for j, b in enumerate(order)}
    return dict(R=R, config=dict(order=list(order)), seconds=1.0)


def test_cl_metrics_by_hand():
    rec = _rec([[0.9, 0.5, 0.6], [0.7, 0.8, 0.6], [0.6, 0.7, 0.9]], [0.4, 0.4, 0.4],
               [[0.1, 0.9, 0.9], [0.3, 0.2, 0.9], [0.4, 0.5, 0.3]])
    m = M.cell_metrics(rec)
    assert m["AP"] == pytest.approx(2.2 / 3)
    assert m["AF"] == pytest.approx(0.2)      # (0.9 - 0.6 + 0.8 - 0.7) / 2
    assert m["BWT"] == pytest.approx(-0.2)    # (0.6 - 0.9 + 0.7 - 0.8) / 2
    assert m["FWT"] == pytest.approx(0.15)    # (0.5 - 0.4 + 0.6 - 0.4) / 2
    assert m["asr_forgetting"] == pytest.approx(0.3)  # (0.4 - 0.1 + 0.5 - 0.2) / 2
    assert m["asr_final"] == pytest.approx(0.4)


def test_choose_picks_best_validation_ap_and_breaks_ties_low():
    def tune_rec(hp, ap):
        r = _rec([[ap] * 3] * 3, [0.0] * 3, [[0.0] * 3] * 3)
        r["config"].update(strategy="twp", hp=dict(hp, lam_t=1.0))
        return r
    recs = [tune_rec({"lam_l": 10.0, "beta": 0.0}, 0.6), tune_rec({"lam_l": 100.0, "beta": 0.0}, 0.8),
            tune_rec({"lam_l": 1000.0, "beta": 0.0}, 0.8), tune_rec({"lam_l": 10.0, "beta": 0.01}, 0.7)]
    ch = RUN.choose(recs)
    assert ch["hp"] == {"beta": 0.0, "lam_l": 100.0} and ch["val_ap"] == pytest.approx(0.8)
    assert len(ch["grid"]) == 4
    assert len(RUN.grid_points("twp")) == 12 and set(TUNED) == {"ewc", "mas", "lwf", "twp"}


# ---------------------------------------------------------------- fingerprints and reuse
def test_fingerprint_tracks_seed_hparams_and_data(exps, tmp_path):
    pl = RUN.Planner(exps, epochs=1, device_type="cpu")
    out = str(tmp_path)
    a = pl.cell(out, "main", "ewc", S.CANONICAL, 0, {"lam": 10.0}, 50)
    assert a["fp"] == pl.cell(out, "main", "ewc", S.CANONICAL, 0, {"lam": 10.0}, 50)["fp"]
    assert a["fp"] != pl.cell(out, "main", "ewc", S.CANONICAL, 1, {"lam": 10.0}, 50)["fp"]
    assert a["fp"] != pl.cell(out, "main", "ewc", S.CANONICAL, 0, {"lam": 100.0}, 50)["fp"]
    assert a["fp"] != pl.cell(out, "main", "ewc", list(reversed(S.CANONICAL)), 0, {"lam": 10.0}, 50)["fp"]
    assert a["fp"] != RUN.Planner(exps, epochs=2, device_type="cpu").cell(out, "main", "ewc", S.CANONICAL, 0, {"lam": 10.0}, 50)["fp"]
    assert a["fp"] != RUN.Planner(exps, epochs=1, device_type="cuda").cell(out, "main", "ewc", S.CANONICAL, 0, {"lam": 10.0}, 50)["fp"]
    # the buffer size only matters for memory strategies
    assert a["fp"] == pl.cell(out, "main", "ewc", S.CANONICAL, 0, {"lam": 10.0}, 20)["fp"]
    assert pl.cell(out, "main", "replay", S.CANONICAL, 0, {}, 50)["fp"] != pl.cell(out, "main", "replay", S.CANONICAL, 0, {}, 20)["fp"]
    # one changed tensor in one training graph changes the data digest
    changed = {b: S.Experience(b, list(e.graphs), e.metas, e.ids, e.tasks) for b, e in exps.items()}
    i = S.splits(exps, 0)["asb"][0][0]
    g = copy.deepcopy(changed["asb"].graphs[i])
    g["action"].x[0, 0] += 1.0
    changed["asb"].graphs[i] = g
    assert RUN.Planner(changed, epochs=1, device_type="cpu").cell(out, "main", "ewc", S.CANONICAL, 0, {"lam": 10.0}, 50)["fp"] != a["fp"]


def test_matching_cell_is_reused_and_stale_cell_is_not(exps, tmp_path, monkeypatch):
    pl = RUN.Planner(exps, epochs=1, device_type="cpu")
    c = pl.cell(str(tmp_path), "main", "finetune", S.CANONICAL, 0, {}, 50)
    assert RUN.load_cell(c) is None
    rec = RUN.compute(pl, c)
    assert RUN.load_cell(c)["fingerprint"] == c["fp"] == rec["fingerprint"]
    stale = dict(c, fp="0" * 64)
    assert RUN.load_cell(stale) is None
    with open(c["path"], "w", encoding="utf-8") as f:
        f.write("{broken")
    assert RUN.load_cell(c) is None


# ---------------------------------------------------------------- the script end to end, then fully reused
def test_run_continual_end_to_end_then_reused(tmp_path):
    out = str(tmp_path / "cl")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_continual.py"), "--devices", "cpu", "--jobs", "1",
           "--seeds", "1", "--epochs", "1", "--strategies", "finetune,lwf", "--sweep", "none", "--outdir", out]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    first = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=900)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    assert "16 planned: 0 reused, 16 computed, 0 missing" in first.stdout  # 4 lwf tuning + 6 finetune + 6 lwf
    for name in ("REPORT.md", "report.html", "continual.json", "hparams.json"):
        assert os.path.exists(os.path.join(out, name)), name
    md = open(os.path.join(out, "REPORT.md"), encoding="utf-8").read()
    assert "## Headline" in md and "| lwf |" in md and "traces 0, groups 0, attacks 0, task_text 0, attacker_tokens 0" in md
    assert "<svg" in open(os.path.join(out, "report.html"), encoding="utf-8").read()
    data = json.load(open(os.path.join(out, "continual.json"), encoding="utf-8"))
    assert data["audit"]["buffer_in_test"] == 0 and data["audit"]["joint_bad"] == 0
    second = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=900)
    assert second.returncode == 0, second.stdout[-3000:]
    assert "16 planned: 16 reused, 0 computed, 0 missing" in second.stdout
    assert not os.path.exists(os.path.join(out, ".lock"))
