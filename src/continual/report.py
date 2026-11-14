"""Aggregate the continual-learning cells into REPORT.md, continual.json and a self-contained report.html.

build() turns fresh cell records into one plain dict, and the writers only format that dict, so the markdown, the
HTML page and the JSON always show the same numbers. The HTML is stdlib-only (inline SVG, no scripts, works offline)
and follows the light/dark palette of the dataviz reference instance.
"""
import glob
import html
import json
import math
import os

import numpy as np

from .metrics import aggregate, cell_metrics, matrix
from .strategies import STRATEGIES, TUNED

KEYS = ["AP", "AF", "BWT", "FWT", "AP_real", "reject_recall", "asr_final", "asr_forgetting", "benign_block"]
LABEL = dict(AP="AP macro-F1", AF="AF (forgetting)", BWT="BWT", FWT="FWT", AP_real="AP real-only",
             reject_recall="reject recall", asr_final="final ASR", asr_forgetting="ASR forgetting",
             benign_block="benign blocked")
MEMORY = ["agem", "ergnn", "replay"]
REGULARISERS = ["ewc", "mas", "lwf", "twp"]


def _fmt(a, digits=2):
    if not a or a["n"] == 0 or math.isnan(a["mean"]):
        return "-"
    return "%.*f±%.*f" % (digits, a["mean"], digits, a["std"])


def _signed(a):
    if not a or a["n"] == 0 or math.isnan(a["mean"]):
        return "-"
    return "%+.2f±%.2f" % (a["mean"], a["std"])


def _mean_matrix(recs, key="macro_f1", part="all"):
    mats = [matrix(r, key, part)[1] for r in recs]
    return np.nanmean(np.array(mats, dtype=float), axis=0).tolist() if mats else None


def inbench_reference(results_dir, model_slug="qwen2.5_1.5b"):
    """GNN macro-F1 of the main suite's inbench/<bench> cells (same test sets), per seed, if those results exist."""
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, model_slug, "experiments_*.json"))
                    + glob.glob(os.path.join(results_dir, model_slug, "legacy", "experiments_*.json"))):
        try:
            R = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for k, runs in R.items():
            if k.startswith("inbench/") and k[8:] not in out:
                out[k[8:]] = dict(values=[r["gnn"]["macro_f1"] for r in runs], source=os.path.relpath(p, results_dir))
    return out


def build(main, sweep, tune, hparams, audit, provenance, results_dir):
    """main / sweep: {strategy: [fresh records]}; tune: {strategy: [records]}; audit: from the orchestrator."""
    strategies = [s for s in STRATEGIES if main.get(s)]
    met = {s: [cell_metrics(r) for r in main[s]] for s in strategies}
    headline = {s: {k: aggregate([m[k] for m in met[s]]) for k in KEYS + ["seconds"]} for s in strategies}

    orders = sorted({"-".join(r["config"]["order"]) for s in strategies for r in main[s]})
    per_order = {s: {o: {k: aggregate([cell_metrics(r)[k] for r in main[s] if "-".join(r["config"]["order"]) == o])
                         for k in ("AP", "AF", "asr_forgetting")} for o in orders} for s in strategies}

    canonical = provenance["canonical"]
    canon = {s: [r for r in main[s] if r["config"]["order"] == canonical] for s in strategies}
    rmat = {s: dict(f1=_mean_matrix(canon[s]), asr=_mean_matrix(canon[s], "asr:asr_defended"), n=len(canon[s]))
            for s in strategies if canon[s]}

    # after stage i, over every order: mean macro-F1 over the benchmarks seen so far, and ASR on the first benchmark
    stages = {}
    for s in strategies:
        seen, first = [], []
        for r in main[s]:
            _, f1 = matrix(r)
            _, asr = matrix(r, "asr:asr_defended")
            seen.append([float(np.mean(f1[i][: i + 1])) for i in range(len(f1))])
            first.append([asr[i][0] for i in range(len(asr))])
        stages[s] = dict(seen=np.nanmean(np.array(seen), axis=0).tolist(),
                         first_asr=np.nanmean(np.array(first, dtype=float), axis=0).tolist())

    # buffer sweep, canonical order: the main cells give M = default, the sweep cells the other sizes
    sweep_rows = {}
    for s in [x for x in MEMORY if sweep.get(x)]:
        recs = canon.get(s, []) + sweep[s]
        by_m = {}
        for r in recs:
            by_m.setdefault(r["config"]["buffer"], []).append(cell_metrics(r))
        if by_m:
            sweep_rows[s] = {m: {k: aggregate([x[k] for x in v]) for k in ("AP", "AF")} for m, v in sorted(by_m.items())}

    regs = [s for s in REGULARISERS if s in headline]
    mems = [s for s in MEMORY if s in headline]
    best_reg = max(regs, key=lambda s: headline[s]["AP"]["mean"]) if regs else None
    best_mem = max(mems, key=lambda s: headline[s]["AP"]["mean"]) if mems else None
    focus = [s for s in ("finetune", "joint", best_reg, best_mem) if s and s in headline]

    ref = inbench_reference(results_dir)
    inbench = {}
    for b in provenance["benches"]:
        firsts = [matrix(r)[1][0][0] for r in main.get("finetune", []) if r["config"]["order"][0] == b]
        inbench[b] = dict(cl=aggregate(firsts), main=aggregate(ref[b]["values"]) if b in ref else None,
                          source=ref.get(b, {}).get("source"))

    tune_rows = {s: hparams[s] for s in TUNED if s in hparams}
    return dict(provenance=provenance, strategies=strategies, headline=headline, orders=orders, per_order=per_order,
                rmat=rmat, stages=stages, sweep=sweep_rows, focus=focus, best_reg=best_reg, best_mem=best_mem,
                inbench=inbench, tune=tune_rows, audit=audit, n_tune={s: len(v) for s, v in tune.items()})


# ---------------------------------------------------------------- markdown
def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def headline_rows(d):
    rows = []
    for s in d["strategies"]:
        h = d["headline"][s]
        rows.append([s, _fmt(h["AP"]), _fmt(h["AF"]), _signed(h["BWT"]), _signed(h["FWT"]), _fmt(h["AP_real"]),
                     _fmt(h["reject_recall"]), _fmt(h["asr_final"]), _signed(h["asr_forgetting"]),
                     _fmt(h["benign_block"]), "%.0f" % h["seconds"]["mean"], h["AP"]["n"]])
    return rows


HEAD = ["strategy", "AP macro-F1 ↑", "AF ↓", "BWT", "FWT", "AP real-only ↑", "reject recall ↑", "final ASR ↓",
        "ASR forgetting ↓", "benign blocked ↓", "s/cell", "runs"]


def markdown(d):
    p, a = d["provenance"], d["audit"]
    md = ["# Continual graph learning report", "",
          "The HeteroGNN guard is trained on one benchmark after another (%s traces only) and "
          "re-tested on every benchmark after each stage. Domain-incremental: one head, labels approve/reject." % p["model"],
          "", "| | |", "|---|---|"]
    for k in ("date", "git commit", "python / torch", "devices / parallel jobs", "seeds / epochs per experience",
              "memory M (agem, replay, ergnn)", "orders", "cells", "command"):
        md.append("| %s | %s |" % (k, p[k]))
    md += ["", "Cells are mean±std over seeds × orders. AP = macro-F1 on every benchmark after the last stage; "
           "AF = average forgetting (best earlier score minus final, ↓); BWT = final minus just-trained score; "
           "FWT = zero-shot score on the next benchmark minus the untrained model; ASR = attack traces with an "
           "attacker action the guard lets through; ASR forgetting = how much ASR on earlier benchmarks rises by "
           "the end (↓). 'real-only' drops the appended synthetic attacker steps.", "",
           "## Headline", "", table(headline_rows(d), HEAD)]
    if d["focus"]:
        md += ["", "Best regulariser by AP: **%s**. Best memory method by AP: **%s**." % (d["best_reg"], d["best_mem"])]

    md += ["", "## Per order (AP / AF / ASR forgetting, mean±std over seeds)", "",
           table([[s] + ["%s / %s / %s" % (_fmt(d["per_order"][s][o]["AP"]), _fmt(d["per_order"][s][o]["AF"]),
                                           _signed(d["per_order"][s][o]["asr_forgetting"])) for o in d["orders"]]
                  for s in d["strategies"]], ["strategy"] + [o.replace("-", " → ") for o in d["orders"]])]

    bn = p["canonical"]
    md += ["", "## Accuracy matrix R, canonical order %s (macro-F1, mean over seeds)" % " → ".join(bn), "",
           "Row = after training on that benchmark; column = test benchmark. Lower triangle = memory, upper = zero-shot."]
    for s, m in d["rmat"].items():
        md += ["", "**%s**" % s, "", table([["after " + bn[i]] + ["%.2f" % v for v in m["f1"][i]] for i in range(len(bn))],
                                           ["", *bn])]
    md += ["", "## ASR on each benchmark after each stage, canonical order (lower is better)", ""]
    md.append(table([[s] + [" / ".join("%.2f" % x for x in d["rmat"][s]["asr"][i]) for i in range(len(bn))]
                     for s in d["rmat"]], ["strategy"] + ["after %s (%s)" % (b, " / ".join(bn)) for b in bn]))

    if d["sweep"]:
        ms = sorted({m for v in d["sweep"].values() for m in v})
        md += ["", "## Memory size sweep, canonical order (AP / AF). Sensitivity only: M=%d was fixed in advance." % p["buffer"],
               "", table([[s] + ["%s / %s" % (_fmt(v[m]["AP"]), _fmt(v[m]["AF"])) if m in v else "-" for m in ms]
                          for s, v in d["sweep"].items()], ["strategy"] + ["M=%d" % m for m in ms])]

    if d["tune"]:
        md += ["", "## Hyperparameters (chosen on the validation stream, never on test)", "",
               "Validation = the main suite's task split applied again inside the seed-0 training traces. "
               "Criterion: final mean macro-F1 over the validation sets, canonical order.", "",
               table([[s, json.dumps(v["hp"]), "%.3f" % v["val_ap"],
                       "; ".join("%s → %.3f" % (json.dumps(g), ap) for g, ap in v["grid"])] for s, v in d["tune"].items()],
                     ["strategy", "chosen", "val AP", "grid (hp → val AP)"])]

    md += ["", "## Data and leakage audit", "",
           "Per seed: train/test traces (attack traces) per benchmark, and overlap counts between the union of all "
           "training sets and the union of all test sets. Every overlap must be 0.", "",
           table([["seed %s" % s] + ["%d (%d) / %d (%d)" % (z[b]["train"], z[b]["train_attack"], z[b]["test"], z[b]["test_attack"])
                                     for b in bn] + [", ".join("%s %d" % kv for kv in a["overlaps"][s].items())]
                  for s, z in a["sizes"].items()], ["split"] + ["%s train / test" % b for b in bn] + ["overlaps"]),
           "", "- Validation split vs tune-train / vs test: %s" % a["tune"],
           "- Memory buffers (agem, replay, ergnn): %d stored graphs checked, %d outside the training sets, %d in a test set."
           % (a["buffer_checked"], a["buffer_not_train"], a["buffer_in_test"]),
           "- Joint: %d steps checked, %d trained on anything other than the union of the training sets so far."
           % (a["joint_checked"], a["joint_bad"])]
    if a.get("caveats"):
        md += ["- " + c for c in a["caveats"]]

    md += ["", "## Sanity check against the main suite", "",
           "Stage-1 finetune score on a benchmark (the model has seen only that benchmark) vs the main suite's GNN "
           "`inbench/<bench>` on the same test sets. Not bit-identical: there the GNN is initialised after the MLP "
           "baseline has used the RNG.", "",
           table([[b, _fmt(v["cl"]), _fmt(v["main"]) if v["main"] else "(not found)", v["source"] or "-"]
                  for b, v in d["inbench"].items()], ["benchmark", "CL stage 1 (finetune)", "main inbench GNN", "source"])]
    if p.get("missing"):
        md += ["", "**Incomplete:** %d planned cells have no fresh result (%s). Re-run the script to finish them."
               % (sum(p["missing"].values()), ", ".join("%s %d" % kv for kv in p["missing"].items()))]
    return "\n".join(md) + "\n"


def terminal(d):
    return "Continual learning, %s, %s\n\n%s\n" % (d["provenance"]["model"], " → ".join(d["provenance"]["canonical"]),
                                                    table(headline_rows(d), HEAD))


# ---------------------------------------------------------------- HTML (inline SVG)
CSS = """
.viz-root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--q0:#cde2fb;--q1:#9ec5f4;--q2:#6da7ec;--q3:#3987e5;--q4:#256abf;--q5:#184f95;--q6:#0d366b;
--qt0:#0b0b0b;--qt1:#0b0b0b;--qt2:#0b0b0b;--qt3:#ffffff;--qt4:#ffffff;--qt5:#ffffff;--qt6:#ffffff}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--q0:#0d366b;--q1:#184f95;--q2:#256abf;--q3:#3987e5;
--q4:#6da7ec;--q5:#9ec5f4;--q6:#cde2fb;--qt0:#ffffff;--qt1:#ffffff;--qt2:#ffffff;--qt3:#ffffff;--qt4:#0b0b0b;
--qt5:#0b0b0b;--qt6:#0b0b0b}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--q0:#0d366b;--q1:#184f95;--q2:#256abf;--q3:#3987e5;--q4:#6da7ec;--q5:#9ec5f4;--q6:#cde2fb;--qt0:#ffffff;
--qt1:#ffffff;--qt2:#ffffff;--qt3:#ffffff;--qt4:#0b0b0b;--qt5:#0b0b0b;--qt6:#0b0b0b}
body{margin:0;background:#f9f9f7}
@media (prefers-color-scheme:dark){body{background:#0d0d0d}}
.viz-root{background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
padding:24px 16px 48px;min-height:100vh;box-sizing:border-box}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}h3{font-size:15px;margin:16px 0 4px}
p,li{color:var(--ink2);max-width:80ch}.muted{color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin:12px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.scroll{overflow-x:auto}.grid2>*,.grid4>*{min-width:0}
table.wrap td{white-space:normal;overflow-wrap:anywhere;text-align:left}table.wrap td:first-child{white-space:nowrap;overflow-wrap:normal}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}
svg{display:block;width:100%;height:auto}svg text{fill:var(--ink2);font-size:12px}
.tick{fill:var(--muted)!important;font-size:11px!important}.gl{stroke:var(--grid);stroke-width:1}
.ax{stroke:var(--axis);stroke-width:1}.legend{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 8px;font-size:13px;color:var(--ink2)}
.key{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.tile{display:flex;flex-direction:column}.tile b{font-size:22px;font-weight:600}.tile span{color:var(--ink2);font-size:13px}
.ok{color:var(--ink)}
"""


def _e(s):
    return html.escape(str(s))


def svg_hbars(rows, title, xmax, better):
    """rows: [(label, aggregate)]; one series, single hue; value at the bar tip, whisker = ±std."""
    W, row_h, left, right = 520, 30, 92, 64
    H = 28 + row_h * len(rows) + 22
    sx = lambda v: left + (W - left - right) * max(0.0, min(v, xmax)) / xmax
    out = ['<svg viewBox="0 0 %d %d" style="max-width:%dpx" role="img" aria-label="%s">' % (W, H, W, _e(title))]
    for t in np.linspace(0, xmax, 5):
        out.append('<line class="gl" x1="%.1f" x2="%.1f" y1="20" y2="%d"/>' % (sx(t), sx(t), H - 22))
        out.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%.2g</text>' % (sx(t), H - 6, t))
    for i, (lab, a) in enumerate(rows):
        y = 28 + i * row_h
        m, s = a["mean"], a["std"]
        if math.isnan(m):
            continue
        x0, x1 = sx(0), sx(m)
        w = max(x1 - x0, 0.0)
        r = min(4.0, w / 2)
        out.append('<g><title>%s: %.3f ± %.3f (%s is better)</title>' % (_e(lab), m, s, better))
        out.append('<path d="M%.1f %.1f h%.1f a%.1f %.1f 0 0 1 %.1f %.1f v%.1f a%.1f %.1f 0 0 1 -%.1f %.1f h-%.1f z" fill="var(--s1)"/>'
                   % (x0, y, w - r, r, r, r, r, 14 - 2 * r, r, r, r, r, w - r))
        out.append('<line x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f" stroke="var(--ink2)" stroke-width="1.5"/>'
                   % (sx(m - s), sx(m + s), y + 7, y + 7))
        out.append('<rect x="0" y="%d" width="%d" height="%d" fill="transparent"/></g>' % (y - 6, W, row_h))
        out.append('<text x="%d" y="%.1f" text-anchor="end">%s</text>' % (left - 8, y + 11, _e(lab)))
        out.append('<text x="%.1f" y="%.1f">%.2f</text>' % (max(x1, sx(m + s)) + 6, y + 11, m))
    out.append('<line class="ax" x1="%.1f" x2="%.1f" y1="20" y2="%d"/></svg>' % (sx(0), sx(0), H - 22))
    return "".join(out)


def svg_lines(series, xlabels, ylim, title, ylab):
    """series: [(name, css var, values)] up to 4; 2px lines, 8px end markers with a surface ring, end labels."""
    W, H, left, right, top, bottom = 520, 270, 44, 110, 28, 34
    lo, hi = ylim
    sx = lambda i: left + (W - left - right) * (i / max(1, len(xlabels) - 1))
    sy = lambda v: top + (H - top - bottom) * (1 - (v - lo) / (hi - lo))
    out = ['<svg viewBox="0 0 %d %d" style="max-width:%dpx" role="img" aria-label="%s">' % (W, H, W, _e(title))]
    for t in np.linspace(lo, hi, 5):
        out.append('<line class="gl" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (left, W - right, sy(t), sy(t)))
        out.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">%.2f</text>' % (left - 6, sy(t) + 4, t))
    for i, lab in enumerate(xlabels):
        out.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (sx(i), H - bottom + 18, _e(lab)))
    out.append('<text class="tick" x="2" y="12">%s</text>' % _e(ylab))
    ends = []
    for name, color, vals in series:
        pts = [(sx(i), sy(v)) for i, v in enumerate(vals) if v is not None and not math.isnan(v)]
        if not pts:
            continue
        out.append('<polyline fill="none" stroke="var(%s)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="%s"/>'
                   % (color, " ".join("%.1f,%.1f" % p for p in pts)))
        for (x, y), v, lab in zip(pts, vals, xlabels):
            out.append('<g><title>%s, %s: %.3f</title><circle cx="%.1f" cy="%.1f" r="9" fill="transparent"/>'
                       '<circle cx="%.1f" cy="%.1f" r="4" fill="var(%s)" stroke="var(--surface)" stroke-width="2"/></g>'
                       % (_e(name), _e(lab), v, x, y, x, y, color))
        ends.append([pts[-1][1], name, vals[-1]])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels at least 14px apart, with a leader line to the end marker
        ends[i].append(ends[i][0])
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
    x_end = sx(len(xlabels) - 1)
    for e in ends:
        y, name, v = e[0], e[1], e[2]
        y0 = e[3] if len(e) > 3 else y
        if abs(y - y0) > 1:
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="var(--muted)" stroke-width="1"/>'
                       % (x_end + 6, y0, x_end + 14, y))
        out.append('<text x="%.1f" y="%.1f">%s %.2f</text>' % (x_end + 16, y + 4, _e(name), v))
    out.append('<line class="ax" x1="%d" x2="%d" y1="%d" y2="%d"/></svg>' % (left, W - right, H - bottom, H - bottom))
    return "".join(out)


HEAT_LO = 0.3  # colour scale of the heatmaps: macro-F1 0.3 (lightest in light mode) to 1.0 (darkest)


def svg_heatmap(mat, labels, title):
    n = len(labels)
    cell, left, top = 56, 92, 40
    W, H = left + cell * n + 8, top + cell * n + 8
    out = ['<svg viewBox="0 0 %d %d" style="max-width:%dpx" role="img" aria-label="%s">' % (W, H, W, _e(title))]
    for j, lab in enumerate(labels):
        out.append('<text class="tick" x="%d" y="%d" text-anchor="middle">%s</text>' % (left + cell * j + cell / 2, top - 22, _e(lab)))
    out.append('<text class="tick" x="%d" y="%d" text-anchor="middle">test on →</text>' % (left + cell * n / 2, top - 6))
    for i, row in enumerate(mat):
        out.append('<text class="tick" x="%d" y="%d" text-anchor="end">after %s</text>' % (left - 6, top + cell * i + cell / 2 + 4, _e(labels[i])))
        for j, v in enumerate(row):
            q = int(min(6, max(0, math.floor((v - HEAT_LO) / (1 - HEAT_LO) * 7)))) if not math.isnan(v) else 0
            x, y = left + cell * j, top + cell * i
            out.append('<g><title>after %s, test %s: macro-F1 %.3f</title><rect x="%d" y="%d" width="%d" height="%d" rx="4" fill="var(--q%d)"/>'
                       '<text x="%d" y="%d" text-anchor="middle" style="fill:var(--qt%d)">%.2f</text></g>'
                       % (_e(labels[i]), _e(labels[j]), v, x + 1, y + 1, cell - 2, cell - 2, q, x + cell / 2, y + cell / 2 + 4, q, v))
    out.append("</svg>")
    return "".join(out)


def html_table(header, rows, wrap=False):
    h = "".join("<th>%s</th>" % _e(c) for c in header)
    b = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % _e(c) for c in r) for r in rows)
    return '<div class="scroll"><table%s><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (
        ' class="wrap"' if wrap else "", h, b)


def page(d):
    p, a = d["provenance"], d["audit"]
    bn = p["canonical"]
    H = d["headline"]
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             "<title>Continual Graph Learning</title><style>%s</style></head><body><div class=\"viz-root\"><div class=\"wrap\">" % CSS,
             "<h1>Continual graph learning</h1><p>HeteroGNN guard trained on %s one after another (%s traces), "
             "re-tested on every benchmark after each stage. Mean±std over seeds × orders "
             "(seeds / epochs per experience: %s). Cells: %s.</p>"
             % (", ".join(bn), _e(p["model"]), _e(p["seeds / epochs per experience"]), _e(p["cells"]))]
    if p.get("missing"):
        parts.append('<p><b>Incomplete:</b> %d planned cells have no fresh result. Re-run the script.</p>' % sum(p["missing"].values()))

    # tiles: the comparison the experiment exists for
    tiles = []
    for s in d["focus"]:
        tiles.append('<div class="card tile"><span>%s</span><b>%.2f</b><span>AP macro-F1 · forgetting %.2f · ASR forgetting %+.2f</span></div>'
                     % (_e(s), H[s]["AP"]["mean"], H[s]["AF"]["mean"], H[s]["asr_forgetting"]["mean"]))
    parts.append('<div class="grid4">%s</div>' % "".join(tiles))

    order = sorted(d["strategies"], key=lambda s: -H[s]["AP"]["mean"])
    parts.append("<h2>Retention vs forgetting</h2><div class=\"grid2\">")
    parts.append('<div class="card"><h3>AP: macro-F1 on all benchmarks after the last stage (higher is better)</h3>%s</div>'
                 % svg_hbars([(s, H[s]["AP"]) for s in order], "AP per strategy", 1.0, "higher"))
    amax = max([H[s]["AF"]["mean"] + H[s]["AF"]["std"] for s in order] + [0.05])
    parts.append('<div class="card"><h3>AF: average forgetting (lower is better)</h3>%s</div></div>'
                 % svg_hbars([(s, H[s]["AF"]) for s in order], "AF per strategy", round(amax * 1.15 + 0.005, 2), "lower"))

    colors = ["--s1", "--s2", "--s3", "--s4"]
    legend = '<div class="legend">%s</div>' % "".join('<span><i class="key" style="background:var(%s)"></i>%s</span>' % (c, _e(s))
                                                     for s, c in zip(d["focus"], colors))
    stage_lab = ["stage %d" % (i + 1) for i in range(len(bn))]
    parts.append("<h2>Over the stream (all orders)</h2>" + legend + '<div class="grid2">')
    seen = [(s, c, d["stages"][s]["seen"]) for s, c in zip(d["focus"], colors)]
    vals = [v for _, _, vs in seen for v in vs]
    lo = math.floor(min(vals + [1.0]) * 10) / 10
    parts.append('<div class="card"><h3>Mean macro-F1 over the benchmarks seen so far</h3>%s</div>'
                 % svg_lines(seen, stage_lab, (lo, 1.0), "macro-F1 over seen benchmarks", "macro-F1"))
    asr = [(s, c, d["stages"][s]["first_asr"]) for s, c in zip(d["focus"], colors)]
    hi = math.ceil(max([v for _, _, vs in asr for v in vs if not math.isnan(v)] + [0.1]) * 10) / 10
    parts.append('<div class="card"><h3>ASR on the first benchmark\'s attacks as the guard moves on (lower is better)</h3>%s</div></div>'
                 % svg_lines(asr, stage_lab, (0.0, hi), "ASR on the first benchmark", "attack success rate"))

    if d["rmat"]:
        parts.append("<h2>Accuracy matrix, canonical order (%s)</h2><p>Row: after training on that benchmark. Column: "
                     "test benchmark. Below the diagonal = memory of earlier benchmarks; above = zero-shot. Darker = higher "
                     "macro-F1 (colour scale %.1f to 1.0; values printed).</p><div class=\"grid4\">" % (_e(" → ".join(bn)), HEAT_LO))
        for s in [x for x in ("finetune", d["best_reg"], "ergnn", "joint") if x in d["rmat"]]:
            parts.append('<div class="card"><h3>%s</h3>%s</div>' % (_e(s), svg_heatmap(d["rmat"][s]["f1"], bn, "R matrix " + s)))
        parts.append("</div>")

    if d["sweep"]:
        ms = sorted({m for v in d["sweep"].values() for m in v})
        ser = [(s, colors[i], [d["sweep"][s][m]["AP"]["mean"] if m in d["sweep"][s] else float("nan") for m in ms])
               for i, s in enumerate(d["sweep"])]
        lo = math.floor(min([v for _, _, vs in ser for v in vs if not math.isnan(v)] + [1.0]) * 10) / 10
        parts.append("<h2>Memory size (canonical order, sensitivity only)</h2>")
        parts.append('<div class="legend">%s</div>' % "".join('<span><i class="key" style="background:var(%s)"></i>%s</span>' % (c, _e(s)) for s, c, _ in ser))
        parts.append('<div class="card">%s</div>' % svg_lines(ser, ["M=%d" % m for m in ms], (lo, 1.0), "AP vs memory size", "AP macro-F1"))

    parts.append("<h2>All strategies</h2><div class=\"card\">%s</div>" % html_table(HEAD, headline_rows(d)))
    if d["tune"]:
        parts.append("<h2>Hyperparameters (validation stream, never test)</h2><div class=\"card\">%s</div>"
                     % html_table(["strategy", "chosen", "val AP", "grid"],
                                  [[s, json.dumps(v["hp"]), "%.3f" % v["val_ap"],
                                    "; ".join("%s → %.3f" % (json.dumps(g), ap) for g, ap in v["grid"])] for s, v in d["tune"].items()],
                                  wrap=True))
    parts.append("<h2>Leakage audit</h2><div class=\"card\">%s<ul><li>Validation split: %s</li>"
                 "<li>Memory buffers: %d stored graphs checked, %d outside training, %d in test</li>"
                 "<li>Joint: %d steps checked, %d trained on anything but the training union so far</li>%s</ul></div>"
                 % (html_table(["split"] + ["%s train / test" % b for b in bn] + ["overlaps (must be 0)"],
                               [["seed %s" % s] + ["%d (%d) / %d (%d)" % (z[b]["train"], z[b]["train_attack"], z[b]["test"], z[b]["test_attack"])
                                                   for b in bn] + [", ".join("%s %d" % kv for kv in a["overlaps"][s].items())]
                                for s, z in a["sizes"].items()], wrap=True),
                    _e(a["tune"]), a["buffer_checked"], a["buffer_not_train"], a["buffer_in_test"], a["joint_checked"], a["joint_bad"],
                    "".join("<li>%s</li>" % _e(c) for c in a.get("caveats", []))))
    parts.append("<h2>Sanity check vs the main suite</h2><div class=\"card\">%s<p class=\"muted\">Same test sets; not bit-identical "
                 "(there the GNN is initialised after the MLP baseline used the RNG).</p></div>"
                 % html_table(["benchmark", "CL stage 1 (finetune)", "main inbench GNN", "source"],
                              [[b, _fmt(v["cl"]), _fmt(v["main"]) if v["main"] else "(not found)", v["source"] or "-"]
                               for b, v in d["inbench"].items()], wrap=True))
    parts.append("<h2>Provenance</h2><div class=\"card\">%s</div>" % html_table(["", ""], [[k, p[k]] for k in (
        "date", "git commit", "python / torch", "devices / parallel jobs", "seeds / epochs per experience",
        "memory M (agem, replay, ergnn)", "orders", "cells", "command")], wrap=True))
    parts.append("</div></div></body></html>")
    return "".join(parts)


def _clean(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def write(d, outdir):
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    for name, text in (("REPORT.md", markdown(d)), ("report.html", page(d)),
                       ("continual.json", json.dumps(_clean(d), indent=1, sort_keys=True))):
        path = os.path.join(outdir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        paths[name] = path
    return paths
