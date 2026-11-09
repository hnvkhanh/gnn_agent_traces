"""Write results/report.html: a self-contained page with charts of the experiment results (no extra dependencies).

    python scripts/report_page.py                    # reads results/, traces/; writes results/report.html
    python scripts/report_page.py --model qwen2.5:1.5b --out results/report.html

Sections: run provenance (which result cells were reused or trained), AgentPoison behaviour, in-benchmark scores,
attack success rate, cross-benchmark transfer, ablations, the headline across trace LLMs, and the leakage audit run on
the stored traces. Every chart has a table next to it.
"""
import argparse
import datetime
import glob
import html
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BENCHES = ["asb", "injecagent", "agentdojo", "agentpoison"]
BENCH_NAME = {"asb": "ASB", "injecagent": "InjecAgent", "agentdojo": "AgentDojo", "agentpoison": "AgentPoison"}
GUARDS = [("gnn", "GNN"), ("mlp", "MLP (no graph)"), ("tool-lookup", "Tool lookup"), ("majority", "Majority")]
ABLATIONS = ["tool_feat", "observation", "memory", "injected", "copies", "mentions"]
ABL_NAME = {"tool_feat": "tool features", "observation": "observations", "memory": "memory", "injected": "injected text",
            "copies": "copies edges", "mentions": "mentions edges"}
SECTIONS = ["inbench", "transfer", "shifts", "ablate"]


def esc(x):
    return html.escape(str(x), quote=True)


def slug(m):
    return m.replace(":", "_")


def mean(v):
    v = [x for x in v if x is not None and x == x]
    return sum(v) / len(v) if v else float("nan")


def std(v):
    m = mean(v)
    v = [x for x in v if x is not None and x == x]
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 if v else float("nan")


def load_results(model_dir):
    """Merged {cell key: runs} and {cell key: provenance} for one trace LLM (per-section files win over legacy)."""
    R, P = {}, {}
    for d in (os.path.join(model_dir, "legacy"), model_dir):
        for p in sorted(glob.glob(os.path.join(d, "experiments_*.json"))):
            R.update(json.load(open(p, encoding="utf-8")))
        for p in sorted(glob.glob(os.path.join(d, "fingerprints_*.json"))):
            P.update(json.load(open(p, encoding="utf-8")))
    return R, P


def f1(runs, guard, key="macro_f1"):
    return [r[guard][key] for r in runs if guard in r]


def rej(runs, guard):
    return [r[guard]["recall"]["reject"] for r in runs if guard in r]


def asr(runs, guard, key):
    return mean([r[guard]["asr"][key] for r in runs if guard in r])


def fmt(v, d=2):
    return "–" if v != v else ("%.*f" % (d, v))


def pm(v):
    return "%s ± %s" % (fmt(mean(v)), fmt(std(v))) if v else "–"


# ---------------------------------------------------------------- traces
def agentpoison_behaviour(model):
    path = os.path.join(ROOT, "traces", "agentpoison__%s.jsonl" % slug(model))
    if not os.path.exists(path):
        return None, []
    trs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = []
    for agent, label in (("ehr", "EHRAgent (memory backdoor, target DeleteDB)"),
                         ("qa", "StrategyQA (knowledge-base backdoor, target Finish(\"I don't know\"))")):
        ts = [t for t in trs if t["group"].startswith("agentpoison|%s|" % agent)]
        att = [t for t in ts if t["attack"]]
        ev = [e for t in ts for e in t["events"]]
        rows.append(dict(
            agent=label, traces=len(ts), attack=len(att), benign=len(ts) - len(att),
            followed=sum(any(e["label"] == "reject" and not e["synthetic"] for e in t["events"]) for t in att),
            approve=sum(e["label"] == "approve" for e in ev), reject=sum(e["label"] == "reject" for e in ev),
            synthetic=sum(e["synthetic"] for e in ev), steps=len(ev) / max(1, len(ts)),
            invalid=sum(t.get("n_invalid", 0) for t in ts) / max(1, sum(t.get("n_invalid", 0) for t in ts) + len(ev))))
    return trs, rows


def leakage_rows(trs):
    from src.leakage import audit
    return [audit(trs, s) for s in range(3)] if trs else []


# ---------------------------------------------------------------- SVG charts
TIP = 'data-tip="%s"'


def grouped_bars(R, benches):
    """Macro-F1 per benchmark, one bar per guard, whisker = ±1 std over seeds."""
    W, H, L, B, T = 760, 300, 44, 46, 16
    gw = (W - L - 8) / max(1, len(benches))
    bw = min(26, (gw - 28) / len(GUARDS))
    y = lambda v: T + (H - T - B) * (1 - v)
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="Macro-F1 by benchmark and guard">' % (W, H)]
    for t in (0, .25, .5, .75, 1):
        out.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (L, W - 8, y(t), y(t)))
        out.append('<text class="tick" x="%d" y="%.1f" text-anchor="end" dy="0.32em">%.2f</text>' % (L - 6, y(t), t))
    for gi, b in enumerate(benches):
        runs = R.get("inbench/" + b, [])
        x0 = L + gi * gw + (gw - bw * len(GUARDS) - 3 * (len(GUARDS) - 1)) / 2
        for k, (g, name) in enumerate(GUARDS):
            v = f1(runs, g)
            if not v:
                continue
            m, s = mean(v), std(v)
            x = x0 + k * (bw + 3)
            tip = "%s · %s: macro-F1 %.2f ± %.2f, reject recall %s" % (BENCH_NAME[b], name, m, s, pm(rej(runs, g)))
            out.append('<path class="s%d" d="M%.1f,%.1f V%.1f Q%.1f,%.1f %.1f,%.1f H%.1f Q%.1f,%.1f %.1f,%.1f V%.1f Z" %s/>'
                       % (k + 1, x, y(0), y(m) + 4, x, y(m), x + 4, y(m), x + bw - 4, x + bw, y(m), x + bw, y(m) + 4, y(0),
                          TIP % esc(tip)))
            out.append('<line class="whisker" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>'
                       % (x + bw / 2, x + bw / 2, y(min(1, m + s)), y(max(0, m - s))))
            if g == "gnn":
                out.append('<text class="val" x="%.1f" y="%.1f" text-anchor="middle">%.2f</text>' % (x + bw / 2, y(min(1, m + s)) - 5, m))
        out.append('<text class="cat" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (L + gi * gw + gw / 2, H - B + 20, BENCH_NAME[b]))
    out.append('<line class="axis" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (L, W - 8, y(0), y(0)))
    out.append("</svg>")
    return "".join(out)


def dumbbells(R, benches):
    """Attack success rate without a guard and with each learned guard."""
    W, rowh, L, T = 760, 54, 120, 26
    H = T + rowh * len(benches) + 26
    x = lambda v: L + (W - L - 24) * v
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="Attack success rate with and without a guard">' % (W, H)]
    for t in (0, .25, .5, .75, 1):
        out.append('<line class="grid" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (x(t), x(t), T - 8, H - 26))
        out.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%d%%</text>' % (x(t), H - 8, t * 100))
    for i, b in enumerate(benches):
        runs = R.get("inbench/" + b, [])
        if not runs:
            continue
        cy = T + i * rowh + rowh / 2 - 6
        u = asr(runs, "gnn", "asr_undefended")
        pts = [(asr(runs, g, "asr_defended"), k + 1, name) for k, (g, name) in enumerate(GUARDS[:3])]
        lo = min([p[0] for p in pts] + [u])
        out.append('<text class="cat" x="%d" y="%.1f" text-anchor="end" dy="0.32em">%s</text>' % (L - 14, cy, BENCH_NAME[b]))
        out.append('<line class="rail" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>' % (x(lo), x(u), cy, cy))
        out.append('<circle class="nog" cx="%.1f" cy="%.1f" r="6" %s/>' % (x(u), cy, TIP % esc("%s · no guard: ASR %.0f%%" % (BENCH_NAME[b], u * 100))))
        for v, k, name in sorted(pts, reverse=True):
            out.append('<circle class="s%d ring" cx="%.1f" cy="%.1f" r="6" %s/>'
                       % (k, x(v), cy, TIP % esc("%s · %s guard: ASR %.0f%% (no guard %.0f%%)" % (BENCH_NAME[b], name, v * 100, u * 100))))
        g = asr(runs, "gnn", "asr_defended")
        out.append('<text class="val" x="%.1f" y="%.1f" text-anchor="middle">%.0f%%</text>' % (x(g), cy - 12, g * 100))
        out.append('<text class="val muted" x="%.1f" y="%.1f" text-anchor="middle">%.0f%%</text>' % (x(u), cy + 20, u * 100))
    out.append("</svg>")
    return "".join(out)


def hbars(settings):
    """Reject recall per setting, one bar per guard (horizontal)."""
    W, L, rowh, T = 760, 230, 58, 8
    H = T + rowh * len(settings) + 26
    x = lambda v: L + (W - L - 40) * v
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="Reject recall by setting">' % (W, H)]
    for t in (0, .25, .5, .75, 1):
        out.append('<line class="grid" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (x(t), x(t), T, H - 24))
        out.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%.2f</text>' % (x(t), H - 8, t))
    for i, (name, runs) in enumerate(settings):
        y0 = T + i * rowh + 8
        out.append('<text class="cat" x="%d" y="%.1f" text-anchor="end" dy="0.32em">%s</text>' % (L - 12, y0 + 19, esc(name)))
        for k, (g, gname) in enumerate(GUARDS[:3]):
            v = mean(rej(runs, g))
            yy = y0 + k * 13
            w = max(0.0, x(v) - x(0))
            tip = "%s · %s: reject recall %.2f, macro-F1 %.2f" % (name, gname, v, mean(f1(runs, g)))
            if w < 1:
                out.append('<circle class="s%d" cx="%.1f" cy="%.1f" r="3" %s/>' % (k + 1, x(0) + 3, yy + 5, TIP % esc(tip)))
            else:
                out.append('<path class="s%d" d="M%.1f,%.1f H%.1f Q%.1f,%.1f %.1f,%.1f V%.1f Q%.1f,%.1f %.1f,%.1f H%.1f Z" %s/>'
                           % (k + 1, x(0), yy, x(v) - 3, x(v), yy, x(v), yy + 3, yy + 7, x(v), yy + 10, x(v) - 3, yy + 10, x(0), TIP % esc(tip)))
            out.append('<text class="val" x="%.1f" y="%.1f" dy="0.32em">%.2f</text>' % (x(v) + 8 if w >= 1 else x(0) + 12, yy + 5, v))
    out.append('<line class="axis" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (x(0), x(0), T, H - 24))
    out.append("</svg>")
    return "".join(out)


SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
       "#184f95", "#104281", "#0d366b"]


def seq_color(v):
    i = max(0, min(len(SEQ) - 1, int(round(v * (len(SEQ) - 1)))))
    return SEQ[i], ("#ffffff" if i >= 6 else "#0b0b0b")


def div_color(d, span):
    """Blue (gain) <-> red (loss), grey midpoint."""
    blue = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf"]
    red = ["#fbd5d4", "#f5a9a8", "#ec7b7a", "#e34948", "#b93636"]
    if abs(d) < 0.005:
        return "var(--mid)", "var(--ink)"
    k = min(4, int(abs(d) / max(span, 1e-9) * 5))
    return (blue if d > 0 else red)[k], ("#ffffff" if k >= 3 else "#0b0b0b")


def heatmap(rows, cols, cell, caption):
    """cell(r, c) -> (fill, ink, label, tip) or None."""
    cw, ch, L, T = 112, 34, 210, 30
    W, H = L + cw * len(cols) + 4, T + ch * len(rows) + 4
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="%s" style="max-width:%dpx">' % (W, H, esc(caption), W)]
    for j, c in enumerate(cols):
        out.append('<text class="cat" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (L + j * cw + cw / 2, T - 10, esc(c)))
    for i, r in enumerate(rows):
        out.append('<text class="cat" x="%d" y="%.1f" text-anchor="end" dy="0.32em">%s</text>' % (L - 10, T + i * ch + ch / 2, esc(r)))
        for j, c in enumerate(cols):
            v = cell(r, c)
            x, yy = L + j * cw, T + i * ch
            if v is None:
                out.append('<rect class="empty" x="%d" y="%d" width="%d" height="%d" rx="3"/>' % (x + 1, yy + 1, cw - 2, ch - 2))
                continue
            fill, ink, label, tip = v
            out.append('<rect x="%d" y="%d" width="%d" height="%d" rx="3" style="fill:%s" %s/>'
                       % (x + 1, yy + 1, cw - 2, ch - 2, fill, TIP % esc(tip)))
            out.append('<text x="%.1f" y="%.1f" text-anchor="middle" dy="0.32em" class="cellv" style="fill:%s">%s</text>'
                       % (x + cw / 2, yy + ch / 2, ink, esc(label)))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- tables
def table(header, rows, cls=""):
    h = "".join("<th>%s</th>" % esc(c) for c in header)
    b = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % c for c in r) for r in rows)
    return '<div class="tbl"><table class="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (cls, h, b)


def git_info():
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


# ---------------------------------------------------------------- page
CSS = """
/* Layout: one reading column of sections; each chart sits beside or above its table. */
:root{
  --page:#f7f8f6; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#7a7873; --grid:#e1e0d9; --axis:#c3c2b7;
  --mid:#e8e7e3; --ring:#fcfcfb; --accent:#1c5cab; --good:#006300; --bad:#b93636;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --nog:#898781;
  --f-display:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;
  --f-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;
  --page:#0f100f; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#9a988f; --grid:#2c2c2a; --axis:#383835;
  --mid:#383835; --ring:#1a1a19; --accent:#86b6ef; --good:#0ca30c; --bad:#e66767;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --nog:#898781;}}
:root[data-theme="dark"]{color-scheme:dark;
  --page:#0f100f; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#9a988f; --grid:#2c2c2a; --axis:#383835;
  --mid:#383835; --ring:#1a1a19; --accent:#86b6ef; --good:#0ca30c; --bad:#e66767;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --nog:#898781;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 var(--f-body);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:980px;margin:0 auto;padding-block:40px 64px;display:grid;gap:44px}
header{display:grid;gap:10px}
.eyebrow{font:500 12px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
h1{font:600 clamp(30px,5vw,44px)/1.05 var(--f-display);margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font:600 24px/1.15 var(--f-display);margin:0;text-wrap:balance}
h3{font:600 15px/1.3 var(--f-body);margin:0}
p{margin:0;max-width:68ch;color:var(--ink2)}
section{display:grid;gap:16px}
.lede{font-size:17px;color:var(--ink2);max-width:70ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--grid);border-radius:6px;padding:14px 16px;display:grid;gap:4px}
.tile .k{font:500 12px/1.2 var(--f-mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.tile .v{font:600 30px/1.1 var(--f-display)}
.tile .s{font-size:13px;color:var(--ink2)}
.panel{background:var(--surface);border:1px solid var(--grid);border-radius:6px;padding:16px;display:grid;gap:12px}
.panel svg{width:100%;height:auto;display:block}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
th{font:500 12px/1.3 var(--f-mono);letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
td{font-variant-numeric:tabular-nums}
td.num,th.num{text-align:right}
code,.mono{font-family:var(--f-mono);font-size:.92em}
.pill{display:inline-block;font:500 11.5px/1 var(--f-mono);padding:4px 7px;border-radius:999px;border:1px solid var(--grid);white-space:nowrap}
.pill.ok{color:var(--good);border-color:currentColor}
.pill.bad{color:var(--bad);border-color:currentColor}
.pill.new{color:var(--accent);border-color:currentColor}
ul.notes{margin:0;padding-left:20px;color:var(--ink2);display:grid;gap:6px;max-width:72ch}
details summary{cursor:pointer;color:var(--ink2);font-size:14px}
details summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
svg text{font-family:var(--f-body)}
svg .tick{font:11px var(--f-mono);fill:var(--muted)}
svg .cat{font-size:12.5px;fill:var(--ink2)}
svg .val{font:600 11.5px var(--f-mono);fill:var(--ink)}
svg .val.muted{fill:var(--muted);font-weight:500}
svg .cellv{font:500 12px var(--f-mono)}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--axis);stroke-width:1}
svg .whisker{stroke:var(--ink2);stroke-width:1.25}
svg .rail{stroke:var(--axis);stroke-width:2}
svg .empty{fill:none;stroke:var(--grid)}
svg .s1{fill:var(--s1)} svg .s2{fill:var(--s2)} svg .s3{fill:var(--s3)} svg .s4{fill:var(--s4)}
svg .nog{fill:var(--surface);stroke:var(--nog);stroke-width:2}
svg .ring{stroke:var(--ring);stroke-width:2}
svg [data-tip]{cursor:default}
svg [data-tip]:hover{opacity:.82}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--page);font:12.5px/1.4 var(--f-body);
  padding:6px 9px;border-radius:4px;max-width:300px;z-index:20}
footer{font-size:13px;color:var(--muted)}
@media (max-width:560px){main{padding-block:24px 48px;gap:36px}.tile .v{font-size:26px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
(function(){var t=document.getElementById('tip');
function show(e){var s=e.target.closest('[data-tip]');if(!s){t.hidden=true;return;}
t.textContent=s.getAttribute('data-tip');t.hidden=false;var x=e.clientX+14,y=e.clientY+14;
var r=t.getBoundingClientRect();if(x+r.width>innerWidth-8)x=e.clientX-r.width-14;if(y+r.height>innerHeight-8)y=e.clientY-r.height-14;
t.style.left=x+'px';t.style.top=y+'px';}
document.addEventListener('mousemove',show);document.addEventListener('mouseleave',function(){t.hidden=true;});})();
"""


def legend(items):
    return '<div class="legend">%s</div>' % "".join('<span><i style="background:var(--%s)"></i>%s</span>' % (c, esc(n)) for c, n in items)


def build(model, outdir):
    mdir = os.path.join(outdir, slug(model))
    R, P = load_results(mdir)
    benches = [b for b in BENCHES if "inbench/" + b in R]
    trs, beh = agentpoison_behaviour(model)
    leak = leakage_rows(trs)
    models = sorted(d for d in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, d)) and d not in ("llm", "legacy"))
    seeds = len(next(iter(R.values()), [])) if R else 0
    parts = []

    # header + tiles
    ap = R.get("inbench/agentpoison", [])
    tiles = []
    if ap:
        tiles.append(("AgentPoison · GNN macro-F1", fmt(mean(f1(ap, "gnn"))), "tool lookup %s, MLP %s" % (fmt(mean(f1(ap, "tool-lookup"))), fmt(mean(f1(ap, "mlp"))))))
        tiles.append(("AgentPoison · reject recall", fmt(mean(rej(ap, "gnn"))), "share of attacker actions the GNN blocks"))
        tiles.append(("AgentPoison · attack success", "%.0f%% → %.0f%%" % (asr(ap, "gnn", "asr_undefended") * 100, asr(ap, "gnn", "asr_defended") * 100),
                      "no guard → GNN guard, held-out tasks and triggers"))
        tiles.append(("AgentPoison · cost", "%.0f%%" % (asr(ap, "gnn", "approve_fpr") * 100), "of legitimate actions rejected by the GNN"))
        held = [R[k] for k in ("transfer/agentpoison:ehr->agentpoison:qa", "transfer/agentpoison:qa->agentpoison:ehr") if k in R]
        if held:
            tiles.append(("Unseen target action · reject recall", fmt(mean([x for r in held for x in rej(r, "gnn")])),
                          "train on one AgentPoison agent, test on the other"))
    n_status = {}
    for v in P.values():
        n_status[v["status"]] = n_status.get(v["status"], 0) + 1
    parts.append("""<header><div class="eyebrow">gnn_trace · trace LLM %s · %d seeds · commit %s</div>
<h1>Per-action guard results with AgentPoison</h1>
<p class="lede">A graph neural network reads each agent trace as a causal graph and labels every tool call as the user's
(<em>approve</em>) or the attacker's (<em>reject</em>). AgentPoison adds backdoored retrieval: a trigger in the question
pulls poisoned examples from EHRAgent's memory or a poisoned passage from StrategyQA's knowledge base. Scores are on
held-out task families and held-out triggers, with no overlap between training and test (audit below). The guards are
near perfect on AgentPoison because its target action never changes, and they fail completely once that action is
held out too.</p></header>""" % (esc(model), seeds, esc(git_info())))
    if tiles:
        parts.append('<div class="tiles">%s</div>' % "".join(
            '<div class="tile"><span class="k">%s</span><span class="v">%s</span><span class="s">%s</span></div>' % (esc(k), esc(v), esc(s))
            for k, v, s in tiles))

    # behaviour
    if beh:
        rows = [[esc(r["agent"]), r["traces"], "%d / %d" % (r["attack"], r["benign"]),
                 "%.0f%% (%d of %d)" % (100 * r["followed"] / max(1, r["attack"]), r["followed"], r["attack"]),
                 "%d / %d" % (r["approve"], r["reject"]), "%d of %d" % (r["synthetic"], r["reject"]), "%.1f" % r["steps"],
                 "%.1f%%" % (100 * r["invalid"])] for r in beh]
        parts.append("<section><h2>How %s behaves under AgentPoison</h2><p>Traces collected in the mock sandbox. "
                     "<em>Backdoor followed</em> counts attack traces where the LLM itself took the target action. When it "
                     "did not, a synthetic attacker step is appended so every attack trace has a reject example.</p>%s</section>"
                     % (esc(model), table(["agent", "traces", "attack / benign", "backdoor followed", "approve / reject actions",
                                           "synthetic rejects", "steps per trace", "invalid replies"], rows)))

    # in-benchmark
    if benches:
        leg = legend([("s%d" % (k + 1), n) for k, (_, n) in enumerate(GUARDS)])
        rows = [[BENCH_NAME[b]] + [pm(f1(R["inbench/" + b], g)) + "<br><span class='mono' style='color:var(--muted)'>rej. recall %s</span>"
                                   % pm(rej(R["inbench/" + b], g)) for g, _ in GUARDS] for b in benches]
        parts.append("""<section><h2>Held-out tasks, by benchmark</h2><p>Macro-F1 over approve/reject, mean over seeds; the
whisker is ±1 standard deviation. Training never shares a task family or an attack with the test set.</p>
<div class="panel">%s%s</div><details><summary>Table: macro-F1 and reject recall</summary>%s</details></section>"""
                     % (leg, grouped_bars(R, benches), table(["benchmark"] + [n for _, n in GUARDS], rows)))

        leg2 = '<div class="legend"><span><i style="background:var(--surface);border:2px solid var(--nog)"></i>no guard</span>%s</div>' % "".join(
            '<span><i style="background:var(--s%d)"></i>%s guard</span>' % (k + 1, esc(n)) for k, (_, n) in enumerate(GUARDS[:3]))
        rows = [[BENCH_NAME[b], "%.0f%%" % (100 * asr(R["inbench/" + b], "gnn", "asr_undefended"))]
                + ["%.0f%%" % (100 * asr(R["inbench/" + b], g, "asr_defended")) for g, _ in GUARDS[:3]]
                + ["%.0f%%" % (100 * asr(R["inbench/" + b], "gnn", "asr_forced")),
                   "%.0f%% / %.0f%%" % (100 * asr(R["inbench/" + b], "gnn", "benign_block"), 100 * asr(R["inbench/" + b], "gnn", "approve_fpr"))]
                for b in benches]
        parts.append("""<section><h2>Attack success rate</h2><p>Share of attack traces in which the LLM took an attacker
action that the guard let through. Read it together with the cost columns: rejecting everything also gives 0%%.</p>
<div class="panel">%s%s</div><details><summary>Table: ASR and cost</summary>%s</details></section>"""
                     % (leg2, dumbbells(R, benches), table(["benchmark", "no guard"] + ["%s guard" % n for _, n in GUARDS[:3]]
                                                            + ["GNN, attack always attempted", "GNN: benign traces blocked / approve actions rejected"], rows)))

    # transfer
    tkeys = [k for k in R if k.startswith("transfer/") and ":" not in k]
    if tkeys:
        srcs, tgts = [], []
        for k in tkeys:
            s, t = k[len("transfer/"):].split("->")
            srcs.append(s) if s not in srcs else None
            tgts.append(t) if t not in tgts else None
        srcs.sort(key=lambda s: (s.count("+"), [BENCHES.index(x) for x in s.split("+")]))
        tgts.sort(key=BENCHES.index)
        name = lambda s: " + ".join(BENCH_NAME[x] for x in s.split("+"))

        def tcell(s, t):
            runs = R.get("transfer/%s->%s" % (s, t))
            if not runs:
                return None
            m = mean(f1(runs, "gnn"))
            fill, ink = seq_color(m)
            return fill, ink, "%.2f" % m, "%s → %s: GNN macro-F1 %.2f ± %.2f, tool lookup %.2f" % (
                name(s), BENCH_NAME[t], m, std(f1(runs, "gnn")), mean(f1(runs, "tool-lookup")))
        rows = [[esc(name(s)), BENCH_NAME[t], pm(f1(R[k], "gnn")), pm(f1(R[k], "tool-lookup")), pm(f1(R[k], "mlp")),
                 "%.0f%% → %.0f%%" % (100 * asr(R[k], "gnn", "asr_undefended"), 100 * asr(R[k], "gnn", "asr_defended"))]
                for s in srcs for t in tgts for k in ["transfer/%s->%s" % (s, t)] if k in R]
        parts.append("""<section><h2>Train on other benchmarks, test on a new one</h2><p>GNN macro-F1 when the test
benchmark is never seen in training. Darker is better; empty cells are pairs that are not run.</p>
<div class="panel">%s</div><details><summary>Table: all guards</summary>%s</details></section>"""
                     % (heatmap([name(s) for s in srcs], [BENCH_NAME[t] for t in tgts],
                                lambda r, c: tcell(srcs[[name(s) for s in srcs].index(r)], tgts[[BENCH_NAME[t] for t in tgts].index(c)]),
                                "Cross-benchmark transfer"),
                        table(["train", "test", "GNN", "tool lookup", "MLP", "GNN ASR"], rows)))

    # held-out target action
    hk = [("transfer/agentpoison:ehr->agentpoison:qa", "EHRAgent → StrategyQA"), ("transfer/agentpoison:qa->agentpoison:ehr", "StrategyQA → EHRAgent")]
    if ap and all(k in R for k, _ in hk):
        settings = [("Held-out triggers (same agent)", ap)] + [(n, R[k]) for k, n in hk]
        parts.append("""<section><h2>Does the guard catch an unseen backdoor target?</h2><p>AgentPoison has a single target
action per agent (<code>DeleteDB</code>, <code>Finish("I don't know")</code>), shared by every trigger. With held-out
triggers the target action is still one the guard saw in training, and every guard recognises it from the action alone:
removing memory, observations or the injected text changes nothing (see the ablations). Training on one agent and
testing on the other also holds out the target action. Then no guard rejects anything.</p>
<div class="panel">%s%s</div>%s</section>"""
                     % (legend([("s%d" % (k + 1), n) for k, (_, n) in enumerate(GUARDS[:3])]), hbars(settings),
                        table(["setting"] + ["%s macro-F1 / reject recall" % n for _, n in GUARDS[:3]] + ["GNN ASR"],
                              [[esc(n)] + ["%s / %s" % (fmt(mean(f1(r, g))), fmt(mean(rej(r, g)))) for g, _ in GUARDS[:3]]
                               + ["%.0f%% → %.0f%%" % (100 * asr(r, "gnn", "asr_undefended"), 100 * asr(r, "gnn", "asr_defended"))]
                               for n, r in settings])))

    # ablations
    if any(k.startswith("ablate/") for k in R):
        ab_b = [b for b in benches if "ablate/none/" + b in R]
        base = {b: mean(f1(R["ablate/none/" + b], "gnn")) for b in ab_b}
        deltas = [abs(mean(f1(R["ablate/%s/%s" % (a, b)], "gnn")) - base[b]) for a in ABLATIONS for b in ab_b if "ablate/%s/%s" % (a, b) in R]
        span = max(deltas + [0.05])

        def acell(r, c):
            a = [x for x in ABLATIONS if ABL_NAME[x] == r][0]
            b = [x for x in ab_b if BENCH_NAME[x] == c][0]
            runs = R.get("ablate/%s/%s" % (a, b))
            if not runs:
                return None
            v = mean(f1(runs, "gnn"))
            d = v - base[b]
            fill, ink = div_color(d, span)
            return fill, ink, "%+.2f" % d, "%s without %s: macro-F1 %.2f (full graph %.2f, change %+.2f)" % (BENCH_NAME[b], ABL_NAME[a], v, base[b], d)
        rows = [["full graph"] + [pm(f1(R["ablate/none/" + b], "gnn")) for b in ab_b]] + [
            ["without " + ABL_NAME[a]] + [pm(f1(R.get("ablate/%s/%s" % (a, b), []), "gnn")) for b in ab_b] for a in ABLATIONS]
        parts.append("""<section><h2>What each graph component contributes</h2><p>Change in GNN macro-F1 when one component is
removed (held-out tasks). Red means the model got worse without it; blue means it did better.</p>
<div class="panel">%s</div><details><summary>Table: absolute macro-F1</summary>%s</details></section>"""
                     % (heatmap([ABL_NAME[a] for a in ABLATIONS], [BENCH_NAME[b] for b in ab_b], acell, "Ablations"),
                        table(["graph"] + [BENCH_NAME[b] for b in ab_b], rows)))

    # headline across LLMs
    hrows = []
    for m in models:
        Rm, _ = load_results(os.path.join(outdir, m))
        if not any(k.startswith("inbench/") for k in Rm):
            continue
        hrows.append([esc(":".join(m.rsplit("_", 1)))] + [
            ("%s <span style='color:var(--muted)'>(%s)</span>" % (pm(f1(Rm["inbench/" + b], "gnn")), fmt(mean(f1(Rm["inbench/" + b], "tool-lookup")))))
            if "inbench/" + b in Rm else "<span style='color:var(--muted)'>no traces</span>" for b in BENCHES])
    if hrows:
        parts.append("<section><h2>GNN across trace LLMs</h2><p>Held-out-task macro-F1 of the GNN, tool-lookup baseline in "
                     "brackets. Each row trains and tests on traces from one LLM. AgentPoison traces exist only for %s.</p>%s</section>"
                     % (esc(model), table(["trace LLM"] + [BENCH_NAME[b] for b in BENCHES], hrows)))

    # leakage
    if leak:
        keys = list(leak[0]["overlaps"])
        nice = {"task_groups": "task families", "attack_ids": "attack ids (agent|trigger)", "trigger_tokens": "trigger words",
                "poison_texts": "poisoned passages / rationale", "questions": "user questions",
                "test_question_in_train_memory": "test questions in training memory"}
        rows = [[nice.get(k, k)] + [('<span class="pill ok">0</span>' if r["overlaps"][k] == 0 else
                                     '<span class="pill bad">%d</span>' % r["overlaps"][k]) for r in leak] for k in keys]
        rows.append(["train / test traces (attack traces)"] + ["%d / %d (%d / %d)" % (r["train"], r["test"], r["train_attacks"], r["test_attacks"]) for r in leak])
        ok = all(not any(r["overlaps"].values()) for r in leak)
        parts.append("""<section><h2>Leakage audit on the collected AgentPoison traces</h2><p>%s Each row counts items shared by
the training and test side of the held-out-task split used above, for each seed. The same checks run in
<code>pytest</code>, along with: triggers share no words with each other or with the data, benign and attacked tasks have
the same structure, and the graph ignores label-bearing trace fields.</p>%s</section>"""
                     % ("Nothing is shared between training and test." if ok else "Overlaps found, see the table.",
                        table(["shared between train and test"] + ["seed %d" % r["seed"] for r in leak], rows)))

    # provenance
    if P:
        origin = lambda v: v.get("origin") or ("trained" if v["status"] == "computed" else "legacy")
        rows = []
        for k in sorted(P, key=lambda k: (origin(P[k]) != "trained", k)):
            v = P[k]
            pill = ('<span class="pill new">trained %s</span>' % esc((v.get("time") or "")[:16].replace("T", " "))
                    if origin(v) == "trained" else '<span class="pill ok">earlier results, inputs unchanged</span>')
            rows.append(["<code>%s</code>" % esc(k), pill, ("%.0f s" % v["seconds"]) if v.get("seconds") else "–",
                         "<code>%s</code>" % esc(v["fp"][:12])])
        n_tr = sum(origin(v) == "trained" for v in P.values())
        parts.append("""<section><h2>Which results were trained for this report</h2><p>%d of %d result cells for %s were
trained for AgentPoison. The other %d come from the earlier runs. Their inputs are unchanged: the fingerprint over the
training code, seeds, epochs and every train and test graph matches, and graph building was checked to give identical
tensors for the old traces before and after the AgentPoison changes. So they were not retrained. The other trace LLMs
retrained nothing.</p><details><summary>All %d cells</summary>%s</details></section>"""
                     % (n_tr, len(P), esc(model), len(P) - n_tr, len(P),
                        table(["cell", "origin", "train time", "fingerprint"], rows)))

    parts.append("""<section><h2>Reading these numbers</h2><ul class="notes">
<li>EHRAgent's target action, <code>DeleteDB</code>, is an attacker-only tool, so the tool-lookup baseline does well on
that half. StrategyQA's target is <code>Finish("I don't know")</code>, the same tool the benign task ends with.</li>
<li>The AgentPoison in-benchmark scores measure recognising a known target action. The held-out-target rows are the
ones to quote for an unseen backdoor.</li>
<li>The 80 triggers are fixed stand-ins (<em>Verb the adjective noun.</em>). AgentPoison optimises its triggers against a
specific retriever and publishes none. Retrieval uses token overlap in place of DPR.</li>
<li>When the LLM does not attack by itself, a synthetic attacker step is appended at the end of the trace, so attack
traces always end with a reject. The real-only scores in <code>results/REPORT.md</code> leave those steps out.</li>
<li>Numbers are means over %d seeds. With a few hundred traces per benchmark, differences under about 0.05 are within noise.</li>
</ul></section>""" % seeds)
    parts.append('<footer>Generated %s by <code>scripts/report_page.py</code> from <code>%s</code> and <code>traces/</code>.</footer>'
                 % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), esc(os.path.relpath(mdir, ROOT))))

    return """<title>gnn_trace AgentPoison Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>%s</style>
<main>%s</main><div id="tip" hidden></div>
<script>%s</script>
""" % (CSS, "\n".join(parts), JS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:1.5b", help="trace LLM whose sections are charted")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=None, help="default: <results>/report.html")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "report.html")
    page = build(a.model, a.results)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    print("wrote", out)


if __name__ == "__main__":
    main()
