"""Interactive trace viewer (Dash) with an optional Cloudflare quick tunnel.

    python scripts/trace_viewer.py                      # serve on :8050 and open a trycloudflare.com tunnel
    python scripts/trace_viewer.py --no-tunnel          # local only
    python scripts/trace_viewer.py --port 9000 data/tracegen_qwen14b

Reads every data/*/traces.jsonl (or the run directories given on the command line). Pick a run, filter the
traces, click a row to see the conversation, the tool calls, the injected text and which calls match the
attacker's ground-truth calls. The tunnel needs `cloudflared` on PATH (or --cloudflared PATH); its public URL is
printed on start. Anyone with that URL can see the traces, so stop the script when you are done.
"""
import argparse
import atexit
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = {}


# ---------------------------------------------------------------- data
def find_runs(dirs):
    paths = dirs or sorted(glob.glob(os.path.join(ROOT, "data", "*")))
    return {os.path.basename(os.path.normpath(p)): os.path.join(p, "traces.jsonl")
            for p in paths if os.path.exists(os.path.join(p, "traces.jsonl"))}


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def is_attacker_call(t, c):
    """A call is flagged when its function is one of the injection task's ground-truth calls."""
    gt = t.get("injection_ground_truth_calls") or []
    return any(g["function"] == c["name"] for g in gt)


def injected_in(t, text):
    n = norm(text)
    return any(norm(v) and norm(v) in n for v in (t.get("injections") or {}).values())


def load(path):
    if path not in CACHE:
        rows = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                t = json.loads(line)
                calls = t.get("tool_calls") or []
                t["_idx"] = i
                t["_n_calls"] = len(calls)
                t["_n_attacker"] = sum(is_attacker_call(t, c) for c in calls) if t["mode"] == "attack" else 0
                t["_inj_seen"] = any(injected_in(t, c.get("result")) for c in calls)
                rows.append(t)
        CACHE[path] = rows
    return CACHE[path]


# ---------------------------------------------------------------- views
def pill(text, color):
    return html.Span(text, style={"background": color, "color": "white", "borderRadius": "10px",
                                  "padding": "1px 8px", "fontSize": "12px", "marginRight": "6px"})


def box(children, border, bg):
    return html.Div(children, style={"borderLeft": "4px solid " + border, "background": bg, "padding": "8px 12px",
                                     "margin": "8px 0", "borderRadius": "4px"})


PRE = {"whiteSpace": "pre-wrap", "wordBreak": "break-word", "margin": "4px 0", "fontFamily": "Consolas, monospace",
       "fontSize": "13px"}


def message_view(t):
    out = []
    for m in t["messages"]:
        role = m["role"]
        if role == "system":
            out.append(html.Details([html.Summary("system prompt"), html.Pre(m["content"], style=PRE)]))
        elif role == "user":
            out.append(box([pill("user", "#2563eb"), html.Pre(m["content"], style=PRE)], "#2563eb", "#eff6ff"))
        elif role == "assistant":
            kids = [pill("assistant", "#7c3aed")]
            if m.get("content"):
                kids.append(html.Pre(m["content"], style=PRE))
            for c in m.get("tool_calls") or []:
                bad = is_attacker_call(t, {"name": c["function"]}) and t["mode"] == "attack"
                kids.append(html.Div([pill("attacker call" if bad else "call", "#dc2626" if bad else "#475569"),
                                      html.Code("%s(%s)" % (c["function"], json.dumps(c["args"], ensure_ascii=False)),
                                                style={"fontSize": "13px"})]))
            out.append(box(kids, "#7c3aed", "#f5f3ff"))
        elif role == "tool":
            inj = injected_in(t, m["content"])
            name = (m.get("tool_call") or {}).get("function", "tool")
            kids = [pill("tool result: " + name, "#d97706" if inj else "#16a34a")]
            if inj:
                kids.append(pill("contains injected text", "#dc2626"))
            if m.get("error"):
                kids.append(pill("error", "#dc2626"))
            kids.append(html.Pre(m["content"], style=PRE))
            out.append(box(kids, "#d97706" if inj else "#16a34a", "#fef2f2" if inj else "#f0fdf4"))
    return out


def flow_figure(t):
    calls = t.get("tool_calls") or []
    fig = go.Figure()
    if not calls:
        fig.update_layout(title="no tool calls", height=160)
        return fig
    xs = list(range(1, len(calls) + 1))
    colors, labels = [], []
    for c in calls:
        bad = t["mode"] == "attack" and is_attacker_call(t, c)
        inj = injected_in(t, c.get("result"))
        colors.append("#dc2626" if bad else "#d97706" if inj else "#16a34a")
        labels.append("%s<br>%s" % (c["name"], json.dumps(c["args"], ensure_ascii=False)[:120]))
    fig.add_trace(go.Scatter(x=xs, y=[0] * len(xs), mode="lines+markers+text", text=[c["name"] for c in calls],
                             textposition="top center", hovertext=labels, hoverinfo="text",
                             marker=dict(size=18, color=colors, line=dict(width=1, color="white")),
                             line=dict(color="#94a3b8")))
    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_xaxes(title="tool call", dtick=1)
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=40), showlegend=False)
    return fig


def detail_view(t):
    meta = [("trace", t["trace_id"]), ("suite", t["suite"]), ("user task", t["task_id"]), ("variant", t["variant_id"]),
            ("agent", t["agent_model"]), ("mode", t["mode"]), ("injection task", t.get("injection_task_id") or "-"),
            ("attack", t.get("attack_name") or "-"), ("utility", t.get("utility")), ("security (attack succeeded)",
             t.get("security")), ("steps", t.get("n_steps")), ("tokens in/out", "%s / %s" % (t.get("tokens_in"),
             t.get("tokens_out"))), ("latency s", t.get("latency_s"))]
    return html.Div([
        html.H3(t["prompt"], style={"marginBottom": "4px"}),
        html.Div([pill(t["mode"], "#dc2626" if t["mode"] == "attack" else "#16a34a"),
                  pill("attack followed" if t.get("security") else "attack not followed", "#dc2626" if t.get("security")
                       else "#64748b") if t["mode"] == "attack" else "",
                  pill("utility ok" if t.get("utility") else "utility failed", "#16a34a" if t.get("utility") else "#64748b")]),
        html.Table([html.Tr([html.Td(k, style={"color": "#64748b", "paddingRight": "16px"}), html.Td(str(v))])
                    for k, v in meta], style={"fontSize": "13px", "margin": "8px 0"}),
        html.H4("Tool-call flow", style={"marginBottom": 0}),
        html.Div("green = benign call, orange = result carries injected text, red = call matches an attacker call",
                 style={"fontSize": "12px", "color": "#64748b"}),
        dcc.Graph(figure=flow_figure(t), config={"displayModeBar": False}),
        html.H4("Conversation"),
        *message_view(t),
        html.H4("Final answer"), html.Pre(t.get("final_answer") or "", style=PRE),
        html.Details([html.Summary("Injected text and attacker ground-truth calls"),
                      html.Pre(json.dumps({"injections": t.get("injections"),
                                           "injection_ground_truth_calls": t.get("injection_ground_truth_calls")},
                                          indent=2, ensure_ascii=False), style=PRE)]),
    ])


# ---------------------------------------------------------------- app
def make_app(runs):
    app = Dash(__name__, title="Trace viewer")
    first = next(iter(runs))
    lbl = {"fontSize": "12px", "color": "#475569", "marginTop": "10px"}
    side = html.Div([
        html.H2("Traces", style={"marginTop": 0}),
        html.Div("run", style=lbl), dcc.Dropdown(list(runs), first, id="run", clearable=False),
        html.Div("mode", style=lbl), dcc.Dropdown(["benign", "attack"], id="mode", multi=True),
        html.Div("suite", style=lbl), dcc.Dropdown(id="suite", multi=True),
        html.Div("injection task", style=lbl), dcc.Dropdown(id="inj", multi=True),
        html.Div("outcome", style=lbl),
        dcc.Checklist(id="flags", options=[
            {"label": " attack followed (security)", "value": "sec"},
            {"label": " injected text seen by agent", "value": "seen"},
            {"label": " utility failed", "value": "nouti"},
            {"label": " originals only", "value": "orig"}], value=[], labelStyle={"display": "block"}),
        html.Div("search prompt / tool / task", style=lbl), dcc.Input(id="q", type="text", debounce=True,
                                                                      style={"width": "100%"}),
        html.Div(id="count", style={"marginTop": "12px", "fontWeight": 600}),
        html.Div(id="summary", style={"fontSize": "12px", "color": "#475569", "marginTop": "6px"}),
    ], style={"width": "270px", "padding": "16px", "borderRight": "1px solid #e2e8f0", "overflowY": "auto",
              "flex": "none"})
    cols = [("suite", "suite"), ("task_id", "task"), ("mode", "mode"), ("injection_task_id", "injection"),
            ("security", "followed"), ("utility", "utility"), ("_n_calls", "calls"), ("_n_attacker", "attacker calls")]
    table = dash_table.DataTable(
        id="tbl", columns=[{"name": n, "id": i} for i, n in cols], page_size=14, row_selectable="single",
        sort_action="native", style_table={"overflowX": "auto"}, selected_rows=[],
        style_cell={"fontSize": "12px", "padding": "4px 8px", "textAlign": "left", "maxWidth": "180px",
                    "overflow": "hidden", "textOverflow": "ellipsis"},
        style_data_conditional=[{"if": {"filter_query": "{followed} = True"}, "backgroundColor": "#fee2e2"}])
    main = html.Div([table, html.Hr(), html.Div(id="detail", children="Select a trace above.")],
                    style={"flex": 1, "padding": "16px", "overflowY": "auto", "minWidth": 0})
    app.layout = html.Div([side, main, dcc.Store(id="ids")],
                          style={"display": "flex", "height": "100vh", "fontFamily": "system-ui, sans-serif"})

    @app.callback(Output("suite", "options"), Output("inj", "options"), Input("run", "value"))
    def options(run):
        rows = load(runs[run])
        return (sorted({r["suite"] for r in rows}),
                sorted({"%s/%s" % (r["suite"], r["injection_task_id"]) for r in rows if r.get("injection_task_id")}))

    @app.callback(Output("tbl", "data"), Output("tbl", "selected_rows"), Output("count", "children"),
                  Output("summary", "children"), Output("ids", "data"),
                  Input("run", "value"), Input("mode", "value"), Input("suite", "value"), Input("inj", "value"),
                  Input("flags", "value"), Input("q", "value"))
    def table_rows(run, mode, suite, inj, flags, q):
        rows = load(runs[run])
        f = []
        for r in rows:
            if mode and r["mode"] not in mode: continue
            if suite and r["suite"] not in suite: continue
            if inj and "%s/%s" % (r["suite"], r.get("injection_task_id")) not in inj: continue
            if "sec" in flags and not r.get("security"): continue
            if "seen" in flags and not r["_inj_seen"]: continue
            if "nouti" in flags and r.get("utility"): continue
            if "orig" in flags and not r.get("is_original"): continue
            if q:
                hay = " ".join([r["prompt"], r["task_id"], r["suite"]] + [c["name"] for c in r.get("tool_calls") or []]).lower()
                if q.lower() not in hay: continue
            f.append(r)
        data = [{**{k: r.get(k) for k in ("suite", "task_id", "mode", "injection_task_id", "utility", "_n_calls",
                                          "_n_attacker")},
                 "followed": r.get("security") if r["mode"] == "attack" else None} for r in f]
        att = [r for r in f if r["mode"] == "attack"]
        s = "%d attacked, %d followed (%.0f%%); %d benign; %d with injected text seen" % (
            len(att), sum(bool(r.get("security")) for r in att), 100 * sum(bool(r.get("security")) for r in att) / max(1, len(att)),
            len(f) - len(att), sum(r["_inj_seen"] for r in f))
        return data, [], "%d traces" % len(f), s, [r["_idx"] for r in f]

    @app.callback(Output("detail", "children"), Input("tbl", "selected_rows"), State("ids", "data"), State("run", "value"))
    def show(sel, ids, run):
        if not sel or not ids:
            return "Select a trace above."
        return detail_view(load(runs[run])[ids[sel[0]]])

    return app


# ---------------------------------------------------------------- tunnel
def start_tunnel(port, exe):
    exe = exe or shutil.which("cloudflared")
    if not exe:
        print("cloudflared not found on PATH; serving locally only (use --cloudflared PATH).")
        return None
    p = subprocess.Popen([exe, "tunnel", "--url", "http://127.0.0.1:%d" % port, "--no-autoupdate"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    atexit.register(lambda: p.poll() is None and p.terminate())

    def watch():
        found = False
        for line in p.stdout:
            m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
            if m and not found:
                found = True
                print("\n  Public URL: %s\n  (anyone with this link can view the traces)\n" % m.group(0), flush=True)
    threading.Thread(target=watch, daemon=True).start()
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="run directories containing traces.jsonl (default: data/*)")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-tunnel", action="store_true", help="serve locally only")
    ap.add_argument("--cloudflared", help="path to the cloudflared executable (default: from PATH)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (the tunnel connects via 127.0.0.1)")
    a = ap.parse_args()
    runs = find_runs(a.dirs)
    if not runs:
        sys.exit("no traces.jsonl found")
    print("runs:", ", ".join(runs))
    if not a.no_tunnel:
        start_tunnel(a.port, a.cloudflared)
    print("Local URL: http://127.0.0.1:%d" % a.port)
    make_app(runs).run(host=a.host, port=a.port, debug=False)


if __name__ == "__main__":
    main()
