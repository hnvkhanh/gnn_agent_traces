"""Interactive viewer for the causal trace graphs (src/graph/build.py + src/graph/files.py).

    python scripts/visualize_graph.py traces/agentdojo__gemma2_9b.jsonl
    python scripts/visualize_graph.py "traces/*.jsonl" --port 8765

Starts a local server (127.0.0.1 only, nothing is written to disk) and opens it in your browser. Pick any trace from the
list (filter: all / attacked / benign / has rejected actions, free-text search), then hover a node to highlight its edges,
click to pin it, and tick edge types on/off (the O(n^2) precedes/next edges are off by default). Ctrl+C stops the server.

The edges drawn are exactly the edge_index tensors of trace_to_graph_files(), so the picture is what the GNN sees.
"""
import argparse
import glob
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.files import file_edges, trace_to_graph_files  # noqa: E402
from src.sandbox.policy import LABELS  # noqa: E402


def load_traces(paths):
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tr = json.loads(line)
                    if tr.get("events"):
                        out.append(tr)
    return out


def short(s, k=160):
    s = " ".join(str(s).split())
    return s if len(s) <= k else s[: k - 1] + "…"


def summary(i, tr):
    return dict(i=i, id=str(tr.get("trace_id", i)), model=tr.get("model", ""), bench=tr.get("benchmark", ""),
                task=short(tr.get("task", ""), 60), attack=tr.get("attack") is not None,
                rejects=sum(e["label"] == "reject" for e in tr["events"]), n=len(tr["events"]))


def graph_json(tr):
    g = trace_to_graph_files(tr)
    keys, _, _ = file_edges(tr)
    ev = tr["events"]
    nodes = []

    def add(type_, idx, label, detail, **kw):
        nodes.append(dict(id="%s%d" % (type_, idx), type=type_, label=short(label, 28), detail=short(detail, 600), **kw))

    add("task", 0, "task", tr["user_task"])
    for i, m in enumerate(tr["memory"]):
        add("memory", i, "memory %d" % i, m)
    for i, t in enumerate(tr["tools"]):
        add("tool", i, t["name"], t["description"])
    for i, e in enumerate(ev):
        add("action", i, "%d. %s" % (i, e["tool"]), "%s(%s)" % (e["tool"], json.dumps(e["args"], sort_keys=True)),
            step=i, rejected=e["label"] == "reject", synthetic=bool(e["synthetic"]), labeltext=e["label"])
        add("observation", i, "obs %d" % i, e["observation"], step=i)
    for i, k in enumerate(keys):
        add("file", i, k, "file node: " + k)
    edges = []
    for (s, r, d), store in g.edge_items():
        for a, b in store.edge_index.t().tolist():
            edges.append(dict(s="%s%d" % (s, a), d="%s%d" % (d, b), rel=r))
    assert g["action"].y.tolist() == [LABELS.index(e["label"]) for e in ev]
    a = tr.get("attack")
    title = "%s | %s | %s | %s" % (tr.get("benchmark", ""), tr.get("model", ""), tr.get("trace_id", ""),
                                  ("attack: %s" % (a.get("id") or a.get("tool") or a.get("mode"))) if a else "benign")
    return dict(title=title, nodes=nodes, edges=edges)


PAGE = r"""<!doctype html><meta charset="utf-8"><title>Trace graph viewer</title>
<style>
:root{--bg:#fff;--fg:#1c1e21;--mut:#6b7280;--line:#d1d5db;--card:#f6f7f9;--task:#7c3aed;--memory:#0891b2;--tool:#a16207;
--approve:#16a34a;--reject:#dc2626;--obs:#475569;--file:#c2410c}
@media(prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e5e7eb;--mut:#9ca3af;--line:#3b4048;--card:#1d2026}}
body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;display:flex;height:100vh}
#side{width:300px;padding:12px;border-right:1px solid var(--line);overflow:auto;box-sizing:border-box;flex:none}
#main{flex:1;overflow:auto;position:relative}
h1{font-size:13px;margin:12px 0 8px;word-break:break-all}h2{font-size:11px;color:var(--mut);margin:14px 0 5px;text-transform:uppercase}
label{display:flex;align-items:center;gap:6px;margin:3px 0;cursor:pointer}.sw{width:18px;height:3px;display:inline-block}
#info{background:var(--card);padding:8px;border-radius:6px;white-space:pre-wrap;word-break:break-word;font-size:12px;min-height:60px;max-height:200px;overflow:auto}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
input[type=text],select{width:100%;box-sizing:border-box;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:4px;margin-bottom:4px}
#list{height:190px;overflow:auto;border:1px solid var(--line);border-radius:4px}
#list div{padding:3px 6px;cursor:pointer;border-bottom:1px solid var(--line);font-size:12px}#list div:hover{background:var(--card)}
#list div.sel{background:var(--task);color:#fff}.atk{color:var(--reject);font-weight:600}#list div.sel .atk{color:#fff}
svg text{font-size:11px;fill:var(--fg);pointer-events:none}
.n{cursor:pointer}.band{fill:var(--card)}.bt{fill:var(--mut)!important;font-size:10px!important;text-transform:uppercase}
</style>
<div id=side><h2 style="margin-top:0">Traces (<span id=cnt></span>)</h2>
<input type=text id=q placeholder="search id / model / task"><select id=flt><option value=all>all</option>
<option value=attack>attacked</option><option value=benign>benign</option><option value=rej>has rejected actions</option></select>
<div id=list></div><h1 id=title></h1><div id=info>hover a node</div>
<h2>Edge types</h2><div id=edges></div><h2>Nodes</h2><div id=legend></div></div><div id=main><svg id=svg></svg></div>
<script>
const COL={task:'--task',memory:'--memory',tool:'--tool',observation:'--obs',file:'--file'};
const REL_COL={proposes:'#7c3aed',recalled:'#0891b2',invoked_as:'#a16207',returns:'#64748b',precedes:'#94a3b8',
 mentions:'#2563eb',copies:'#db2777',next:'#cbd5e1',targets:'#c2410c',appears_in:'#fb923c'};
const REL_ON=new Set(['proposes','invoked_as','returns','mentions','copies','targets','appears_in']);
const relState={};Object.keys(REL_COL).forEach(r=>relState[r]=REL_ON.has(r));
const svg=document.getElementById('svg'),info=document.getElementById('info'),NS='http://www.w3.org/2000/svg';
const mk=(t,a,p)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);(p||svg).appendChild(e);return e;};
let edgeEls=[],nodeEls={},pinned=null;
function render(G){
 svg.innerHTML='';edgeEls=[];nodeEls={};pinned=null;
 document.getElementById('title').textContent=G.title;
 const by={};G.nodes.forEach(n=>by[n.type]=(by[n.type]||[]).concat(n));
 const nAct=(by.action||[]).length,SP=118,W=Math.max(900,nAct*SP+160),X0=80;
 const Y={ctx:50,action:190,observation:300,file:430},H=500,pos={};
 const spread=(list,y)=>list.forEach((n,i)=>pos[n.id]={x:X0+(i+.5)*(W-2*X0)/list.length,y});
 spread([].concat(by.task||[],by.memory||[],by.tool||[]),Y.ctx);
 (by.action||[]).forEach(n=>pos[n.id]={x:X0+n.step*SP,y:Y.action});
 (by.observation||[]).forEach(n=>pos[n.id]={x:X0+n.step*SP,y:Y.observation});
 spread(by.file||[],Y.file);
 svg.setAttribute('width',W);svg.setAttribute('height',H);
 [['context',Y.ctx],['actions',Y.action],['observations',Y.observation],['files',Y.file]].forEach(([t,y])=>{
  mk('rect',{x:0,y:y-32,width:W,height:64,class:'band',opacity:.45});mk('text',{x:6,y:y-36,class:'bt'}).textContent=t;});
 const eg=mk('g',{});
 G.edges.forEach(e=>{const a=pos[e.s],b=pos[e.d],dy=b.y-a.y,c=Math.abs(b.x-a.x)*.15+20,s=dy>0?c:-c;
  const p=mk('path',{d:`M${a.x},${a.y} C${a.x},${a.y+s} ${b.x},${b.y-s} ${b.x},${b.y}`,fill:'none',stroke:REL_COL[e.rel],'stroke-width':1.3,opacity:.55},eg);
  p._e=e;edgeEls.push(p);});
 G.nodes.forEach(n=>{
  const g=mk('g',{class:'n',transform:`translate(${pos[n.id].x},${pos[n.id].y})`});
  const col=n.type=='action'?(n.rejected?'var(--reject)':'var(--approve)'):`var(${COL[n.type]})`;
  const w=Math.min(104,Math.max(44,n.label.length*6.2+10));
  mk('rect',{x:-w/2,y:-13,width:w,height:26,rx:n.type=='action'?13:5,fill:col,'fill-opacity':n.type=='observation'?.16:.9,
   stroke:col,'stroke-width':n.synthetic?2:1,'stroke-dasharray':n.synthetic?'4 2':''},g);
  mk('text',{'text-anchor':'middle',y:4,style:`fill:${n.type=='observation'?'var(--fg)':'#fff'}`},g).textContent=n.label;
  g.onmouseenter=()=>focus(n);g.onmouseleave=()=>focus(pinned);g.onclick=()=>{pinned=pinned===n?null:n;focus(pinned)};
  nodeEls[n.id]=g;});
 const cnt={};G.edges.forEach(e=>cnt[e.rel]=(cnt[e.rel]||0)+1);
 const box=document.getElementById('edges');
 box.innerHTML=Object.keys(REL_COL).filter(r=>cnt[r]).map(r=>
  `<label><input type=checkbox data-r=${r} ${relState[r]?'checked':''}><span class=sw style="background:${REL_COL[r]}"></span>${r} (${cnt[r]})</label>`).join('');
 box.querySelectorAll('input').forEach(i=>i.onchange=()=>{relState[i.dataset.r]=i.checked;focus(pinned)});
 focus(null);}
function focus(n){
 const vis=p=>relState[p._e.rel];
 edgeEls.forEach(p=>{p.style.display=vis(p)?'':'none';const hit=n&&(p._e.s==n.id||p._e.d==n.id);
  p.setAttribute('opacity',n?(hit?1:.06):.55);p.setAttribute('stroke-width',hit?2.4:1.3);});
 Object.values(nodeEls).forEach(g=>g.style.opacity=1);
 if(n){const nb=new Set([n.id]);edgeEls.forEach(p=>{if(vis(p)&&(p._e.s==n.id||p._e.d==n.id)){nb.add(p._e.s);nb.add(p._e.d)}});
  Object.entries(nodeEls).forEach(([k,g])=>g.style.opacity=nb.has(k)?1:.25);
  info.textContent=`${n.type}${n.labeltext?' ['+n.labeltext+(n.synthetic?', synthetic':'')+']':''}\n${n.detail}`;}
 else info.textContent='hover a node, click to pin';}
document.getElementById('legend').innerHTML=[['task','--task'],['memory','--memory'],['tool','--tool'],['action (approve)','--approve'],
 ['action (reject)','--reject'],['observation','--obs'],['file','--file']].map(([t,c])=>`<label><span class=dot style="background:var(${c})"></span>${t}</label>`).join('')
 +'<label style="color:var(--mut)">dashed outline = synthetic action</label>';
let LIST=[],cur=-1;
const list=document.getElementById('list');
function drawList(){
 const q=document.getElementById('q').value.toLowerCase(),f=document.getElementById('flt').value;
 const rows=LIST.filter(t=>(f=='all'||(f=='attack'&&t.attack)||(f=='benign'&&!t.attack)||(f=='rej'&&t.rejects))&&
  (t.id+' '+t.model+' '+t.task+' '+t.bench).toLowerCase().includes(q));
 document.getElementById('cnt').textContent=rows.length+'/'+LIST.length;
 list.innerHTML='';rows.slice(0,500).forEach(t=>{const d=document.createElement('div');if(t.i==cur)d.className='sel';
  d.innerHTML=`<span class="${t.attack?'atk':''}">${t.attack?'ATK':'ben'}</span> ${t.n} steps${t.rejects?' / '+t.rejects+' rej':''} · ${t.model}<br><span style="opacity:.7">${t.id}</span>`;
  d.onclick=()=>openTrace(t.i);list.appendChild(d);});}
async function openTrace(i){cur=i;drawList();render(await (await fetch('/api/graph?i='+i)).json());}
document.getElementById('q').oninput=drawList;document.getElementById('flt').onchange=drawList;
fetch('/api/list').then(r=>r.json()).then(l=>{LIST=l;drawList();if(l.length)openTrace(l[0].i);});
</script>"""


def serve(traces, host, port, open_browser):
    summaries = [summary(i, t) for i, t in enumerate(traces)]
    cache = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype, code=200):
            b = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(PAGE, "text/html")
            elif u.path == "/api/list":
                self._send(json.dumps(summaries), "application/json")
            elif u.path == "/api/graph":
                try:
                    i = int(parse_qs(u.query)["i"][0])
                    if i not in cache:
                        cache[i] = json.dumps(graph_json(traces[i]))
                    self._send(cache[i], "application/json")
                except (KeyError, ValueError, IndexError) as e:
                    self._send(json.dumps({"error": str(e)}), "application/json", 400)
            else:
                self._send("not found", "text/plain", 404)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    url = "http://%s:%d/" % (host, srv.server_address[1])
    print("serving %d traces at %s  (Ctrl+C to stop)" % (len(traces), url), flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="trace .jsonl file(s) or globs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765, help="0 = pick a free port")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    a = ap.parse_args()
    paths = sorted(p for pat in a.traces for p in (glob.glob(pat) or [pat]))
    traces = load_traces(paths)
    if not traces:
        raise SystemExit("no traces with events found in %s" % paths)
    serve(traces, a.host, a.port, not a.no_open)


if __name__ == "__main__":
    main()
