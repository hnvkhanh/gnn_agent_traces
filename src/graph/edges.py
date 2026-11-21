"""Edge graph: artifacts are nodes, actions are edges (src/graph/build.py and files.py are left unchanged).

    g = trace_to_graph_edges(tr, ablate)

Nodes: agent (the assistant, generic), task (user prompt), memory items, tools, files (src/graph/files.py rules) and
entities: argument values such as an IBAN, an email address, a channel, a URL or a hotel name. An entity is keyed by its
lowercased value, so the IBAN of `send_money(recipient=...)` and the same IBAN in an earlier tool result are one node.

A past action j is a group of edges, each with edge_attr = build.feat(text, step j):
  (agent, calls, tool)                  tool + arguments
  (tool, acts_on, entity | file)        one per argument value / file argument (arg key + tool)
  (tool, returns, agent)                the observation of j (the tool result goes back to the agent)
  (tool, returned, entity | file)       entities / files that the observation of j names
Context: (task, mentions, entity | file) and (memory, mentions, entity) for values the prompt / memory names.
Every edge type also has a reverse `rev_<relation>` (same attr), so tool, file and entity nodes absorb context.

Causality: one block-diagonal snapshot per action. Snapshot k is built from user_task, memory, tools, events[:k] with
their observations and the arguments of events[k] only (its observation is blanked). Nothing at or after action k's
result can reach snapshot k, whatever the edge direction. The action being scored is the `query` of its snapshot:
g["query"].x = feat(tool + args, step), g["query"].y its label, and read-only index edges (agent|task|tool, q_*, query)
and (entity|file, q_arg, query) point at its endpoints (they are not used for message passing).

Ablations: directed (no rev_ edges), entities (no entity nodes), returned, task_mentions, observation (no returns /
returned edges), attacker_values (files.mask_attacker_values first).
"""
import json

import torch
from torch_geometric.data import HeteroData

from ..sandbox.policy import LABELS
from .build import _ei, _stack, feat
from .files import ID_KEYS, NAME_KEYS, file_edges, mask_attacker_values, _names

NODE_TYPES = ["agent", "task", "memory", "tool", "file", "entity", "query"]
CTX_EDGE_TYPES = [
    ("agent", "calls", "tool"),
    ("tool", "acts_on", "entity"),
    ("tool", "acts_on", "file"),
    ("tool", "returns", "agent"),
    ("tool", "returned", "entity"),
    ("tool", "returned", "file"),
    ("task", "mentions", "entity"),
    ("task", "mentions", "file"),
    ("memory", "mentions", "entity"),
]
REV_EDGE_TYPES = [(d, "rev_" + r, s) for s, r, d in CTX_EDGE_TYPES]
EDGE_TYPES = CTX_EDGE_TYPES + REV_EDGE_TYPES          # message passing
QUERY_EDGE_TYPES = [("agent", "q_agent", "query"), ("task", "q_task", "query"), ("tool", "q_tool", "query"),
                    ("entity", "q_arg", "query"), ("file", "q_arg", "query")]   # readout only
EDGE_ABLATIONS = ["directed", "entities", "returned", "task_mentions", "observation", "attacker_values"]

MIN_LEN, MAX_LEN = 4, 64      # entity values: same lower bound as attacker_values; longer text stays in edge features
FILE_KEYS = NAME_KEYS | ID_KEYS | {"attachments", "attachment"}


def arg_values(args):
    """[(arg key, value)] of the leaf argument values that become entity nodes (file arguments excluded)."""
    out = []

    def walk(k, v):
        if isinstance(v, dict):
            for kk, vv in v.items():
                if kk.lower() not in FILE_KEYS and kk.lower() != "type":
                    walk(kk, vv)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(k, x)
        elif v is not None:
            s = str(v).strip()
            if MIN_LEN <= len(s) <= MAX_LEN:
                out.append((k, s.lower()))
    walk("", args or {})
    return out


def _call_text(e):
    return e["tool"] + " " + json.dumps(e["args"], sort_keys=True, default=str)


def snapshot(tr, k, ablate=frozenset()):
    """Node features and edges of snapshot k (action k is the query). Returns a dict of plain lists."""
    ev = tr["events"][:k + 1]
    ev = ev[:k] + [dict(ev[k], observation="")]          # the query's own result is not known at decision time
    tk = dict(tr, events=ev)
    task_l = (tr["user_task"] or "").lower()
    mem_l = [(m or "").lower() for m in tr["memory"]]
    tools = {t["name"]: t for t in tr["tools"]}

    # nodes
    tool_ix, tool_x = {}, []
    for e in ev:
        if e["tool"] not in tool_ix:
            tool_ix[e["tool"]] = len(tool_x)
            t = tools.get(e["tool"], {"name": e["tool"], "description": ""})
            tool_x.append(feat(t["name"] + " " + t.get("description", "")))
    ent_ix, ent_x, ent_val = {}, [], []
    per_action_ents = []
    for e in ev:
        mine = []
        if "entities" not in ablate:
            for key, val in arg_values(e["args"]):
                if val not in ent_ix:
                    ent_ix[val] = len(ent_x)
                    ent_x.append(feat(key + " " + val))
                    ent_val.append(val)
                mine.append((key, ent_ix[val]))
        per_action_ents.append(mine)
    fkeys, ftargets, fappears = file_edges(tk)            # prefix only: snapshot k never sees later files
    file_x = [feat(f) for f in fkeys]

    E = {et: ([], []) for et in CTX_EDGE_TYPES}            # et -> (pairs, attr rows)

    def add(et, s, d, text, step):
        E[et][0].append([s, d])
        E[et][1].append(feat(text, step))

    for j, e in enumerate(ev[:k]):                         # past actions = edge groups
        t, st = tool_ix[e["tool"]], e["step"]
        add(("agent", "calls", "tool"), 0, t, _call_text(e), st)
        for key, n in per_action_ents[j]:
            add(("tool", "acts_on", "entity"), t, n, key + " " + e["tool"], st)
        if "observation" not in ablate:
            add(("tool", "returns", "agent"), t, 0, e["observation"], st)
            if "returned" not in ablate:
                obs_l = (e["observation"] or "").lower()
                for n, val in enumerate(ent_val):
                    if val in obs_l:
                        add(("tool", "returned", "entity"), t, n, e["observation"], st)
    for f, i in ftargets:
        if i < k:
            add(("tool", "acts_on", "file"), tool_ix[ev[i]["tool"]], f, "file " + ev[i]["tool"], ev[i]["step"])
    if "observation" not in ablate and "returned" not in ablate:
        for f, j in fappears:
            if j < k:
                add(("tool", "returned", "file"), tool_ix[ev[j]["tool"]], f, ev[j]["observation"], ev[j]["step"])
    if "task_mentions" not in ablate:
        for n, val in enumerate(ent_val):
            if val in task_l:
                add(("task", "mentions", "entity"), 0, n, "task mentions", 0)
        task_files = set(_names(tr["user_task"]))
        for f, key in enumerate(fkeys):
            if key in task_files:
                add(("task", "mentions", "file"), 0, f, "task mentions", 0)
    for m, text in enumerate(mem_l):
        for n, val in enumerate(ent_val):
            if val in text:
                add(("memory", "mentions", "entity"), m, n, "memory mentions", 0)

    q = ev[k]
    query = dict(x=feat(_call_text(q), q["step"]), y=LABELS.index(q["label"]), tool=tool_ix[q["tool"]],
                 ents=sorted({n for _, n in per_action_ents[k]}), files=sorted({f for f, i in ftargets if i == k}))
    return dict(tool=tool_x, entity=ent_x, file=file_x, memory=[feat(m) for m in tr["memory"]], edges=E, query=query)


def trace_to_graph_edges(tr, ablate=frozenset()):
    ablate = frozenset(ablate)
    if not tr["events"]:
        return None
    if "attacker_values" in ablate:
        tr = mask_attacker_values(tr)
    X = {nt: [] for nt in NODE_TYPES}
    E = {et: ([], []) for et in EDGE_TYPES + QUERY_EDGE_TYPES}
    y = []
    agent_x, task_x = feat("assistant"), feat(tr["user_task"])
    for k in range(len(tr["events"])):
        s = snapshot(tr, k, ablate)
        off = {nt: len(X[nt]) for nt in NODE_TYPES}
        X["agent"].append(agent_x)
        X["task"].append(task_x)
        for nt in ("memory", "tool", "file", "entity"):
            X[nt].extend(s[nt])
        for (src, rel, dst), (pairs, attrs) in s["edges"].items():
            for (a, b), at in zip(pairs, attrs):
                a, b = a + off[src], b + off[dst]
                E[(src, rel, dst)][0].append([a, b])
                E[(src, rel, dst)][1].append(at)
                if "directed" not in ablate:
                    E[(dst, "rev_" + rel, src)][0].append([b, a])
                    E[(dst, "rev_" + rel, src)][1].append(at)
        qi, q = len(X["query"]), s["query"]
        X["query"].append(q["x"])
        y.append(q["y"])
        E[("agent", "q_agent", "query")][0].append([off["agent"], qi])
        E[("task", "q_task", "query")][0].append([off["task"], qi])
        E[("tool", "q_tool", "query")][0].append([off["tool"] + q["tool"], qi])
        E[("entity", "q_arg", "query")][0].extend([[off["entity"] + n, qi] for n in q["ents"]])
        E[("file", "q_arg", "query")][0].extend([[off["file"] + f, qi] for f in q["files"]])
    g = HeteroData()
    for nt in NODE_TYPES:
        g[nt].x = _stack(X[nt])
    g["query"].y = torch.tensor(y, dtype=torch.long)
    for et, (pairs, attrs) in E.items():
        g[et].edge_index = _ei(pairs)
        if et in EDGE_TYPES:
            g[et].edge_attr = _stack(attrs)
    return g
