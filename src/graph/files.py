"""File entity nodes on top of src/graph/build.py (which is left unchanged, so stored results keep their fingerprints).

    g = trace_to_graph_files(tr, ablate)   # trace_to_graph(tr, ablate) + node type `file` and two edge types

A `file` node is one file the trace refers to: named in the user task, in an action's arguments (file_path, filename,
file_id, attachments, ...) or in an observation (a name with a file extension, or an `id_` / `filename` pair of a
workspace file listing). Its key is the filename when known, else `id:<N>`. For action i, a file_id is resolved to a
filename only through observations j < i, so a later observation can never merge or rename the node action i uses.

Feature: build.feat(key), the shared 66-d layout (step 0). `g["file"].time` (file_times) records when the node first
exists; only the transformer reads it.
Edges (file nodes have NO incoming edges, so their state never depends on any event):
  (file, targets, action)          action i's arguments reference file f
  (file, appears_in, observation)  observation j names file f (observation j feeds only actions k > j)
Causality therefore holds exactly as in docs/GRAPH.md: nothing at or after action i reaches action i.

Ablations (in addition to build.ABLATIONS):
  files            drop both file edge types (file nodes stay, isolated)
  attacker_values  control: replace every attacker-chosen argument string of the suite (tr['suite_attacker_values'],
                   set by src/benchmarks/tracegen.py) with one placeholder token in task / action / observation text,
                   in every trace, so the model cannot memorise attacker constants such as the banking attacker IBAN
"""
import copy
import re

import torch

from .build import ABLATIONS, EDGE_TYPES, NODE_TYPES, _ei, _stack, feat, trace_to_graph

FILE_NODE_TYPES = NODE_TYPES + ["file"]
FILE_EDGE_TYPES = EDGE_TYPES + [("file", "targets", "action"), ("file", "appears_in", "observation")]
FILE_ABLATIONS = ABLATIONS + ["files", "attacker_values"]

EXT = r"(?:txt|docx?|xlsx?|pptx?|pdf|csv|md|json|ya?ml|png|jpe?g|zip)"
NAME_RE = re.compile(r"(?<![\w.-])([A-Za-z0-9][\w\-.]{0,80}?\." + EXT + r")(?![\w])", re.I)
NAME_KEYS = {"file_path", "filename", "file_name", "path", "file"}
ID_KEYS = {"file_id"}
MASK = "attackervalue"


def _names(text):
    return [m.group(1).strip().lower() for m in NAME_RE.finditer(text or "")]


def listing_pairs(text):
    """{id: filename} from YAML file listings / file objects (keys of one item share the item's top indent)."""
    out = {}
    for chunk in re.split(r"\n(?=- )", text or ""):
        top = re.match(r"^(- |\s*)", chunk).group(1)
        ind = len(top)
        fn = re.search(r"^(?:- |\s{%d})filename: ['\"]?([^'\"\n]+?)['\"]?\s*$" % ind, chunk, re.M)
        fid = re.search(r"^(?:- |\s{%d})id_: ['\"]?(\w+)['\"]?\s*$" % ind, chunk, re.M)
        if fn and fid:
            out[fid.group(1)] = fn.group(1).strip().lower()
    return out


def obs_files(text):
    """Filenames an observation names, in order of first appearance."""
    return list(dict.fromkeys(_names(text) + list(listing_pairs(text).values())))


def arg_refs(args):
    """[('name', filename) | ('id', id)] referenced by one action's arguments."""
    out = []

    def walk(k, v):
        k = (k or "").lower()
        if isinstance(v, dict):
            for kk, vv in v.items():   # {"type": "file", "file_id": "4"}: the id keeps its own key
                if kk.lower() != "type":
                    walk(kk, vv)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(k if k != "attachments" else "attachment", x)
        elif v is not None:
            s = str(v).strip()
            if k in ID_KEYS or (k == "attachment" and re.fullmatch(r"\w+", s) and not NAME_RE.fullmatch(s)):
                out.append(("id", s))
            elif k in NAME_KEYS or k == "attachment":
                out.extend(("name", n) for n in _names(s))
    for k, v in (args or {}).items():
        walk(k, v)
    return out


def file_edges(tr):
    """(keys, targets, appears_in): node keys in order of first appearance and the two edge lists."""
    ev = tr["events"]
    keys, idx = [], {}

    def node(k):
        if k not in idx:
            idx[k] = len(keys)
            keys.append(k)
        return idx[k]

    for n in _names(tr["user_task"]):
        node(n)
    known = {}  # file id -> filename, from observations strictly before the current action
    targets, appears = [], []
    for i, e in enumerate(ev):
        for kind, v in arg_refs(e["args"]):
            k = v if kind == "name" else known.get(v, "id:%s" % v)
            targets.append([node(k), i])
        for n in obs_files(e["observation"]):
            appears.append([node(n), i])
        known.update(listing_pairs(e["observation"]))  # usable from action i+1 on
    return keys, sorted(set(map(tuple, targets))), sorted(set(map(tuple, appears)))


def file_times(keys, targets, appears, task_names=()):
    """When each file node first exists, on the transformer's clock (src/model/transformer.py): -1 = named in the user
    task, 2i = first named by action i's arguments, 2j+1 = first named by observation j. Taken before any edge ablation,
    so a file that only a later event names is never visible earlier. The GNN ignores it (files have no in-edges)."""
    t = [-1 if k in task_names else None for k in keys]
    for f, i in targets:
        t[f] = 2 * i if t[f] is None else min(t[f], 2 * i)
    for f, j in appears:
        t[f] = 2 * j + 1 if t[f] is None else min(t[f], 2 * j + 1)
    return torch.tensor([-1 if x is None else x for x in t], dtype=torch.long)


def mask_attacker_values(tr):
    vals = sorted({v for v in tr.get("suite_attacker_values", []) if len(v) >= 4}, key=len, reverse=True)
    if not vals:
        return tr
    rx = re.compile("|".join(re.escape(v) for v in vals), re.I)
    sub = lambda t: rx.sub(MASK, t) if isinstance(t, str) else t

    def deep(x):
        if isinstance(x, dict):
            return {k: deep(v) for k, v in x.items()}
        if isinstance(x, list):
            return [deep(v) for v in x]
        return sub(x)
    t = copy.copy(tr)
    t["user_task"] = sub(tr["user_task"])
    t["memory"] = [sub(m) for m in tr["memory"]]
    t["injected"] = sub(tr.get("injected", ""))
    t["events"] = [dict(e, args=deep(e["args"]), observation=sub(e["observation"])) for e in tr["events"]]
    return t


def trace_to_graph_files(tr, ablate=frozenset()):
    ablate = frozenset(ablate)
    if "attacker_values" in ablate:
        tr = mask_attacker_values(tr)
    g = trace_to_graph(tr, ablate & set(ABLATIONS))
    if g is None:
        return None
    src = tr
    if "injected" in ablate and tr.get("injected"):  # same stripping as build.trace_to_graph
        inj = tr["injected"]
        src = dict(tr, user_task=tr["user_task"].replace(inj, ""),
                   events=[dict(e, observation=e["observation"].replace(inj, "")) for e in tr["events"]])
    keys, targets, appears = file_edges(src)
    g["file"].x = _stack([feat(k) for k in keys])
    g["file"].time = file_times(keys, targets, appears, set(_names(src["user_task"])))
    drop = "files" in ablate
    g[("file", "targets", "action")].edge_index = _ei([] if drop else [list(p) for p in targets])
    g[("file", "appears_in", "observation")].edge_index = _ei([] if drop else [list(p) for p in appears])
    if "observation" in ablate:  # observations are cut off from actions; keep file -> observation harmless anyway
        g[("file", "appears_in", "observation")].edge_index = torch.zeros((2, 0), dtype=torch.long)
    return g
