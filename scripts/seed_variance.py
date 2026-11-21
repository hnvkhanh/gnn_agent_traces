"""Seed-variance diagnostic: replicate src/tracegen_gnn.fit_select on the fixed split, but log val/test AUPRC every epoch
and separate the init seed (weights + dropout) from the order seed (DataLoader shuffle).

    python scripts/seed_variance.py --device cuda:0 --guards edge-gnn --runs 0:0,0:1,1:0 --epochs 20 --out runs_a.jsonl
    --deterministic   torch.use_deterministic_algorithms(True, warn_only=True) + cudnn deterministic
"""
import argparse
import json
import os
import sys
import time

ap_ = argparse.ArgumentParser()
ap_.add_argument("--device", default="cuda:0")
ap_.add_argument("--guards", default="edge-gnn,gnn+files")
ap_.add_argument("--runs", default="0:0", help="init:order pairs, comma separated (r suffix = repeat tag, e.g. 0:0:b)")
ap_.add_argument("--epochs", type=int, default=20)
ap_.add_argument("--lr", type=float, default=3e-3)
ap_.add_argument("--out", required=True)
ap_.add_argument("--clip", type=float, default=float("inf"), help="gradient-norm clip (inf = off, as in the repo)")
ap_.add_argument("--deterministic", action="store_true")
ap_.add_argument("--cv", action="store_true", help="also score CV cell 0-0 (trains on the CV cell's train set)")
a = ap_.parse_args()
os.environ["GNN_DEVICE"] = a.device
if a.deterministic:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from src.tracegen_gnn import Data, LABELS, _scores, _ys, _y, cell_spec  # noqa: E402
from src.rank_metrics import average_precision  # noqa: E402
from src.model.edge_gnn import EdgeGNN  # noqa: E402
from src.model.gnn import ActionMLP, HeteroGNN  # noqa: E402
from src.model.gnn_files import FileHeteroGNN  # noqa: E402
from src.model.transformer import GraphTransformer  # noqa: E402

if a.deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEV = torch.device(a.device)
t0 = time.time()
data = Data(os.path.join(ROOT, "data", "tracegen_all"), os.path.join(ROOT, "data", "agentdojo"))
key = "cv/5/0-0" if a.cv else "main"
tr, va, ev, _, _ = cell_spec(data, key)
te = ev["test"]
print("data loaded %.0fs, train %d val %d test %d" % (time.time() - t0, len(tr), len(va), len(te)), flush=True)

MODELS = {"mlp": lambda h: ActionMLP(h, 2), "gnn": lambda h: HeteroGNN(h, classes=2),
          "gnn+files": lambda h: FileHeteroGNN(h, classes=2), "transformer": lambda h: GraphTransformer(h, classes=2),
          "edge-gnn": lambda h: EdgeGNN(h, classes=2)}
KIND = {"gnn": "plain", "edge-gnn": "edges"}

out = open(a.out, "a", encoding="utf-8")
for gname in a.guards.split(","):
    G = data.graphs(KIND.get(gname, "files"), ())
    gtr, gva, gte = [G[i] for i in tr], [G[i] for i in va], [G[i] for i in te]
    ytr = _ys(gtr)
    cnt = torch.bincount(ytr, minlength=2).float().clamp(min=1)
    w = (cnt.sum() / (2 * cnt)).to(DEV)
    yv, yt = _ys(gva).numpy(), _ys(gte).numpy()
    for spec in a.runs.split(","):
        parts = spec.split(":")
        init, order, tag = int(parts[0]), int(parts[1]), (parts[2] if len(parts) > 2 else "")
        torch.manual_seed(init)
        np.random.seed(init)
        model = MODELS[gname](64).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
        gen = torch.Generator()
        gen.manual_seed(order)
        loader = DataLoader(gtr, batch_size=32, shuffle=True, generator=gen)
        rec = dict(guard=gname, init=init, order=order, tag=tag, device=a.device, det=a.deterministic, cell=key,
                   lr=a.lr, clip=a.clip,
                   epochs=[], n_val_rej=int(yv.sum()), n_test_rej=int(yt.sum()))
        t1 = time.time()
        for ep in range(1, a.epochs + 1):
            model.train()
            tot, nb, gn = 0.0, 0, 0.0
            for b in loader:
                b = b.to(DEV)
                opt.zero_grad()
                loss = Fn.cross_entropy(model(b), _y(b), weight=w)
                loss.backward()
                gn += float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))
                opt.step()
                tot += float(loss)
                nb += 1
            sv, st = _scores(model, gva, DEV), _scores(model, gte, DEV)
            rec["epochs"].append(dict(ep=ep, loss=tot / nb, gnorm=gn / nb, val_ap=float(average_precision(yv, sv)),
                                      test_ap=float(average_precision(yt, st)),
                                      test=[round(float(x), 5) for x in st]))
            print("%s init %d order %d%s ep %2d loss %.4f gnorm %.2f val %.3f test %.3f  (%.0fs)" % (
                gname, init, order, tag, ep, tot / nb, gn / nb, rec["epochs"][-1]["val_ap"], rec["epochs"][-1]["test_ap"],
                time.time() - t1), flush=True)
        rec["y_test"] = yt.tolist()
        out.write(json.dumps(rec) + "\n")
        out.flush()
