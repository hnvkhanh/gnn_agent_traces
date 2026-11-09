"""Continual-learning strategies for the HeteroGNN guard (domain-incremental: one head, labels approve/reject).

The CGLB baseline set (Zhang et al., NeurIPS 2022):
  finetune  naive sequential training (lower bound)
  joint     re-initialise and train on the union of all training sets so far (upper bound)
  ewc       Elastic Weight Consolidation (Kirkpatrick et al. 2017), empirical Fisher, one penalty per past experience
  mas       Memory Aware Synapses (Aljundi et al. 2018), importance |d ||f(x)||^2 / d theta|
  lwf       Learning without Forgetting (Li & Hoiem 2017), distillation from the previous model on the same head
  agem      Averaged GEM (Chaudhry et al. 2019): project the gradient so the loss on a memory batch does not increase.
            GEM itself needs a QP solver, which is not installed.
  twp       Topology-aware Weight Preserving (Liu et al. 2021). Importance lambda_l |grad L|^2 + lambda_t |grad T|^2, plus
            beta * ||grad L||_1 on the current loss. T is the GAT attention norm in the paper. SAGEConv has no attention,
            so T is the norm of the message-passed action embeddings. The reference code detaches the beta term (a
            no-op); here it is differentiated (create_graph=True), as the paper describes, and beta is tuned together
            with lambda_l over {0, 1e-3, 1e-2} (0 = the reference code's behaviour; 1e-2 can collapse training).
  ergnn     ER-GNN experience replay (Zhou & Cao 2021) with mean-of-feature selection. An action's computation graph is
            its whole trace, so whole trace graphs are stored. The total memory M is shared equally by the seen
            experiences. Loss (1 - beta) L_cur + beta L_mem, beta = |M| / (|M| + |D_k|).
  replay    the same replay loss with a reservoir buffer: it differs from ergnn only in how the buffer is selected.

Memories are filled only from the training graphs of finished experiences, never from test graphs.
"""
import copy
import random

import torch
import torch.nn.functional as Fn
from torch_geometric.data import Batch

MEMORY_BATCH = 32


def batches(graphs, size=64):
    """Collate graphs into new Batch objects (never moves the stored graphs; uses no RNG)."""
    for i in range(0, len(graphs), size):
        yield Batch.from_data_list(graphs[i:i + size])


def params(model):
    return [p for p in model.parameters() if p.requires_grad]


def flat_grad(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params(model)])


def set_flat_grad(model, g):
    k = 0
    for p in params(model):
        n = p.numel()
        p.grad = g[k:k + n].view_as(p).clone()
        k += n


class ActionEmbeddings:
    """Captures the action-node embeddings that enter model.head (a forward pre-hook; the model is not edited)."""

    def __init__(self, model):
        self.h = None
        self.handle = model.head.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self.h = inputs[0]

    def close(self):
        self.handle.remove()


class Strategy:
    name = "finetune"
    grid = None  # {hparam: [values]} tuned on the validation stream, or None
    uses_buffer = False

    def __init__(self, hp=None, buffer=50, seed=0, make_model=None):
        self.hp = dict(hp or {})
        self.buffer = buffer
        self.rng = random.Random(seed)
        self.make_model = make_model
        self.k = -1

    def start(self, k, model, cur, past):
        """Before experience k. cur: its training graphs; past: training graphs of the earlier experiences.
        Returns (model, graphs to train on)."""
        self.k = k
        return model, cur

    def replay_graphs(self):
        """Stored graphs mixed into the loss (they also count for the class weights)."""
        return []

    def loss(self, model, batch, out, base, w):
        return base

    def after_backward(self, model, w):
        pass

    def end(self, k, model, cur, cur_ids, w):
        """After training on experience k. cur_ids: (bench, trace_id) of each graph in cur."""

    def memory_ids(self):
        """(bench, trace_id) of every stored graph, for the leakage audit."""
        return []


class Finetune(Strategy):
    name = "finetune"


class Joint(Strategy):
    name = "joint"

    def start(self, k, model, cur, past):
        self.k = k
        if k == 0:
            return model, cur
        return self.make_model(), [g for p in past for g in p] + cur


class _Penalty(Strategy):
    """Quadratic penalty sum_k sum_i I_k,i (theta_i - theta*_k,i)^2 with one (I, theta*) per past experience."""
    scale_key = "lam"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.terms = []

    def penalty(self, model):
        if not self.terms:
            return 0.0
        tot = 0.0
        for imp, star in self.terms:
            for p, i, s in zip(params(model), imp, star):
                tot = tot + (i * (p - s).pow(2)).sum()
        return tot

    def loss(self, model, batch, out, base, w):
        return base + self.hp[self.scale_key] * self.penalty(model) if self.terms else base

    def importance(self, model, cur, w):
        raise NotImplementedError

    def end(self, k, model, cur, cur_ids, w):
        model.eval()
        imp = self.importance(model, cur, w)
        self.terms.append(([x.detach() for x in imp], [p.detach().clone() for p in params(model)]))


def _per_graph(model, cur, fn, squared):
    """Mean over graphs of grad^2 (squared=True) or |grad| of fn(model output, graph), one graph at a time."""
    device = next(model.parameters()).device
    acc = [torch.zeros_like(p) for p in params(model)]
    for b in batches(cur, 1):
        b = b.to(device)
        model.zero_grad()
        fn(model(b), b).backward()
        for a, p in zip(acc, params(model)):
            if p.grad is not None:
                a += p.grad.detach().pow(2) if squared else p.grad.detach().abs()
    model.zero_grad()
    return [a / max(1, len(cur)) for a in acc]


class EWC(_Penalty):
    name = "ewc"
    grid = {"lam": [1e1, 1e2, 1e3, 1e4]}

    def importance(self, model, cur, w):
        # empirical Fisher: mean of squared per-graph gradients of the log-likelihood
        return _per_graph(model, cur, lambda out, b: Fn.cross_entropy(out, b["action"].y), squared=True)


class MAS(_Penalty):
    name = "mas"
    grid = {"lam": [0.1, 1.0, 10.0, 100.0]}

    def importance(self, model, cur, w):
        # mean |gradient| of the squared L2 norm of the output (unsupervised)
        return _per_graph(model, cur, lambda out, b: out.pow(2).sum(1).mean(), squared=False)


class TWP(_Penalty):
    name = "twp"
    grid = {"lam_l": [1e1, 1e2, 1e3, 1e4], "beta": [0.0, 1e-3, 1e-2]}
    scale_key = None

    def loss(self, model, batch, out, base, w):
        loss = base
        if self.terms:
            loss = loss + self.penalty(model)
        beta = self.hp.get("beta", 0.01)
        if beta:
            ps = params(model)
            grads = torch.autograd.grad(base, ps, create_graph=True, allow_unused=True)
            loss = loss + beta * sum(g.abs().sum() for g in grads if g is not None)
        return loss

    def importance(self, model, cur, w):
        """Full-data gradients (accumulated over batches), squared: lambda_l |grad L|^2 + lambda_t |grad T|^2."""
        dev = next(model.parameters()).device
        lam_l = self.hp["lam_l"]
        lam_t = self.hp.get("lam_t", lam_l / 100.0)
        ps = params(model)
        n = sum(int(g["action"].y.numel()) for g in cur)
        model.zero_grad()
        for b in batches(cur):
            b = b.to(dev)
            out = model(b)
            (Fn.cross_entropy(out, b["action"].y, weight=w) * (b["action"].y.numel() / n)).backward()
        g_loss = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in ps]
        model.zero_grad()
        emb = ActionEmbeddings(model)
        total = 0.0
        try:
            for b in batches(cur):
                model(b.to(dev))
                sq = emb.h.pow(2).sum()
                sq.backward()
                total += float(sq.detach())
        finally:
            emb.close()
        norm = max(total, 1e-12) ** 0.5  # T = ||h||; dT/dtheta = d(sum ||h_b||^2)/dtheta / (2 T)
        g_topo = [(p.grad.detach() / (2 * norm) if p.grad is not None else torch.zeros_like(p)) for p in ps]
        model.zero_grad()
        return [lam_l * a.pow(2) + lam_t * t.pow(2) for a, t in zip(g_loss, g_topo)]


class LwF(Strategy):
    name = "lwf"
    grid = {"alpha": [0.5, 1.0, 2.0, 5.0]}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.old = None

    def start(self, k, model, cur, past):
        self.k = k
        if k > 0:
            self.old = copy.deepcopy(model).eval()
            for p in self.old.parameters():
                p.requires_grad_(False)
        return model, cur

    def distill(self, model, batch, out):
        T = self.hp.get("T", 2.0)
        with torch.no_grad():
            target = Fn.softmax(self.old(batch) / T, dim=1)
        return Fn.kl_div(Fn.log_softmax(out / T, dim=1), target, reduction="batchmean") * T * T

    def loss(self, model, batch, out, base, w):
        if self.old is None:
            return base
        return base + self.hp["alpha"] * self.distill(model, batch, out)


class _Memory(Strategy):
    """A buffer of (bench, trace_id, graph) from finished experiences."""
    uses_buffer = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.mem = []
        self.seen = 0

    def reservoir(self, cur, cur_ids):
        order = list(range(len(cur)))
        self.rng.shuffle(order)
        for i in order:
            item = (cur_ids[i][0], cur_ids[i][1], cur[i])
            if len(self.mem) < self.buffer:
                self.mem.append(item)
            else:
                j = self.rng.randrange(self.seen + 1)
                if j < self.buffer:
                    self.mem[j] = item
            self.seen += 1

    def sample(self, device):
        graphs = [x[2] for x in self.rng.sample(self.mem, min(MEMORY_BATCH, len(self.mem)))]
        return Batch.from_data_list(graphs).to(device)

    def memory_ids(self):
        return [(b, t) for b, t, _ in self.mem]


class _Replay(_Memory):
    def replay_graphs(self):
        return [x[2] for x in self.mem]

    def start(self, k, model, cur, past):
        self.k, self.n_cur = k, len(cur)
        return model, cur

    def loss(self, model, batch, out, base, w):
        if not self.mem:
            return base
        mb = self.sample(out.device)
        beta = len(self.mem) / (len(self.mem) + self.n_cur)
        return (1 - beta) * base + beta * Fn.cross_entropy(model(mb), mb["action"].y, weight=w)


class Replay(_Replay):
    name = "replay"

    def end(self, k, model, cur, cur_ids, w):
        self.reservoir(cur, cur_ids)


class ERGNN(_Replay):
    name = "ergnn"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ranked = []  # per experience: [(bench, trace_id, graph)] best first

    def end(self, k, model, cur, cur_ids, w):
        """Mean-of-feature selection: per class, graphs whose actions of that class lie closest to the class mean."""
        model.eval()
        dev = next(model.parameters()).device
        emb = ActionEmbeddings(model)
        hs, ys = [], []
        try:
            with torch.no_grad():
                for b in batches(cur, 1):
                    model(b.to(dev))
                    hs.append(emb.h.detach().cpu())
                    ys.append(b["action"].y)
        finally:
            emb.close()
        rank = {}
        for c in (0, 1):
            pts = [h[y == c] for h, y in zip(hs, ys)]
            allc = [p for p in pts if len(p)]
            if not allc:
                continue
            mu = torch.cat(allc).mean(0)
            d = [(float((p.mean(0) - mu).norm()), i) for i, p in enumerate(pts) if len(p)]
            rank[c] = [i for _, i in sorted(d)]
        order, seen = [], set()
        for pos in range(max((len(r) for r in rank.values()), default=0)):
            for c in sorted(rank):
                if pos < len(rank[c]) and rank[c][pos] not in seen:
                    seen.add(rank[c][pos])
                    order.append(rank[c][pos])
        self.ranked.append([(cur_ids[i][0], cur_ids[i][1], cur[i]) for i in order])
        n = len(self.ranked)
        quota = [self.buffer // n + (1 if i < self.buffer % n else 0) for i in range(n)]
        self.mem = [x for r, q in zip(self.ranked, quota) for x in r[:q]]


class AGEM(_Memory):
    name = "agem"

    def end(self, k, model, cur, cur_ids, w):
        self.reservoir(cur, cur_ids)

    def after_backward(self, model, w):
        if not self.mem:
            return
        g = flat_grad(model)
        model.zero_grad()
        mb = self.sample(g.device)
        Fn.cross_entropy(model(mb), mb["action"].y, weight=w).backward()
        ref = flat_grad(model)
        dot = torch.dot(g, ref)
        if dot < 0:
            g = g - dot / torch.dot(ref, ref).clamp(min=1e-12) * ref
        set_flat_grad(model, g)


STRATEGIES = {c.name: c for c in (Finetune, Joint, EWC, MAS, LwF, AGEM, TWP, ERGNN, Replay)}
TUNED = [n for n, c in STRATEGIES.items() if c.grid]
DEFAULT_HP = {"twp": {"lam_t_ratio": 0.01, "beta": 0.01}, "lwf": {"T": 2.0}}


def make(name, hp, buffer, seed, make_model):
    if name not in STRATEGIES:
        raise ValueError("unknown strategy %r (one of %s)" % (name, ", ".join(STRATEGIES)))
    full = dict(DEFAULT_HP.get(name, {}), **(hp or {}))
    if name == "twp" and "lam_l" in full:
        full.setdefault("lam_t", full["lam_l"] * full.pop("lam_t_ratio", 0.01))
    full.pop("lam_t_ratio", None)
    return STRATEGIES[name](full, buffer, seed, make_model)


def resolved_hp(name, hp):
    """The full hyperparameter dict a strategy will use (goes into the fingerprint)."""
    return make(name, hp, 0, 0, None).hp
