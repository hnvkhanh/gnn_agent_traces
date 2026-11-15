"""Stages 3-5: entity check → bidirectional NLI → optional LLM judge. Writes verified/rejected variants."""
import logging
import re

from tracegen import config as C
from tracegen import gpu, io, ollama
from tracegen.stages import entities

log = logging.getLogger("tracegen")

JUDGE_PROMPT = """Request A: "{a}"
Request B: "{b}"
Would these two requests require exactly the same actions to complete? Answer YES or NO."""


class NLI:
    """Bidirectional entailment with an MNLI cross-encoder on an explicit device."""

    def __init__(self, model_name, device, batch_size=16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = "cuda:%d" % device if torch.cuda.is_available() and device is not None else "cpu"
        if self.device == "cpu":
            log.warning("NLI on CPU (CUDA unavailable)")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # fp32: DeBERTa's disentangled attention fails under fp16 in transformers 5 (Float/Half mismatch)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float32).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.ent = next(i for name, i in labels.items() if name.startswith("entail"))
        self.bs, self.name = batch_size, model_name
        log.info("NLI %s on %s (entailment label %d)", model_name, self.device, self.ent)

    def entail(self, premises, hypotheses):
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(premises), self.bs):
                enc = self.tok(premises[i:i + self.bs], hypotheses[i:i + self.bs], return_tensors="pt",
                               padding=True, truncation=True, max_length=512).to(self.device)
                p = self.model(**enc).logits.float().softmax(-1)[:, self.ent]
                out.extend(p.tolist())
        return out

    def close(self):
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


def judge_once(url, model, a, b, seed=0, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": JUDGE_PROMPT.format(a=a, b=b)}]
    text, _ = ollama.chat(url, model, msgs,
                          temperature=0.0, seed=seed, num_predict=8, timeout=300)
    m = re.search(r"\b(YES|NO)\b", text.upper())
    return (m.group(1) if m else "UNPARSEABLE"), text


class Verifier:
    def __init__(self, cfg, judge=None, nli=None):
        self.cfg, v = cfg, cfg["verify"]
        self.judge = v["judge"] if judge is None else judge
        self.out, self.rej = C.out_path(cfg, "verified_variants.jsonl"), C.out_path(cfg, "rejected_variants.jsonl")
        self.done = io.keys(self.out, lambda r: r["variant_id"]) | io.keys(self.rej, lambda r: r["variant_id"])
        self._nli = nli
        self.judge_url, self.judge_model = C.judge_url(cfg), v["judge_model"]
        self.judge_tag = None

    @property
    def nli(self):
        if self._nli is None:
            v = self.cfg["verify"]
            self._nli = NLI(v["nli_model"], C.nli_device(self.cfg), v["nli_batch_size"])
        return self._nli

    def run(self, cands):
        """Verify a batch of candidates. Returns the verified records (already appended to the output)."""
        cfg, v = self.cfg, self.cfg["verify"]
        todo = [dict(c) for c in cands if c["variant_id"] not in self.done]
        if not todo:
            return []
        survivors = []
        for c in todo:
            c.update(entity_ok=None, entities=None, nli_fwd=None, nli_bwd=None, judge=None, reject_reason=None)
            if v["entities"]:
                ents = entities.extract(c["original"])
                ok, missing = entities.check(c["original"], c["paraphrase"], ents)
                c["entities"], c["entity_ok"] = [list(e) for e in ents], ok
                if not ok:
                    c["reject_reason"] = "entity:" + ";".join(missing)
                    self._emit(c, False)
                    continue
            survivors.append(c)
        if survivors:
            fwd = self.nli.entail([c["original"] for c in survivors], [c["paraphrase"] for c in survivors])
            bwd = self.nli.entail([c["paraphrase"] for c in survivors], [c["original"] for c in survivors])
            tau, nxt = v["nli_tau"], []
            for c, f, b in zip(survivors, fwd, bwd):
                c.update(nli_fwd=round(f, 4), nli_bwd=round(b, 4), nli_model=v["nli_model"], nli_tau=tau)
                if f >= tau and b >= tau:
                    nxt.append(c)
                else:
                    c["reject_reason"] = "nli:%s" % ("fwd+bwd" if f < tau and b < tau else "fwd" if f < tau else "bwd")
                    self._emit(c, False)
            survivors = nxt
        if self.judge and survivors:
            if self.judge_tag is None:
                self.judge_tag = ollama.model_info(self.judge_url, self.judge_model)["tag"]
            for c in survivors:
                ans, raw = judge_once(self.judge_url, self.judge_model, c["original"], c["paraphrase"],
                                      system=v.get("judge_system"))
                c.update(judge=ans, judge_raw=raw, judge_model=self.judge_model, judge_model_tag=self.judge_tag)
                if ans != "YES":
                    c["reject_reason"] = "judge:%s" % ans
            kept = [c for c in survivors if c["reject_reason"] is None]
            for c in survivors:
                if c["reject_reason"] is not None:
                    self._emit(c, False)
            survivors = kept
        for c in survivors:
            self._emit(c, True)
        return survivors

    def _emit(self, c, ok):
        io.append_jsonl(self.out if ok else self.rej, c)
        self.done.add(c["variant_id"])

    def close(self):
        if self._nli is not None:
            self._nli.close()
            self._nli = None


def verify(cfg, suites, judge=None, limit=None):
    from tracegen.stages.generate import originals
    keep = {(r["suite"], r["task_id"]) for r in originals(cfg, suites, limit)}
    cands = [c for c in io.read_jsonl(C.out_path(cfg, "candidates.jsonl")) if (c["suite"], c["task_id"]) in keep]
    ver = Verifier(cfg, judge=judge)
    try:
        if any(c["variant_id"] not in ver.done for c in cands):
            if C.nli_device(cfg) is not None:
                gpu.check_fit(gpu.probe(), [C.nli_device(cfg)], [("NLI " + cfg["verify"]["nli_model"], gpu.NLI_BYTES)])
        kept = ver.run(cands)
    finally:
        ver.close()
    log.info("verify: %d candidates, %d newly verified", len(cands), len(kept))
