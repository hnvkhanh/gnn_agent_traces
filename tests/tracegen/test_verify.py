from tracegen import config as C
from tracegen import io
from tracegen.stages import verify as V


class StubNLI:
    def __init__(self, scores):
        self.scores = scores   # paraphrase -> (fwd, bwd)

    def entail(self, premises, hyps):
        # original → paraphrase: hypothesis is the paraphrase (fwd); paraphrase → original: premise is (bwd)
        return [self.scores[h][0] if h in self.scores else self.scores[p][1] for p, h in zip(premises, hyps)]

    def close(self):
        pass


def cand(i, para):
    return {"variant_id": "banking/user_task_3/p%d" % i, "suite": "banking", "task_id": "user_task_3",
            "original": "Send 12.00 to GB29NWBK60161331926819.", "paraphrase": para}


def test_verifier_routes_and_resume(tmp_path, monkeypatch):
    cfg = C.load("/nonexistent.yaml", {"paths": {"out_dir": str(tmp_path)}})
    cs = [cand(0, "Please transfer 12.00 to GB29NWBK60161331926819."),      # passes everything
          cand(1, "Transfer 12 to GB29NWBK60161331926819."),               # entity: number changed
          cand(2, "Transfer 12.00 to GB29NWBK60161331926819 tomorrow."),   # NLI bwd too low
          cand(3, "Wire 12.00 over to GB29NWBK60161331926819.")]           # judge says NO
    nli = StubNLI({cs[0]["paraphrase"]: (0.95, 0.9), cs[2]["paraphrase"]: (0.9, 0.3), cs[3]["paraphrase"]: (0.9, 0.9)})
    monkeypatch.setattr(V.ollama, "model_info", lambda u, m: {"tag": m + "@j", "size": 1})
    monkeypatch.setattr(V, "judge_once", lambda u, m, a, b, seed=0, system=None: ("NO", "NO") if "Wire" in b else ("YES", "Yes."))
    ver = V.Verifier(cfg, judge=True, nli=nli)
    kept = ver.run(cs)
    assert [k["variant_id"] for k in kept] == ["banking/user_task_3/p0"]
    assert kept[0]["nli_fwd"] == 0.95 and kept[0]["judge"] == "YES" and kept[0]["judge_model_tag"] == "gemma2:9b@j"
    rej = {r["variant_id"][-2:]: r["reject_reason"] for r in io.read_jsonl(tmp_path / "rejected_variants.jsonl")}
    assert rej["p1"].startswith("entity:missing:number:12.00") and rej["p2"] == "nli:bwd" and rej["p3"] == "judge:NO"
    assert V.Verifier(cfg, judge=True, nli=nli).run(cs) == []    # resume: nothing re-verified
    assert len(io.read_jsonl(tmp_path / "verified_variants.jsonl")) == 1
