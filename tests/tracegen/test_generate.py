import pytest

from tracegen.stages.generate import clean, generate_one, parse_paraphrases


def test_parse_variants():
    assert parse_paraphrases('["a", "b"]') == ["a", "b"]
    assert parse_paraphrases('```json\n["a", "b"]\n```') == ["a", "b"]
    assert parse_paraphrases('{"paraphrases": ["a"]}') == ["a"]
    assert parse_paraphrases('Sure! Here you go: ["a", "b"] Hope it helps') == ["a", "b"]
    assert parse_paraphrases('<think>hmm [1]</think>["x"]') == ["x"]


@pytest.mark.parametrize("bad", ["", "not json", "[1, 2]", "[]", '{"a": ["x"], "b": ["y"]}', '["a", 3]'])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse_paraphrases(bad)


def test_retry_then_success_uses_new_seed():
    replies = iter(["garbage", "{oops", '["one", "two"]'])
    seeds = []

    def chat(msgs, seed):
        seeds.append(seed)
        assert "Rewrite the following request in 2 different ways." in msgs[0]["content"]
        return next(replies)

    paras, attempts, used, err = generate_one(chat, "pay the bill", 2, 100, 3)
    assert paras == ["one", "two"] and attempts == 3 and used == 102 and err is None
    assert seeds == [100, 101, 102]


def test_retry_exhausted():
    calls = []
    paras, attempts, used, err = generate_one(lambda m, s: calls.append(s) or "nope", "x", 3, 0, 3)
    assert paras == [] and attempts == 3 and used is None and "no JSON list" in err and len(calls) == 3


def test_clean_dedup_and_identical():
    out = clean("Pay the bill.", ["Pay the bill.", " Settle the bill. ", "Settle the bill.", "", "\"Cover it.\"",
                                  "a", "b", "c"], 3)
    assert out == ["Settle the bill.", "Cover it.", "a"]
