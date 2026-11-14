import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from confidence_analysis import ece, quadrant, quadrant_counts, selective


def test_quadrants():
    rows = [dict(true=1, pred=1, conf=.95), dict(true=1, pred=0, conf=.99), dict(true=0, pred=1, conf=.92),
            dict(true=0, pred=0, conf=.6), dict(true=1, pred=0, conf=.55)]
    c = quadrant_counts(rows, .9)
    assert (c["cc"], c["cw_fn"], c["cw_fp"], c["uc"], c["uw"], c["n"]) == (1, 1, 1, 1, 1, 5)
    assert quadrant(1, 0, .9, .9) == "cw_fn"


def test_ece_and_selective():
    e, bins = ece([1.0, 1.0, .5, .5], [1, 0, 0, 1])
    assert abs(e - 0.25) < 1e-9 and sum(b[2] for b in bins) == 4
    assert selective([.9, .6], [1, 0], 2) == [(.5, 1.0), (1.0, .5)]


def test_temperature_scaling():
    import numpy as np
    from confidence_analysis import fit_temperature, nll
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    flip = rng.random(2000) < 0.2                       # 80%-accurate but logits saturated at +-20
    z = np.where((y ^ flip)[:, None] == np.array([0, 1]), 20.0, -20.0)
    T = fit_temperature(z, y)
    assert T > 5 and nll(z, y, T) < nll(z, y, 1.0)
    assert fit_temperature(z, np.zeros(2000, int)) == 1.0   # single-class validation: refuse to fit
    e1, _ = ece(np.abs(1 / (1 + np.exp(-z[:, 1] + z[:, 0])) - .5) + .5, (z.argmax(1) == y))
    p = 1 / (1 + np.exp(-(z[:, 1] - z[:, 0]) / T))
    e2, _ = ece(np.maximum(p, 1 - p), (z.argmax(1) == y))
    assert e2 < e1 / 3
