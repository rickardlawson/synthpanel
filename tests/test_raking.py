import numpy as np
import pandas as pd
import pytest

from synthpanel.calibrate.raking import effective_sample_size, rake


def _toy():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "kjonn": rng.choice(["m", "k"], 2000),
        "region": rng.choice(["nord", "sor", "vest"], 2000),
    })


def test_rake_hits_margins():
    df = _toy()
    m1 = pd.Series({"m": 600.0, "k": 400.0})
    m2 = pd.Series({"nord": 100.0, "sor": 500.0, "vest": 400.0})
    res = rake(df, {"kjonn": m1, "region": m2}, tol=1e-8)
    assert res.converged
    w = pd.Series(res.weights)
    assert w.groupby(df["kjonn"]).sum().round(4).to_dict() == m1.to_dict()
    assert w.groupby(df["region"]).sum().round(4).to_dict() == m2.to_dict()


def test_rake_joint_margin():
    df = _toy()
    joint = df.groupby(["kjonn", "region"]).size().astype(float) * 3
    res = rake(df, {("kjonn", "region"): joint}, tol=1e-8)
    assert res.converged
    assert pd.Series(res.weights).sum() == pytest.approx(joint.sum())


def test_rake_refuses_large_missing_category():
    df = _toy()
    m = pd.Series({"m": 500.0, "k": 400.0, "x": 100.0})  # "x" har ingen agenter
    with pytest.raises(ValueError):
        rake(df, {"kjonn": m})


def test_effective_sample_size():
    assert effective_sample_size(np.ones(100)) == pytest.approx(100)
    assert effective_sample_size(np.array([1.0, 1.0, 10.0])) < 3
