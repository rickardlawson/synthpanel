import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from synthpanel import config
from synthpanel.household import economy

built = (config.PROCESSED_DIR / "agents.parquet").exists()


def test_tilt_hits_target_mixture():
    p = {"a": np.array([0.5, 0.3, 0.2]), "b": np.array([0.1, 0.3, 0.6])}
    w = {"a": 3.0, "b": 1.0}
    target = np.array([0.2, 0.3, 0.5])
    q = economy.tilt(p, w, target)
    mix = (3 * q["a"] + q["b"]) / 4
    assert mix == pytest.approx(target, abs=1e-3)
    assert all(v.sum() == pytest.approx(1.0) for v in q.values())


@pytest.mark.skipif(not economy.available(), reason="kjør `make fetch` først")
def test_income_kr_within_decile_bounds():
    b = economy.decile_bounds()
    d = pd.Series(np.repeat(np.arange(1, 11), 200))
    kr = economy.income_kr(d, np.random.default_rng(1))
    for dec in range(1, 10):
        x = kr[d.to_numpy() == dec]
        lo = 100_000 if dec == 1 else b[dec - 2]
        assert x.min() >= lo - 1000 and x.max() <= b[dec - 1] + 1000
    assert kr[d.to_numpy() == 10].min() >= b[8] - 1000


@pytest.mark.skipif(not built, reason="kjør `make build` først")
def test_economy_and_geo_endpoints():
    from synthpanel.api.main import app
    client = TestClient(app)
    cols = set(pd.read_parquet(config.PROCESSED_DIR / "agents.parquet").columns)
    if "inntekt_kr" not in cols:
        pytest.skip("husholdningslaget er ikke bygget")
    e = client.get("/population/economy", params={"fylke": "03"}).json()
    assert e["inntekt"]["median_befolkning"] > 300_000
    g = client.get("/population/geo", params={"utdanning": "uh_lang", "fordeling": "husholdning",
                                               "snitt": "inntekt_kr"}).json()
    assert len(g["kommuner"]) > 300
    tot = sum(r["personer"] for r in g["kommuner"])
    assert tot == pytest.approx(g["personer"], rel=1e-3)
    oslo = next(r for r in g["kommuner"] if r["kode"] == "0301")
    assert oslo["lift"] > 1.3                        # lang høyere utdanning er overrepresentert i Oslo
    assert sum(oslo["kategorier"].values()) == pytest.approx(1.0, abs=0.02)
    assert client.get("/population/geo", params={"snitt": "passord"}).status_code == 422
