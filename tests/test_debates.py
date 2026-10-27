"""Debattkoblingen (krever ikke bygget populasjon – bruker små kunstige agenter)."""
import json

import numpy as np
import pandas as pd
import pytest

from synthpanel import config
from synthpanel.debates.model import Debate, exposure, pull, react


def _agents(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "vekt": np.full(n, 100.0),
        "fylke_navn": rng.choice(["Oslo - Oslove", "Finnmark - Finnmárku - Finmarkku"], n),
        "alder": rng.integers(18, 90, n),
        "eierstatus": rng.choice(["selveier", "leier"], n),
        "lavinntekt": rng.choice(["ja", "nei"], n, p=[0.1, 0.9]),
        "utdanning": rng.choice(["grunnskole", "videregaende", "uh_kort", "uh_lang"], n),
        "score_fellesskap": rng.normal(size=n), "score_trygghet": rng.normal(size=n),
        "score_apenhet": rng.normal(size=n), "score_tillit": rng.normal(size=n),
        "politisk_interesse": rng.choice(["høy", "lav"], n),
    })


def _debate(arguments, articles, total=None):
    return Debate("x", "Test", {"-10": "mot", "+10": "for"}, arguments, articles,
                  article_total=total or len(articles))


def test_local_article_reaches_only_its_county():
    from synthpanel.debates.model import Article
    a = _agents()
    e = exposure(a, _debate({}, [Article("1", "Finnmark", "Lokale medier", 20000)]))
    finnmark = a["fylke_navn"].str.startswith("Finnmark").to_numpy()
    assert e[~finnmark].max() == 0
    assert e[finnmark].min() > 0


def test_political_interest_raises_exposure():
    from synthpanel.debates.model import Article
    a = _agents()
    e = exposure(a, _debate({}, [Article("1", None, "Nasjonale medier", 300000)]))
    hoy = (a["politisk_interesse"] == "høy").to_numpy()
    assert e[hoy].mean() > e[~hoy].mean()


def test_fairness_argument_pulls_egalitarians_hardest():
    a = _agents()
    p = pull(a, _debate({"Fairness / equality": 8.0}, []))
    assert (p > 0).all() and (p <= 1).all()
    top = a["score_fellesskap"] > a["score_fellesskap"].quantile(0.75)
    bottom = a["score_fellesskap"] < a["score_fellesskap"].quantile(0.25)
    assert p[top.to_numpy()].mean() > p[bottom.to_numpy()].mean()


def test_opposing_arguments_split_by_values():
    """Rettferdighet trekker mot +, orden mot −: de med høy bevaring ender på minus-siden."""
    a = _agents()
    p = pull(a, _debate({"Fairness / equality": 6.0, "Order / security / authority": -6.0}, []))
    assert (p > 0).any() and (p < 0).any()
    corr = np.corrcoef(p, a["score_trygghet"] - a["score_fellesskap"])[0, 1]
    assert corr < -0.5


def test_react_index_and_samples_parse():
    raw = json.loads((config.CONFIG_DIR / "debates_samples.json").read_text(encoding="utf-8"))
    debates = [Debate.from_signalist(d) for d in raw["debatter"]]
    assert len(debates) == 3 and all(len(d.arguments) == 8 for d in debates)
    out = react(_agents(), debates[1], ["fylke_navn"])
    assert set(out.columns) >= {"andel_eksponert", "snitt_dragning", "indeks_dragning", "agenter"}
    assert (out["andel_eksponert"].between(0, 1)).all()
    w = out["eksponerte_personer"]
    assert np.average(out["indeks_dragning"], weights=w) == pytest.approx(100, rel=0.01)


@pytest.mark.skipif(not (config.PROCESSED_DIR / "agents.parquet").exists(), reason="kjør `make build` først")
def test_debate_api():
    from fastapi.testclient import TestClient
    from synthpanel.api.main import app
    client = TestClient(app)
    samples = client.get("/debates/samples").json()
    assert len(samples) == 3
    r = client.post("/debates/react", json={"claim_id": samples[2]["claim_id"], "by": ["fylke_navn"]})
    assert r.status_code == 200
    body = r.json()
    assert body["status"].startswith("simulering")
    assert all("presisjon" in g for g in body["grupper"])
    assert client.post("/debates/react", json={"claim_id": "finnes-ikke"}).status_code == 404
    assert client.post("/debates/react", json={"claim_id": samples[0]["claim_id"], "by": ["drop table"]}).status_code == 422
