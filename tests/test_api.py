import pytest
from fastapi.testclient import TestClient

from synthpanel import config
from synthpanel.api.main import app

pytestmark = pytest.mark.skipif(
    not (config.PROCESSED_DIR / "agents.parquet").exists(), reason="kjør `make build` først"
)
client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_size_total_and_filter():
    total = client.get("/population/size").json()
    assert total["andel_av_voksne"] == pytest.approx(1.0)
    oslo = client.get("/population/size", params={"fylke": "03"}).json()
    assert 0.1 < oslo["andel_av_voksne"] < 0.15


def test_breakdown_shares_sum_to_one():
    rows = client.get("/population/breakdown", params={"by": ["sentralitet"]}).json()
    assert sum(r["andel"] for r in rows) == pytest.approx(1.0)


def test_rejects_unknown_dimension():
    assert client.get("/population/breakdown", params={"by": ["drop table"]}).status_code == 422
    assert client.get("/lookups/passord").status_code == 404


def test_estimate_contract_exists():
    r = client.post("/estimate", json={"entity": "Tine", "entity_type": "brand", "stimulus": "x"})
    assert r.status_code == 501
