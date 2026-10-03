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


def test_frontend_served_and_docs_still_work():
    r = client.get("/")
    assert r.status_code == 200 and "Synthpanel" in r.text
    assert client.get("/docs").status_code == 200


def _parallel_over_http(app_obj, n_rounds=10):
    """Start en ekte uvicorn-server og send mange samtidige kall, slik nettsiden gjør."""
    import socket
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app_obj, port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    dims = ["kjonn", "aldersband", "fylke_navn", "sentralitet", "utdanning", "bakgrunn"] * n_rounds

    def call(d):
        r = httpx.get(f"http://127.0.0.1:{port}/population/breakdown", params={"by": d}, timeout=30)
        return r.status_code == 200 and all(d in row for row in r.json())

    try:
        with ThreadPoolExecutor(12) as ex:
            return list(ex.map(call, dims))
    finally:
        server.should_exit = True


def test_parallel_requests_do_not_mix_results():
    """Røyktest for samtidige kall mot ekte server.

    Bakgrunn: en delt DuckDB-tilkobling ga blandede svar når nettsiden hentet
    seks fordelinger parallelt. Feilen viste seg i nettleser, men denne testen
    klarer ikke å gjenskape den pålitelig – den sikrer bare at samtidige kall
    gir gyldige svar med dagens kode.
    """
    assert all(_parallel_over_http(app))
