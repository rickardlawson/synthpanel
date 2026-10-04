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


def test_dimensions_config_matches_data():
    cfg = client.get("/dimensions").json()
    assert [t["id"] for t in cfg["tiers"]] == ["hygiene", "motivasjon", "verdi"]
    assert all(s["tier"] in ("hygiene", "motivasjon", "verdi") for s in cfg["sections"])
    secs = cfg["sections"]
    keys = [d["key"] for s in secs for d in s["dims"]]
    assert {"kjonn", "fylke", "parti_2025"} <= set(keys)
    fylke = next(d for s in secs for d in s["dims"] if d["key"] == "fylke")
    assert len(fylke["values"]) >= 15  # hentet fra data med navn


def test_places_and_profile():
    q = {"kjonn": "mann", "aldersband": ["18-19", "20-24", "25-29"]}
    pl = client.get("/population/places", params=q).json()
    assert pl["storst"][0]["navn"] == "Oslo"
    assert all(r["agenter"] >= 25 for r in pl["tettest"])
    pr = client.get("/population/profile", params=q).json()
    assert pr["over"] and all(r["lift"] >= 1.2 for r in pr["over"])
    assert not any(r["dim"] in ("kjonn", "aldersband") for r in pr["over"] + pr["under"])


def test_personas_cover_segment_and_respect_filters():
    q = {"kjonn": "kvinne", "aldersband": ["67-79"]}
    d = client.get("/population/personas", params=q).json()
    ps = d["personas"]
    assert 5 <= len(ps) <= 10
    assert sum(p["andel"] for p in ps) == pytest.approx(1.0)
    assert [p["andel"] for p in ps] == sorted((p["andel"] for p in ps), reverse=True)
    assert all(p["kjonn"] == "kvinne" and 67 <= p["alder"] <= 79 for p in ps)
    assert len({p["navn"] for p in ps}) == len(ps)
    assert set(ps[0]["profil"]) == {"hygiene", "motivasjon", "verdi"}
    from synthpanel.api.main import _personas_cached
    _personas_cached.cache_clear()
    again = client.get("/population/personas", params=q).json()["personas"]
    assert [p["id"] for p in again] == [p["id"] for p in ps]  # samme utvalg -> samme personas


def test_personas_small_segment_is_refused():
    d = client.get("/population/personas", params={"fylke": "56", "parti_2025": "MDG", "aldersband": "80+"}).json()
    assert d["personas"] == [] and d["melding"]


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


# --- Personaene svarer (språkmodellen byttes ut med en attrapp) ----------------
def _fake_call(system, messages, max_tokens=500, tools=None, tool_choice=None):
    assert "Du er " in system  # briefen er med
    if tools:
        return {"content": [{"type": "tool_use", "name": "reaksjon", "input": {
            "holdning": 1 if "kvinne" in system else -1, "sitat": "Hm.", "treffer": "", "skurrer": "", "handling": "ingen"}}]}
    return {"content": [{"type": "text", "text": f"Svar på: {messages[-1]['content']}"}]}


def test_personas_respond_and_chat(monkeypatch):
    from synthpanel.personas import voice
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(voice, "_call", _fake_call)
    f = {"aldersband": ["30-39"]}
    r = client.post("/personas/respond", json={"filters": f, "modus": "spørsmål", "tekst": "Hei?"}).json()
    assert len(r["svar"]) >= 5 and all(s["svar"] == "Svar på: Hei?" for s in r["svar"])
    b = client.post("/personas/respond", json={"filters": f, "modus": "budskap", "tekst": "Kjøp melk"}).json()
    sm = b["sammendrag"]
    assert -1 <= sm["vektet_holdning"] <= 1 and sm["andel_positive"] + sm["andel_negative"] == pytest.approx(1)
    pid = r["svar"][0]["id"]
    c = client.post("/personas/chat", json={"filters": f, "id": pid,
                                            "meldinger": [{"role": "user", "content": "Hvem er du?"}]}).json()
    assert c["svar"] == "Svar på: Hvem er du?"


def test_personas_respond_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/personas/respond", json={"filters": {}, "modus": "spørsmål", "tekst": "Hei?"})
    assert r.status_code == 503
    assert client.get("/personas/status").json()["språkmodell"] is False
