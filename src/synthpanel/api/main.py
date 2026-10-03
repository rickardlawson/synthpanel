"""Synthpanel API.

Speiler konvensjonene i Signalist-API-et (FastAPI, JSON, /health, /lookups,
/docs), slik at panelet kan kobles på som egen tjeneste senere.

Start lokalt:  uvicorn synthpanel.api.main:app --port 8090
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated

import duckdb
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from synthpanel import __version__, config

app = FastAPI(
    title="Synthpanel",
    version=__version__,
    description=(
        "Syntetisk panel av den voksne norske befolkningen (18+). "
        "L0: befolkningsramme kalibrert mot SSB. Alle tall er vektede estimater."
    ),
)

# Dimensjoner som kan filtreres og grupperes på (hviteliste – brukes i SQL).
DIMENSIONS = {
    "kjonn": "Kjønn",
    "aldersband": "Aldersgruppe",
    "fylke": "Fylkesnummer (SSB 2024)",
    "fylke_navn": "Fylke",
    "kommune": "Kommunenummer",
    "kommune_navn": "Kommune",
    "sentralitet": "SSB sentralitetsklasse 01 (mest sentral) – 06",
    "utdanning": "Høyeste fullførte utdanning",
    "bakgrunn": "Egen/foreldres landbakgrunn (verdensdel) eller norsk",
    "innvkat": "Innvandringskategori: innvandrer, norskfødt med innvandrerforeldre, øvrige",
    "arbeidsstatus": "Hovedstatus: sysselsatt, student, pensjonist, aap_ufor, arbeidsledig, tiltak, annet",
    "husholdning": "Husholdningstype personen bor i",
    "lavinntekt": "Bor i husholdning med lavinntekt (EU-skala 60 %): ja/nei",
    "inntektsdesil": "Husholdningens inntekt etter skatt, nasjonal desil 1 (lavest) – 10 (høyest)",
}


class Filters(BaseModel):
    """Filtre kan gjentas, f.eks. ?fylke=03&fylke=32. Innen ett filter betyr flere verdier «eller»."""
    kjonn: list[str] = []
    aldersband: list[str] = []
    fylke: list[str] = []
    kommune: list[str] = []
    sentralitet: list[str] = []
    utdanning: list[str] = []
    bakgrunn: list[str] = []
    innvkat: list[str] = []
    arbeidsstatus: list[str] = []
    husholdning: list[str] = []
    lavinntekt: list[str] = []
    inntektsdesil: list[str] = []


class BreakdownParams(Filters):
    by: list[str] = Field(["kjonn"], description="Én eller to dimensjoner å gruppere på")
    limit: int = 500


@lru_cache
def _db() -> duckdb.DuckDBPyConnection:
    path = config.PROCESSED_DIR / "agents.parquet"
    if not path.exists():
        raise RuntimeError("Fant ikke agents.parquet – kjør `make build` først.")
    con = duckdb.connect()
    con.execute(f"CREATE VIEW agents AS SELECT * FROM read_parquet('{path.as_posix()}')")
    return con


def _con() -> duckdb.DuckDBPyConnection:
    """Egen cursor per kall. En DuckDB-tilkobling er ikke trådsikker, og FastAPI
    kjører synkrone endepunkter parallelt i en trådpool."""
    return _db().cursor()


def _where(f: Filters) -> tuple[str, list]:
    clauses, params = [], []
    for col, values in f.model_dump(include=set(Filters.model_fields)).items():
        if values:
            assert col in DIMENSIONS  # kolonnenavn kommer fra modellen, aldri fra bruker
            clauses.append(f"CAST({col} AS VARCHAR) IN ({', '.join('?' for _ in values)})")
            params.extend(values)
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    """Kilder, perioder og kalibreringsrapport for gjeldende bygg."""
    out = {}
    for name, path in [("sources", config.RAW_DIR / "manifest.json"),
                       ("build", config.PROCESSED_DIR / "build_report.json")]:
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


@app.get("/lookups/{dimension}")
def lookups(dimension: str):
    """Gyldige verdier for en dimensjon (til nedtrekkslister og filtre)."""
    if dimension not in DIMENSIONS:
        raise HTTPException(404, f"Ukjent dimensjon. Gyldige: {sorted(DIMENSIONS)}")
    rows = _con().execute(f"SELECT DISTINCT {dimension} FROM agents ORDER BY 1").fetchall()
    return [r[0] for r in rows]


@app.get("/population/size")
def size(f: Annotated[Filters, Query()]):
    """Hvor mange voksne passer beskrivelsen?"""
    where, params = _where(f)
    con = _con()
    total = con.execute("SELECT SUM(vekt) FROM agents").fetchone()[0]
    persons, n = con.execute(f"SELECT COALESCE(SUM(vekt),0), COUNT(*) FROM agents {where}", params).fetchone()
    return {
        "personer": round(persons),
        "andel_av_voksne": persons / total,
        "agenter": n,
        "presisjon": _precision_note(n),
    }


@app.get("/population/breakdown")
def breakdown(f: Annotated[BreakdownParams, Query()]):
    """Fordeling av (filtrert) befolkning etter én eller to dimensjoner."""
    by, limit = f.by, f.limit
    bad = [b for b in by if b not in DIMENSIONS]
    if bad or not 1 <= len(by) <= 2:
        raise HTTPException(422, f"`by` må være 1–2 av {sorted(DIMENSIONS)}")
    where, params = _where(f)
    cols = ", ".join(by)
    rows = _con().execute(
        f"""
        SELECT {cols}, SUM(vekt) AS personer, COUNT(*) AS agenter,
               SUM(vekt) / SUM(SUM(vekt)) OVER () AS andel
        FROM agents {where}
        GROUP BY {cols} ORDER BY personer DESC LIMIT ?
        """,
        params + [limit],
    ).fetchall()
    return [
        {**dict(zip(by, r[: len(by)])), "personer": round(r[-3]), "agenter": r[-2], "andel": r[-1],
         "presisjon": _precision_note(r[-2])}
        for r in rows
    ]


def _precision_note(n_agents: int) -> str:
    if n_agents >= 400:
        return "god"
    if n_agents >= 100:
        return "moderat"
    return "lav – for få agenter til å si noe sikkert"


# ---------------------------------------------------------------------------
# L4 – Scenariomotor (kontrakt definert, ikke implementert ennå)
# ---------------------------------------------------------------------------
class EstimateRequest(BaseModel):
    entity: str = Field(..., examples=["Tine"])
    entity_type: str = Field(..., pattern="^(brand|debate|person)$", examples=["brand"])
    stimulus: str = Field(..., examples=["Tine lanserer plantebasert yoghurt under hovedmerket."])
    question: str | None = Field(None, examples=["Hvor sannsynlig er det at du vil prøve produktet?"])
    segment_by: list[str] = Field(default=["arketype"], examples=[["arketype", "fylke_navn"]])


@app.post("/estimate", status_code=501)
def estimate(req: EstimateRequest):
    """Scenarioestimat per segment. Kommer i L4 – her ligger kontrakten."""
    raise HTTPException(501, "Scenariomotoren (L4) er ikke bygget ennå.")


# ---------------------------------------------------------------------------
# Frontend: befolkningsutforskeren på http://localhost:8090/
# Montert sist, så API-rutene over har forrang.
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
