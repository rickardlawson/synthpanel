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

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, create_model

from synthpanel import __version__, config

app = FastAPI(
    title="Synthpanel",
    version=__version__,
    description=(
        "Syntetisk panel av den voksne norske befolkningen (18+). "
        "L0: befolkningsramme kalibrert mot SSB. Alle tall er vektede estimater."
    ),
)

app.add_middleware(GZipMiddleware, minimum_size=2000)

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
    "eierstatus": "Selveier, andelseier eller leier (husholdningens bolig)",
    "boligtype": "Enebolig, tomannsbolig, rekkehus/småhus, blokk, annen",
    "verdi_apenhet": "Verdier (ESS): åpenhet for endring – selvstendighet, stimulans, nytelse (lav/middels/høy)",
    "verdi_trygghet": "Verdier (ESS): bevaring – trygghet, regler, tradisjon (lav/middels/høy)",
    "verdi_selvhevdelse": "Verdier (ESS): selvhevdelse – makt, suksess (lav/middels/høy)",
    "verdi_fellesskap": "Verdier (ESS): selvoverskridelse – omsorg, likeverd, natur (lav/middels/høy)",
    "tillit": "Tillit til Storting, rettsvesen, politi og politikere (ESS, lav/middels/høy)",
    "risikovilje": "«Søker eventyr og tar sjanser» (ESS, lav/middels/høy)",
    "politisk_sted": "Plassering på venstre–høyre-skala (ESS)",
    "religiositet": "Hvor religiøs (ESS, lav/middels/høy)",
    "politisk_interesse": "Interesse for politikk (ESS, høy/lav)",
    "klimabekymring": "Bekymring for klimaendringer (ESS, lav/middels/høy)",
    "stemmerett": "Stemmerett ved stortingsvalg (ja/nei)",
    "stemte_2025": "Stemte ved stortingsvalget 2025 (ja/nei/ikke_stemmerett)",
    "parti_2025": "Parti ved stortingsvalget 2025, eller stemte_ikke / ikke_stemmerett",
    "partisympati": "Nærmeste parti (også for dem som ikke stemte)",
}
from synthpanel.media.layer import COLUMNS as _MEDIA  # noqa: E402
DIMENSIONS.update({c: f"Medie/netthandel: {lab} (ja/nei)" for c, lab in _MEDIA.items()})
from synthpanel.leisure.layer import COLUMNS as _LEISURE  # noqa: E402
DIMENSIONS.update({c: f"Trening/friluftsliv siste 12 mnd: {lab} (ja/nei)" for c, lab in _LEISURE.items()})
DIMENSIONS["treningsfrekvens"] = "Hvor ofte de trener: ukentlig, av_og_til (månedlig), sjelden"
from synthpanel.household import economy as _eco  # noqa: E402
DIMENSIONS.update({
    "inntekt_gruppe": "Husholdningens inntekt etter skatt (kr): u300, 300-500, 500-750, 750-1000, 1000-1500, o1500 (tusen kr)",
    "fritidsforbruk": "Husholdningens forbruk på fritid, sport og kultur: lav/middels/høy (tredeler blant husholdningene)",
    "reiseforbruk": "Husholdningens forbruk på reiser: lav/middels/høy (tredeler blant husholdningene)",
    "forbruksniva": "Husholdningens samlede forbruk: lav/middels/høy (tredeler blant husholdningene)",
    **{c: f"Kjøp: {lab} (ja/nei)" for c, lab in _eco.PURCHASE_COLS.items()},
})
DIMENSIONS.update({
    "arketype": "Arketype (Jungs tolv, fra Schwartz-verdiene) – se /archetypes",
    "verdikart": "Felt i verdikartet: tradisjonell/moderne × materialist/idealist",
    "samfunnsrolle": "Samfunnsrolle (kapital, tillit, utrygghet)",
    "resiliens": "Kriseresiliens (tillit, nettverk, økonomisk buffer)",
})


# Filtermodellen genereres fra DIMENSIONS, så nye dimensjoner bare trenger én linje over.
Filters = create_model(
    "Filters",
    __doc__="Filtre kan gjentas, f.eks. ?fylke=03&fylke=32. Innen ett filter betyr flere verdier «eller».",
    **{k: (list[str], Field(default_factory=list, description=v)) for k, v in DIMENSIONS.items()},
)


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


def _cond(f: Filters) -> tuple[str, list, list[str]]:
    """SQL-betingelse for filtrene (uten WHERE), parametere og hvilke kolonner som er filtrert."""
    clauses, params, cols = [], [], []
    for col, values in f.model_dump(include=set(Filters.model_fields)).items():
        if values:
            assert col in DIMENSIONS  # kolonnenavn kommer fra modellen, aldri fra bruker
            clauses.append(f"CAST({col} AS VARCHAR) IN ({', '.join('?' for _ in values)})")
            params.extend(values)
            cols.append(col)
    return (" AND ".join(clauses) if clauses else "TRUE"), params, cols


def _where(f: Filters) -> tuple[str, list]:
    cond, params, cols = _cond(f)
    return ("WHERE " + cond) if cols else "", params


class PlacesParams(Filters):
    level: str = Field("kommune", pattern="^(kommune|fylke)$", description="kommune eller fylke")
    limit: int = 8
    min_agenter: int = Field(25, description="Minste antall agenter i segmentet for å regnes med i «tettest»")


class ProfileParams(Filters):
    limit: int = 8
    min_andel: float = Field(0.08, description="Minste andel (av segmentet eller befolkningen) for å vises")


@lru_cache
def dimension_config() -> dict:
    """dimensions.yaml, med verdilister fra archetypes.yaml der det står «archetypes:…»."""
    import copy
    cfg = copy.deepcopy(config.load("dimensions"))
    arch = config.load("archetypes")
    for t in cfg["tiers"]:
        for sec in t["sections"]:
            for d in sec["dims"]:
                v = d.get("values")
                if isinstance(v, str) and v.startswith("archetypes:"):
                    node = arch
                    for part in v.split(":", 1)[1].split("."):
                        node = node[part]
                    d["values"] = {k: x["navn"] for k, x in node.items()}
    return cfg


def config_sections() -> list[dict]:
    """Alle seksjoner i hierarkiets rekkefølge, hver merket med laget (tier) den hører til."""
    return [{**sec, "tier": t["id"]} for t in dimension_config()["tiers"] for sec in t["sections"]]


@app.get("/dimensions")
def dimensions():
    """Verdihierarkiet (lag → seksjoner → dimensjoner) og etiketter som grensesnittet bygges fra.
    `sections` er den samme lista flatet ut, med `tier` på hver seksjon."""
    cfg = dimension_config()
    out = {"tiers": [{k: v for k, v in t.items() if k != "sections"} for t in cfg["tiers"]],
           "sections": [], "binary_groups": cfg.get("binary_groups", {})}
    cols = set(_con().execute("SELECT * FROM agents LIMIT 0").df().columns)
    for sec in config_sections():
        dims = []
        for d in sec["dims"]:
            if d["key"] not in cols:
                continue  # laget er ikke bygget (f.eks. uten ESS-data)
            d = dict(d)
            if d.get("values") == "from_data":
                rows = _con().execute(
                    f"SELECT DISTINCT {d['key']}, {d['label_column']} FROM agents ORDER BY 2").fetchall()
                d["values"] = {k: v.split(" - ")[0] for k, v in rows}
            dims.append(d)
        if dims:
            out["sections"].append({**{k: v for k, v in sec.items() if k != "dims"}, "dims": dims})
    return out


@app.get("/population/places")
def places(f: Annotated[PlacesParams, Query()]):
    """Hvor bor segmentet? «Størst» = flest personer; «tettest» = høyest lift
    (andel av stedets voksne delt på andelen i hele landet)."""
    cond, params, _ = _cond(f)
    geo, name = ("kommune", "kommune_navn") if f.level == "kommune" else ("fylke", "fylke_navn")
    con = _con()
    total, seg_total = con.execute(f"SELECT SUM(vekt), SUM(vekt) FILTER (WHERE {cond}) FROM agents", params).fetchone()
    seg_total = seg_total or 0.0
    base = seg_total / total if total else 0.0
    rows = con.execute(
        f"""
        SELECT {geo}, ANY_VALUE({name}),
               COALESCE(SUM(vekt) FILTER (WHERE {cond}), 0) AS personer,
               COUNT(*) FILTER (WHERE {cond}) AS agenter,
               SUM(vekt) AS voksne
        FROM agents GROUP BY 1
        """, params + params).fetchall()
    items = []
    for kode, navn, pers, n, voksne in rows:
        andel = pers / voksne if voksne else 0.0
        items.append({"kode": kode, "navn": navn.split(" - ")[0], "personer": round(pers), "agenter": n,
                      "andel_av_stedet": andel, "lift": (andel / base) if base else None})
    storst = sorted(items, key=lambda r: -r["personer"])[: f.limit]
    tettest = sorted([r for r in items if r["agenter"] >= f.min_agenter], key=lambda r: -(r["lift"] or 0))[: f.limit]
    return {"segment_personer": round(seg_total), "andel_av_voksne": base, "storst": storst, "tettest": tettest,
            "steder_med_nok_data": sum(1 for r in items if r["agenter"] >= f.min_agenter)}


@app.get("/population/profile")
def profile(f: Annotated[ProfileParams, Query()]):
    """Hva kjennetegner segmentet? Kategorier som er klart over- eller underrepresentert
    sammenlignet med hele befolkningen (lift = andel i segmentet / andel i befolkningen)."""
    cond, params, filtered = _cond(f)
    con = _con()
    cols = set(con.execute("SELECT * FROM agents LIMIT 0").df().columns)
    seg_total, total, n = con.execute(
        f"SELECT SUM(vekt) FILTER (WHERE {cond}), SUM(vekt), COUNT(*) FILTER (WHERE {cond}) FROM agents",
        params + params).fetchone()
    if not seg_total:
        return {"over": [], "under": [], "agenter": 0}
    over, under = [], []
    for sec in config_sections():
        for d in sec["dims"]:
            key = d["key"]
            geo_filtered = bool({"fylke", "kommune", "sentralitet"} & set(filtered))
            if key in filtered or key not in cols or key == "kommune" or (geo_filtered and key in ("fylke", "sentralitet")):
                continue
            rows = con.execute(
                f"""SELECT CAST({key} AS VARCHAR), COALESCE(SUM(vekt) FILTER (WHERE {cond}), 0), SUM(vekt)
                    FROM agents GROUP BY 1""", params).fetchall()
            for val, s, p in rows:
                if val in ("ukjent", None) or (d.get("binary") and val != "ja"):
                    continue
                a_seg, a_pop = s / seg_total, p / total
                lift = a_seg / a_pop if a_pop else 0
                item = {"dim": key, "seksjon": sec["id"], "lag": sec["tier"], "verdi": val, "andel_segment": a_seg,
                        "andel_befolkning": a_pop, "lift": lift, "personer": round(s)}
                if a_seg >= f.min_andel and lift >= 1.2:
                    over.append(item)
                elif a_pop >= f.min_andel and lift <= 0.8:
                    under.append(item)
    over.sort(key=lambda r: -r["lift"])
    under.sort(key=lambda r: r["lift"])
    return {"over": over[: f.limit], "under": under[: f.limit], "agenter": n,
            "presisjon": _precision_note(n)}


class PersonaParams(Filters):
    n: int = Field(10, ge=1, le=12, description="Antall personas (største grupperinger)")


def _labels() -> dict:
    out = {"_titles": {}, "_phrases": {}}
    for sec in config_sections():
        for d in sec["dims"]:
            out["_titles"][d["key"]] = d["title"]
            if d.get("phrase"):
                out["_phrases"][d["key"]] = d["phrase"]
            if isinstance(d.get("values"), dict):
                out[d["key"]] = d["values"]
    return out


@lru_cache(maxsize=256)
def _personas_cached(key: str, n: int, cond: str, params: tuple, filtered: tuple, portraits_stamp: float) -> dict:
    from synthpanel.personas import engine
    seg = _con().execute(f"SELECT * FROM agents WHERE {cond}", list(params)).df()
    tiers = {d["key"]: s["tier"] for s in config_sections() for d in s["dims"]}
    return engine.build(seg, set(filtered), _labels(), tiers, key, k=n)


@app.get("/population/personas")
def personas(f: Annotated[PersonaParams, Query()]):
    """«Ti på gata»: de største grupperingene i utvalget, hver vist som én representativ
    syntetisk person med navn, AI-generert portrett og profil etter verdihierarkiet.
    Grupperingene dekker til sammen hele utvalget (`andel` summerer til 1)."""
    return _get_personas(f)


def _get_personas(f: PersonaParams) -> dict:
    cond, params, filtered = _cond(f)
    key = json.dumps(sorted((c, sorted(map(str, getattr(f, c)))) for c in filtered), ensure_ascii=False)
    from synthpanel.personas import portraits
    stamp = portraits.DIR.stat().st_mtime if portraits.DIR.exists() else 0.0  # nye portretter -> nytt valg
    return _personas_cached(key, f.n, cond, tuple(params), tuple(sorted(filtered)), stamp)


# ---------------------------------------------------------------------------
# L4 v0 – personaene svarer (krever ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------
class PanelRequest(BaseModel):
    filters: dict[str, list[str]] = Field(default_factory=dict, examples=[{"kjonn": ["kvinne"], "aldersband": ["67-79"]}])
    ids: list[str] | None = Field(None, description="Hvilke personas (agent-id). Tomt = alle i galleriet")
    modus: str = Field("spørsmål", pattern="^(spørsmål|budskap)$")
    tekst: str = Field(..., min_length=2, max_length=2000, examples=["Hva tenker du om plantebasert yoghurt fra Tine?"])


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    filters: dict[str, list[str]] = Field(default_factory=dict)
    id: str
    meldinger: list[ChatMessage] = Field(..., min_length=1, max_length=40)


def _panel(filters: dict[str, list[str]]) -> list[dict]:
    bad = [k for k in filters if k not in DIMENSIONS]
    if bad:
        raise HTTPException(422, f"Ukjente filtre: {bad}")
    return _get_personas(PersonaParams(**filters))["personas"]


def _voice_errors(fn):
    from synthpanel.personas import voice
    try:
        return fn()
    except voice.NoApiKey as e:
        raise HTTPException(503, "Språkmodell er ikke satt opp: legg ANTHROPIC_API_KEY i .env og start serveren på nytt.") from e
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e


@app.get("/personas/status")
def personas_status():
    """Om personaene kan svare (språkmodell konfigurert)."""
    import os
    from synthpanel.personas import voice
    return {"språkmodell": bool(os.environ.get("ANTHROPIC_API_KEY")), "modell": voice._model()}


@app.post("/personas/respond")
def personas_respond(req: PanelRequest):
    """Still alle personaene samme spørsmål (`modus=spørsmål`) eller test et budskap
    (`modus=budskap`: holdning −2…+2, sitat, hva treffer/skurrer, sannsynlig handling).
    Svarene er AI-simuleringer med personaens profil som grunnlag – ikke målinger."""
    from synthpanel.personas import voice
    ps = _panel(req.filters)
    if req.ids:
        ps = [p for p in ps if p["id"] in set(req.ids)]
    if not ps:
        raise HTTPException(404, "Ingen personas for dette utvalget.")
    fn = voice.ask if req.modus == "spørsmål" else voice.react
    svar = _voice_errors(lambda: voice.run_all(ps, fn, req.tekst))
    out = {"modus": req.modus, "modell": voice._model(), "svar": svar}
    if req.modus == "budskap":
        out["sammendrag"] = voice.summarize(ps, svar)
    return out


@app.post("/personas/chat")
def personas_chat(req: ChatRequest):
    """Samtale med én persona. Send hele samtalen hittil i `meldinger`."""
    from synthpanel.personas import voice
    p = next((p for p in _panel(req.filters) if p["id"] == req.id), None)
    if p is None:
        raise HTTPException(404, "Fant ikke personaen i dette utvalget.")
    return {"id": p["id"], "svar": _voice_errors(lambda: voice.chat(p, [m.model_dump() for m in req.meldinger]))}


@app.get("/archetypes")
def archetypes_config():
    """Definisjonene bak arketypelaget (navn, korte beskrivelser, profiler)."""
    return config.load("archetypes")


@app.get("/population/archetypes")
def population_archetypes(f: Annotated[Filters, Query()]):
    """Arketypehjul, verdikart, samfunnsroller og kriseresiliens for utvalget,
    med andel i utvalget, andel i befolkningen og lift. Verdikartet har i tillegg
    et 16×16-rutenett (andel av utvalget/befolkningen per rute, akser i standardavvik)."""
    cond, params, _ = _cond(f)
    con = _con()
    cfg = config.load("archetypes")
    cols = set(con.execute("SELECT * FROM agents LIMIT 0").df().columns)
    if "arketype" not in cols:
        raise HTTPException(404, "Arketypelaget er ikke bygget (krever ESS-data).")
    seg_total, total, n = con.execute(
        f"SELECT SUM(vekt) FILTER (WHERE {cond}), SUM(vekt), COUNT(*) FILTER (WHERE {cond}) FROM agents",
        params + params).fetchone()
    seg_total = seg_total or 0.0

    def shares(col: str, spec: dict) -> list[dict]:
        rows = dict((k, (s or 0.0, p)) for k, s, p in con.execute(
            f"SELECT {col}, SUM(vekt) FILTER (WHERE {cond}), SUM(vekt) FROM agents GROUP BY 1", params).fetchall())
        out = []
        for k, meta in spec.items():
            s, p = rows.get(k, (0.0, 0.0))
            a_s, a_p = (s / seg_total if seg_total else 0.0), p / total
            out.append({"kode": k, **{x: meta[x] for x in ("navn", "kort", "motiv") if x in meta},
                        "andel_segment": a_s, "andel_befolkning": a_p, "lift": a_s / a_p if a_p else None})
        return out

    N, E = 16, 2.5
    def grid(where: str, prm: list) -> list[list[float]]:
        rows = con.execute(f"""
            SELECT LEAST({N - 1}, GREATEST(0, CAST(FLOOR((verdikart_x + {E}) / {2 * E / N}) AS INT))) AS gx,
                   LEAST({N - 1}, GREATEST(0, CAST(FLOOR((verdikart_y + {E}) / {2 * E / N}) AS INT))) AS gy,
                   SUM(vekt) FROM agents WHERE verdikart_x IS NOT NULL AND {where} GROUP BY 1, 2""", prm).fetchall()
        g = [[0.0] * N for _ in range(N)]
        tot = sum(r[2] for r in rows) or 1.0
        for gx, gy, v in rows:
            g[gy][gx] = v / tot
        return g
    mean = con.execute(f"SELECT SUM(verdikart_x * vekt) / SUM(vekt), SUM(verdikart_y * vekt) / SUM(vekt) "
                       f"FROM agents WHERE verdikart_x IS NOT NULL AND {cond}", params).fetchone()
    med = con.execute("SELECT median(verdikart_x), median(verdikart_y) FROM agents WHERE verdikart_x IS NOT NULL").fetchone()
    return {
        "agenter": n, "presisjon": _precision_note(n),
        "arketyper": shares("arketype", cfg["arketyper"]),
        "verdikart": {"felt": shares("verdikart", cfg["verdikart"]["felt"]),
                      "rutenett": {"n": N, "utstrekning": E, "segment": grid(cond, params), "befolkning": grid("TRUE", [])},
                      "snitt_segment": {"x": mean[0], "y": mean[1]}, "median": {"x": med[0], "y": med[1]}},
        "samfunnsroller": shares("samfunnsrolle", cfg["samfunnsroller"]),
        "resiliens": shares("resiliens", cfg["resiliens"]),
    }


# ---------------------------------------------------------------------------
# Husholdningens økonomi og kart
# ---------------------------------------------------------------------------
@app.get("/population/economy")
def population_economy(f: Annotated[Filters, Query()]):
    """Husholdningens økonomi for utvalget mot hele befolkningen: inntekt etter skatt,
    forventet årlig forbruk per gruppe (kr, prisjustert) og kjøpsrater.
    Alle tall gjelder husholdningen personene bor i, og er snitt over personer."""
    cond, params, _ = _cond(f)
    con = _con()
    cols = set(con.execute("SELECT * FROM agents LIMIT 0").df().columns)
    if "inntekt_kr" not in cols:
        raise HTTPException(404, "Husholdningslaget er ikke bygget – kjør `make fetch` og `make build`.")
    n = con.execute(f"SELECT COUNT(*) FROM agents WHERE {cond}", params).fetchone()[0]

    def avg(expr: str) -> tuple[float | None, float | None]:
        seg, pop = con.execute(
            f"SELECT SUM(({expr}) * vekt) FILTER (WHERE {cond}) / SUM(vekt) FILTER (WHERE {cond}), "
            f"SUM(({expr}) * vekt) / SUM(vekt) FROM agents", params + params).fetchone()
        return seg, pop

    def item(key: str, title: str, expr: str, unit: str) -> dict:
        seg, pop = avg(expr)
        return {"key": key, "title": title, "enhet": unit, "segment": seg, "befolkning": pop,
                "lift": (seg / pop) if seg is not None and pop else None}

    med_seg, med_pop = con.execute(
        f"SELECT quantile_cont(inntekt_kr, 0.5) FILTER (WHERE {cond}), quantile_cont(inntekt_kr, 0.5) FROM agents",
        params).fetchone()
    spend = [item(c, lab, c, "kr") for c, (_, lab, _) in _eco.SPEND.items() if c in cols]
    kjop = [item(c, lab, f"CASE WHEN {c} = 'ja' THEN 1.0 ELSE 0.0 END", "andel")
            for c, lab in _eco.PURCHASE_COLS.items() if c in cols]
    return {"agenter": n, "presisjon": _precision_note(n),
            "inntekt": {**item("inntekt_kr", "Inntekt etter skatt (snitt)", "inntekt_kr", "kr"),
                        "median_segment": med_seg, "median_befolkning": med_pop},
            "forbruk": spend, "kjop": kjop,
            "note": "Forventet årlig utgift for husholdningen (Forbruksundersøkelsen 2022, prisjustert med KPI). "
                    "Kjøpsrater fra førstegangsregistreringer og bilparken per kommune."}


GEO_NUMERIC = {"inntekt_kr": "Husholdningsinntekt etter skatt (kr)", **{c: v[1] for c, v in _eco.SPEND.items()},
               "p_nybil": "Sannsynlighet for å kjøpe ny bil", "p_elbil": "Sannsynlighet for å ha elbil"}
SHRINK = 50   # «pseudo-agenter» fra fylket i glattingen (empirisk Bayes)


class GeoParams(Filters):
    fordeling: str | None = Field(None, description="Kategorisk dimensjon å vise fordelingen av per kommune (f.eks. arketype)")
    snitt: str | None = Field(None, description=f"Tallvariabel å vise snitt av per kommune: {sorted(GEO_NUMERIC)}")


@app.get("/population/geo")
def population_geo(f: Annotated[GeoParams, Query()]):
    """Segmentet per kommune, til kartet. For hver kommune: voksne, personer i segmentet,
    andel, lift og antall agenter – både rått og *glattet* (andelen trekkes mot fylkets
    med vekt som 50 agenter, så små kommuner ikke gir tilfeldige utslag).
    Med `fordeling` kommer andelen per kategori i segmentet og den mest overrepresenterte
    kategorien; med `snitt` kommer segmentets snitt av en tallvariabel."""
    if f.fordeling and f.fordeling not in DIMENSIONS:
        raise HTTPException(422, f"Ukjent dimensjon: {f.fordeling}")
    if f.snitt and f.snitt not in GEO_NUMERIC:
        raise HTTPException(422, f"`snitt` må være en av {sorted(GEO_NUMERIC)}")
    cond, params, _ = _cond(f)
    con = _con()
    rows = con.execute(f"""
        SELECT kommune, ANY_VALUE(kommune_navn), ANY_VALUE(fylke), SUM(vekt), COUNT(*),
               COALESCE(SUM(vekt) FILTER (WHERE {cond}), 0), COUNT(*) FILTER (WHERE {cond})
        FROM agents GROUP BY 1""", params + params).fetchall()
    df = {r[0]: {"kode": r[0], "navn": r[1].split(" - ")[0], "fylke": r[2], "voksne": r[3], "agenter_alle": r[4],
                 "personer": r[5], "agenter": r[6]} for r in rows}
    tot = sum(r["voksne"] for r in df.values())
    seg_tot = sum(r["personer"] for r in df.values())
    base = seg_tot / tot if tot else 0.0
    fy = {}
    for r in df.values():
        g = fy.setdefault(r["fylke"], [0.0, 0.0])
        g[0] += r["personer"]; g[1] += r["voksne"]

    def smooth(p: float, n: int, parent: float) -> float:
        return (n * p + SHRINK * parent) / (n + SHRINK)

    for r in df.values():
        raw = r["personer"] / r["voksne"] if r["voksne"] else 0.0
        parent = fy[r["fylke"]][0] / fy[r["fylke"]][1]
        r["andel"] = raw
        r["andel_glattet"] = smooth(raw, r["agenter_alle"], parent)
        r["lift"] = r["andel_glattet"] / base if base else None
        r["presisjon"] = _precision_note(r["agenter"])
    out = {"andel_land": base, "personer": round(seg_tot), "glatting_agenter": SHRINK}

    if f.fordeling:
        col = f.fordeling
        cats = con.execute(f"""
            SELECT kommune, CAST({col} AS VARCHAR), SUM(vekt) FILTER (WHERE {cond}), COUNT(*) FILTER (WHERE {cond})
            FROM agents GROUP BY 1, 2""", params + params).fetchall()
        land = {}
        fyc = {}
        for k, c, w, _n in cats:
            if c is None or w is None:
                continue
            land[c] = land.get(c, 0.0) + w
            fyc.setdefault(df[k]["fylke"], {}).setdefault(c, 0.0)
            fyc[df[k]["fylke"]][c] += w
        lt = sum(land.values()) or 1.0
        out["kategorier_land"] = {c: v / lt for c, v in land.items()}
        per = {}
        for k, c, w, _n in cats:
            if c is not None and w:
                per.setdefault(k, {})[c] = w
        for k, r in df.items():
            seg = per.get(k, {})
            s = sum(seg.values())
            parent = fyc.get(r["fylke"], {})
            ps = sum(parent.values()) or 1.0
            shares = {c: smooth((seg.get(c, 0.0) / s) if s else 0.0, r["agenter"], parent.get(c, 0.0) / ps)
                      for c in land}
            r["kategorier"] = {c: round(v, 4) for c, v in shares.items()}
            # «Mest overrepresentert» blant kategorier med minst 2 % i landet (små kategorier gir støy).
            cand = [c for c in land if out["kategorier_land"][c] >= 0.02] or list(land)
            top = max(cand, key=lambda c: shares[c] / out["kategorier_land"][c]) if cand else None
            r["topp"] = {"kode": top, "andel": shares[top], "lift": shares[top] / out["kategorier_land"][top]} if top else None

    if f.snitt:
        col = f.snitt
        vals = con.execute(f"""
            SELECT kommune, SUM({col} * vekt) FILTER (WHERE {cond}), SUM(vekt) FILTER (WHERE {cond} AND {col} IS NOT NULL)
            FROM agents GROUP BY 1""", params + params).fetchall()
        land_s = sum(v[1] or 0 for v in vals) / (sum(v[2] or 0 for v in vals) or 1)
        fys = {}
        for k, s, w in vals:
            g = fys.setdefault(df[k]["fylke"], [0.0, 0.0])
            g[0] += s or 0; g[1] += w or 0
        for k, s, w in vals:
            r = df[k]
            parent = fys[r["fylke"]][0] / fys[r["fylke"]][1] if fys[r["fylke"]][1] else land_s
            raw = (s / w) if w else parent
            r["snitt"] = smooth(raw, r["agenter"], parent)
        out["snitt_land"] = land_s
        out["snitt_tittel"] = GEO_NUMERIC[col]

    out["kommuner"] = list(df.values())
    return out


@lru_cache
def _geo_file(name: str) -> bytes:
    path = config.RAW_DIR / name
    if not path.exists():
        raise HTTPException(404, "Geodata mangler – kjør `python -m synthpanel.geo.fetch`.")
    return path.read_bytes()


@app.get("/geo/kommuner.geojson")
def geo_kommuner():
    """Kommunegrenser 2024 (Kartverket via robhop/fylker-og-kommuner, CC BY 4.0), forenklet."""
    return Response(_geo_file("geo_kommuner.geojson"), media_type="application/geo+json")


@app.get("/geo/fylker.geojson")
def geo_fylker():
    return Response(_geo_file("geo_fylker.geojson"), media_type="application/geo+json")


@lru_cache
def _grid_payload() -> dict:
    import pandas as pd
    path = config.RAW_DIR / "geo_grid_1km.parquet"
    if not path.exists():
        raise HTTPException(404, "Rutenettet mangler – kjør `python -m synthpanel.geo.fetch`.")
    g = pd.read_parquet(path)
    g = g[g["bosatte"] > 0]
    codes = sorted(g["kommune"].unique())
    ix = {c: i for i, c in enumerate(codes)}
    return {"kilde": "SSB, befolkning på rutenett 1 km, 1.1.2026", "kommuner": codes,
            "kolonner": ["lon", "lat", "bosatte", "kommune_idx", "snittalder"],
            "ruter": [[r.lon, r.lat, int(r.bosatte), ix[r.kommune], None if r.snittalder != r.snittalder else r.snittalder]
                      for r in g.itertuples(index=False)]}


@app.get("/geo/grid")
def geo_grid():
    """Befolkning på 1 km-rutenett (SSB), med kommunen hver rute ligger i."""
    return _grid_payload()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    """Kilder, perioder og kalibreringsrapport for gjeldende bygg."""
    out = {}
    for name, path in [("sources", config.RAW_DIR / "manifest.json"),
                       ("build", config.PROCESSED_DIR / "build_report.json"),
                       ("validation", config.PROCESSED_DIR / "validation_report.json"),
                       ("fasit_kundebarometer", config.PROCESSED_DIR / "fasit_kundebarometer.json")]:
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
