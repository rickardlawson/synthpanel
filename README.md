# Synthpanel

Et syntetisk panel av den voksne norske befolkningen (18+), bygget lag for lag
på offisiell statistikk. Målet er et prediksjonsverktøy som kan svare på
*hvordan ulike deler av befolkningen trolig reagerer* på en lansering, et
budskap eller en debatt – og hvor store, hvor bosatte og hvor viktige
segmentene er.

> Status: **v0.1 – L0 befolkningsramme ferdig.** 50 751 syntetiske agenter som
> til sammen gjenskaper SSB-tallene for kjønn, alder, kommune, sentralitet,
> utdanning og landbakgrunn. Scenariomotoren (L4) har definert kontrakt, men er
> ikke bygget ennå. Se [docs/architecture.md](docs/architecture.md).

## Kom i gang

**Med Docker (server):**

```bash
git clone <repo> && cd synthpanel
docker compose up -d --build      # første oppstart henter SSB-data og bygger (~1 min)
curl localhost:8090/health
```

API-dokumentasjon med «Try it out»: `http://localhost:8090/docs`
(tjenesten lytter kun på localhost som standard – se `docker-compose.yml`).

**Lokalt (utvikling):**

```bash
make install   # pip install -e ".[dev]"
make data      # hent fra SSB + bygg populasjon
make test
make serve     # http://localhost:8090/docs
```

Oppdatere med ferske SSB-tall: `make rebuild-data` (Docker) eller `make data`.

## Eksempler

```bash
# Hvor mange voksne bor i Oslo?
curl "localhost:8090/population/size?fylke=03"

# Kvinner 25–39 i Oslo med lang høyere utdanning
curl "localhost:8090/population/size?fylke=03&kjonn=kvinne&aldersband=25-29&aldersband=30-39&utdanning=uh_lang"

# Befolkningen fordelt på sentralitet
curl "localhost:8090/population/breakdown?by=sentralitet"

# Utdanning i Oslo vs Finnmark
curl "localhost:8090/population/breakdown?by=fylke_navn&by=utdanning&fylke=03&fylke=56"
```

Hvert svar har `agenter` (hvor mange syntetiske personer tallet bygger på) og
`presisjon` (god / moderat / lav), slik at smale segmenter ikke presenteres
som sikrere enn de er.

## Struktur

```
configs/frame.yaml          kilder, aldersbånd, kodelister
src/synthpanel/
  ssb.py                    klient for SSB PxWebApi v2 og Klass
  frame/fetch.py            henter rådata  -> data/raw/*.parquet
  frame/build.py            bygger agenter -> data/processed/agents.parquet
  calibrate/raking.py       raking/IPF – vekter agenter mot kjente totaler
  api/main.py               FastAPI
tests/                      kalibrering, SSB-avstemming, API
docs/architecture.md        modellen L0–L5, prinsipper og veikart
```

## Kompatibilitet med Signalist

Bevisst samme stack som Signalist-backend: Python, FastAPI med auto-generert
`/docs`, YAML-konfig, Parquet + DuckDB, Docker. Panelet er en frittstående
tjeneste med eget API, slik at Signalist (eller andre) kan kalle det uten at
noe må skrives om. Port 8090 for å ikke kollidere med Signalist på 8080.

## Datakilder

Kun åpne data i v0.1: SSB-tabell 07459 (befolkning 1.1.2026), 08921
(utdanningsnivå 2025), 07111 (innvandrere og norskfødte med innvandrerforeldre
1.1.2026) og Klass 128 (sentralitet). Se `GET /meta` for nøyaktige perioder og
hentetidspunkt.
