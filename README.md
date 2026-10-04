# Synthpanel

Et syntetisk panel av den voksne norske befolkningen (18+), bygget lag for lag
på offisiell statistikk. Målet er et prediksjonsverktøy som kan svare på
*hvordan ulike deler av befolkningen trolig reagerer* på en lansering, et
budskap eller en debatt – og hvor store, hvor bosatte og hvor viktige
segmentene er.

> Status: **v0.3 – befolkningsramme (L0) og verdilag (L2).** 50 751 syntetiske
> agenter som gjenskaper SSB-tallene for kjønn, alder, kommune, sentralitet,
> utdanning, landbakgrunn, innvandringskategori, hovedstatus, husholdning,
> lavinntekt, inntekt og bolig – og som har fått verdier og holdninger fra
> European Social Survey (testet mot respondenter modellen ikke har sett),
> partivalg ved stortingsvalget 2025 (kalibrert mot valgresultatet per fylke)
> og bruk av sosiale medier, strømming og netthandel.
> Scenariomotoren (L4) har definert kontrakt, men er ikke bygget ennå. Se [docs/architecture.md](docs/architecture.md).

## Kom i gang

**Med Docker (server):**

```bash
git clone <repo> && cd synthpanel
docker compose up -d --build      # første oppstart henter SSB-data og bygger (~1 min)
curl localhost:8090/health
```

Befolkningsutforsker: `http://localhost:8090/` – bygg en målgruppe til venstre
(søk, eller kryss av i kategoriene Hvem · Livssituasjon · Verdier · Politikk ·
Medier) og se til høyre hvor stor den er, hvor den bor (størst og tettest, med
lift) og hva som kjennetegner den. Kategoriene og etikettene styres fra
`configs/dimensions.yaml`.

API-dokumentasjon med «Try it out»: `http://localhost:8090/docs`
(tjenesten lytter kun på localhost som standard – se `docker-compose.yml`).

**Verdilaget (ESS):** Opprett gratis bruker på https://ess.sikt.no, finn din
bruker-ID under https://ess.sikt.no/en/api og legg den i en fil `.env` i
prosjektmappen (filen sjekkes ikke inn i git):

```bash
echo "ESS_USER_ID=din-id-her" > .env
```

Uten ID bygges panelet som før, bare uten verdier og holdninger.

**Lokalt (utvikling):**

```bash
make install   # pip install -e ".[dev]"
make data      # hent fra SSB + bygg populasjon
make validate  # testsett: verdilaget mot ESS-respondenter panelet ikke har sett
make test
make serve     # http://localhost:8090 (utforsker) og /docs (API)
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
configs/dimensions.yaml     kategorier, etiketter og formuleringer for API og grensesnitt
src/synthpanel/
  ssb.py                    klient for SSB PxWebApi v2 og Klass
  frame/fetch.py            henter rådata  -> data/raw/*.parquet
  frame/build.py            bygger agenter -> data/processed/agents.parquet
  calibrate/raking.py       raking/IPF – vekter agenter mot kjente totaler
  values/ess.py             henter ESS via API (Norge, runde 9–11)
  values/match.py           verdilag: statistisk matching av ESS-respondenter
  values/validate.py        testsett mot holdte ESS-respondenter
  politics/                 valgresultat 2025, velgerstrømmer, deltakelse -> parti per agent
  media/layer.py            sosiale medier, strømming, netthandel
  api/main.py               FastAPI
  web/index.html            befolkningsutforskeren (ren HTML/JS, ingen byggesteg)
tests/                      kalibrering, SSB-avstemming, API
docs/architecture.md        modellen L0–L5, prinsipper og veikart
```

## Kompatibilitet med Signalist

Bevisst samme stack som Signalist-backend: Python, FastAPI med auto-generert
`/docs`, YAML-konfig, Parquet + DuckDB, Docker. Panelet er en frittstående
tjeneste med eget API, slik at Signalist (eller andre) kan kalle det uten at
noe må skrives om. Port 8090 for å ikke kollidere med Signalist på 8080.

## Datakilder

Åpne data: 28 SSB-tabeller pluss Klass 128 (sentralitet), Valgdirektoratets
resultater for stortingsvalget 2025 (alle kommuner), og European Social
Survey runde 9–11 (4 154 norske respondenter, 2018–2024). ESS-vilkårene skiller
mellom forsknings- og kommersiell bruk – avklar før panelet selges. Oversikt over
hvilken tabell som brukes til hva står i [docs/architecture.md](docs/architecture.md). Se `GET /meta` for nøyaktige perioder og
hentetidspunkt.
