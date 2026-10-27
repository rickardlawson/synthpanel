# Synthpanel – arbeidsregler for Claude

Syntetisk panel av den voksne norske befolkningen (50 751 agenter) bygget på
SSB, ESS og valgdata. Signalist viser hva media sier; Synthpanel estimerer hva
befolkningen trolig mener. Full arkitektur: `docs/architecture.md`.

Rickard eier produktet og koder ikke selv. Forklar valg kort på norsk, og vis
resultater (tall, skjermbilde, testutfall) heller enn å beskrive innsats.

## Kommandoer

```bash
make install    # pip install -e ".[dev]"
make data       # hent rådata (SSB, valg, ESS hvis ESS_USER_ID) + bygg populasjonen
make build      # bygg populasjonen på nytt fra rådata
make test       # pytest -q
make validate   # verdilaget mot holdte ESS-respondenter (testsettet)
make fasit      # personaene mot BI Norsk kundebarometer (krever ANTHROPIC_API_KEY)
make serve      # http://localhost:8090 og /docs
```

**NB:** De fleste testene hoppes over i stillhet når `data/` mangler. «Alle
tester grønne» uten bygget populasjon betyr nesten ingenting – kjør `make data`
først, og si fra hvis testene ble hoppet over.

## Struktur

- `src/synthpanel/frame/` – L0 befolkningsramme (fetch → build, raking)
- `src/synthpanel/values/` – L2 verdilag (ESS-donorer, testsett)
- `src/synthpanel/archetypes/` – L3 arketyper (regler i `configs/archetypes.yaml`)
- `src/synthpanel/personas/` – «Ti på gata», navn, portretter, `voice.py` (LLM)
- `src/synthpanel/politics|media|leisure|household|geo/` – øvrige lag
- `src/synthpanel/api/main.py` – FastAPI; `src/synthpanel/web/index.html` – grensesnittet
- `configs/dimensions.yaml` – verdihierarkiet styrer API, UI og personaprofiler

## Prinsipper som ikke skal brytes

1. **Telling og tenking holdes adskilt.** Andeler, antall og geografi kommer fra
   populasjonen. LLM-en gir retning, argumenter og sitater – den produserer
   aldri prosentfordelinger direkte.
2. **Kalibrer mot ankre.** Nye mål (f.eks. bankbytte) trenger en målt fasit før
   tallene vises som estimat. Uten anker: merk det tydelig som simulering.
3. **Presisjon vises alltid.** Antall agenter bak hvert tall følger svaret.
4. **Ikke rase.** Landbakgrunn (verdensdel) brukes, og varsomt.
5. **Partitilhørighet og samfunnsroller** er ikke for politisk målretting.
6. **Deterministisk:** samme filter gir samme personas (frø fra filtrene).
   Ikke bryt det uten grunn.

## Arbeidsflyt

- Jobb på en egen branch og åpne PR mot `main`. Ikke push direkte til `main`.
- Små, avgrensede endringer per PR, med kort beskrivelse på norsk.
- Nye lag eller kilder: oppdater tabellene i `docs/architecture.md` (kilde,
  SSB-tabellnummer, kontroll mot fasit, kjente svakheter) i samme PR.
- Legg til test når du endrer populasjonsbyggingen eller et API-endepunkt.

## Aldri i git

- `data/` (bygges fra kildene), `.env`, API-nøkler (`ANTHROPIC_API_KEY`,
  `ESS_USER_ID`, senere Signalist-nøkler).
- Rådata eller tekst fra Signalist/Opoint og BI-tall utover det som allerede
  ligger i `configs/fasit/` – bruken er avtalebasert.

## Data og vilkår

- ESS skiller forsknings- og kommersiell bruk – ikke avklart ennå.
- BI Norsk kundebarometer brukes kun til intern validering.
- Portretter er AI-genererte og skal merkes slik i grensesnittet.
