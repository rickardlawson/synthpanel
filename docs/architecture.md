# Arkitektur og modell

## Idé i én setning

Signalist viser hva **media** sier; Synthpanel estimerer hva **befolkningen**
trolig mener – per segment, med størrelse, geografi og usikkerhet. Det mest
verdifulle er ofte **gapet** mellom de to.

## Lagene

| Lag | Innhold | Kilder | Status |
|---|---|---|---|
| **L0 Befolkningsramme** | Kjønn, alder, kommune, fylke, sentralitet, utdanning, landbakgrunn. Neste: inntekt, husholdning, yrke, livsfase | SSB | ✅ v0.1 |
| **L1 Fasitbibliotek – atferd** | Valgresultater, mediebruk, forbruk, dagligvare, kjente meningsmålinger. Lagres som *fordelinger med metadata* (kilde, år, spørsmål, populasjon, usikkerhet) | SSB, Valgundersøkelsen, Medietilsynet, publiserte målinger | ⏳ |
| **L2 Verdilag** | Tillit, frivillighet/dugnad, tro, Schwartz-verdier, politisk engasjement | ESS, Norsk medborgerpanel, WVS (ev. Norsk Monitor på lisens) | ⏳ |
| **L3 Arketyper (8–12)** | Utledes statistisk fra L2 (latent klasseanalyse), navngis etterpå. Koblet til L0 → størrelse, geografi, kjøpekraft | Avledet | ⏳ |
| **L4 Scenariomotor** | Stimulus inn → reaksjon per segment ut, med usikkerhet, volumvekt og spredning over tid (first movers → etternølere) | LLM + kalibrering | Kontrakt i `POST /estimate` |
| **L5 Mediekobling** | Fersk kontekst fra Signalist (Brands / People / Debates) | Signalist API | Senere – via avtale |

## Prinsipper (lærdom fra Pew 2026 og Hiasynth)

1. **Telling og tenking holdes adskilt.** Populasjonen (statistikk) sier hvor
   mange og hvor. LLM-en gir retning og argumenter – den produserer aldri
   prosentfordelinger direkte.
2. **Kalibrer mot ankre.** Hvert scenario forankres i målte spørsmål som ligner.
3. **Flere modeller, vist spredning.** Pew fant at modellvalg alene endret
   historien (GPT ekstremt, Opus midt-på-treet). Spredning mellom modeller er
   en ærlig usikkerhet.
4. **Tving inn variasjon.** Pew: 47 % av spørsmålene hadde svaralternativ ingen
   syntetiske respondenter valgte, og «vet ikke» var 4× for sjelden.
5. **Testsett før produkt.** Mål gjennomsnittlig absolutt feil mot norske data
   modellen ikke har sett. Pew-baseline: ~12 prosentpoeng.
6. **Presisjon vises alltid.** Antall agenter bak hvert tall følger svaret.
7. **Ikke rase – SSB registrerer ikke det.** Landbakgrunn (verdensdel) brukes,
   og estimater per gruppe behandles varsomt (stereotypirisiko).

## L0 – hvordan populasjonen bygges

1. Eksakte antall per kommune × kjønn × ettårig alder (07459).
2. Utdanningsandeler (08921) og landbakgrunn (07111) per fylke × kjønn ×
   aldersgruppe knyttes til hver ettårig alder → forventede marginaler.
3. Agenter fordeles på celler kommune × kjønn × aldersbånd (minst én per
   celle) og får trukket alder, utdanning og bakgrunn.
4. **Raking** justerer vektene til seks marginaler samtidig. Siste marginal
   (kommune × kjønn × aldersbånd) treffes eksakt; de øvrige ligger under
   0,02 % feilplassert befolkning. Effektiv utvalgsstørrelse ≈ 49 400 av 50 751.

### Kjente svakheter i v0.1

- **Utdanning og landbakgrunn antas uavhengige** gitt fylke/kjønn/alder. Det
  stemmer dårlig (innvandrere har mer todelt utdanningsfordeling). Fikses ved
  å rake mot en nasjonal tabell utdanning × innvandringskategori.
- **Innvandrere og norskfødte med innvandrerforeldre er slått sammen** (07111).
- **Utdanning for 18–19 år** bruker andelene for 16–19 år; **67–79 og 80+**
  deler andelene for 67+.
- **Utdanning er fra 2025**, befolkning fra 1.1.2026.
- Ingen inntekt, husholdning eller yrke ennå.
- Svært små kommuner har 1–2 agenter per celle – kommunetall er riktige,
  men kombinasjoner på kommunenivå er lite presise (vises som `presisjon: lav`).

## Kobling til Signalist (senere)

Signalist kategoriserer medietreff i **Brands, People og Debates**.

| Signalist | Media sier | Panelet legger til |
|---|---|---|
| Brands | dekning, tone, saker | kjennskap, sympati, kjøpsintensjon per segment |
| Debates | hvem snakker, hvilke argumenter | standpunktfordeling, hvem bryr seg |
| People | hvem omtales | kjennskap/tillit (sist – mest sensitivt) |

Koblingen går begge veier: Signalist gir fersk kontekst (Pews største svakhet
var saker etter modellens kunnskapsgrense), panelet gir en «befolkningen
mener»-modul. Signalist-API-et bruker 15 fylker med egne `county_id`;
mapping lages mot SSB-fylkesnummer.

Panelet bygges på åpne data og egen infrastruktur. Bruk av Signalist- og
Opoint-data avtales eksplisitt før kobling.

## Pilotcaser (ufarlige, med fasit å teste mot)

1. **Produktlansering:** Tine lanserer plantebasert yoghurt under hovedmerket.
2. **Budskapstest:** «Spis mindre kjøtt for klimaet» vs «Velg kortreist og norsk».
3. **Holdningsendring:** kontantbruk/kontantberedskap, elbil 2015 → i dag.
4. **Debatt:** mobilforbud i skolen; sommertid/vintertid.

## Veikart

1. ✅ L0 befolkningsramme + API
2. Testsett: 30–50 spørsmål med norske svar per undergruppe (Medborgerpanelet, ESS)
3. L0b: inntekt, husholdning, livsfase; fiks utdanning × bakgrunn
4. L2 verdilag og L3 arketyper v0
5. L4 første scenario ende-til-ende (Tine), målt mot testsettet
6. L5 kobling til Signalist
