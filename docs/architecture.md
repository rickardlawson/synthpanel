# Arkitektur og modell

## Idé i én setning

Signalist viser hva **media** sier; Synthpanel estimerer hva **befolkningen**
trolig mener – per segment, med størrelse, geografi og usikkerhet. Det mest
verdifulle er ofte **gapet** mellom de to.

## Lagene

| Lag | Innhold | Kilder | Status |
|---|---|---|---|
| **L0 Befolkningsramme** | Kjønn, alder, kommune, fylke, sentralitet, utdanning, landbakgrunn, innvandringskategori, hovedstatus (arbeid/studier/pensjon/trygd), husholdningstype, lavinntekt, inntektsdesil. Neste: yrke/næring, bolig | SSB | ✅ v0.2 |
| **L1 Fasitbibliotek – atferd** | Valgresultater, mediebruk, forbruk, dagligvare, kjente meningsmålinger. Lagres som *fordelinger med metadata* (kilde, år, spørsmål, populasjon, usikkerhet) | SSB, Valgundersøkelsen, Medietilsynet, publiserte målinger | ⏳ |
| **L2 Verdilag** | Schwartz-verdier (fire hovedretninger), tillit, risikovilje, klimabekymring, religiøsitet, politisk ståsted og interesse. Neste: frivillighet, mediebruk, Medborgerpanelet | ESS runde 9–11 (ev. Norsk Monitor på lisens) | ✅ v0.3 |
| **L3 Arketyper / personas** | v0: «Ti på gata» – de største grupperingene i hvilket som helst utvalg, vist som representative personer (se under). Neste: faste arketyper (latent klasseanalyse på L2) | Avledet | 🟡 v0 |
| **L4 Scenariomotor** | Stimulus inn → reaksjon per segment ut, med usikkerhet, volumvekt og spredning over tid (first movers → etternølere) | LLM + kalibrering | Kontrakt i `POST /estimate` |
| **L5 Mediekobling** | Fersk kontekst fra Signalist (Brands / People / Debates) | Signalist API | Senere – via avtale |

## Verdihierarkiet

Alle egenskaper er ordnet i tre lag, fra det målbare til det dype. Hierarkiet
styres fra `configs/dimensions.yaml` (lag → seksjoner → dimensjoner) og brukes
likt i API (`/dimensions`, `lag` på kjennetegn), grensesnitt og persona-profiler.

| Lag | Spørsmål | Innhold | Presisjon |
|---|---|---|---|
| **1 Hygienefaktorer** | Hvem er de, hvordan lever de? | Demografi og bosted · utdanning og arbeid · husholdning, økonomi og bolig | Registerdata (SSB) |
| **2 Motivasjonsfaktorer** | Hva bruker de tid, oppmerksomhet og stemme på? | Medievaner · forbruk og netthandel · politisk engasjement (parti, interesse) | Estimert fra undersøkelser |
| **3 Verdifaktorer** | Hva tror de på, hvordan møter de verden? | Grunnverdier (Schwartz) · risikovilje, tillit, klimabekymring, religiøsitet, politisk ståsted | Estimert fra ESS, testet |

## «Ti på gata» – personas (L3 v0)

`GET /population/personas` tar samme filtre som resten av API-et og returnerer de
(inntil) ti største grupperingene i utvalget.

1. Agentene kodes på tvers av hierarkiet (alder, husholdning, status, utdanning,
   økonomi, bolig, bosted · parti, medier, netthandel · verdier og holdninger),
   med vekter per lag (1,0 / 0,7 / 0,8). Egenskaper som er låst av filteret, utelates.
2. Vektet k-means (k ≤ 10, minst 15 agenter per gruppe) på et utvalg på inntil
   6 000 agenter. Grupperingene dekker hele utvalget; `andel` summerer til 1.
3. Hver gruppering vises som den *faktiske* agenten nærmest gruppens midtpunkt –
   en sammenhengende person, ikke et gjennomsnitt. «Typisk for grupperingen» er
   egenskaper med andel ≥ 45 % og lift ≥ 1,35 mot resten av utvalget.
4. **Navn:** norsk bakgrunn – fornavn trukket blant navn gitt til barn født
   samme år ± 2 (SSB 10467), etternavn blant vanlige etternavn (SSB 12891).
   Annen bakgrunn – kuratert liste per opphavsland (`configs/names.yaml`), som
   følger portrettet slik at navn og ansikt henger sammen.
5. **Portretter:** AI-genererte (Realistic Vision 5.1 + LCM-LoRA, åpne lisenser)
   per kjønn × 7 aldersgrupper × landbakgrunn, laget én gang med
   `scripts/generate_portraits.py` og sjekket inn. Merket «AI-generert» i
   grensesnittet. Mangler et passende bilde, vises initial – aldri et ansikt
   med feil alder.
6. **Brief:** hver persona har en tekst (`brief`) som beskriver personen etter
   hierarkiet – grunnlaget når personaene skal kunne svare på spørsmål og
   reagere på budskap (L4).

Samme utvalg gir alltid samme personas (frø fra filtrene).

**Svakheter:** k-means på blandede data gir grupper som er gode til å
oppsummere, men ikke nødvendigvis «naturlige» segmenter; verdiene og medievanene
er koblet til demografien gjennom statistisk matching, så en enkelt persona kan
ha kombinasjoner som er mindre typiske enn gruppen den representerer.

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

Egenskapene trekkes i en rekkefølge der hver kan avhenge av de forrige, og
vektene kalibreres (raking) mot 15 marginaler fra SSB i to pass.

| Egenskap | Trekkes betinget på | Kilde | Kalibreres mot |
|---|---|---|---|
| Kommune, kjønn, alder | – | 07459 (1.1.2026) | kommune × kjønn × aldersbånd (eksakt), kjønn × ettårig alder |
| Landbakgrunn | fylke, kjønn, alder | 07111 | fylke × kjønn, kjønn × alder |
| Innvandringskategori | kjønn, alder (blant utenlandsk bakgrunn) | 09599 | via utdanningsmarginal |
| Utdanning | fylke, kjønn, alder, innvandringskategori | 08921, 09599 | fylke × kjønn; kjønn × alder × innvkat; innvandrere per fylke (12934) |
| Hovedstatus | kjønn, alder, utdanning, innvandrer; 62+ justert per ettårig alder | 12424–12426, 06161 | kjønn × alder × status; sysselsatt per ettårig alder 62–74 og 75+; aktiv per fylke (13678) |
| Husholdningstype | fylke, kjønn, alder | 06071, 12836, 10986 | kjønn × alder (18–29, 30–44, 45–61, 62–66, 67+) |
| Lavinntekt (EU-60) | husholdning, status, innvandring, utdanning | 12599, 09570 | hver av gruppene |
| Inntektsdesil | fylke, husholdningstype, lavinntekt | 12563 | – (trekkes etter kalibrering) |
| Eierstatus og boligtype | fylke, husholdningstype, inntektskvartil (dempet) | 14901, 14900, 14921 | – (dempingen tilpasses så eierstatus per kvartil treffer 14900) |
| Verdier og holdninger | kjønn, alder, utdanning, inntekt, region, innvandring, status, bosted, aleneboende | ESS 9–11 | – (statistisk matching, se under) |
| Stemmerett, valgdeltakelse, parti 2025 | ESS-donorens partivalg, kjønn, alder, utdanning, innvandring, status | Valgdirektoratet, 11666, 10440, 13818, 13446, 13554 | frammøte og partier per fylke; parti × kjønn × alder |
| Medier og netthandel | kjønn, alder | 14511, 14512, 07001 | – |

Resultat v0.2: 50 751 agenter, effektiv utvalgsstørrelse ≈ 45 500. Kommunetall
er eksakte. Alle marginaler ligger under 0,2 % feilplassert befolkning, unntatt
de to utdanningsmarginalene for innvandrere (≈ 0,45 %), der kildene er uenige.

**Uavhengig kontroll:** innvandreres utdanning etter kjønn per fylke (12934) er
ikke brukt i kalibreringen. Snittavvik i andel med høyere utdanning: 1,5
prosentpoeng; under 3 pp for alle grupper med minst 300 agenter.

## L2 – verdilaget

Hver agent får en «donor»: en ekte norsk ESS-respondent med lik demografi
(nærmeste nabo på kjønn, alder, utdanning, inntektsdesil, region, innvandring,
hovedstatus, bosted og om personen bor alene; trekkes blant de 15 nærmeste
etter ESS-vekt og hvor ny runden er). Agenten arver donorens svar. Slik bevares
sammenhengen *mellom* verdiene, ikke bare fordelingen av hver.

Avledede mål: Schwartz' fire hovedretninger (åpenhet for endring, bevaring,
selvhevdelse, selvoverskridelse) beregnes etter ESS-anbefaling (sentrert per
person) og deles i nasjonale tertiler. Tillit = snitt av fire institusjoner.
Klimabekymring finnes bare i runde 10–11 og matches separat.

### Testsett (`make validate`)

30 % av norske respondenter i runde 11 holdes helt utenfor. Panelet bygges med
resten og skal anslå andelen «høy» (o.l.) i 11 indikatorer for undergrupper
(kjønn, alder, utdanning, region, innvandring, status og kryss). 330 celler.

| | Snittfeil |
|---|---|
| Landssnitt uten demografi | 7,1 pp |
| **Synthpanel** | **5,2 pp** |
| Støygulv (testgruppens størrelse) | 4,5 pp |

Panelet slår landssnittet på 10 av 11 indikatorer. Til sammenligning fant Pew
12 pp snittfeil for LLM-genererte «digitale tvillinger» – men det er ulike
spørsmål og oppsett, så tallene er ikke direkte sammenlignbare. Det viktige er
prinsippet: tallene kommer fra ekte respondenter, ikke fra språkmodellen.

### Kjente svakheter i verdilaget

- **Få respondenter.** 4 154 nordmenn; hver donor brukes i snitt ~13 ganger.
  Smale segmenter arver verdiene fra et lite antall personer.
- **Tidsspenn 2018–2024.** Holdninger endrer seg; nyere runder vektes høyere,
  men runde 9 er med.
- **Runde 9 bruker gamle regioner** (NUTS 2016); delte regioner matcher bredt.
- **Verdier henger sammen med demografi bare gjennom matching-variablene.**
  Lokale forskjeller utover region og bosted fanges ikke.
- **ESS-vilkårene** skiller mellom forsknings- og kommersiell bruk. Må avklares
  før panelet selges.

## Politisk lag

1. **Partipreferanse:** ESS-donorens partivalg (2017 eller 2021) føres frem til
   2025 med SSBs velgerstrømmer fra Valgundersøkelsen (11666): 2017→2021→2025.
   Donorer uten partivalg bruker partiet de står nærmest, ellers landsresultatet.
2. **Stemmerett:** innvandrere får sannsynlighet for statsborgerskap fra 13446.
3. **Valgdeltakelse:** 10440 (kjønn × alder × utdanning), justert for
   innvandringskategori og status (13818), forskjøvet per fylke til faktisk frammøte.
4. **Partivalg:** preferansene skaleres vekselvis mot valgresultatet per fylke
   (Valgdirektoratet, alle 357 kommuner) og partivalg etter kjønn × alder (13554).

**Kontroll:** Før 13554 ble tatt inn, traff panelet partivalg etter kjønn × alder
med 2,0 pp snittfeil (landssnitt 2,7 pp; 80 celler). Det bommet på det nye i
2025 – unge menn til FrP (38 % faktisk, 25 % i panelet). Etter kalibrering
treffer kjønn × alder og fylke. Uavhengig sjekk (13698, ikke brukt): Høyre øker
og SV faller med inntekt i panelet som i Valgundersøkelsen, men forskjellene er
svakere (Høyre 13→16 % fra laveste til høyeste desil, mot 12→27 % fra laveste
til høyeste personinntekt i 13698 – ulike inntektsmål).

**Svakheter:** små partier bygger på få respondenter; inntektsforskjellene er
for svake; partisympati for ikke-velgere er en modell, ikke målt.
**Bruk:** partitilhørighet er sensitivt. Agentene er syntetiske, men bruk til
politisk målretting bør vurderes eksplisitt før kommersiell lansering.

## Medielag

Daglig bruk av Facebook, Instagram, Snapchat, TikTok, YouTube, LinkedIn,
NRK TV, Netflix, TV 2 Play, Viaplay og Disney+ (Norsk mediebarometer 2025), og
netthandel siste 12 mnd (dagligvarer, klær, reiser, take-away, kosmetikk;
nyeste år med tall, 2024/2025). SSB publiserer bare etter kjønn og alder, så
laget følger bare disse. Mikrodata fra Mediebarometeret (Sikt) vil gi
sammenheng med utdanning, bosted og verdier, og mellom tjenestene.

### Kjente svakheter i boliglaget

- Tabellene gjelder husholdninger; agentene er personer. Husholdningsstørrelse
  er tilnærmet per type ved tilpasning.
- Andelen leiere i andre inntektskvartil ligger ~10 pp over SSB (desiler og
  SSBs kvartiler er ikke helt samme mål).

### Kjente svakheter i v0.2

- **Inntektsdesil er husholdningens inntekt, ikke personens,** og fordelingen
  per husholdningstype gjelder husholdninger, ikke personer. Par uten barn
  plasseres etter egen alder (SSB bruker eldste person).
- **Inntekt henger sammen med utdanning og status bare via lavinntekt.**
  SSB publiserer ikke desiler etter utdanning. Over lavinntektsgrensen er desil
  uavhengig av utdanning gitt husholdningstype og fylke.
- **Lavinntekt etter utdanning** bygger på *vedvarende* lavinntekt (09570),
  skalert til årlig nivå.
- **Europa-bakgrunn ≈ «EU/EØS m.fl.» i lavinntektsgruppene.** Innvandrere fra
  Europa utenfor EU/EØS (f.eks. Ukraina) havner dermed i feil lavinntektsgruppe.
- **Husholdning 18–29 år** er en samlet gruppe: 18-åringer og 29-åringer har
  samme sannsynlighet for å bo med foreldre.
- **Sysselsetting 75+** er utledet som rest (12426 minus 06161 for 67–74).
- **Innvandrere og norskfødte** skilles bare med nasjonale andeler per alder.
- **Små fylker og kommuner** har få agenter i smale grupper – svarene merkes
  `presisjon: moderat/lav`.
- Utdanning, status og inntekt er fra 2024/2025; befolkning fra 1.1.2026.

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

1. ✅ L0 befolkningsramme + API + utforsker
2. ✅ L0b: innvandringskategori, hovedstatus, husholdning, lavinntekt, inntekt
3. ✅ L2 verdilag fra ESS + første testsett; bolig
4. ✅ Politisk lag (valg 2025) og medielag (kjønn × alder)
4c. ✅ Verdihierarki (hygiene / motivasjon / verdi) og «Ti på gata»-personas med navn og AI-portretter
4b. ✅ Nytt grensesnitt: målgruppebygger med kategorier (styrt av `configs/dimensions.yaml`), resultatpanel med steder (størst/tettest + lift) og kjennetegn (`/population/places`, `/population/profile`)
5. Forbruksprofil (SSB forbruksundersøkelse), fritid; Mediebarometer-mikrodata (Sikt)
6. Utvidet testsett: Norsk medborgerpanel og publiserte målinger
7. L3 faste arketyper (latent klasseanalyse på verdilaget)
8. L4: still personaene spørsmål og test budskap på tvers av galleriet (språkmodell med briefen som grunnlag)
9. L4 første scenario ende-til-ende (Tine), målt mot testsettet
10. L5 kobling til Signalist
