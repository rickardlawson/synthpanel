"""Henter valgdata: stortingsvalget 2025 per kommune (Valgdirektoratet) og
SSB-tabeller om velgerstrømmer, valgdeltakelse og stemmerett.

Kjør:  python -m synthpanel.politics.fetch
"""
from __future__ import annotations

import time

import httpx
import pandas as pd

from synthpanel import config, ssb

VALG_API = "https://valgresultat.no/api"
PARTIES = ["RØDT", "SV", "A", "SP", "MDG", "KRF", "V", "H", "FRP"]


def _get(path: str) -> dict:
    for attempt in range(4):
        r = httpx.get(VALG_API + path, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(1 + attempt)
    r.raise_for_status()
    return {}


def _kommune_row(href: str, dist_name: str) -> dict:
    k = _get(href)
    row = {"kommune": k["id"]["nr"], "kommune_navn": k["id"]["navn"], "valgdistrikt": dist_name,
           "stemmeberettigede": k.get("antallsb"), "godkjente": k["stemmegiving"]["totalGodkjente"]}
    andre = blanke = 0
    for p in k["partier"]:
        cat, code = p["id"]["partikategori"], p["id"]["partikode"].upper()
        n = (p["stemmer"]["resultat"].get("antall") or {}).get("total") or 0
        if cat == 0:                 # sumlinjer («Andre») – ville telt småpartiene dobbelt
            continue
        if code == "BLANKE":
            blanke += n
        elif code in PARTIES:
            row[code] = row.get(code, 0) + n
        else:
            andre += n
    row["ANDRE"] = andre
    row["blanke"] = blanke
    return row


def fetch_results(year: int = 2025) -> pd.DataFrame:
    """Stemmer per parti i hver kommune (stortingsvalg). Blanke holdes utenfor partiandelene."""
    from concurrent.futures import ThreadPoolExecutor

    land = _get(f"/{year}/st")
    jobs = []
    for dist in land["_links"]["related"]:
        d = _get(dist["href"])
        jobs += [(kom["href"], dist["navn"]) for kom in d["_links"]["related"]]
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(lambda j: _kommune_row(*j), jobs))
    return pd.DataFrame(rows).fillna({p: 0 for p in PARTIES})


def fetch_flows() -> pd.DataFrame:
    return ssb.get_table("11666", [
        {"variableCode": "PolitPartiNeste", "valueCodes": ["*"]},
        {"variableCode": "PolitPartiDette", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Velgere"]},
        {"variableCode": "Tid", "valueCodes": ["2021", "2025"]},
    ])


def fetch_turnout() -> pd.DataFrame:
    return ssb.get_table("10440", [
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "Alder", "valueCodes": ["*"]},
        {"variableCode": "UtdNivaa", "valueCodes": ["1-2", "3-5", "6-8"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Deltakelse"]},
        {"variableCode": "Tid", "valueCodes": ["2025"]},
    ])


def fetch_turnout_immigrant() -> pd.DataFrame:
    return ssb.get_table("13818", [
        {"variableCode": "ArbStyrkStatus", "valueCodes": ["*"]},
        {"variableCode": "InnvandrKat", "valueCodes": ["Ialt", "B", "C"]},
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "ContentsCode", "valueCodes": ["ValgDeltakelse"]},
        {"variableCode": "Tid", "valueCodes": ["2025"]},
    ])


def fetch_eligible() -> pd.DataFrame:
    return ssb.get_table("13446", [
        {"variableCode": "Region", "valueCodes": ["0"]},
        {"variableCode": "Alder", "valueCodes": ["18-29", "30-39", "40-49", "50-59", "60+"]},
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "InnvandrKat", "valueCodes": ["B", "C", "Rest"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Stemmeberettigede"]},
        {"variableCode": "Tid", "valueCodes": ["2025"]},
    ])


def fetch_holdout_13554() -> pd.DataFrame:
    """Partivalg etter kjønn og alder (Valgundersøkelsen 2025) – brukes i kalibreringen."""
    return ssb.get_table("13554", [
        {"variableCode": "PolitParti", "valueCodes": ["*"]},
        {"variableCode": "Kjonn", "valueCodes": ["0", "1", "2"]},
        {"variableCode": "Alder", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Velgere"]},
        {"variableCode": "Tid", "valueCodes": ["2025"]},
    ])


def fetch_holdout_13698() -> pd.DataFrame:
    """KUN TIL KONTROLL: oppslutning etter kjønn, alder og bruttoinntekt (Valgundersøkelsen 2025)."""
    return ssb.get_table("13698", [
        {"variableCode": "PolitParti", "valueCodes": ["*"]},
        {"variableCode": "Kjonn", "valueCodes": ["0"]},
        {"variableCode": "Alder", "valueCodes": ["999A"]},
        {"variableCode": "BruttoInnte", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Velgere"]},
        {"variableCode": "Tid", "valueCodes": ["2025"]},
    ])


JOBS = {
    "valg_resultat_2025": fetch_results,
    "valg_flows_11666": fetch_flows,
    "valg_turnout_10440": fetch_turnout,
    "valg_turnout_innv_13818": fetch_turnout_immigrant,
    "valg_eligible_13446": fetch_eligible,
    "valg_kontroll_13554": fetch_holdout_13554,
    "valg_kontroll_13698": fetch_holdout_13698,
}


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in JOBS.items():
        df = fn()
        df.to_parquet(config.RAW_DIR / f"{name}.parquet", index=False)
        print(f"{name}: {len(df)} rader")


if __name__ == "__main__":
    main()
