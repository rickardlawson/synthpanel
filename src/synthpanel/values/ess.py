"""Henting og harmonisering av European Social Survey (ESS) for Norge.

Kjør:  ESS_USER_ID=... python -m synthpanel.values.ess
"""
from __future__ import annotations

import io
import os

import httpx
import numpy as np
import pandas as pd

from synthpanel import config

API = "https://api.ess.sikt.no/v1/data/dataFile"
RAW_FILE = "ess_norway.parquet"

# De 21 verdispørsmålene (Schwartz). I runde 11 har de suffikset «a».
VALUE_ITEMS = ["ipcrtiv", "imprich", "ipeqopt", "ipshabt", "impsafe", "impdiff", "ipfrule",
               "ipudrst", "ipmodst", "ipgdtim", "impfree", "iphlppl", "ipsuces", "ipstrgv",
               "ipadvnt", "ipbhprp", "iprspot", "iplylfr", "impenv", "imptrad", "impfun"]

OTHER_ITEMS = ["ppltrst", "trstprl", "trstlgl", "trstplc", "trstplt", "polintr", "lrscale",
               "rlgdgr", "happy", "wrclmch", "ccrdprs",
               # Arketypelaget (samfunnsroller, kriseresiliens):
               "sclmeet",   # hvor ofte man treffer venner/familie sosialt (1 aldri – 7 hver dag)
               "inprdsc",   # antall man kan snakke fortrolig med (0 ingen – 6 ti eller flere)
               "hincfel",   # opplevd inntekt (1 lever komfortabelt – 4 svært vanskelig)
               "stfdem"]    # tilfredshet med demokratiet (0–10)

DEMOGRAPHICS = ["idno", "essround", "gndr", "agea", "eisced", "hinctnta", "region", "domicil",
                "mnactic", "brncntr", "facntr", "mocntr", "hhmmb", "pspwght", "anweight", "vote"]

# Partikoder varierer mellom runder (verifisert mot verdietikettene i SPSS-filene).
# Runde 9–10 spør om valget i 2017, runde 11 om valget i 2021.
PARTY_VARS = {9: ("prtvtbno", "prtclbno", 2017), 10: ("prtvtbno", "prtclbno", 2017), 11: ("prtvtcno", "prtclcno", 2021)}
_COMMON = {1: "RØDT", 2: "SV", 3: "A", 4: "V", 5: "KRF", 6: "SP", 7: "H", 8: "FRP", 11: "ANDRE"}
PARTY_CODES = {2017: {**_COMMON, 9: "ANDRE", 10: "MDG"},   # 9 = Kystpartiet
               2021: {**_COMMON, 9: "MDG", 10: "ANDRE"}}   # 10 = Pasientfokus

SCALE_MAX = {**{v: 6 for v in VALUE_ITEMS}, "ppltrst": 10, "trstprl": 10, "trstlgl": 10, "trstplc": 10,
             "trstplt": 10, "lrscale": 10, "rlgdgr": 10, "happy": 10, "polintr": 4, "wrclmch": 5,
             "ccrdprs": 10, "sclmeet": 7, "inprdsc": 6, "hincfel": 4, "stfdem": 10}
SCALE_MIN = {"inprdsc": 0}


def _download(doi: str, user_id: str) -> pd.DataFrame:
    prefix = config.load("values")["ess"]["doi_prefix"]
    r = httpx.get(f"{API}/{prefix}/{doi}", params={"userId": user_id, "fileFormat": "parquet"},
                  follow_redirects=True, timeout=300)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def harmonize(d: pd.DataFrame, rnd: int | None = None) -> pd.DataFrame:
    """Plukk ut Norge, gi verdispørsmålene felles navn og sett manglende svar til NaN."""
    d = d[d["cntry"] == "NO"].copy()
    if rnd in PARTY_VARS:
        vote_var, close_var, year = PARTY_VARS[rnd]
        codes = PARTY_CODES[year]
        d["siste_valg_aar"] = year
        d["siste_parti"] = d[vote_var].map(codes) if vote_var in d else np.nan
        d.loc[d["vote"] == 2, "siste_parti"] = "ikke_stemt"
        d.loc[d["vote"] == 3, "siste_parti"] = "ikke_stemmerett"
        d["naermeste_parti"] = d[close_var].map(codes) if close_var in d else np.nan
    d = d.rename(columns={f"{v}a": v for v in VALUE_ITEMS if f"{v}a" in d.columns and v not in d.columns})
    cols = [c for c in DEMOGRAPHICS + VALUE_ITEMS + OTHER_ITEMS + ["siste_valg_aar", "siste_parti", "naermeste_parti"]
            if c in d.columns]
    d = d[cols].copy()
    for c in VALUE_ITEMS + OTHER_ITEMS:
        if c not in d.columns:
            d[c] = np.nan
            continue
        top = SCALE_MAX[c]
        lo = SCALE_MIN.get(c, 0 if top == 10 else 1)
        d[c] = d[c].where(d[c].between(lo, top))   # alt utenfor skalaen er «vet ikke», «nekter» o.l.
    d["hinctnta"] = d["hinctnta"].where(d["hinctnta"].between(1, 10))
    d["eisced"] = d["eisced"].where(d["eisced"].between(1, 7))
    d["agea"] = d["agea"].where(d["agea"].between(15, 110))
    return d


def fetch() -> pd.DataFrame:
    user_id = os.environ.get("ESS_USER_ID")
    if not user_id:
        raise SystemExit("Sett ESS_USER_ID (se https://ess.sikt.no/en/api) for å hente ESS-data.")
    rounds = config.load("values")["ess"]["rounds"]
    parts = []
    for rnd, spec in rounds.items():
        d = harmonize(_download(spec["doi"], user_id), int(rnd))
        d["essround"] = int(rnd)
        parts.append(d)
        print(f"ESS runde {rnd}: {len(d)} norske respondenter")
    out = pd.concat(parts, ignore_index=True)
    out["resp_id"] = out["essround"].astype(str) + "-" + out["idno"].astype(str)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.RAW_DIR / RAW_FILE, index=False)
    return out


def load() -> pd.DataFrame | None:
    path = config.RAW_DIR / RAW_FILE
    return pd.read_parquet(path) if path.exists() else None


if __name__ == "__main__":
    fetch()
