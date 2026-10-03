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
               "rlgdgr", "happy", "wrclmch", "ccrdprs"]

DEMOGRAPHICS = ["idno", "essround", "gndr", "agea", "eisced", "hinctnta", "region", "domicil",
                "mnactic", "brncntr", "facntr", "mocntr", "hhmmb", "pspwght", "anweight"]

# Koder for «vet ikke», «nekter» osv. per skala.
MISSING = {6: {7, 8, 9}, 10: {77, 88, 99}, 5: {7, 8, 9}, 4: {7, 8, 9}}
SCALE_MAX = {**{v: 6 for v in VALUE_ITEMS}, "ppltrst": 10, "trstprl": 10, "trstlgl": 10, "trstplc": 10,
             "trstplt": 10, "lrscale": 10, "rlgdgr": 10, "happy": 10, "polintr": 4, "wrclmch": 5,
             "ccrdprs": 10}


def _download(doi: str, user_id: str) -> pd.DataFrame:
    prefix = config.load("values")["ess"]["doi_prefix"]
    r = httpx.get(f"{API}/{prefix}/{doi}", params={"userId": user_id, "fileFormat": "parquet"},
                  follow_redirects=True, timeout=300)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def harmonize(d: pd.DataFrame) -> pd.DataFrame:
    """Plukk ut Norge, gi verdispørsmålene felles navn og sett manglende svar til NaN."""
    d = d[d["cntry"] == "NO"].copy()
    d = d.rename(columns={f"{v}a": v for v in VALUE_ITEMS if f"{v}a" in d.columns and v not in d.columns})
    cols = [c for c in DEMOGRAPHICS + VALUE_ITEMS + OTHER_ITEMS if c in d.columns]
    d = d[cols].copy()
    for c in VALUE_ITEMS + OTHER_ITEMS:
        if c not in d.columns:
            d[c] = np.nan
            continue
        top = SCALE_MAX[c]
        bad = MISSING.get(top, set()) | MISSING.get(6 if top == 6 else top, set())
        d[c] = d[c].where(~d[c].isin(bad) & d[c].between(0 if top == 10 else 1, top))
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
        d = harmonize(_download(spec["doi"], user_id))
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
