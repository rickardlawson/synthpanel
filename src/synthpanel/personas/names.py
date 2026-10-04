"""Navn til personas.

Norsk bakgrunn: fornavn trekkes blant navn gitt til barn født samme år (± 2 år)
som personaen (SSB 10467), etternavn blant etternavn brukt av minst 200 personer
(SSB 12891) – begge vektet etter hvor vanlige de er. Andre bakgrunner: kuratert
liste i configs/names.yaml (SSB publiserer ikke navn etter landbakgrunn).

Henting:  python -m synthpanel.personas.names
"""
from __future__ import annotations

import random
from functools import lru_cache

import pandas as pd

from synthpanel import config, ssb

RAW = config.RAW_DIR
FIRST = RAW / "names_first_10467.parquet"
LAST = RAW / "names_last_12891.parquet"

# Etternavn i SSB-lista som ikke er typisk norske – holdes utenfor for personas med norsk bakgrunn.
def _foreign_surnames() -> set[str]:
    out = set()
    for spec in config.load("names").values():
        for k in ("etternavn", "etternavn_f", "etternavn_m"):
            out |= {n.upper() for n in spec.get(k, [])}
    return out | {"ALI", "AHMED", "HUSSAIN", "HUSAIN", "MOHAMMED", "MOHAMED", "MOHAMMAD", "HASSAN", "KHAN", "NGUYEN",
                  "TRAN", "LE", "PHAM", "SINGH", "KAUR", "NOWAK", "ABDI", "OMAR", "IBRAHIM", "YUSUF", "MAHMOUD",
                  "ABDULLAHI", "ISMAIL", "RAHMAN", "AKHTAR", "IQBAL", "BUTT", "MALIK", "SHAH", "ARSHAD", "ASLAM",
                  "HOANG", "VU", "DO", "HUYNH", "SMITH", "JOHNSON", "SANTOS", "GARCIA", "LEE", "KIM", "PARK", "CHOI",
                  "JAMA", "OSMAN", "NUR", "FARAH", "WARSAME", "HAJI", "ADAN", "MUSE", "AWAD"}


def fetch() -> None:
    first = ssb.get_table("10467", [
        {"variableCode": "Fornavn", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Personer"]},
        {"variableCode": "Tid", "valueCodes": ["from(1925)"]}])
    first = first[first["value"].fillna(0) > 0]
    pd.DataFrame({"kjonn": first["Fornavn"].str[0].map({"1": "kvinne", "2": "mann"}),
                  "navn": first["Fornavn_tekst"], "ar": first["Tid"].astype(int),
                  "antall": first["value"].astype(int)}).to_parquet(FIRST, index=False)
    last = ssb.get_table("12891", [
        {"variableCode": "Etternavn", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["*"]},
        {"variableCode": "Tid", "valueCodes": ["top(1)"]}])
    last = last[last["value"].fillna(0) > 0]
    pd.DataFrame({"navn": last["Etternavn_tekst"].str.title(), "antall": last["value"].astype(int)}
                 ).to_parquet(LAST, index=False)
    print(f"10467: {len(first)} navn×år  12891: {len(last)} etternavn")


@lru_cache(maxsize=1)
def _first() -> pd.DataFrame | None:
    return pd.read_parquet(FIRST) if FIRST.exists() else None


@lru_cache(maxsize=1)
def _last() -> pd.DataFrame | None:
    if not LAST.exists():
        return None
    d = pd.read_parquet(LAST)
    # Vanlige navn (minst 1 000 bærere) uten de typiske innvandrernavnene på lista
    return d[(d["antall"] >= 1000) & ~d["navn"].str.upper().isin(_foreign_surnames())]


_FALLBACK = {"kvinne": ["Anne", "Inger", "Kari", "Marit", "Ingrid", "Hilde", "Nina", "Ida", "Emma"],
             "mann": ["Jan", "Per", "Bjørn", "Ole", "Lars", "Kjetil", "Thomas", "Martin", "Jonas"]}
_FALLBACK_LAST = ["Hansen", "Johansen", "Olsen", "Larsen", "Andersen", "Pedersen", "Nilsen", "Kristiansen"]


def norwegian(sex: str, birth_year: int, rng: random.Random) -> tuple[str, str]:
    f, l = _first(), _last()
    if f is None:
        first = rng.choice(_FALLBACK[sex])
    else:
        pool = f[(f["kjonn"] == sex) & f["ar"].between(birth_year - 2, birth_year + 2)]
        pool = pool.groupby("navn")["antall"].sum()
        pool = pool[pool >= pool.sum() * 0.002]  # typiske navn for årskullet, ikke sjeldne
        first = rng.choices(list(pool.index), weights=list(pool.values ** 1.2))[0] if len(pool) else rng.choice(_FALLBACK[sex])
    last = rng.choice(_FALLBACK_LAST) if l is None else rng.choices(list(l["navn"]), weights=list(l["antall"]))[0]
    return first, last


def foreign(sex: str, origin: str, rng: random.Random) -> tuple[str, str]:
    spec = config.load("names").get(origin)
    if not spec:
        return rng.choice(_FALLBACK[sex]), rng.choice(_FALLBACK_LAST)
    first = rng.choice(spec["f" if sex == "kvinne" else "m"])
    last = rng.choice(spec.get("etternavn") or spec["etternavn_f" if sex == "kvinne" else "etternavn_m"])
    return first, last


def name_for(sex: str, birth_year: int, origin: str, key: str, taken: set[str]) -> str:
    """Typisk navn for personaen; unngår at to i samme galleri heter det samme."""
    rng = random.Random(key)
    for _ in range(20):
        first, last = norwegian(sex, birth_year, rng) if origin == "Norwegian" else foreign(sex, origin, rng)
        full = f"{first} {last}"
        if full not in taken and first not in {t.split()[0] for t in taken}:
            taken.add(full)
            return full
    taken.add(full)
    return full


if __name__ == "__main__":
    fetch()
