"""Medielag: daglig bruk av sosiale medier og strømmetjenester (Norsk mediebarometer)
og netthandel siste 12 måneder (SSBs IKT-undersøkelse).

SSB publiserer disse bare etter kjønn og alder. Hver agent får derfor
ja/nei per tjeneste trukket uavhengig, gitt kjønn og alder.

Svakhet: sammenhengen *mellom* tjenestene (de som bruker TikTok, bruker ofte
Snapchat) fanges ikke, og utdanning/bosted påvirker ikke. Mikrodata fra
Mediebarometeret (Sikt) vil kunne løse dette.

Kjør henting:  python -m synthpanel.media.layer
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from synthpanel import config, ssb

RAW = config.RAW_DIR

SOCIAL = {"Facebook": "facebook", "Instagram": "instagram", "Snapchat": "snapchat", "TikTok": "tiktok",
          "Linkedin": "linkedin", "YouTube": "youtube"}
STREAM = {"01": "nrktv", "02": "netflix", "05": "tv2play", "04": "viaplay", "06": "disney"}
SHOP = {"Kjøpt/bestillt mat/kolonialvarer": "dagligvarer",
        "Kjøpt/bestilt klær/sportsartikler": "klaer",
        "Kjøpt/bestilt reiser/innkvartering": "reiser",
        "Kjøpt/bestilt levering/henting/take-away fra restauranter, fastfood el.l": "takeaway",
        "Kjøpt/bestilt kosmetikk, skjønnhets- eller velværeprodukter": "kosmetikk"}

# Kolonnenavn i agenttabellen og visningsnavn.
COLUMNS = {**{f"daglig_{v}": k for k, v in SOCIAL.items()},
           "daglig_nrktv": "NRK TV", "daglig_netflix": "Netflix", "daglig_tv2play": "TV 2 Play",
           "daglig_viaplay": "Viaplay", "daglig_disney": "Disney+",
           **{f"netthandel_{v}": v for v in SHOP.values()}}


def fetch() -> None:
    latest = {"variableCode": "Tid", "valueCodes": ["top(1)"]}
    social = ssb.get_table("14511", [
        {"variableCode": "AppNettsted", "valueCodes": ["*"]},
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "Alder", "valueCodes": ["16-24", "25-44", "45-64", "65-79", "80+"]},
        {"variableCode": "ContentsCode", "valueCodes": ["*"]}, latest])
    stream = ssb.get_table("14512", [
        {"variableCode": "StrommeTjeneste", "valueCodes": list(STREAM)},
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "Alder", "valueCodes": ["16-24", "25-44", "45-64", "65-79", "80+"]},
        {"variableCode": "ContentsCode", "valueCodes": ["AndelPersoner"]}, latest])
    shop = ssb.get_table("07001", [
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "Alder", "valueCodes": ["16-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-79"]},
        {"variableCode": "ContentsCode", "valueCodes": ["*"]},
        {"variableCode": "Tid", "valueCodes": ["top(5)"]}])   # spørsmålene roterer mellom år
    social.to_parquet(RAW / "media_social_14511.parquet", index=False)
    stream.to_parquet(RAW / "media_stream_14512.parquet", index=False)
    shop.to_parquet(RAW / "media_shop_07001.parquet", index=False)
    for n, d in [("14511", social), ("14512", stream), ("07001", shop)]:
        print(f"{n}: {len(d)} rader, periode {sorted(d['Tid'].unique())}")


def _band(age: pd.Series, bands) -> pd.Series:
    out = pd.Series(index=age.index, dtype=object)
    for lo, hi, lab in bands:
        out[age.between(lo, hi)] = lab
    return out


def rates() -> pd.DataFrame:
    """Lang tabell: kolonne, kjønn, aldersgruppe (med grensene), andel."""
    rows = []
    s = pd.read_parquet(RAW / "media_social_14511.parquet")
    s = s[s["AppNettsted_tekst"].isin(SOCIAL)]
    for _, r in s.iterrows():
        rows.append((f"daglig_{SOCIAL[r['AppNettsted_tekst']]}", r["Kjonn"], r["Alder"], r["value"]))
    st = pd.read_parquet(RAW / "media_stream_14512.parquet")
    for _, r in st.iterrows():
        rows.append((f"daglig_{STREAM[r['StrommeTjeneste']]}", r["Kjonn"], r["Alder"], r["value"]))
    sh = pd.read_parquet(RAW / "media_shop_07001.parquet")
    sh = sh[sh["ContentsCode_tekst"].isin(SHOP) & sh["value"].notna()]
    # Nyeste år med tall for hver kategori.
    newest = sh.groupby("ContentsCode_tekst")["Tid"].transform("max")
    sh = sh[sh["Tid"] == newest]
    for _, r in sh.iterrows():
        rows.append((f"netthandel_{SHOP[r['ContentsCode_tekst']]}", r["Kjonn"], r["Alder"], r["value"]))
    d = pd.DataFrame(rows, columns=["kolonne", "kjonn", "alder", "prosent"])
    d["kjonn"] = d["kjonn"].map({"1": "mann", "2": "kvinne"})
    return d


BANDS_MEDIA = [(16, 24, "16-24"), (25, 44, "25-44"), (45, 64, "45-64"), (65, 79, "65-79"), (80, 200, "80+")]
BANDS_SHOP = [(16, 24, "16-24"), (25, 34, "25-34"), (35, 44, "35-44"), (45, 54, "45-54"), (55, 64, "55-64"),
              (65, 74, "65-74"), (75, 200, "75-79")]   # 80+ har ikke tall; bruker 75–79


def attach(a: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if not (RAW / "media_social_14511.parquet").exists():
        return a
    r = rates().set_index(["kolonne", "kjonn", "alder"])["prosent"] / 100
    out = a.copy()
    bm, bs = _band(a["alder"], BANDS_MEDIA), _band(a["alder"], BANDS_SHOP)
    for col in COLUMNS:
        b = bs if col.startswith("netthandel_") else bm
        p = r.reindex(pd.MultiIndex.from_arrays([[col] * len(a), a["kjonn"], b])).to_numpy()
        p = np.nan_to_num(p, nan=0.0)
        out[col] = np.where(rng.random(len(a)) < p, "ja", "nei")
    return out


if __name__ == "__main__":
    fetch()
