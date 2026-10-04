"""Arketypelaget: fire «linser» regnet ut fra det panelet allerede vet.

1. **Verdikartet** – to akser fra Schwartz' verdisirkel: tradisjonell–moderne
   (bevaring vs åpenhet for endring) og materialistisk–idealistisk
   (selvhevdelse vs selvoverskridelse). Fire felt delt ved befolkningens median.
2. **Arketypehjulet** – Jungs tolv arketyper (Mark & Pearson) uttrykt som profiler
   over de ti grunnverdiene; personen får den profilen hen ligner mest på.
3. **Samfunnsroller** – økonomisk og kulturell kapital, tillit og utrygghet
   (inspirert av Bourdieu).
4. **Kriseresiliens** – tillit til systemet, sosialt nettverk og økonomisk buffer.

Verdier, tillit, nettverk og opplevd inntekt kommer fra agentens ESS-donor;
inntekt, utdanning, bolig, status og bosted fra registerlaget. Ingenting
trekkes tilfeldig – samme agent får alltid samme arketyper. Definisjonene står
i configs/archetypes.yaml.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from synthpanel import config
from synthpanel.values.ess import VALUE_ITEMS

BASIC = {"SD": ["ipcrtiv", "impfree"], "ST": ["impdiff", "ipadvnt"], "HE": ["ipgdtim", "impfun"],
         "AC": ["ipshabt", "ipsuces"], "PO": ["imprich", "iprspot"], "SE": ["impsafe", "ipstrgv"],
         "CO": ["ipfrule", "ipbhprp"], "TR": ["ipmodst", "imptrad"], "BE": ["iphlppl", "iplylfr"],
         "UN": ["ipeqopt", "ipudrst", "impenv"]}
COLUMNS = ["verdikart", "verdikart_x", "verdikart_y", "arketype", "arketype_2", "samfunnsrolle", "resiliens"]


def basic_values(e: pd.DataFrame) -> pd.DataFrame:
    """Schwartz' ti grunnverdier per ESS-respondent, sentrert per person (ESS-anbefaling)."""
    imp = 7 - e[VALUE_ITEMS]
    ok = imp.notna().sum(axis=1) >= 15
    centered = imp.sub(imp.mean(axis=1), axis=0)
    out = pd.DataFrame({k: centered[v].mean(axis=1) for k, v in BASIC.items()}, index=e.index)
    out[~ok] = np.nan
    return out


def donor_frame(e: pd.DataFrame) -> pd.DataFrame:
    v = basic_values(e)
    v["moderne"] = v[["SD", "ST"]].mean(axis=1) - v[["SE", "CO", "TR"]].mean(axis=1)
    v["idealist"] = v[["UN", "BE"]].mean(axis=1) - v[["AC", "PO"]].mean(axis=1)
    v["TILLIT"] = e[["trstprl", "trstlgl", "trstplc", "trstplt"]].mean(axis=1)
    v["LYKKE"] = e["happy"]
    for c in ["ppltrst", "sclmeet", "inprdsc", "hincfel", "stfdem"]:
        v[c] = e[c] if c in e else np.nan
    v["resp_id"] = e["resp_id"].to_numpy()
    return v.set_index("resp_id")


def _z(s: pd.Series, w: np.ndarray) -> pd.Series:
    x = s.to_numpy(float)
    m = ~np.isnan(x)
    mu = np.average(x[m], weights=w[m])
    sd = np.sqrt(np.average((x[m] - mu) ** 2, weights=w[m])) or 1.0
    return pd.Series((x - mu) / sd, index=s.index)


def _wmedian(x: np.ndarray, w: np.ndarray) -> float:
    m = ~np.isnan(x)
    o = np.argsort(x[m])
    c = np.cumsum(w[m][o]) / w[m].sum()
    return float(x[m][o][np.searchsorted(c, 0.5)])


def attach(a: pd.DataFrame, donors: pd.DataFrame) -> pd.DataFrame:
    if "ess_donor" not in a.columns:
        return a
    cfg = config.load("archetypes")
    d = donor_frame(donors).reindex(a["ess_donor"])
    d.index = a.index
    w = a["vekt"].to_numpy(float)
    out = a.copy()

    # 1. Verdikart
    zx, zy = _z(d["moderne"], w), _z(d["idealist"], w)
    mx, my = _wmedian(zx.to_numpy(), w), _wmedian(zy.to_numpy(), w)
    out["verdikart_x"], out["verdikart_y"] = zx.round(3), zy.round(3)
    out["verdikart"] = np.where(zx.isna(), "ukjent", np.where(
        zx >= mx, np.where(zy >= my, "moderne_idealist", "moderne_materialist"),
        np.where(zy >= my, "tradisjonell_idealist", "tradisjonell_materialist")))

    # 2. Arketypehjul: cosinus mellom personens standardiserte profil og arketypeprofilene
    dims = list(BASIC) + ["TILLIT", "LYKKE"]
    Z = pd.DataFrame({k: _z(d[k], w) for k in dims}).fillna(0.0).to_numpy()
    names = list(cfg["arketyper"])
    T = np.array([[cfg["arketyper"][n]["profil"].get(k, 0.0) for k in dims] for n in names], float)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    S = Z @ T.T
    order = np.argsort(-S, axis=1)
    missing = d["SD"].isna().to_numpy()
    out["arketype"] = np.where(missing, "ukjent", np.array(names)[order[:, 0]])
    out["arketype_2"] = np.where(missing, "ukjent", np.array(names)[order[:, 1]])

    # 3. Samfunnsroller (prioritert rekkefølge – første regel som slår til)
    trust = _z(d["TILLIT"], w)
    dem = d["stfdem"]
    hinc = d["hincfel"]
    edu, dec, own = a["utdanning"], a["inntektsdesil"].astype(int), a["eierstatus"] != "leier"
    working_age = a["alder"] < 67
    tekno = (dec >= 9) & own & (edu.eq("uh_lang") | (edu.eq("uh_kort") & (dec == 10))) & (trust > -0.3)
    kultur = (edu.eq("uh_lang") | (edu.eq("uh_kort") & out["verdikart"].eq("moderne_idealist"))) & \
        a["sentralitet"].isin(["01", "02", "03"]) & out["verdikart"].isin(["moderne_idealist", "tradisjonell_idealist"])
    skeptic = (trust <= -0.8) & ((dem <= 5) | (d["ppltrst"] <= 4))
    precarious = working_age & ((a["lavinntekt"] == "ja") | (hinc >= 3) |
                                a["arbeidsstatus"].isin(["arbeidsledig", "tiltak", "aap_ufor"]))
    out["samfunnsrolle"] = np.select([tekno, kultur, skeptic, precarious],
                                     ["teknokraten", "kulturelle_eliten", "systemskeptikeren", "utrygge_arbeideren"],
                                     "etablerte_midten")

    # 4. Kriseresiliens
    net = pd.concat([_z(d["sclmeet"], w), _z(d["inprdsc"], w), _z(d["ppltrst"], w)], axis=1).mean(axis=1)
    buf = pd.concat([_z(-hinc, w), _z(a["inntektsdesil"].astype(float), w)], axis=1).mean(axis=1) \
        - 0.6 * (a["lavinntekt"] == "ja")
    vulnerable = (net <= -0.6) & ((trust <= -0.3) | (buf <= -0.5))
    optimist = (trust >= 0.3) & (buf >= 0.0)
    community = (net >= 0.4) & (trust < 0.3)
    out["resiliens"] = np.select([vulnerable, optimist, community],
                                 ["sarbare_individualister", "tillitsfulle_optimister", "fellesskapsbyggerne"],
                                 "stodige_midten")
    return out
