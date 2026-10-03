"""L2 – knytter ESS-respondenter (donorer) til syntetiske agenter.

Metode: statistisk matching (nearest-neighbour hot-deck). For hver agent finnes
de k ESS-respondentene som ligner mest på demografi, og én trekkes med
sannsynlighet etter ESS-vekt og hvor ny runden er. Agenten arver donorens svar.
Fordelen fremfor én modell per spørsmål er at sammenhengen mellom svarene
bevares – den som vektlegger trygghet, vektlegger typisk også tradisjon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from synthpanel import config
from synthpanel.values.ess import VALUE_ITEMS

# --- Felles trekk mellom agent og ESS-respondent ----------------------------

EDU_AGENT = {"grunnskole": 1, "videregaende": 2, "fagskole": 3, "uh_kort": 4, "uh_lang": 5, "uoppgitt": np.nan}
# ES-ISCED: 1–2 grunnskole, 3–4 videregående, 5 høyere yrkesfaglig/kort, 6 bachelor, 7 master+
EDU_ESS = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3.5, 6: 4, 7: 5}

FYLKE_NUTS = {"03": "NO08", "31": "NO08", "32": "NO08", "33": "NO08",
              "39": "NO09", "40": "NO09", "42": "NO09",
              "11": "NO0A", "46": "NO0A", "15": "NO0A",
              "34": "NO02", "50": "NO06", "18": "NO07", "55": "NO07", "56": "NO07"}
# Runde 9 bruker NUTS 2016. Regioner som ble delt, matcher flere nye regioner.
NUTS2016 = {"NO01": {"NO08"}, "NO02": {"NO02"}, "NO03": {"NO08", "NO09"}, "NO04": {"NO09", "NO0A"},
            "NO05": {"NO0A"}, "NO06": {"NO06"}, "NO07": {"NO07"}}

STATUS_ESS = {1: "sysselsatt", 2: "student", 3: "arbeidsledig", 4: "arbeidsledig", 5: "aap_ufor",
              6: "pensjonist", 7: "annet", 8: "annet", 9: "annet"}
SENTRAL_NUM = {"01": 1.5, "02": 2.5, "03": 3.0, "04": 3.5, "05": 4.0, "06": 4.5}


def ess_features(e: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=e.index)
    f["kjonn"] = e["gndr"].map({1: "mann", 2: "kvinne"})
    f["alder"] = e["agea"]
    f["edu"] = e["eisced"].map(EDU_ESS)
    f["inc"] = e["hinctnta"]
    f["regions"] = e["region"].map(lambda r: NUTS2016.get(r, {r}) if isinstance(r, str) else set())
    foreign_parents = (e["facntr"] == 2) & (e["mocntr"] == 2)
    f["innv"] = np.where(e["brncntr"] == 2, "innvandrer",
                         np.where(foreign_parents, "norskfodt_innv_foreldre", "ovrige"))
    f["status"] = e["mnactic"].map(STATUS_ESS)
    f["bosted"] = e["domicil"].where(e["domicil"].between(1, 5))
    f["alene"] = e["hhmmb"] == 1
    return f


def agent_features(a: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=a.index)
    f["kjonn"] = a["kjonn"]
    f["alder"] = a["alder"]
    f["edu"] = a["utdanning"].map(EDU_AGENT)
    f["inc"] = a["inntektsdesil"]
    f["region"] = a["fylke"].map(FYLKE_NUTS)
    f["innv"] = a["innvkat"]
    f["status"] = a["arbeidsstatus"].replace({"tiltak": "annet"})
    f["bosted"] = a["sentralitet"].map(SENTRAL_NUM)
    f["alene"] = a["husholdning"] == "aleneboende"
    return f


def _distance(af: pd.DataFrame, ef: pd.DataFrame, w: dict) -> np.ndarray:
    """Avstandsmatrise (agenter × donorer). Manglende verdier gir halv straff."""
    def num(col, scale, weight):
        x = af[col].to_numpy(float)[:, None]
        y = ef[col].to_numpy(float)[None, :]
        d = np.abs(x - y) / scale
        return weight * np.where(np.isnan(d), 0.5, d)

    def cat(col, weight):
        x = af[col].to_numpy()[:, None]
        y = ef[col].to_numpy()[None, :]
        return weight * (x != y)

    D = (cat("kjonn", w["kjonn"]) + num("alder", 10, w["alder_per_10ar"])
         + num("edu", 1, w["utdanning_per_niva"]) + num("inc", 3, w["inntekt_per_3_desiler"])
         + cat("innv", w["innvandring"]) + cat("status", w["status"])
         + num("bosted", 2, w["bosted"]) + cat("alene", w["alene"]))
    # Region: treff hvis agentens region er blant donorens mulige regioner.
    reg = af["region"].to_numpy()
    donor_regs = ef["regions"].to_numpy()
    uniq = pd.unique(reg)
    reg_pen = np.zeros(D.shape)
    for r in uniq:
        rows = reg == r
        miss = np.array([r not in s for s in donor_regs])
        reg_pen[np.ix_(rows, miss)] = w["region"]
    return D + reg_pen


def match(a: pd.DataFrame, donors: pd.DataFrame, rng: np.random.Generator, chunk: int = 1500) -> np.ndarray:
    """Returnerer donorindeks (posisjon i `donors`) for hver agent."""
    cfg = config.load("values")
    w, k = cfg["matching"]["weights"], cfg["matching"]["k"]
    recency = {int(r): s["recency"] for r, s in cfg["ess"]["rounds"].items()}
    ef = ess_features(donors).reset_index(drop=True)
    dw = (donors["pspwght"].fillna(1.0) * donors["essround"].map(recency).fillna(0.5)).to_numpy()

    af = agent_features(a)
    # Agenter med lik profil får samme kandidatliste – regn avstand per unik profil.
    key_cols = ["kjonn", "alder", "edu", "inc", "region", "innv", "status", "bosted", "alene"]
    codes, uniques = pd.factorize(pd.MultiIndex.from_frame(af[key_cols].astype(object).fillna("NA")))
    first = pd.Series(np.arange(len(af))).groupby(codes).first().to_numpy()
    upf = af.iloc[first].reset_index(drop=True)

    cand = np.empty((len(upf), k), dtype=int)
    for s in range(0, len(upf), chunk):
        D = _distance(upf.iloc[s:s + chunk], ef, w)
        cand[s:s + chunk] = np.argpartition(D, k, axis=1)[:, :k]

    out = np.empty(len(af), dtype=int)
    for i in range(len(af)):
        c = cand[codes[i]]
        p = dw[c] / dw[c].sum()
        out[i] = rng.choice(c, p=p)
    return out


# --- Avledede mål ------------------------------------------------------------

HIGHER_ORDER = {
    "apenhet": ["ipcrtiv", "impfree", "impdiff", "ipadvnt", "ipgdtim", "impfun"],
    "trygghet": ["impsafe", "ipstrgv", "ipfrule", "ipbhprp", "ipmodst", "imptrad"],
    "selvhevdelse": ["imprich", "iprspot", "ipshabt", "ipsuces"],
    "fellesskap": ["ipeqopt", "ipudrst", "impenv", "iphlppl", "iplylfr"],
}


def derive(e: pd.DataFrame) -> pd.DataFrame:
    """Lag tolkbare mål fra ESS-svarene (én rad per respondent)."""
    out = pd.DataFrame(index=e.index)
    imp = 7 - e[VALUE_ITEMS]                     # 6 = «svært lik meg»
    centered = imp.sub(imp.mean(axis=1), axis=0)  # sentrering per person (ESS-anbefaling)
    for name, items in HIGHER_ORDER.items():
        out[f"score_{name}"] = centered[items].mean(axis=1)
    out["score_tillit"] = e[["trstprl", "trstlgl", "trstplc", "trstplt"]].mean(axis=1)
    out["risikovilje"] = pd.cut(e["ipadvnt"], [0, 2, 4, 6], labels=["høy", "middels", "lav"]).astype(object)
    out["politisk_sted"] = pd.cut(e["lrscale"], [-1, 3, 6, 10], labels=["venstre", "sentrum", "høyre"]).astype(object)
    out["religiositet"] = pd.cut(e["rlgdgr"], [-1, 3, 6, 10], labels=["lav", "middels", "høy"]).astype(object)
    out["politisk_interesse"] = e["polintr"].map({1: "høy", 2: "høy", 3: "lav", 4: "lav"})
    out["klimabekymring"] = e["wrclmch"].map({1: "lav", 2: "lav", 3: "middels", 4: "høy", 5: "høy"})
    return out


def tertile_labels(score: pd.Series, weights: pd.Series) -> tuple[float, float]:
    """Vektede tertilgrenser (lav / middels / høy)."""
    s = score.dropna()
    w = weights.reindex(s.index).fillna(1.0)
    order = np.argsort(s.to_numpy())
    cum = np.cumsum(w.to_numpy()[order]) / w.sum()
    vals = s.to_numpy()[order]
    return float(vals[np.searchsorted(cum, 1 / 3)]), float(vals[np.searchsorted(cum, 2 / 3)])


def attach(a: pd.DataFrame, donors: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Legg verdilag på agentene. Klimaspørsmålet finnes bare i runde 10–11, så det
    får egen donor fra den delen av utvalget."""
    d = derive(donors).reset_index(drop=True)
    w = donors["pspwght"].fillna(1.0).reset_index(drop=True)
    cuts = {n: tertile_labels(d[f"score_{n}"], w) for n in list(HIGHER_ORDER) + ["tillit"]}

    idx = match(a, donors, rng)
    out = a.copy()
    out["ess_donor"] = donors["resp_id"].to_numpy()[idx]
    for n in list(HIGHER_ORDER) + ["tillit"]:
        sc = d[f"score_{n}"].to_numpy()[idx]
        lo, hi = cuts[n]
        out[f"score_{n}"] = sc
        out[f"verdi_{n}" if n != "tillit" else "tillit"] = np.select(
            [np.isnan(sc), sc < lo, sc < hi], ["ukjent", "lav", "middels"], "høy")
    for col in ["risikovilje", "politisk_sted", "religiositet", "politisk_interesse"]:
        out[col] = d[col].to_numpy()[idx]

    has_klima = donors["wrclmch"].notna().to_numpy()
    kd = donors[has_klima].reset_index(drop=True)
    kidx = match(a, kd, rng)
    out["klimabekymring"] = derive(kd)["klimabekymring"].to_numpy()[kidx]
    for col in ["risikovilje", "politisk_sted", "religiositet", "politisk_interesse", "klimabekymring"]:
        out[col] = out[col].fillna("ukjent")
    return out
