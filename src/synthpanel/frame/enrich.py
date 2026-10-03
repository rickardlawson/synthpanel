"""L0b – beriking av agentene med innvandringskategori, arbeidsstatus,
husholdningstype og husholdningsinntekt.

Hver funksjon tar agenttabellen, trekker en ny egenskap fra en betinget
fordeling hentet fra SSB, og returnerer også forventede totaler
(marginaler) som raking-steget kalibrerer mot.

Alle antakelser er samlet i docs/architecture.md under «Kjente svakheter».
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from synthpanel import config

RAW = config.RAW_DIR

# ---------------------------------------------------------------------------
# Hjelpere
# ---------------------------------------------------------------------------

def band(age: pd.Series, bands: list[tuple[int, int, str]]) -> pd.Series:
    out = pd.Series(index=age.index, dtype="object")
    for lo, hi, label in bands:
        out[(age >= lo) & (age <= hi)] = label
    return out


def draw_categorical(
    a: pd.DataFrame, probs: pd.DataFrame, keys: list[str], col: str, rng: np.random.Generator,
    weight_col: str = "p",
) -> pd.Series:
    """Trekk `col` for hver agent fra `probs` (kolonner: keys + col + weight_col)."""
    out = pd.Series(index=a.index, dtype="object")
    indexed = probs.set_index(keys).sort_index()
    for key, idx in a.groupby(keys).groups.items():
        try:
            dist = indexed.loc[key]
        except KeyError as e:
            raise KeyError(f"Mangler fordeling for {dict(zip(keys, key if isinstance(key, tuple) else (key,)))}") from e
        p = dist[weight_col].to_numpy(dtype=float)
        p = np.nan_to_num(p, nan=0.0)
        if p.sum() <= 0:
            raise ValueError(f"Tom fordeling for {key} i {col}")
        out.loc[idx] = rng.choice(dist[col].to_numpy(), size=len(idx), p=p / p.sum())
    return out


# ---------------------------------------------------------------------------
# 1. Innvandringskategori + utdanning justert for innvandringskategori
# ---------------------------------------------------------------------------

EDU_09599 = {"00": "uoppgitt", "1-2": "grunnskole", "3-5a": "videregaende", "11": "fagskole",
             "6": "uh_kort", "7-8": "uh_lang", "9": "uoppgitt"}
INNV_09599 = {"B": "innvandrer", "C": "norskfodt_innv_foreldre", "Rest": "ovrige"}
BANDS_09599 = [(16, 19, "16-19"), (20, 24, "20-24"), (25, 29, "25-29"), (30, 34, "30-34"),
               (35, 39, "35-39"), (40, 49, "40-49"), (50, 59, "50-59"), (60, 66, "60-66"),
               (67, 200, "067+")]


def edu_immigrant_table() -> pd.DataFrame:
    d = pd.read_parquet(RAW / "edu_immigrant_age_09599.parquet")
    d = d.assign(utdanning=d["UtdanNivaa"].map(EDU_09599), innvkat=d["InnvandrKat"].map(INNV_09599),
                 b09599=d["Alder"], kjonn=d["Kjonn"])
    return d.groupby(["kjonn", "b09599", "innvkat", "utdanning"], as_index=False)["value"].sum()


def draw_innvkat(a: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Skill innvandrere fra norskfødte med innvandrerforeldre blant agenter med
    utenlandsk landbakgrunn. Andelen hentes fra 09599 per kjønn × alder."""
    t = edu_immigrant_table()
    t = t[t["innvkat"] != "ovrige"].groupby(["kjonn", "b09599", "innvkat"], as_index=False)["value"].sum()
    t = t.rename(columns={"value": "p"})
    out = pd.Series("ovrige", index=a.index, dtype="object")
    foreign = a["bakgrunn"] != "norsk"
    out.loc[foreign] = draw_categorical(a[foreign], t, ["kjonn", "b09599"], "innvkat", rng)
    return out


def edu_adjustment() -> pd.DataFrame:
    """Forholdstall P(utdanning | innvkat, kjønn, alder) / P(utdanning | kjønn, alder)."""
    t = edu_immigrant_table()
    within = t.assign(p_innv=t["value"] / t.groupby(["kjonn", "b09599", "innvkat"])["value"].transform("sum"))
    allp = t.groupby(["kjonn", "b09599", "utdanning"], as_index=False)["value"].sum()
    allp["p_all"] = allp["value"] / allp.groupby(["kjonn", "b09599"])["value"].transform("sum")
    m = within.merge(allp[["kjonn", "b09599", "utdanning", "p_all"]], on=["kjonn", "b09599", "utdanning"])
    m["ratio"] = np.where(m["p_all"] > 0, m["p_innv"] / m["p_all"], 0.0)
    return m[["kjonn", "b09599", "innvkat", "utdanning", "ratio"]]


def draw_education(a: pd.DataFrame, edu_fylke: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Utdanning trekkes fra fylkesfordelingen (08921) justert med forholdstallet
    for innvandringskategori (09599), og normaliseres."""
    adj = edu_adjustment()
    combo = a[["fylke", "kjonn", "edu_band", "b09599", "innvkat"]].drop_duplicates()
    probs = combo.merge(edu_fylke.rename(columns={"andel": "p_base"}), on=["fylke", "kjonn", "edu_band"])
    probs = probs.merge(adj, on=["kjonn", "b09599", "innvkat", "utdanning"], how="left")
    probs["p"] = probs["p_base"] * probs["ratio"].fillna(1.0)
    return draw_categorical(a, probs, ["fylke", "kjonn", "edu_band", "b09599", "innvkat"], "utdanning", rng)


def edu_immigrant_margin(exp_bg: pd.DataFrame) -> pd.Series:
    """Mål for kjønn × alder(09599) × innvkat × utdanning.

    Gruppestørrelsene (norsk / utenlandsk bakgrunn) hentes fra samme kilde som
    bakgrunnsmarginalen (07111, via `exp_bg` per ettårig alder), slik at de to
    marginalene ikke motsier hverandre. 09599 brukes bare til (a) delingen mellom
    innvandrere og norskfødte og (b) utdanningsfordelingen innen hver gruppe."""
    t = edu_immigrant_table()
    t["p_edu"] = t["value"] / t.groupby(["kjonn", "b09599", "innvkat"])["value"].transform("sum")

    bgb = exp_bg.assign(b09599=band(exp_bg["alder"], BANDS_09599),
                        norsk=np.where(exp_bg["bakgrunn"] == "norsk", "ovrige", "utenlandsk"))
    sizes = bgb.groupby(["kjonn", "b09599", "norsk"])["n"].sum().unstack("norsk")

    split = t[t["innvkat"] != "ovrige"].groupby(["kjonn", "b09599", "innvkat"])["value"].sum()
    split = split / split.groupby(level=["kjonn", "b09599"]).transform("sum")

    def group_size(row):
        if row.innvkat == "ovrige":
            return sizes.loc[(row.kjonn, row.b09599), "ovrige"]
        return sizes.loc[(row.kjonn, row.b09599), "utenlandsk"] * split.loc[(row.kjonn, row.b09599, row.innvkat)]

    g = t[["kjonn", "b09599", "innvkat"]].drop_duplicates()
    g["N"] = g.apply(group_size, axis=1)
    t = t.merge(g, on=["kjonn", "b09599", "innvkat"])
    t["n"] = t["p_edu"] * t["N"]
    return t.groupby(["kjonn", "b09599", "innvkat", "utdanning"])["n"].sum()


def immigrant_edu_fylke_margin(exp_bg: pd.DataFrame) -> pd.Series:
    """Mål for fylke × (utdanning for innvandrere | «ikke_innvandrer»), fra 12934.

    Nasjonale forholdstall alene overvurderer innvandreres utdanning i Oslo (der
    alle har høy utdanning) og undervurderer den andre steder. Bare innvandrere
    kalibreres her: norskfødte med innvandrerforeldre er unge, og 12934 tar med
    16–17-åringer, som ville skjevfordelt dem mot grunnskole."""
    v = pd.read_parquet(RAW / "edu_immigrant_fylke_12934.parquet")
    v = v[(v["Region"] != "0") & v["InnvandrKat"].isin(["B", "C"])]
    v = v.assign(utdanning=v["UtdanNivaa"].map(EDU_09599))
    counts = v.groupby(["Region", "InnvandrKat", "utdanning"])["value"].sum()
    b_share = (counts.xs("B", level="InnvandrKat").groupby(level="Region").sum()
               / counts.groupby(level="Region").sum())
    p_edu = counts.xs("B", level="InnvandrKat")
    p_edu = p_edu / p_edu.groupby(level="Region").transform("sum")

    by_f = exp_bg.assign(utl=exp_bg["bakgrunn"] != "norsk").groupby(["fylke", "utl"])["n"].sum().unstack("utl")
    n_innv = by_f[True] * b_share.reindex(by_f.index)
    rows = []
    for f in by_f.index:
        for edu, p in p_edu.loc[f].items():
            rows.append((f, edu, n_innv[f] * p))
        rows.append((f, "ikke_innvandrer", by_f.loc[f].sum() - n_innv[f]))
    out = pd.DataFrame(rows, columns=["fylke", "innv_utd", "n"])
    return out.groupby(["fylke", "innv_utd"])["n"].sum()


def innv_utd_key(a: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(a["innvkat"] == "innvandrer", a["utdanning"], "ikke_innvandrer"), index=a.index)


# ---------------------------------------------------------------------------
# 2. Arbeidsstatus
# ---------------------------------------------------------------------------

STATUS = {"A.01": "sysselsatt", "A.09": "arbeidsledig", "U.01": "tiltak", "U.03": "student",
          "U.04-U.05": "aap_ufor", "U.06-U.07": "pensjonist",
          "U.90B": "annet", "U.90C": "annet", "U.90D": "annet"}
EDU4 = {"grunnskole": "1-2", "videregaende": "3-5", "fagskole": "3-5",
        "uh_kort": "6-8", "uh_lang": "6-8", "uoppgitt": "0_9"}
BANDS_STATUS = [(15, 19, "15-19"), (20, 24, "20-24"), (25, 29, "25-29"), (30, 54, "30-54"),
                (55, 61, "55-61"), (62, 66, "62-66"), (67, 200, "67+")]
AKTIV = {"sysselsatt", "student", "tiltak"}


def labour_table() -> pd.DataFrame:
    d = pd.read_parquet(RAW / "labour_status_1242x.parquet")
    d = d[d["HovArbStyrkStatus"].isin(STATUS)]
    d = d.assign(arbeidsstatus=d["HovArbStyrkStatus"].map(STATUS), kjonn=d["Kjonn"],
                 b_status=d["Alder"], edu4=d["UtdNivaa"], innv2=d["InnvandrKat"])
    return d.groupby(["kjonn", "b_status", "edu4", "innv2", "arbeidsstatus"], as_index=False)["value"].sum()


def draw_labour(a: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    t = labour_table().rename(columns={"value": "p"})
    a2 = a.assign(edu4=a["utdanning"].map(EDU4),
                  innv2=np.where(a["innvkat"] == "innvandrer", "B", "A_C-G"))
    # Små celler (f.eks. innvandrere 67+ med uoppgitt utdanning) kan være tomme;
    # fall da tilbake til samme gruppe uten innvandrerskille.
    pooled = t.groupby(["kjonn", "b_status", "edu4", "arbeidsstatus"], as_index=False)["p"].sum()
    sizes = t.groupby(["kjonn", "b_status", "edu4", "innv2"])["p"].sum()
    thin = sizes[sizes < 200].index
    rows = []
    for k, b, e, i in thin:
        rows.append(pooled[(pooled.kjonn == k) & (pooled.b_status == b) & (pooled.edu4 == e)].assign(innv2=i))
    if rows:
        t = pd.concat([t.set_index(["kjonn", "b_status", "edu4", "innv2"]).drop(index=thin).reset_index(),
                       *rows], ignore_index=True)
    return draw_categorical(a2, t, ["kjonn", "b_status", "edu4", "innv2"], "arbeidsstatus", rng)


def labour_margin(pop: pd.DataFrame) -> pd.Series:
    """Mål for kjønn × aldersgruppe × arbeidsstatus, skalert til vår befolkning."""
    t = labour_table().groupby(["kjonn", "b_status", "arbeidsstatus"], as_index=False)["value"].sum()
    t["andel"] = t["value"] / t.groupby(["kjonn", "b_status"])["value"].transform("sum")
    ours = pop.assign(b_status=band(pop["alder"], BANDS_STATUS)).groupby(["kjonn", "b_status"])["personer"].sum()
    t["n"] = t["andel"] * ours.reindex(pd.MultiIndex.from_frame(t[["kjonn", "b_status"]])).to_numpy()
    return t.groupby(["kjonn", "b_status", "arbeidsstatus"])["n"].sum()


BANDS_13678 = [(15, 19, "15-19"), (20, 24, "20-24"), (25, 29, "25-29"), (30, 39, "30-39"),
               (40, 49, "40-49"), (50, 61, "50-61"), (62, 200, "62+")]


def activity_margin(pop: pd.DataFrame) -> pd.Series:
    """Mål for fylke × kjønn × aktiv (i arbeid, utdanning eller tiltak) fra 13678."""
    d = pd.read_parquet(RAW / "activity_fylke_13678.parquet")
    d = d[d["Region"] != "0"]
    piv = d.pivot_table(index=["Region", "Kjonn", "Alder"], columns="HovArbStyrkStatus", values="value", aggfunc="sum")
    piv["andel"] = piv["A.01xU.01xU.03"] / piv["TOT"]
    share = piv["andel"].rename_axis(["fylke", "kjonn", "b13678"])
    p = pop.assign(b13678=band(pop["alder"], BANDS_13678)).groupby(["fylke", "kjonn", "b13678"])["personer"].sum()
    s = share.reindex(p.index)
    aktiv = (p * s).groupby(level=["fylke", "kjonn"]).sum()
    total = p.groupby(level=["fylke", "kjonn"]).sum()
    out = pd.concat({"ja": aktiv, "nei": total - aktiv}, names=["aktiv"])
    return out.reorder_levels(["fylke", "kjonn", "aktiv"]).sort_index()


def employment_rate_by_age(pop: pd.DataFrame) -> pd.DataFrame:
    """Sysselsettingsrate per kjønn og ettårig alder.

    06161 dekker 15–74 år. For 75+ utledes raten som det som gjenstår av 12426
    sin andel sysselsatte 67+ når 67–74 er trukket fra."""
    e = pd.read_parquet(RAW / "employment_age_06161.parquet")
    e = e.assign(kjonn=e["Kjonn"], alder=e["Alder"].astype(int), rate=e["value"] / 100)[["kjonn", "alder", "rate"]]
    lab = labour_table()
    s67 = lab[lab["b_status"] == "67+"].groupby(["kjonn", "arbeidsstatus"])["value"].sum()
    s67 = (s67.xs("sysselsatt", level="arbeidsstatus") / s67.groupby(level="kjonn").sum())
    rows = []
    for kj in ["1", "2"]:
        p = pop[pop["kjonn"] == kj].groupby("alder")["personer"].sum()
        n67, n75 = p[p.index >= 67].sum(), p[p.index >= 75].sum()
        known = e[(e["kjonn"] == kj) & e["alder"].between(67, 74)].set_index("alder")["rate"]
        r75 = max((s67[kj] * n67 - (known * p.reindex(known.index)).sum()) / n75, 0.0)
        rows += [(kj, a, r75) for a in p.index if a >= 75]
    return pd.concat([e, pd.DataFrame(rows, columns=["kjonn", "alder", "rate"])], ignore_index=True)


def adjust_elderly_employment(a: pd.DataFrame, pop: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Statustabellene har 62–66 og 67+ samlet. Uten justering blir en 85-åring like
    ofte «i arbeid» som en 67-åring. Her flyttes agenter 62+ mellom sysselsatt og
    pensjonist slik at sannsynligheten følger ettårig alder (06161)."""
    status = a["arbeidsstatus"].copy()
    rates = employment_rate_by_age(pop).set_index(["kjonn", "alder"])["rate"]
    old = a["alder"] >= 62
    for (kj, b), idx in a[old].groupby(["kjonn", "b_status"]).groups.items():
        p0 = (status.loc[idx] == "sysselsatt").mean()
        target = rates.reindex(pd.MultiIndex.from_arrays([a.loc[idx, "kjonn"], a.loc[idx, "alder"]])).to_numpy()
        is_emp = (status.loc[idx] == "sysselsatt").to_numpy()
        u = rng.random(len(idx))
        down = is_emp & (target < p0) & (u < 1 - target / max(p0, 1e-9))
        up = ~is_emp & (target > p0) & (u < (target - p0) / max(1 - p0, 1e-9))
        ids = np.asarray(idx)
        status.loc[ids[down]] = "pensjonist"
        status.loc[ids[up]] = "sysselsatt"
    return status


def employment_age_margin(pop: pd.DataFrame) -> pd.Series:
    """Mål for kjønn × alder (62–74 ettårig, 75+ samlet) × sysselsatt.
    Under 62 legges i én gruppe med mål fra statusmarginalen."""
    rates = employment_rate_by_age(pop)
    p = pop.groupby(["kjonn", "alder"])["personer"].sum().reset_index().merge(rates, on=["kjonn", "alder"], how="left")
    p["b_emp"] = np.where(p["alder"] < 62, "u62", np.where(p["alder"] >= 75, "75+", p["alder"].astype(str)))
    lab = labour_margin(pop).reset_index()
    young = lab[lab["b_status"].isin(["15-19", "20-24", "25-29", "30-54", "55-61"])]
    young_emp = young[young["arbeidsstatus"] == "sysselsatt"].groupby("kjonn")["n"].sum()
    rows = []
    for (kj, b), g in p.groupby(["kjonn", "b_emp"]):
        n = g["personer"].sum()
        emp = young_emp[kj] if b == "u62" else (g["personer"] * g["rate"]).sum()
        rows += [(kj, b, "ja", emp), (kj, b, "nei", n - emp)]
    out = pd.DataFrame(rows, columns=["kjonn", "b_emp", "syss", "n"])
    return out.set_index(["kjonn", "b_emp", "syss"])["n"]


def emp_keys(a: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    b = np.where(a["alder"] < 62, "u62", np.where(a["alder"] >= 75, "75+", a["alder"].astype(str)))
    return pd.Series(b, index=a.index), pd.Series(np.where(a["arbeidsstatus"] == "sysselsatt", "ja", "nei"), index=a.index)


# ---------------------------------------------------------------------------
# 3. Husholdningstype
# ---------------------------------------------------------------------------

# Internt skilles flerfamiliehusholdninger etter barn, så alderskorreksjonen
# under kan bruke dem; de slås sammen til «flerfamilie» i ferdig populasjon.
HH_06071 = {"001": "aleneboende", "002": "par_uten_barn", "003": "par_smaa_barn", "004": "par_store_barn",
            "005": "enslig_smaa_barn", "006": "enslig_store_barn", "007": "voksne_barn",
            "008": "flerfamilie_ingen", "009": "flerfamilie_smaa", "010": "flerfamilie_store", "000": "annet"}
HH_10986 = {"1.1": "aleneboende", "1.2": "par_uten_barn", "1.3": "par_smaa_barn", "1.4": "par_store_barn",
            "1.5": "enslig_smaa_barn", "1.6": "enslig_store_barn", "1.7": "voksne_barn",
            "2.1": "flerfamilie_ingen", "2.2": "flerfamilie_smaa", "2.3": "flerfamilie_store"}
KIDS_GROUP = {"par_smaa_barn": "smaa", "enslig_smaa_barn": "smaa", "flerfamilie_smaa": "smaa",
              "par_store_barn": "store", "enslig_store_barn": "store", "flerfamilie_store": "store"}
BANDS_HH = [(16, 29, "18-29"), (30, 44, "30-44"), (45, 61, "45-61"), (62, 66, "62-66"), (67, 200, "67+")]
STORE_BARN_TYPES = {"004": "par_store_barn", "006": "enslig_store_barn", "010": "flerfamilie_store"}


def collapse_household(s: pd.Series) -> pd.Series:
    return s.where(~s.str.startswith("flerfamilie"), "flerfamilie")


def household_national(pop_all: pd.DataFrame) -> pd.DataFrame:
    """Personer 18+ per kjønn × aldersgruppe × husholdningstype (intern inndeling).

    1. 06071 gir fordelingen for 16–29, 30–66 og 67+. 16–17-åringene trekkes ut
       ved å fordele dem som barn i husholdninger med store barn (fordeling fra 0–15 år).
    2. 30–66 er for grovt: en 32-åring og en 64-åring ville fått like stor sjanse for
       småbarn. Gruppen deles i 30–44, 45–61 og 62–66, og andelen med små barn,
       store barn og uten barn settes fra 12836 (30–44 utledes fra 25–44). For 62–66
       utledes andelen som rest, slik at totalen for 30–66 fortsatt stemmer med 06071.
    """
    d = pd.read_parquet(RAW / "household_persons_06071.parquet")
    d = d.assign(kjonn=d["Kjonn"], b=d["Alder"], husholdning=d["HusholdType"].map(HH_06071))
    d["value"] = d["value"].astype(float)

    kids = d[(d["b"] == "00-15") & d["HusholdType"].isin(STORE_BARN_TYPES)]
    kids = kids.groupby(["kjonn", "husholdning"])["value"].sum()
    kids = kids / kids.groupby(level="kjonn").transform("sum")
    n1617 = pop_all[pop_all["alder"].isin([16, 17])].groupby("kjonn")["personer"].sum()

    prior = d[d["b"] != "00-15"].groupby(["kjonn", "b", "husholdning"])["value"].sum()
    for kj in ["1", "2"]:
        for hh, share in kids.loc[kj].items():
            prior.loc[(kj, "16-29", hh)] = max(prior.loc[(kj, "16-29", hh)] - share * n1617[kj], 0.0)

    fam = pd.read_parquet(RAW / "family_kids_age_12836.parquet")
    fam = fam.assign(g=fam["FamilieType"].map({"2.1+2.3": "smaa", "2.2+2.4": "store", "3.1+3.2+3.3": "ingen"}))
    fam = fam.groupby(["Kjonn", "Alder", "g"])["value"].sum()
    fam = fam / fam.groupby(level=["Kjonn", "Alder"]).transform("sum")

    def n_age(kj, lo, hi):
        p = pop_all[(pop_all["kjonn"] == kj) & pop_all["alder"].between(lo, hi)]
        return float(p["personer"].sum())

    rows = []
    for kj in ["1", "2"]:
        for b_src, b_out in [("16-29", "18-29"), ("067+", "67+")]:
            for hh, v in prior.loc[(kj, b_src)].items():
                rows.append((kj, b_out, hh, v))
        base = prior.loc[(kj, "30-66")]
        base = base / base.sum()
        grp = base.groupby(lambda h: KIDS_GROUP.get(h, "ingen")).sum()
        n25, n30, n45, n62 = n_age(kj, 25, 29), n_age(kj, 30, 44), n_age(kj, 45, 61), n_age(kj, 62, 66)
        # 12836 har 25–44 samlet. Trekk ut 25–29 (andel fra 18–29-fordelingen over)
        # slik at 30–44 ikke undervurderes og overskuddet havner feil i 62–66.
        young = prior.loc[(kj, "16-29")]
        t25 = young.groupby(lambda h: KIDS_GROUP.get(h, "ingen")).sum() / young.sum()
        t30 = ((fam.loc[(kj, "25-44")] * (n25 + n30) - t25.reindex(fam.loc[(kj, "25-44")].index).fillna(0) * n25) / n30).clip(lower=0)
        t30 = t30 / t30.sum()
        t45 = fam.loc[(kj, "45-61")]
        t62 = ((grp * (n30 + n45 + n62) - t30 * n30 - t45 * n45) / n62).clip(lower=0)
        t62 = t62 / t62.sum()
        for b_out, target, n in [("30-44", t30, n30), ("45-61", t45, n45), ("62-66", t62, n62)]:
            for hh, p in base.items():
                g = KIDS_GROUP.get(hh, "ingen")
                rows.append((kj, b_out, hh, p * target[g] / grp[g] * n if grp[g] > 0 else 0.0))
    return pd.DataFrame(rows, columns=["kjonn", "hh_band", "husholdning", "value"])


def household_regional_factor() -> pd.DataFrame:
    """Hvor mye vanligere hver husholdningstype er i et fylke enn i landet (10986, alle aldre)."""
    d = pd.read_parquet(RAW / "household_fylke_10986.parquet")
    d = d.assign(husholdning=d["HushType"].map(HH_10986))
    d = d.groupby(["Region", "husholdning"], as_index=False)["value"].sum()
    d["share"] = d["value"] / d.groupby("Region")["value"].transform("sum")
    nat = d[d["Region"] == "0"].set_index("husholdning")["share"]
    d = d[d["Region"] != "0"].copy()
    d["faktor"] = d["share"] / d["husholdning"].map(nat)
    return d.rename(columns={"Region": "fylke"})[["fylke", "husholdning", "faktor"]]


def draw_household(a: pd.DataFrame, pop_all: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Returnerer intern husholdningstype (flerfamilie delt etter barn)."""
    nat = household_national(pop_all).rename(columns={"value": "p_nat"})
    fac = household_regional_factor()
    combo = a[["fylke", "kjonn", "hh_band"]].drop_duplicates()
    probs = combo.merge(nat, on=["kjonn", "hh_band"]).merge(fac, on=["fylke", "husholdning"], how="left")
    probs["p"] = probs["p_nat"] * probs["faktor"].fillna(1.0)
    return draw_categorical(a, probs, ["fylke", "kjonn", "hh_band"], "husholdning", rng)


def household_margin(pop: pd.DataFrame, pop_all: pd.DataFrame) -> pd.Series:
    nat = household_national(pop_all)
    nat["husholdning"] = collapse_household(nat["husholdning"])
    nat = nat.groupby(["kjonn", "hh_band", "husholdning"], as_index=False)["value"].sum()
    nat["andel"] = nat["value"] / nat.groupby(["kjonn", "hh_band"])["value"].transform("sum")
    ours = pop.assign(hh_band=band(pop["alder"], BANDS_HH)).groupby(["kjonn", "hh_band"])["personer"].sum()
    nat["n"] = nat["andel"] * ours.reindex(pd.MultiIndex.from_frame(nat[["kjonn", "hh_band"]])).to_numpy()
    return nat.groupby(["kjonn", "hh_band", "husholdning"])["n"].sum()


# ---------------------------------------------------------------------------
# 3b. Lavinntekt (EU-skala 60 %)
# ---------------------------------------------------------------------------

def _lowinc_rates() -> pd.Series:
    d = pd.read_parquet(RAW / "low_income_groups_12599.parquet")
    piv = d.pivot_table(index="Forbruksenhet", columns="ContentsCode", values="value")
    return piv


def lowinc_groups(a: pd.DataFrame) -> pd.DataFrame:
    """Gruppenøkler som lavinntektsratene i 12599/09570 er publisert for."""
    age, hh = a["alder"], a["husholdning"]
    g = pd.DataFrame(index=a.index)
    g["lav_hh"] = "ovrig"
    alene = hh == "aleneboende"
    g.loc[alene & (age < 35), "lav_hh"] = "alene_u35"
    g.loc[alene & age.between(35, 49), "lav_hh"] = "alene_35_49"
    g.loc[alene & age.between(50, 66), "lav_hh"] = "alene_50_66"
    g.loc[alene & (age >= 67), "lav_hh"] = "alene_67"
    g.loc[hh.isin(["enslig_smaa_barn", "enslig_store_barn"]), "lav_hh"] = "enslig_forsorger"
    g.loc[hh.isin(["par_smaa_barn", "par_store_barn"]), "lav_hh"] = "par_med_barn"
    g["lav_status"] = a["arbeidsstatus"].where(a["arbeidsstatus"].isin(["aap_ufor", "pensjonist"]), "ovrig")
    eu = a["bakgrunn"] == "europa"
    g["lav_innv"] = "ovrige"
    g.loc[(a["innvkat"] == "innvandrer") & eu, "lav_innv"] = "innv_eu"
    g.loc[(a["innvkat"] == "innvandrer") & ~eu, "lav_innv"] = "innv_andre"
    g.loc[(a["innvkat"] == "norskfodt_innv_foreldre") & eu, "lav_innv"] = "nf_eu"
    g.loc[(a["innvkat"] == "norskfodt_innv_foreldre") & ~eu, "lav_innv"] = "nf_andre"
    edu = a["utdanning"].map({"grunnskole": "0-2", "videregaende": "3-5", "fagskole": "3-5",
                              "uh_kort": "6", "uh_lang": "7-8", "uoppgitt": "9"})
    g["lav_edu"] = np.where(age.between(18, 66), edu, "67+")
    return g


HH_CODES = {"alene_u35": "20", "alene_35_49": "65", "alene_50_66": "70", "alene_67": "75",
            "enslig_forsorger": "79", "par_med_barn": "06"}
INNV_CODES = {"innv_eu": "83", "innv_andre": "85", "nf_eu": "84", "nf_andre": "86"}


def lowinc_target_rates() -> dict[str, dict[str, float]]:
    """Lavinntektsrater per gruppe. «ovrig» settes som rest i marginalene."""
    r = _lowinc_rates()["EUskala60"] / 100
    n = _lowinc_rates()["AntPersoner"]
    aap_ufor = (r["80"] * n["80"] + r["51"] * n["51"]) / (n["80"] + n["51"])
    e = pd.read_parquet(RAW / "low_income_education_09570.parquet").set_index("UtdNivaa")["value"]
    adult_annual = (r["77"] * n["77"] + r["78"] * n["78"]) / (n["77"] + n["78"])
    return {
        "total": float(r["76"]),
        "lav_hh": {k: float(r[c]) for k, c in HH_CODES.items()},
        "lav_status": {"aap_ufor": float(aap_ufor), "pensjonist": float(r["82b"])},
        "lav_innv": {k: float(r[c]) for k, c in INNV_CODES.items()},
        # 09570 er vedvarende lavinntekt; forholdet mellom utdanningsnivåene brukes
        # til å fordele den årlige raten for 18–66 år.
        "lav_edu": {k: float(adult_annual * e[k] / e["TOT"]) for k in ["0-2", "3-5", "6", "7-8", "9"]},
    }


def draw_lowinc(a: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Trekk lavinntekt med odds satt sammen av husholdning, status, innvandring og
    utdanning (hver gruppe relativt til totalen). Raking justerer deretter mot
    gruppenes publiserte rater."""
    g = lowinc_groups(a)
    t = lowinc_target_rates()
    logit = lambda p: np.log(p / (1 - p))
    base = logit(t["total"])
    z = np.full(len(a), base)
    for dim in ["lav_hh", "lav_status", "lav_innv", "lav_edu"]:
        lift = g[dim].map({k: logit(v) - base for k, v in t[dim].items()}).fillna(0.0).to_numpy()
        z += lift
    p = 1 / (1 + np.exp(-z))
    return pd.Series(np.where(rng.random(len(a)) < p, "ja", "nei"), index=a.index)


def lowinc_margins(a_groups: pd.DataFrame, group_sizes: dict[str, pd.Series], total_n: float) -> dict:
    """Marginaler (gruppe, lavinntekt). Publiserte grupper får sin rate; «ovrig»
    får resten slik at totalen blir 12599 sin rate for 18+."""
    t = lowinc_target_rates()
    total_low = t["total"] * total_n
    out = {}
    for dim in ["lav_hh", "lav_status", "lav_innv", "lav_edu"]:
        sizes = group_sizes[dim]
        rows = []
        known_low = 0.0
        for grp, n in sizes.items():
            if grp in t[dim]:
                low = t[dim][grp] * n
                known_low += low
                rows += [(grp, "ja", low), (grp, "nei", n - low)]
        rest = [g for g in sizes.index if g not in t[dim]]
        rest_n = sizes[rest].sum()
        rest_low = min(max(total_low - known_low, 0.0), rest_n)
        for grp in rest:
            share = sizes[grp] / rest_n
            rows += [(grp, "ja", rest_low * share), (grp, "nei", (rest_n - rest_low) * share)]
        out[(dim, "lavinntekt")] = pd.DataFrame(rows, columns=[dim, "lavinntekt", "n"]).set_index([dim, "lavinntekt"])["n"]
    return out


# ---------------------------------------------------------------------------
# 4. Husholdningsinntekt (desil av inntekt etter skatt, nasjonale grenser)
# ---------------------------------------------------------------------------

def income_type(a: pd.DataFrame) -> pd.Series:
    """Oversett agentens husholdning til typene i 12563. For par uten barn brukes
    agentens egen alder som stedfortreder for eldste person."""
    age = a["alder"]
    t = pd.Series("2", index=a.index, dtype="object")
    alene, par = a["husholdning"] == "aleneboende", a["husholdning"] == "par_uten_barn"
    t[alene & (age < 45)] = "1.1.1-1.1.2"
    t[alene & age.between(45, 66)] = "1.1.3"
    t[alene & (age >= 67)] = "1.1.4"
    t[par & (age < 45)] = "1.2.1-1.2.2"
    t[par & age.between(45, 66)] = "1.2.3"
    t[par & (age >= 67)] = "1.2.4"
    t[a["husholdning"] == "par_smaa_barn"] = "1.3"
    t[a["husholdning"] == "par_store_barn"] = "1.4"
    t[a["husholdning"] == "enslig_smaa_barn"] = "1.5"
    t[a["husholdning"] == "enslig_store_barn"] = "1.6"
    t[a["husholdning"] == "voksne_barn"] = "1.7.1-1.7.2"
    return t


def _income_probs(a2: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_parquet(RAW / "income_deciles_12563.parquet")
    d = d.rename(columns={"Region": "fylke", "HushType": "inntektstype", "Desiler": "desil", "value": "p"})
    nat = d[d["fylke"] == "0"]
    reg = d[d["fylke"] != "0"]
    # Fylker/typer uten publiserte tall (prikket eller manglende) får landsfordelingen.
    complete = reg.groupby(["fylke", "inntektstype"])["p"].apply(lambda s: s.notna().all() and s.sum() > 0)
    reg = reg.set_index(["fylke", "inntektstype"]).loc[complete[complete].index].reset_index()
    combo = a2[["fylke", "inntektstype"]].drop_duplicates()
    have = set(map(tuple, reg[["fylke", "inntektstype"]].drop_duplicates().to_numpy()))
    missing = combo[[tuple(x) not in have for x in combo.to_numpy()]]
    fill = missing.merge(nat.drop(columns=["fylke"]), on="inntektstype")
    return pd.concat([reg[["fylke", "inntektstype", "desil", "p"]], fill[["fylke", "inntektstype", "desil", "p"]]])


def draw_income(a: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Trekk inntektsdesil fra fordelingen for husholdningstype og fylke (12563).

    Innen hver celle får personer med lavinntekt den nederste delen av fordelingen
    (like stor som cellens lavinntektsandel), og de andre resten. Slik henger desil
    sammen med lavinntekt – og dermed med status, utdanning og innvandring."""
    a2 = a.assign(inntektstype=income_type(a))
    probs = _income_probs(a2).set_index(["fylke", "inntektstype"]).sort_index()
    out = pd.Series(0, index=a.index, dtype=int)
    w = a2["_w"] if "_w" in a2 else pd.Series(1.0, index=a.index)
    for key, idx in a2.groupby(["fylke", "inntektstype"]).groups.items():
        dist = probs.loc[key].sort_values("desil")
        p = np.nan_to_num(dist["p"].to_numpy(dtype=float)); p = p / p.sum()
        deciles = dist["desil"].astype(int).to_numpy()
        cdf = np.concatenate([[0.0], np.cumsum(p)])
        low = (a2.loc[idx, "lavinntekt"] == "ja").to_numpy()
        L = float((w.loc[idx][low]).sum() / w.loc[idx].sum())
        u = rng.random(len(idx))
        # lavinntekt: u i [0, L), andre: u i [L, 1)
        q = np.where(low, u * L, L + u * (1 - L))
        out.loc[idx] = deciles[np.clip(np.searchsorted(cdf, q, side="right") - 1, 0, len(deciles) - 1)]
    return out
