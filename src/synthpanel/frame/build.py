"""Bygg den syntetiske populasjonen (L0) fra rådata i data/raw.

Fremgangsmåte
1. Forventede antall per kommune × kjønn × ettårig alder (07459, eksakt).
2. Utdanningsandeler per fylke × kjønn × aldersgruppe (08921) og
   landbakgrunn per fylke × kjønn × aldersgruppe (07111) knyttes på hver
   ettårig alder. Fra dette regnes forventede marginaler ut.
3. Agenter fordeles på celler (kommune × kjønn × aldersbånd) proporsjonalt
   med folketall, minst én per celle, og får trukket alder, utdanning og
   bakgrunn fra de betingede fordelingene.
4. Vektene rakes slik at alle marginalene treffer SSB-tallene.

Antakelse (v0): utdanning og landbakgrunn er uavhengige gitt fylke, kjønn
og alder. Se docs/architecture.md for kjente svakheter.

Kjør:  python -m synthpanel.frame.build
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from synthpanel import config
from synthpanel.calibrate.raking import effective_sample_size, rake
from synthpanel.frame import enrich

EDU_BANDS = [(16, 19, "16-19"), (20, 24, "20-24"), (25, 29, "25-29"), (30, 39, "30-39"),
             (40, 49, "40-49"), (50, 59, "50-59"), (60, 66, "60-66"), (67, 200, "067+")]
BG_BANDS = [(0, 0, "000"), (1, 5, "01-05"), (6, 12, "06-12"), (13, 15, "13-15"), (16, 19, "16-19"),
            (20, 44, "20-44"), (45, 66, "45-66"), (67, 79, "67-79"), (80, 200, "080+")]


def _band(age: pd.Series, bands) -> pd.Series:
    out = pd.Series(index=age.index, dtype="object")
    for lo, hi, label in bands:
        out[(age >= lo) & (age <= hi)] = label
    return out


def load_population(min_age: int) -> pd.DataFrame:
    p = pd.read_parquet(config.RAW_DIR / "population_07459.parquet")
    p = p[p["value"] > 0].copy()
    p["alder"] = p["Alder"].str.rstrip("+").astype(int)  # "105+" -> 105
    p = p.rename(columns={"Region": "kommune", "Region_tekst": "kommune_navn", "Kjonn": "kjonn"})
    p["fylke"] = p["kommune"].str[:2]
    return p[["kommune", "kommune_navn", "fylke", "kjonn", "alder", "value"]].rename(columns={"value": "personer"})


def education_shares(labels: dict) -> pd.DataFrame:
    e = pd.read_parquet(config.RAW_DIR / "education_08921.parquet")
    e = e[e["Region"] != "0"].rename(columns={"Region": "fylke", "Kjonn": "kjonn", "Alder": "edu_band"})
    e["utdanning"] = e["UtdanNivaa"].map(labels)
    e = e.groupby(["fylke", "kjonn", "edu_band", "utdanning"], as_index=False)["value"].sum()
    e["andel"] = e["value"] / e.groupby(["fylke", "kjonn", "edu_band"])["value"].transform("sum")
    return e[["fylke", "kjonn", "edu_band", "utdanning", "andel"]]


def background_shares(pop_all_ages: pd.DataFrame, labels: dict) -> pd.DataFrame:
    b = pd.read_parquet(config.RAW_DIR / "background_07111.parquet")
    b = b[b["Region"] != "0"].rename(columns={"Region": "fylke", "Kjonn": "kjonn", "Alder": "bg_band"})
    b["bakgrunn"] = b["Landbakgrunn"].map(labels)
    b = b.groupby(["fylke", "kjonn", "bg_band", "bakgrunn"], as_index=False)["value"].sum()

    pop = pop_all_ages.copy()
    pop["bg_band"] = _band(pop["alder"], BG_BANDS)
    denom = pop.groupby(["fylke", "kjonn", "bg_band"], as_index=False)["personer"].sum()
    b = b.merge(denom, on=["fylke", "kjonn", "bg_band"], how="left")
    b["andel"] = b["value"] / b["personer"]

    norsk = b.groupby(["fylke", "kjonn", "bg_band"], as_index=False)["andel"].sum()
    norsk["andel"] = (1 - norsk["andel"]).clip(lower=0)
    norsk["bakgrunn"] = "norsk"
    out = pd.concat([b[["fylke", "kjonn", "bg_band", "bakgrunn", "andel"]], norsk], ignore_index=True)
    return out


def build() -> dict:
    cfg = config.load("frame")
    pcfg = cfg["population"]
    rng = np.random.default_rng(pcfg["seed"])
    bands = [(lo, hi, config.age_band_label(lo, hi)) for lo, hi in cfg["age_bands"]]

    pop_all = load_population(0)
    pop = pop_all[pop_all["alder"] >= pcfg["min_age"]].copy()
    pop["aldersband"] = _band(pop["alder"], bands)
    pop["edu_band"] = _band(pop["alder"], EDU_BANDS)
    pop["bg_band"] = _band(pop["alder"], BG_BANDS)

    edu = education_shares(cfg["education_labels"])
    bg = background_shares(pop_all, cfg["background_labels"])

    centr = pd.read_parquet(config.RAW_DIR / "centrality_klass128.parquet")
    fylke_navn = (
        pd.read_parquet(config.RAW_DIR / "education_08921.parquet")[["Region", "Region_tekst"]]
        .drop_duplicates().set_index("Region")["Region_tekst"]
    )

    # --- Forventede marginaler (fasit fra SSB) ---
    exp_edu = pop.merge(edu, on=["fylke", "kjonn", "edu_band"])
    exp_edu["n"] = exp_edu["personer"] * exp_edu["andel"]
    exp_bg = pop.merge(bg, on=["fylke", "kjonn", "bg_band"])
    exp_bg["n"] = exp_bg["personer"] * exp_bg["andel"]

    # --- Fordel agenter på celler ---
    cells = pop.groupby(["kommune", "kjonn", "aldersband"], as_index=False)["personer"].sum()
    total = cells["personer"].sum()
    cells["n_agents"] = np.maximum(1, np.round(pcfg["n_agents"] * cells["personer"] / total)).astype(int)

    # Trekk ettårig alder innen hver celle
    pop_sorted = pop.sort_values(["kommune", "kjonn", "aldersband", "alder"])
    agents = []
    for (kom, kj, band), g in pop_sorted.groupby(["kommune", "kjonn", "aldersband"], sort=False):
        agents.append((kom, kj, band, g["alder"].to_numpy(), g["personer"].to_numpy()))
    n_lookup = cells.set_index(["kommune", "kjonn", "aldersband"])["n_agents"]

    rows_kom, rows_kj, rows_age = [], [], []
    for kom, kj, band, ages, counts in agents:
        n = int(n_lookup[(kom, kj, band)])
        drawn = rng.choice(ages, size=n, p=counts / counts.sum())
        rows_kom += [kom] * n
        rows_kj += [kj] * n
        rows_age.append(drawn)
    a = pd.DataFrame({"kommune": rows_kom, "kjonn": rows_kj, "alder": np.concatenate(rows_age)})
    a["fylke"] = a["kommune"].str[:2]
    a["aldersband"] = _band(a["alder"], bands)
    a["edu_band"] = _band(a["alder"], EDU_BANDS)
    a["bg_band"] = _band(a["alder"], BG_BANDS)

    # Trekk egenskaper i en rekkefølge der hver kan avhenge av de forrige:
    # bakgrunn -> innvandringskategori -> utdanning -> arbeidsstatus -> husholdning -> inntekt
    def draw(attr_df: pd.DataFrame, keys: list[str], col: str) -> pd.Series:
        return enrich.draw_categorical(a, attr_df.rename(columns={"andel": "p"}), keys, col, rng)

    a["bakgrunn"] = draw(bg, ["fylke", "kjonn", "bg_band"], "bakgrunn")
    a["b09599"] = enrich.band(a["alder"], enrich.BANDS_09599)
    a["innvkat"] = enrich.draw_innvkat(a, rng)
    a["utdanning"] = enrich.draw_education(a, edu, rng)
    a["b_status"] = enrich.band(a["alder"], enrich.BANDS_STATUS)
    a["arbeidsstatus"] = enrich.draw_labour(a, rng)
    a["arbeidsstatus"] = enrich.adjust_elderly_employment(a, pop, rng)
    a["b_emp"], a["syss"] = enrich.emp_keys(a)
    a["aktiv"] = np.where(a["arbeidsstatus"].isin(enrich.AKTIV), "ja", "nei")
    a["innv_utd"] = enrich.innv_utd_key(a)
    a["hh_band"] = enrich.band(a["alder"], enrich.BANDS_HH)
    a["husholdning"] = enrich.collapse_household(enrich.draw_household(a, pop_all, rng))
    a = a.join(enrich.lowinc_groups(a))
    a["lavinntekt"] = enrich.draw_lowinc(a, rng)

    # --- Forventede marginaler (fasit fra SSB) ---
    # Rekkefølgen betyr noe: siste marginal treffes eksakt (geografi × kjønn × alder).
    exp_edu = pop.merge(edu, on=["fylke", "kjonn", "edu_band"])
    exp_edu["n"] = exp_edu["personer"] * exp_edu["andel"]
    exp_bg = pop.merge(bg, on=["fylke", "kjonn", "bg_band"])
    exp_bg["n"] = exp_bg["personer"] * exp_bg["andel"]
    margins = {
        ("kjonn", "b09599", "innvkat", "utdanning"): enrich.edu_immigrant_margin(exp_bg),
        ("kjonn", "aldersband", "bakgrunn"): exp_bg.groupby(["kjonn", "aldersband", "bakgrunn"])["n"].sum(),
        ("kjonn", "b_status", "arbeidsstatus"): enrich.labour_margin(pop),
        ("kjonn", "b_emp", "syss"): enrich.employment_age_margin(pop),
        ("kjonn", "hh_band", "husholdning"): enrich.household_margin(pop, pop_all),
        ("fylke", "kjonn", "aktiv"): enrich.activity_margin(pop),
        ("fylke", "innv_utd"): enrich.immigrant_edu_fylke_margin(exp_bg),
        ("fylke", "kjonn", "utdanning"): exp_edu.groupby(["fylke", "kjonn", "utdanning"])["n"].sum(),
        ("fylke", "kjonn", "bakgrunn"): exp_bg.groupby(["fylke", "kjonn", "bakgrunn"])["n"].sum(),
        ("kjonn", "alder"): pop.groupby(["kjonn", "alder"])["personer"].sum(),
        ("kommune", "kjonn", "aldersband"): pop.groupby(["kommune", "kjonn", "aldersband"])["personer"].sum(),
    }

    # Basisvekt: celle-folketall / antall agenter i cellen
    cell_pop = cells.set_index(["kommune", "kjonn", "aldersband"])["personer"]
    key = pd.MultiIndex.from_frame(a[["kommune", "kjonn", "aldersband"]])
    base = cell_pop.reindex(key).to_numpy() / n_lookup.reindex(key).to_numpy()

    # Pass 1: kalibrer alt unntatt lavinntekt.
    res = rake(a, margins, base_weights=base, bounds=(0.1, 10.0), max_iter=500, tol=5e-3)
    # Pass 2: gruppestørrelsene for lavinntekt hentes fra pass 1, og lavinntekt
    # legges inn før geografien (som fortsatt kalibreres sist og treffes eksakt).
    w1 = pd.Series(res.weights, index=a.index)
    sizes = {dim: w1.groupby(a[dim]).sum() for dim in ["lav_hh", "lav_status", "lav_innv", "lav_edu"]}
    lav = enrich.lowinc_margins(a, sizes, float(w1.sum()))
    geo_key = ("kommune", "kjonn", "aldersband")
    margins2 = {k: v for k, v in margins.items() if k != geo_key} | lav | {geo_key: margins[geo_key]}
    res = rake(a, margins2, base_weights=base, bounds=(0.1, 10.0), max_iter=500, tol=5e-3)  # kildene er uenige på 0,1–0,5 %; se docs/architecture.md
    a["vekt"] = res.weights
    a["inntektsdesil"] = enrich.draw_income(a.assign(_w=a["vekt"]), rng)

    # Beriking
    a = a.merge(centr[["kommune", "kommune_navn", "sentralitet"]], on="kommune", how="left")
    a["fylke_navn"] = a["fylke"].map(fylke_navn)
    a["kjonn"] = a["kjonn"].map({"1": "mann", "2": "kvinne"})
    a.insert(0, "agent_id", [f"NO-{i:06d}" for i in range(len(a))])
    a = a[["agent_id", "kommune", "kommune_navn", "fylke", "fylke_navn", "sentralitet",
           "kjonn", "alder", "aldersband", "utdanning", "bakgrunn", "innvkat",
           "arbeidsstatus", "husholdning", "lavinntekt", "inntektsdesil", "vekt"]]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    a.to_parquet(config.PROCESSED_DIR / "agents.parquet", index=False)

    report = {
        "agents": len(a),
        "population_18plus": float(pop["personer"].sum()),
        "weighted_total": float(a["vekt"].sum()),
        "raking_iterations": res.iterations,
        "raking_max_misallocated": res.max_misallocated,
        "raking_misallocated": res.misallocated,
        "raking_converged": res.converged,
        "raking_dropped_share": res.dropped_share,
        "effective_sample_size": effective_sample_size(a["vekt"].to_numpy()),
        "weight_min": float(a["vekt"].min()),
        "weight_max": float(a["vekt"].max()),
    }
    with open(config.PROCESSED_DIR / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
