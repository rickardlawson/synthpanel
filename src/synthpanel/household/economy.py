"""Husholdningslaget: inntekt i kroner, forbruk og kjøpsrater.

Alle tall gjelder *husholdningen agenten bor i*. Agentene er personer, så når
tall sammenlignes med SSBs husholdningstabeller, vektes hver agent med
vekt / husholdningsstørrelse (se `household_weight`).

1. Inntekt per kommune (12558)
   Inntektsdesilen trekkes som før fra fylke × husholdningstype (12563), men
   fordelingen vippes per kommune slik at andelen husholdninger i hver nasjonale
   desil treffer kommunetallene i 12558 (IPF over husholdningstype × desil).
   Bærum får dermed flere husholdninger i toppdesilen enn Vardø, selv om de
   har samme husholdningstyper.

2. Inntekt i kroner (12558, desilgrenser)
   Innen desilen trekkes et beløp jevnt mellom de nasjonale grensene. Desil 1
   starter på 100 000 kr (skjønn – negative og svært lave inntekter finnes, men
   er få), desil 10 trekkes fra en Pareto-hale (alfa = 3, skjønn) over grensen
   for desil 9. Kontroll: median per kommune mot 06944 (ikke brukt i trekkingen).

3. Forbruk (Forbruksundersøkelsen 2022: 14157, 14161, 14227, 14100)
   Forventet utgift per vare-/tjenestegruppe = snitt for husholdningstype ×
   inntektskvartil (14157), justert for sentralitet (14161) og – for
   aleneboende – alder (14227), begge dempet (0,6) fordi de delvis forklares av
   inntekt og husholdningstype. Skalert så landssnittet treffer 14100, og
   prisjustert med KPI fra 2022 til siste hele år (14709). Hver husholdning får
   en felles «forbruksfaktor» (lognormal, sd 0,25) og et eget avvik per gruppe
   (sd 0,3; 0,6 for store, sjeldne kjøp som bil og pakketur). Tallene er
   *forventet årlig utgift*, ikke faktiske kjøp et bestemt år.

4. Kjøpsrater (12906, 13370, 06944)
   Sannsynlighet for å kjøpe ny bil neste 12 mnd = nye personbiler i kommunen
   siste år / husholdninger i kommunen (Poisson: p = 1 − e^−λ), fordelt mellom
   husholdningene etter hvor mye husholdningstypen og inntektskvartilen bruker
   på kjøp av kjøretøy (14157). Andelen elbil blant nye biler følger kommunen.
   «Har elbil» bruker privateide elbiler per husholdning (13370) på samme måte,
   med dempet (^0,5) inntektsgradient.

Svakheter: forbrukstallene er fra 2022 (strømpris-året), leasing- og firmabiler
er med i nybilraten, og hyttetilgang mangler (SSB publiserer ikke eierskap til
fritidsbolig etter eierens bosted i åpne tabeller).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from synthpanel import config

RAW = config.RAW_DIR

# Husholdningstype i panelet -> type i 06944 (antall husholdninger). «rest» = alle andre.
TYPE_06944 = {"aleneboende": "0001", "par_uten_barn": "0002", "par_smaa_barn": "0003", "par_store_barn": "0003",
              "enslig_smaa_barn": "0004", "enslig_store_barn": "0004"}

# kolonne -> (COICOP-koder som summeres, visningsnavn, individuelt avvik)
SPEND = {
    "forbruk_totalt": (["00"], "Forbruk i alt", 0.0),
    "forbruk_mat": (["01.1"], "Matvarer", 0.15),
    "forbruk_klaer": (["03"], "Klær og sko", 0.3),
    "forbruk_innbo": (["05"], "Møbler og innbo", 0.4),
    "forbruk_bilkjop": (["07.1"], "Kjøp av kjøretøy", 0.6),
    "forbruk_elektronikk": (["08.1"], "TV, PC, mobil og annet utstyr", 0.4),
    "forbruk_fritid": (["09"], "Fritid, sport og kultur", 0.3),
    "forbruk_fritidsutstyr": (["09.1"], "Varig fritidsutstyr (båt, sykkel, ski …)", 0.6),
    "forbruk_fritidstjenester": (["09.4"], "Trening, kurs og fritidstjenester", 0.4),
    "forbruk_kultur": (["09.6"], "Kino, konsert og kultur", 0.4),
    "forbruk_reiser": (["07.3", "09.8", "11.2"], "Reiser (billetter, pakketur, overnatting)", 0.5),
    "forbruk_restaurant": (["11.1"], "Restaurant og kafé", 0.35),
}
SHARED_SD = 0.25
DAMP = 0.6
LEVEL_COLS = {"forbruk_fritid": "fritidsforbruk", "forbruk_reiser": "reiseforbruk", "forbruk_totalt": "forbruksniva"}
INCOME_BANDS = [(0, 300_000, "u300"), (300_000, 500_000, "300-500"), (500_000, 750_000, "500-750"),
                (750_000, 1_000_000, "750-1000"), (1_000_000, 1_500_000, "1000-1500"), (1_500_000, np.inf, "o1500")]
PURCHASE_COLS = {"kjop_nybil": "Kjøper ny bil (12 mnd)", "kjop_nybil_el": "Kjøper ny elbil (12 mnd)", "har_elbil": "Har elbil"}
NUMERIC = ["inntekt_kr", *SPEND, "p_nybil", "p_elbil"]

# Husholdningstype -> gruppe i 14157 («0» = alle husholdninger)
TYPE_14157 = {"aleneboende": "1", "par_uten_barn": "2", "par_smaa_barn": "3-4", "par_store_barn": "3-4"}
QUARTILE = {1: "41", 2: "41", 3: "41", 4: "42", 5: "42", 6: "43", 7: "43", 8: "44", 9: "44", 10: "44"}


def adults_per_household(a: pd.DataFrame) -> pd.Series:
    """Voksne (18+) per husholdning for hver type: personer i panelet / husholdninger i 06944.
    Agentene er voksne, så en husholdning «telles» like mange ganger som den har voksne."""
    m = pd.read_parquet(RAW / "hh_income_median_06944.parquet")
    m = m[(m["Region"] == "0") & (m["ContentsCode"] == "AntallHushold")].set_index("HusholdType")["value"]
    grp = a["husholdning"].map(TYPE_06944).fillna("rest")
    persons = a["vekt"].groupby(grp).sum()
    hh = m.reindex(["0001", "0002", "0003", "0004"])
    hh["rest"] = m["0000"] - hh.sum()
    return grp.map(persons / hh)


def household_weight(a: pd.DataFrame) -> pd.Series:
    if "_hhw" in a:
        return a["_hhw"]
    return a["vekt"] / adults_per_household(a)


# ---------------------------------------------------------------------------
# 1–2. Inntekt
# ---------------------------------------------------------------------------

def _deciles() -> pd.DataFrame:
    return pd.read_parquet(RAW / "hh_income_deciles_12558.parquet")


def available() -> bool:
    return (RAW / "hh_income_deciles_12558.parquet").exists()


def kommune_decile_targets() -> pd.DataFrame:
    """Andel husholdninger per nasjonal desil (kolonner 1–10) per kommune, bare kommuner med fullstendige tall."""
    d = _deciles()
    d = d[(d["ContentsCode"] == "AndelHush") & (d["Region"].str.len() == 4)]
    t = d.pivot_table(index="Region", columns="Desiler", values="value")
    t.columns = [int(c) for c in t.columns]
    t = t.dropna()
    t = t[t.sum(axis=1) > 0]
    return t.div(t.sum(axis=1), axis=0)


def decile_bounds() -> np.ndarray:
    """Øvre grense (kr) for desil 1–9, nasjonalt."""
    d = _deciles()
    d = d[(d["Region"] == "0") & (d["ContentsCode"] == "VerdiDesil")].sort_values("Desiler")
    return d["value"].dropna().to_numpy()[:9]


def tilt(p_by_type: dict, w_by_type: dict, target: np.ndarray, iters: int = 60) -> dict:
    """IPF: vipp fordelingene per husholdningstype (samme faktor per desil) slik at
    blandingen, vektet med husholdninger, treffer `target`."""
    r = np.ones(len(target))
    W = sum(w_by_type.values())
    for _ in range(iters):
        mix = np.zeros(len(target))
        for t, p in p_by_type.items():
            q = p * r
            mix += w_by_type[t] / W * q / q.sum()
        ok = mix > 0
        r[ok] *= np.where(target[ok] > 0, target[ok] / mix[ok], 1.0)
        r = np.clip(r, 1e-3, 1e3)
    out = {}
    for t, p in p_by_type.items():
        q = p * r
        out[t] = q / q.sum()
    return out


def income_kr(decile: pd.Series, rng: np.random.Generator) -> np.ndarray:
    b = decile_bounds()
    lo = np.concatenate([[100_000.0], b])            # nedre grense per desil 1..10
    hi = np.concatenate([b, [np.nan]])
    d = decile.to_numpy(dtype=int) - 1
    u = rng.random(len(d))
    kr = lo[d] + u * (np.nan_to_num(hi[d]) - lo[d])
    top = d == 9
    kr[top] = np.minimum(lo[9] * (1 - u[top]) ** (-1 / 3.0), 20_000_000)   # Pareto, alfa = 3
    return np.round(kr, -3)


def income_band(kr: pd.Series) -> pd.Series:
    out = pd.Series("u300", index=kr.index, dtype=object)
    for lo, hi, lab in INCOME_BANDS:
        out[(kr >= lo) & (kr < hi)] = lab
    return out


def income_check(a: pd.DataFrame) -> dict:
    """Kontroll mot 06944 (ikke brukt i trekkingen): vektet median husholdningsinntekt per kommune."""
    m = pd.read_parquet(RAW / "hh_income_median_06944.parquet")
    m = m[(m["HusholdType"] == "0000") & (m["ContentsCode"] == "InntSkatt")].set_index("Region")["value"]
    hw = household_weight(a)
    rows = []
    for k, g in a.groupby("kommune"):
        if k not in m.index or np.isnan(m[k]) or len(g) < 100:
            continue
        o = np.argsort(g["inntekt_kr"].to_numpy())
        cw = np.cumsum(hw.loc[g.index].to_numpy()[o])
        med = g["inntekt_kr"].to_numpy()[o][np.searchsorted(cw, cw[-1] / 2)]
        rows.append((k, med, m[k]))
    r = pd.DataFrame(rows, columns=["kommune", "panel", "ssb"])
    rel = (r["panel"] / r["ssb"] - 1).abs()
    nat = float(m.get("0", np.nan))
    return {"kommuner_med_minst_100_agenter": len(r),
            "snitt_relativt_avvik_median": float(rel.mean()),
            "korrelasjon_kommunemedian": float(np.corrcoef(r["panel"], r["ssb"])[0, 1]) if len(r) > 2 else None,
            "ssb_landsmedian": nat}


# ---------------------------------------------------------------------------
# 3. Forbruk
# ---------------------------------------------------------------------------

def _spend(name: str) -> pd.DataFrame:
    d = pd.read_parquet(RAW / f"{name}.parquet")
    return d[d["ContentsCode"] == "Utgift"]


def kpi_factor() -> tuple[float, str]:
    k = pd.read_parquet(RAW / "hh_kpi_14709.parquet").dropna(subset=["value"]).set_index("Tid")["value"]
    last = max(k.index)
    return float(k[last] / k["2022"]), last


def expected_spend(a: pd.DataFrame) -> dict[str, np.ndarray]:
    """Forventet årlig utgift per agent og kolonne (kr, 2022-priser, før skalering)."""
    tq = _spend("hh_spend_type_quartile_14157")
    ce = _spend("hh_spend_centrality_14161")
    al = _spend("hh_spend_alone_age_14227")
    tg = a["husholdning"].map(TYPE_14157).fillna("0")
    qv = a["inntektsdesil"].astype(int).map(QUARTILE)
    sent = a["sentralitet"].replace({"05": "05-06", "06": "05-06"})
    age = pd.cut(a["alder"], [0, 44, 64, 200], labels=["-44", "45-64", "65+"]).astype(str)
    alone = (a["husholdning"] == "aleneboende").to_numpy()
    out = {}
    for col, (codes, _, _) in SPEND.items():
        total = np.zeros(len(a))
        for code in codes:
            t = tq[tq["VareTjenesteGruppe"] == code].set_index(["HusholdType", "Inntektskvartil"])["value"]
            # Prikkede celler: bruk typens snitt × kvartilens forhold for alle husholdninger.
            allq = t.xs("0", level="HusholdType")
            ratio = allq / allq.get("0", np.nan)
            base = t.reindex(pd.MultiIndex.from_arrays([tg, qv])).to_numpy()
            fill = t.reindex(pd.MultiIndex.from_arrays([tg, pd.Series("0", index=a.index)])).to_numpy() * ratio.reindex(qv).to_numpy()
            base = np.where(np.isnan(base), fill, base)
            c = ce[ce["VareTjenesteGruppe"] == code].set_index("SentralitetKomm")["value"]
            fs = (c.reindex(sent).to_numpy() / c.get("0", np.nan)) ** DAMP
            g = al[al["VareTjenesteGruppe"] == code].set_index("Alder")["value"]
            fa = np.where(alone, (g.reindex(age).to_numpy() / g.get("999A", np.nan)) ** DAMP, 1.0)
            total += np.nan_to_num(base * np.nan_to_num(fs, nan=1.0) * np.nan_to_num(fa, nan=1.0))
        out[col] = total
    return out


def attach_spend(a: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    exp = expected_spend(a)
    nat = _spend("hh_spend_total_14100").set_index("VareTjenesteGruppe")["value"]
    kpi, _ = kpi_factor()
    hw = household_weight(a).to_numpy()
    shared = rng.standard_normal(len(a))
    out = a.copy()
    for col, (codes, _, sd) in SPEND.items():
        target = float(sum(nat[c] for c in codes))
        e = exp[col] * target / np.average(exp[col], weights=hw)      # landssnittet treffer 14100
        z = SHARED_SD * shared + sd * rng.standard_normal(len(a))
        val = e * np.exp(z - (SHARED_SD ** 2 + sd ** 2) / 2) * kpi    # lognormal med samme forventning
        out[col] = np.round(val, -2)
    for col, lab in LEVEL_COLS.items():
        q = _weighted_quantiles(out[col].to_numpy(), hw, [1 / 3, 2 / 3])
        out[lab] = np.where(out[col] <= q[0], "lav", np.where(out[col] <= q[1], "middels", "høy"))
    return out


def _weighted_quantiles(x: np.ndarray, w: np.ndarray, qs: list[float]) -> list[float]:
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(x[o][np.searchsorted(cw, q)]) for q in qs]


# ---------------------------------------------------------------------------
# 4. Kjøpsrater
# ---------------------------------------------------------------------------

def _households_per_kommune() -> pd.Series:
    m = pd.read_parquet(RAW / "hh_income_median_06944.parquet")
    m = m[(m["HusholdType"] == "0000") & (m["ContentsCode"] == "AntallHushold")]
    return m.set_index("Region")["value"]


def car_rates() -> pd.DataFrame:
    """Per kommune: nye personbiler, andel elbil blant nye, privateide elbiler og husholdninger."""
    n = pd.read_parquet(RAW / "hh_cars_new_12906.parquet").dropna(subset=["value"])
    new = n.groupby("Region")["value"].sum()
    new_el = n[n["DrivstoffType"] == "5"].groupby("Region")["value"].sum()
    f = pd.read_parquet(RAW / "hh_cars_fleet_13370.parquet").dropna(subset=["value"])
    priv_el = f[(f["Eierform"] == "01") & (f["DrivstoffType"] == "5")].groupby("Region")["value"].sum()
    hh = _households_per_kommune()
    df = pd.DataFrame({"nye": new, "nye_el": new_el, "elbiler_privat": priv_el, "husholdninger": hh}).dropna()
    df = df[(df["husholdninger"] > 0) & (df.index.str.len() == 4)]
    df["lambda_ny"] = df["nye"] / df["husholdninger"]
    df["andel_el_ny"] = (df["nye_el"] / df["nye"]).fillna(0)
    df["lambda_elbil"] = df["elbiler_privat"] / df["husholdninger"]
    return df


def _purchase_intensity(a: pd.DataFrame) -> np.ndarray:
    """Relativ kjøpsintensitet for kjøretøy etter husholdningstype × inntektskvartil (14157, kode 07.1)."""
    exp = expected_spend(a.assign(sentralitet="00"))  # sentralitet ligger allerede i kommunetallene
    return exp["forbruk_bilkjop"] / np.nanmean(exp["forbruk_bilkjop"])


def attach_purchases(a: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rates = car_rates()
    inten = _purchase_intensity(a)
    hw = household_weight(a).to_numpy()
    out = a.copy()
    p_new = np.full(len(a), np.nan)
    p_el = np.full(len(a), np.nan)
    share_el = np.zeros(len(a))
    nat = rates[["nye", "nye_el", "elbiler_privat", "husholdninger"]].sum()
    for k, idx in a.groupby("kommune").groups.items():
        ix = a.index.get_indexer(idx)
        r = rates.loc[k] if k in rates.index else None
        lam_new = r["lambda_ny"] if r is not None else nat["nye"] / nat["husholdninger"]
        lam_el = r["lambda_elbil"] if r is not None else nat["elbiler_privat"] / nat["husholdninger"]
        share_el[ix] = r["andel_el_ny"] if r is not None else nat["nye_el"] / nat["nye"]
        f = inten[ix] / np.average(inten[ix], weights=hw[ix])          # snitt 1 i kommunen
        fe = f ** 0.5 / np.average(f ** 0.5, weights=hw[ix])
        p_new[ix] = 1 - np.exp(-lam_new * f)
        p_el[ix] = 1 - np.exp(-lam_el * fe)
    buy = rng.random(len(a)) < p_new
    out["p_nybil"] = np.round(p_new, 4)
    out["p_elbil"] = np.round(p_el, 4)
    out["kjop_nybil"] = np.where(buy, "ja", "nei")
    out["kjop_nybil_el"] = np.where(buy & (rng.random(len(a)) < share_el), "ja", "nei")
    out["har_elbil"] = np.where(rng.random(len(a)) < p_el, "ja", "nei")
    return out


def purchase_check(a: pd.DataFrame) -> dict:
    rates = car_rates()
    hw = household_weight(a)
    exp_new = float((a["p_nybil"] * hw).sum())
    return {"forventede_nybilkjop_panel": round(exp_new),
            "nye_personbiler_ssb": float(rates["nye"].sum()),
            "husholdninger_panel": round(float(hw.sum())),
            "husholdninger_ssb": float(rates["husholdninger"].sum())}


# ---------------------------------------------------------------------------

def attach(a: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Legg til inntekt i kroner, forbruk og kjøpsrater. Inntektsdesilen må allerede være trukket."""
    if not available():
        return a, {}
    out = a.copy()
    out["inntekt_kr"] = income_kr(out["inntektsdesil"], rng)
    out["inntekt_gruppe"] = income_band(out["inntekt_kr"])
    report = {"inntekt": income_check(out)}
    if (RAW / "hh_spend_type_quartile_14157.parquet").exists():
        out = attach_spend(out, rng)
        kpi, year = kpi_factor()
        report["forbruk"] = {"kpi_faktor_2022": round(kpi, 4), "prisnivaa": year}
    if (RAW / "hh_cars_new_12906.parquet").exists():
        out = attach_purchases(out, rng)
        report["kjop"] = purchase_check(out)
    return out, report


COLUMNS = ["inntekt_kr", "inntekt_gruppe", *SPEND, *LEVEL_COLS.values(), "p_nybil", "p_elbil", *PURCHASE_COLS]
