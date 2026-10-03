"""Politisk lag: stemmerett, valgdeltakelse og partivalg ved stortingsvalget 2025.

Fremgangsmåte per agent:
1. **Partipreferanse.** Agenten har en ESS-donor (fra verdilaget) med kjent partivalg
   ved valget i 2017 eller 2021. Det føres frem til 2025 med SSBs velgerstrømmer
   (11666, fra Valgundersøkelsen): 2017→2021→2025. Donorer som ikke stemte eller
   ikke hadde stemmerett, bruker partiet de føler seg nærmest; ellers landsresultatet.
2. **Stemmerett.** Innvandrere får sannsynlighet for stemmerett (statsborgerskap)
   fra 13446 per kjønn og alder. Øvrige regnes som stemmeberettigede.
3. **Valgdeltakelse** fra 10440 (kjønn × alder × utdanning), justert for
   innvandringskategori og hovedstatus (13818), og deretter forskjøvet per fylke
   slik at frammøtet treffer det faktiske (Valgdirektoratet).
4. **Partivalg 2025.** Preferansene skaleres per fylke slik at velgernes
   fordeling treffer det faktiske valgresultatet i fylket.

5. **Kjønn og alder.** Velgerstrømmene antar at alle grupper flyttet seg likt
   (de gjorde ikke det i 2025 – f.eks. unge menn mot FrP). Preferansene skaleres
   derfor også mot Valgundersøkelsens partivalg etter kjønn × alder (13554),
   vekselvis med fylkesresultatet.

Kontroll: oppslutning etter inntekt (13698) brukes ikke her og er holdt av til test.
Før 13554 ble tatt inn, var snittfeilen mot 13554 2,0 pp (landssnitt: 2,7 pp).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from synthpanel import config

RAW = config.RAW_DIR
PARTIES = ["RØDT", "SV", "A", "SP", "MDG", "KRF", "V", "H", "FRP", "ANDRE"]
FLOW_FROM = {"F55": "RØDT", "F06": "SV", "F01": "A", "F05": "SP", "F08": "MDG", "F04": "KRF",
             "F07": "V", "F03": "H", "F02": "FRP", "F92": "ANDRE"}
FLOW_TO = {"T55": "RØDT", "T06": "SV", "T01": "A", "T05": "SP", "T08": "MDG", "T04": "KRF",
           "T07": "V", "T03": "H", "T02": "FRP", "T92": "ANDRE"}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(z):
    return 1 / (1 + np.exp(-z))


# --- Data ---------------------------------------------------------------------

def flows() -> dict[int, pd.DataFrame]:
    """Overgangsmatriser (rader: fra-parti, kolonner: til-parti) blant dem som stemte begge ganger."""
    d = pd.read_parquet(RAW / "valg_flows_11666.parquet")
    d = d[d["PolitPartiNeste"].isin(FLOW_TO) & d["PolitPartiDette"].isin(FLOW_FROM)]
    out = {}
    for year, g in d.groupby("Tid"):
        m = g.pivot_table(index="PolitPartiDette", columns="PolitPartiNeste", values="value", aggfunc="sum")
        m = m.rename(index=FLOW_FROM, columns=FLOW_TO).reindex(index=PARTIES, columns=PARTIES).fillna(0)
        m = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        # Rader uten data (få velgere): behold partiet.
        for p in PARTIES:
            if m.loc[p].sum() == 0:
                m.loc[p, p] = 1.0
        out[int(year)] = m
    return out


SURVEY_CODES = {"01": "A", "02": "FRP", "03": "H", "04": "KRF", "08": "MDG", "55": "RØDT",
                "05": "SP", "06": "SV", "07": "V", "92": "ANDRE"}
AGE4 = [(18, 34, "18-34"), (35, 49, "35-49"), (50, 69, "50-69"), (70, 200, "70+")]


def survey_by_sex_age() -> pd.DataFrame:
    """Partivalg blant velgere etter kjønn × alder (Valgundersøkelsen 2025, 13554)."""
    d = pd.read_parquet(RAW / "valg_kontroll_13554.parquet")
    d = d[d["Kjonn"].isin(["1", "2"])].assign(parti=d["PolitParti"].map(SURVEY_CODES),
                                              kjonn=d["Kjonn"].map({"1": "mann", "2": "kvinne"}))
    m = d.pivot_table(index=["kjonn", "Alder"], columns="parti", values="value").reindex(columns=PARTIES).fillna(0)
    return m.div(m.sum(axis=1), axis=0)


def results_by_fylke() -> pd.DataFrame:
    r = pd.read_parquet(RAW / "valg_resultat_2025.parquet")
    r["fylke"] = r["kommune"].str[:2]
    g = r.groupby("fylke")[PARTIES + ["godkjente", "blanke", "stemmeberettigede"]].sum()
    shares = g[PARTIES].div(g[PARTIES].sum(axis=1), axis=0)
    shares["frammote"] = g["godkjente"] / g["stemmeberettigede"]
    return shares


# --- 1. Partipreferanse ---------------------------------------------------------

def preference(a: pd.DataFrame, donors: pd.DataFrame) -> np.ndarray:
    """Sannsynlighetsvektor over PARTIES for hver agent, ut fra donorens partivalg."""
    fl = flows()
    nat = results_by_fylke()
    r = pd.read_parquet(RAW / "valg_resultat_2025.parquet")
    nat_share = r[PARTIES].sum() / r[PARTIES].sum().sum()

    d = donors.set_index("resp_id")
    vec = {}
    for rid, row in d.iterrows():
        party = row.get("siste_parti")
        year = row.get("siste_valg_aar")
        if party not in PARTIES:                      # stemte ikke / ikke stemmerett / ukjent
            party = row.get("naermeste_parti")
            year = 2021 if party in PARTIES else None
        if party in PARTIES:
            v = pd.Series(0.0, index=PARTIES)
            v[party] = 1.0
            if year == 2017:
                v = v @ fl[2021]
            v = v @ fl[2025]
            vec[rid] = v.to_numpy()
        else:
            vec[rid] = nat_share.reindex(PARTIES).to_numpy()
    return np.vstack([vec[rid] for rid in a["ess_donor"]])


# --- 2. Stemmerett ----------------------------------------------------------------

def eligibility(a: pd.DataFrame) -> np.ndarray:
    e = pd.read_parquet(RAW / "valg_eligible_13446.parquet")
    e = e[e["InnvandrKat"] == "B"].assign(kjonn=lambda d: d["Kjonn"].map({"1": "mann", "2": "kvinne"}))
    bands = [(18, 29, "18-29"), (30, 39, "30-39"), (40, 49, "40-49"), (50, 59, "50-59"), (60, 200, "60+")]
    ab = pd.Series("", index=a.index)
    for lo, hi, lab in bands:
        ab[a["alder"].between(lo, hi)] = lab
    imm = a["innvkat"] == "innvandrer"
    n_imm = a[imm].groupby([a.loc[imm, "kjonn"], ab[imm]])["vekt"].sum()
    elig = e.set_index(["kjonn", "Alder"])["value"]
    rate = (elig / n_imm.rename_axis(["kjonn", "Alder"])).clip(upper=1.0)
    p = np.ones(len(a))
    key = pd.MultiIndex.from_arrays([a.loc[imm, "kjonn"], ab[imm]])
    p[imm.to_numpy()] = rate.reindex(key).fillna(rate.mean()).to_numpy()
    return p


# --- 3. Valgdeltakelse --------------------------------------------------------------

def turnout(a: pd.DataFrame) -> np.ndarray:
    t = pd.read_parquet(RAW / "valg_turnout_10440.parquet")
    t = t[t["value"].notna()].assign(kjonn=lambda d: d["Kjonn"].map({"1": "mann", "2": "kvinne"}))
    # Ikke-overlappende aldersgrupper i 10440 (25 år legges i 26–29).
    def band(age):
        for lo, hi, lab in [(18, 19, "18-19"), (20, 24, "20-24"), (25, 29, "26-29"), (30, 39, "30-39"),
                            (40, 49, "40-49"), (50, 59, "50-59"), (60, 79, "60-79"), (80, 200, "80+")]:
            if lo <= age <= hi:
                return lab
    ab = a["alder"].map(band)
    edu = a["utdanning"].map({"grunnskole": "1-2", "videregaende": "3-5", "fagskole": "3-5",
                              "uh_kort": "6-8", "uh_lang": "6-8", "uoppgitt": "1-2"})
    rate = t.set_index(["kjonn", "Alder", "UtdNivaa"])["value"] / 100
    base = rate.reindex(pd.MultiIndex.from_arrays([a["kjonn"], ab, edu])).to_numpy()
    base = np.where(np.isnan(base), np.nanmean(base), base)

    # Justering (log-odds) for innvandringskategori og hovedstatus fra 13818.
    s = pd.read_parquet(RAW / "valg_turnout_innv_13818.parquet")
    s = s.assign(kjonn=s["Kjonn"].map({"1": "mann", "2": "kvinne"}))
    tot = s[(s["ArbStyrkStatus"] == "TOT2") & (s["InnvandrKat"] == "Ialt")].set_index("kjonn")["value"] / 100
    innv = s[(s["ArbStyrkStatus"] == "TOT2")].set_index(["kjonn", "InnvandrKat"])["value"] / 100
    stat = s[s["InnvandrKat"] == "Ialt"].set_index(["kjonn", "ArbStyrkStatus"])["value"] / 100
    cat = a["innvkat"].map({"innvandrer": "B", "norskfodt_innv_foreldre": "C"})
    st = a["arbeidsstatus"].map({"sysselsatt": "A.01", "arbeidsledig": "A.09", "tiltak": "U.01", "student": "U.03",
                                 "aap_ufor": "U.04-U.05", "pensjonist": "U.06-U.07", "annet": "U.90A"})
    t0 = _logit(tot.reindex(a["kjonn"]).to_numpy())
    li = _logit(innv.reindex(pd.MultiIndex.from_arrays([a["kjonn"], cat])).to_numpy())
    ls = _logit(stat.reindex(pd.MultiIndex.from_arrays([a["kjonn"], st])).to_numpy())
    z = _logit(base) + np.nan_to_num(li - t0) + 0.5 * np.nan_to_num(ls - t0)  # status delvis forklart av alder/utdanning
    return _expit(z)


def _shift_to_target(p: np.ndarray, w: np.ndarray, target: float) -> np.ndarray:
    """Finn konstant c slik at vektet snitt av expit(logit(p)+c) = target."""
    lo, hi = -5.0, 5.0
    z = _logit(p)
    for _ in range(60):
        c = (lo + hi) / 2
        if np.average(_expit(z + c), weights=w) < target:
            lo = c
        else:
            hi = c
    return _expit(z + (lo + hi) / 2)


# --- 4. Samlet tilordning --------------------------------------------------------------

def attach(a: pd.DataFrame, donors: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    res = results_by_fylke()
    q = preference(a, donors)
    elig = eligibility(a)
    t = turnout(a)
    w = a["vekt"].to_numpy()

    # Frammøte per fylke blant stemmeberettigede.
    for f, idx in a.groupby("fylke").groups.items():
        ix = a.index.get_indexer(idx)
        t[ix] = _shift_to_target(t[ix], w[ix] * elig[ix], float(res.loc[f, "frammote"]))

    # Partifordeling blant velgere: skaler preferansene vekselvis mot fylkesresultatet
    # og mot partivalg etter kjønn × alder (13554). Faktorene er multiplikative per
    # parti og gruppe, så rangeringen mellom agenter innen en gruppe beholdes.
    voter_w = w * elig * t
    groups = {"fylke": a["fylke"].to_numpy()}
    age4 = np.empty(len(a), dtype=object)
    for lo, hi, lab in AGE4:
        age4[a["alder"].between(lo, hi).to_numpy()] = lab
    groups["kjonn_alder"] = (a["kjonn"] + "|" + pd.Series(age4, index=a.index)).to_numpy()
    sex_age = survey_by_sex_age()
    targets = {"fylke": {f: res.loc[f, PARTIES].to_numpy(float) for f in res.index},
               "kjonn_alder": {f"{k}|{al}": sex_age.loc[(k, al)].to_numpy(float) for k, al in sex_age.index}}
    for _ in range(40):
        for dim in ["kjonn_alder", "fylke"]:
            for g, tgt in targets[dim].items():
                ix = np.flatnonzero(groups[dim] == g)
                if len(ix) == 0:
                    continue
                got = (q[ix] * voter_w[ix, None]).sum(axis=0)
                got = got / got.sum()
                q[ix] = q[ix] * np.where(got > 0, tgt / got, 1.0)
                q[ix] = q[ix] / q[ix].sum(axis=1, keepdims=True)

    u = rng.random(len(a))
    has_right = rng.random(len(a)) < elig
    voted = has_right & (u < t)
    cum = q.cumsum(axis=1)
    party = np.array(PARTIES)[(rng.random(len(a))[:, None] > cum).sum(axis=1).clip(0, len(PARTIES) - 1)]

    out = a.copy()
    out["stemmerett"] = np.where(has_right, "ja", "nei")
    out["stemte_2025"] = np.where(~has_right, "ikke_stemmerett", np.where(voted, "ja", "nei"))
    out["parti_2025"] = np.where(voted, party, out["stemte_2025"].map({"nei": "stemte_ikke", "ikke_stemmerett": "ikke_stemmerett"}))
    # Partisympati for alle (også ikke-velgere), fra samme kalibrerte preferanse.
    out["partisympati"] = party
    return out
