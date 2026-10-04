"""Sjekker at den bygde populasjonen stemmer med SSB-rådataene."""
import pandas as pd
import pytest

from synthpanel import config

AGENTS = config.PROCESSED_DIR / "agents.parquet"
pytestmark = pytest.mark.skipif(not AGENTS.exists(), reason="kjør `make build` først")


@pytest.fixture(scope="module")
def agents():
    return pd.read_parquet(AGENTS)


@pytest.fixture(scope="module")
def pop():
    p = pd.read_parquet(config.RAW_DIR / "population_07459.parquet")
    p = p[p["value"] > 0].copy()
    p["alder"] = p["Alder"].str.rstrip("+").astype(int)
    return p[p["alder"] >= 18]


def test_total_matches_ssb(agents, pop):
    assert agents["vekt"].sum() == pytest.approx(pop["value"].sum(), rel=1e-9)


def test_every_kommune_matches_ssb_exactly(agents, pop):
    sim = agents.groupby("kommune")["vekt"].sum()
    ssb = pop.groupby("Region")["value"].sum()
    assert len(sim) == len(ssb) == 357
    assert (sim / ssb.reindex(sim.index) - 1).abs().max() < 1e-9


def test_gender_by_age_matches_ssb(agents, pop):
    sim = agents.groupby(["kjonn", "alder"])["vekt"].sum()
    ssb = pop.assign(kjonn=pop["Kjonn"].map({"1": "mann", "2": "kvinne"})).groupby(["kjonn", "alder"])["value"].sum()
    # Ettårig alder er trukket innen aldersbånd, så her tillates litt støy.
    rel = (sim / ssb.reindex(sim.index) - 1).abs()
    assert rel[ssb.reindex(sim.index) > 20000].max() < 0.05


def test_education_by_fylke_matches_ssb(agents):
    e = pd.read_parquet(config.RAW_DIR / "education_08921.parquet")
    e = e[(e["Region"] != "0") & (e["Alder"] != "16-19")]
    labels = config.load("frame")["education_labels"]
    e["utdanning"] = e["UtdanNivaa"].map(labels)
    ssb = e.groupby(["Region", "utdanning"])["value"].sum()
    ssb = ssb / ssb.groupby(level=0).transform("sum")
    a = agents[agents["alder"] >= 20]
    sim = a.groupby(["fylke", "utdanning"])["vekt"].sum()
    sim = sim / sim.groupby(level=0).transform("sum")
    diff = (sim - ssb.reindex(sim.index)).abs()
    assert diff.max() < 0.005, diff.sort_values().tail()  # under 0,5 prosentpoeng


def test_every_agent_has_complete_profile(agents):
    # score_* er kontinuerlige verdimål og kan mangle når donoren ikke svarte;
    # kategoriene står da som «ukjent».
    cols = [c for c in agents.columns if not c.startswith("score_")]
    assert agents[cols].notna().all().all()


# --- L0b ---------------------------------------------------------------------

def _share(df, col):
    s = df.groupby(col)["vekt"].sum()
    return s / s.sum()


def test_labour_status_matches_ssb(agents):
    from synthpanel.frame import enrich
    t = enrich.labour_table()
    for band, (lo, hi) in {"30-54": (30, 54), "67+": (67, 200)}.items():
        ssb = t[t["b_status"] == band].groupby("arbeidsstatus")["value"].sum()
        ssb = ssb / ssb.sum()
        sim = _share(agents[agents["alder"].between(lo, hi)], "arbeidsstatus")
        assert (sim.reindex(ssb.index).fillna(0) - ssb).abs().max() < 0.005, band


def test_household_matches_ssb(agents):
    from synthpanel.frame import enrich
    pop = pd.read_parquet(config.RAW_DIR / "population_07459.parquet")
    pop = pop[pop["value"] > 0].assign(alder=lambda d: d["Alder"].str.rstrip("+").astype(int),
                                       kjonn=lambda d: d["Kjonn"]).rename(columns={"value": "personer"})
    nat = enrich.household_national(pop)
    nat["husholdning"] = enrich.collapse_household(nat["husholdning"])
    for band, (lo, hi) in {"67+": (67, 200), "30-44": (30, 44), "45-61": (45, 61)}.items():
        ssb = nat[nat["hh_band"] == band].groupby("husholdning")["value"].sum()
        ssb = ssb / ssb.sum()
        sim = _share(agents[agents["alder"].between(lo, hi)], "husholdning")
        assert (sim.reindex(ssb.index).fillna(0) - ssb).abs().max() < 0.01, band


def test_small_children_parents_are_mostly_30_to_49(agents):
    """Regresjon: grov aldersgruppe (30–66) i 06071 ga 36 % småbarnsforeldre over 50."""
    p = agents[agents["husholdning"] == "par_smaa_barn"]
    share = p.loc[p["alder"].between(30, 49), "vekt"].sum() / p["vekt"].sum()
    old = p.loc[p["alder"] >= 55, "vekt"].sum() / p["vekt"].sum()
    assert share > 0.75 and old < 0.03


def test_immigrant_categories_consistent(agents):
    foreign = agents["bakgrunn"] != "norsk"
    assert (agents.loc[foreign, "innvkat"] != "ovrige").all()
    assert (agents.loc[~foreign, "innvkat"] == "ovrige").all()


def test_income_deciles_valid(agents):
    assert agents["inntektsdesil"].between(1, 10).all()
    # Personvektet: større husholdninger har høyere inntekt, så toppdesilene skal være størst.
    s = _share(agents, "inntektsdesil")
    assert s.loc[10] > s.loc[1]


def test_heldout_immigrant_education_by_sex(agents):
    """Uavhengig kontroll: kjønnsfordelingen i 12934 brukes ikke i kalibreringen."""
    from synthpanel.frame import enrich
    v = pd.read_parquet(config.RAW_DIR / "edu_immigrant_fylke_12934.parquet")
    v = v[(v["Region"] != "0") & (v["InnvandrKat"] == "B")].assign(utd=lambda d: d["UtdanNivaa"].map(enrich.EDU_09599))
    devs = []
    for kj, name in [("1", "mann"), ("2", "kvinne")]:
        ssb = v[v["Kjonn"] == kj].groupby(["Region", "utd"])["value"].sum()
        ssb = ssb / ssb.groupby(level=0).transform("sum")
        sub = agents[(agents["innvkat"] == "innvandrer") & (agents["kjonn"] == name)]
        sim = sub.groupby(["fylke", "utdanning"])["vekt"].sum()
        sim = sim / sim.groupby(level=0).transform("sum")
        for f, n in sub.groupby("fylke").size().items():
            uh = lambda x, f=f: x.loc[f].reindex(["uh_kort", "uh_lang"]).fillna(0).sum()
            devs.append((n, abs(uh(sim) - uh(ssb))))
    assert sum(d for _, d in devs) / len(devs) < 0.03          # snitt under 3 prosentpoeng
    assert max(d for n, d in devs if n >= 300) < 0.03           # store grupper under 3 pp


def test_elderly_employment_follows_single_year_age(agents):
    """Regresjon: 67+ samlet ga 12 % «i arbeid» blant 80+."""
    from synthpanel.frame import enrich
    e = pd.read_parquet(config.RAW_DIR / "employment_age_06161.parquet")
    e = e.assign(alder=e["Alder"].astype(int), kjonn=e["Kjonn"].map({"1": "mann", "2": "kvinne"}))
    for _, r in e[e["alder"].isin([62, 67, 70, 74])].iterrows():
        sub = agents[(agents["alder"] == r["alder"]) & (agents["kjonn"] == r["kjonn"])]
        sim = sub.loc[sub["arbeidsstatus"] == "sysselsatt", "vekt"].sum() / sub["vekt"].sum()
        assert abs(sim - r["value"] / 100) < 0.02, (r["alder"], r["kjonn"])
    old = agents[agents["alder"] >= 80]
    assert old.loc[old["arbeidsstatus"] == "sysselsatt", "vekt"].sum() / old["vekt"].sum() < 0.08


def test_low_income_matches_ssb_groups(agents):
    from synthpanel.frame import enrich
    t = enrich.lowinc_target_rates()
    lav = lambda d: d.loc[d["lavinntekt"] == "ja", "vekt"].sum() / d["vekt"].sum()
    assert abs(lav(agents) - t["total"]) < 0.005
    assert abs(lav(agents[agents["arbeidsstatus"] == "aap_ufor"]) - t["lav_status"]["aap_ufor"]) < 0.01
    assert abs(lav(agents[agents["arbeidsstatus"] == "pensjonist"]) - t["lav_status"]["pensjonist"]) < 0.01
    alone = agents[(agents["husholdning"] == "aleneboende") & (agents["alder"] < 35)]
    assert abs(lav(alone) - t["lav_hh"]["alene_u35"]) < 0.015


def test_income_correlates_with_low_income_and_status(agents):
    m = lambda d: (d["inntektsdesil"] * d["vekt"]).sum() / d["vekt"].sum()
    assert m(agents[agents["lavinntekt"] == "ja"]) < 3.5 < m(agents[agents["lavinntekt"] == "nei"])
    assert m(agents[agents["arbeidsstatus"] == "aap_ufor"]) < m(agents[agents["arbeidsstatus"] == "sysselsatt"])


# --- Bolig og verdier ------------------------------------------------------------

def test_housing_tenure_by_income_quartile(agents):
    """Eierstatus per inntektskvartil (omregnet til husholdninger) mot 14900."""
    from synthpanel.frame import enrich
    d = pd.read_parquet(config.RAW_DIR / "housing_income_14900_14921.parquet")
    t = d[(d["Region"] == "0") & (d["dim"] == "eierstatus") & (d["Inntekstgruppe"].isin(["41", "44"]))]
    t = t.assign(kat=t["kode"].map(enrich.EIER)).pivot_table(index="Inntekstgruppe", columns="kat", values="value")
    t = t.div(t.sum(axis=1), axis=0)
    hw = agents["vekt"] / agents["husholdning"].map(enrich.HH_SIZE)
    for q, decs in {"41": [1, 2], "44": [9, 10]}.items():
        m = agents["inntektsdesil"].isin(decs)
        sim = hw[m & (agents["eierstatus"] == "leier")].sum() / hw[m].sum()
        assert abs(sim - t.loc[q, "leier"]) < 0.05, q


def test_oslo_lives_in_apartments(agents):
    oslo = agents[agents["fylke"] == "03"]
    share = oslo.loc[oslo["boligtype"] == "blokk", "vekt"].sum() / oslo["vekt"].sum()
    assert share > 0.45


@pytest.mark.skipif(not (config.RAW_DIR / "ess_norway.parquet").exists(), reason="ESS ikke hentet")
def test_values_attached_and_balanced(agents):
    for col in ["verdi_apenhet", "verdi_trygghet", "tillit"]:
        s = agents.groupby(col)["vekt"].sum() / agents["vekt"].sum()
        for lvl in ["lav", "middels", "høy"]:
            assert 0.2 < s[lvl] < 0.45, (col, lvl)
    # Kjent mønster: åpenhet for endring faller med alder.
    young = agents[agents["alder"] < 30]
    old = agents[agents["alder"] >= 67]
    hi = lambda d: d.loc[d["verdi_apenhet"] == "høy", "vekt"].sum() / d["vekt"].sum()
    assert hi(young) > hi(old) + 0.2


@pytest.mark.skipif(not (config.PROCESSED_DIR / "validation_report.json").exists(), reason="kjør `make validate`")
def test_value_layer_beats_national_average_on_holdout():
    import json
    r = json.loads((config.PROCESSED_DIR / "validation_report.json").read_text(encoding="utf-8"))
    assert r["snittfeil_panel_pp"] < r["snittfeil_landssnitt_pp"]
    assert r["snittfeil_panel_pp"] < r["stoygulv_pp"] + 1.5


# --- Politikk og medier ---------------------------------------------------------

def _pol_ready():
    return (config.RAW_DIR / "valg_resultat_2025.parquet").exists()


@pytest.mark.skipif(not _pol_ready(), reason="valgdata ikke hentet")
def test_election_2025_national_and_oslo(agents):
    from synthpanel.politics import assign
    r = pd.read_parquet(config.RAW_DIR / "valg_resultat_2025.parquet")
    nat = r[assign.PARTIES].sum() / r[assign.PARTIES].sum().sum()
    voters = agents[agents["stemte_2025"] == "ja"]
    sim = voters.groupby("parti_2025")["vekt"].sum() / voters["vekt"].sum()
    assert (sim.reindex(nat.index).fillna(0) - nat).abs().max() < 0.008
    eligible = agents.loc[agents["stemmerett"] == "ja", "vekt"].sum()
    assert abs(eligible - r["stemmeberettigede"].sum()) / r["stemmeberettigede"].sum() < 0.02
    assert abs(voters["vekt"].sum() / eligible - r["godkjente"].sum() / r["stemmeberettigede"].sum()) < 0.01
    oslo = voters[voters["fylke"] == "03"]
    ro = r[r["kommune"] == "0301"]
    assert abs(oslo.loc[oslo["parti_2025"] == "A", "vekt"].sum() / oslo["vekt"].sum()
               - float(ro["A"].iloc[0] / ro[assign.PARTIES].sum(axis=1).iloc[0])) < 0.02


@pytest.mark.skipif(not _pol_ready(), reason="valgdata ikke hentet")
def test_young_men_frp_matches_election_survey(agents):
    """13554 er brukt i kalibreringen – dette sikrer at den virker (unge menn: FrP 38 % i 2025)."""
    v = agents[(agents["stemte_2025"] == "ja") & (agents["kjonn"] == "mann") & agents["alder"].between(18, 34)]
    share = v.loc[v["parti_2025"] == "FRP", "vekt"].sum() / v["vekt"].sum()
    assert 0.33 < share < 0.43


@pytest.mark.skipif(not _pol_ready(), reason="valgdata ikke hentet")
def test_heldout_income_gradient_direction(agents):
    """Uavhengig (13698 ikke brukt): Høyre øker og SV faller med inntekt."""
    v = agents[agents["stemte_2025"] == "ja"]
    sh = lambda d, p: d.loc[d["parti_2025"] == p, "vekt"].sum() / d["vekt"].sum()
    low, high = v[v["inntektsdesil"] <= 3], v[v["inntektsdesil"] == 10]
    assert sh(high, "H") > sh(low, "H")
    assert sh(low, "SV") > sh(high, "SV")


def test_media_rates_follow_age(agents):
    if "daglig_tiktok" not in agents.columns:
        pytest.skip("medielag ikke bygget")
    y = agents[(agents["kjonn"] == "kvinne") & agents["alder"].between(18, 24)]
    o = agents[agents["alder"] >= 80]
    share = lambda d: d.loc[d["daglig_tiktok"] == "ja", "vekt"].sum() / d["vekt"].sum()
    assert abs(share(y) - 0.80) < 0.06 and share(o) < 0.05
    assert (agents["netthandel_dagligvarer"] == "ja").mean() > 0.05


def test_leisure_matches_ssb_and_gradients(agents):
    if "friluft_jakt" not in agents.columns:
        pytest.skip("fritidslag ikke bygget")
    import pandas as pd
    from synthpanel import config
    ka = pd.read_parquet(config.RAW_DIR / "leisure_trening_kjonn_alder_13388.parquet")
    target = ka[(ka["TreningsAkt"] == "04") & (ka["Kjonn"] == "2") & (ka["Alder"] == "25-44")]["value"].iloc[0] / 100
    g = agents[(agents["kjonn"] == "kvinne") & agents["alder"].between(25, 44)]
    share = lambda d, c: d.loc[d[c] == "ja", "vekt"].sum() / d["vekt"].sum()  # noqa: E731
    assert abs(share(g, "trening_styrke") - target) < 0.03          # treffer SSB per kjønn × alder
    rural, urban = agents[agents["sentralitet"].isin(["05", "06"])], agents[agents["sentralitet"] == "01"]
    assert share(rural, "friluft_jakt") > 1.4 * share(urban, "friluft_jakt")   # jakt er distrikt
    weekly = agents[agents["treningsfrekvens"] == "ukentlig"]
    rare = agents[agents["treningsfrekvens"] == "sjelden"]
    assert share(weekly, "trening_lop") > 1.5 * share(rare, "trening_lop")     # aktive gjør mer
