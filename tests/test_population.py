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
    assert agents.drop(columns=["vekt"]).notna().all().all()


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
