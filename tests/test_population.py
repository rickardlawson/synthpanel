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
