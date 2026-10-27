"""Debattkobling (L5 v0): hvordan en Signalist-debatt treffer befolkningen.

En debatt fra Signalist (`/claims`) har en standpunktakse (−10 … +10), en
aggregert score per argumentdimensjon og artiklene bak. For hver agent regnes:

1. **Eksponering** – sannsynligheten for å ha fått med seg saken. Nasjonale
   medier når alle, lokale bare fylket. Rekkevidde fra nettstedets lesertall,
   skalert med politisk interesse (ESS).
2. **Dragning** – hvilken vei argumentene trekker *denne* personen (−1 mot
   aksens −10-pol, +1 mot +10-polen). Argumentene vektes med hvor mye agenten
   bryr seg om den dimensjonen: rettferdighet → selvoverskridelse, orden →
   bevaring, frihet → åpenhet, institusjoner → tillit, økonomi → egen økonomi.

Dette er *ikke* et estimat av hva folk mener. Det sier hvem debatten når og
hvem argumentene treffer hardest – i tråd med prinsippet om at telling (hvor
mange, hvor) kommer fra populasjonen og ikke fra en språkmodell. Mening krever
et startpunkt (anker) per sak, som ennå ikke finnes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from synthpanel import config

DIMENSIONS = [
    "Health / harm", "Economy / cost", "Freedom / autonomy", "Fairness / equality",
    "Morality / decency", "Order / security / authority", "Evidence / effectiveness",
    "Institution / law / legitimacy",
]


@dataclass
class Article:
    uid: str
    county: str | None
    media_type: str | None
    readership: float | None


@dataclass
class Debate:
    claim_id: str
    statement: str
    axis: dict[str, str]
    arguments: dict[str, float]          # dimensjon → score (−10…+10, mot aksen)
    articles: list[Article] = field(default_factory=list)
    article_total: int = 0               # artikler i perioden (kan være flere enn utvalget)
    stance_label: str | None = None

    @classmethod
    def from_signalist(cls, d: dict) -> "Debate":
        samples = d.get("sample_articles") or []
        if isinstance(samples, dict):     # paginert form: {total, articles: [...]}
            total, samples = samples.get("total"), samples.get("articles", [])
        else:
            total = None
        summary = d.get("summary") or {}
        arts = [Article(a.get("article_uid", ""), a.get("county_name"), a.get("media_type_name"),
                        a.get("site_readership")) for a in samples]
        return cls(
            claim_id=d["claim_id"], statement=d.get("statement", ""), axis=d.get("stance_axis", {}),
            arguments={x["argument_name"]: float(x["score"]) for x in d.get("arguments_aggregate", [])},
            articles=arts,
            article_total=int(total or summary.get("article_amount") or len(arts)),
            stance_label=summary.get("overall_stance_label"),
        )


def _fylke(name: str | None) -> str | None:
    """«Troms - Romsa - Tromssa» → «Troms» (Signalist bruker norsk navn alene)."""
    return name.split(" - ")[0].strip() if isinstance(name, str) else None


def _z(s: pd.Series, w: pd.Series) -> pd.Series:
    ok = s.notna()
    m = np.average(s[ok], weights=w[ok])
    sd = np.sqrt(np.average((s[ok] - m) ** 2, weights=w[ok])) or 1.0
    return ((s - m) / sd).fillna(0.0)


def traits(agents: pd.DataFrame) -> pd.DataFrame:
    """Standardiserte egenskaper argumentene kan treffe (0 = befolkningssnittet)."""
    w = agents["vekt"]
    out = pd.DataFrame(index=agents.index)
    for col in ["score_fellesskap", "score_trygghet", "score_apenhet", "score_tillit"]:
        out[col] = _z(agents[col], w) if col in agents else 0.0
    # Egen økonomi: boliglån (eier og i arbeidsfør alder) og lavinntekt gjør kostnad mer følbart.
    debt = agents["eierstatus"].isin(["selveier", "andelseier"]) & (agents["alder"] < 67)
    raw = debt.astype(float) + (agents["lavinntekt"] == "ja").astype(float)
    out["relevans_okonomi"] = _z(raw, w)
    edu = agents["utdanning"].map({"uh_lang": 1.0, "uh_kort": 0.5, "grunnskole": -1.0}).fillna(0.0)
    out["utdanning"] = _z(edu, w)
    return out


def exposure(agents: pd.DataFrame, debate: Debate, cfg: dict | None = None) -> np.ndarray:
    """Sannsynlighet for at hver agent har fått med seg debatten."""
    cfg = cfg or config.load("debates")
    w = agents["vekt"]
    fylke = agents["fylke_navn"].map(_fylke)
    pop_total = w.sum()
    pop_fylke = w.groupby(fylke).sum()
    share = cfg["andel_av_lesere_per_sak"]
    missing = cfg["lesere_mangler"]
    cap = cfg["maks_rekkevidde_per_sak"]
    hazard = np.zeros(len(agents))
    for a in debate.articles:
        local = _fylke(a.county)
        readers = a.readership or missing["lokal" if local else "nasjonal"]
        if local and local in pop_fylke.index:
            reach = min(cap, share * readers / pop_fylke[local])
            mask = (fylke == local).to_numpy()
        else:
            reach = min(cap, share * readers / pop_total)
            mask = np.ones(len(agents), dtype=bool)
        hazard[mask] += -np.log1p(-reach)
    # Artikler utenfor utvalget: antas nasjonale med standard rekkevidde (lokale
    # artikler i utvalget skal ikke ganges opp til hele perioden).
    unseen = max(debate.article_total - len(debate.articles), 0)
    if unseen:
        hazard += -np.log1p(-min(cap, share * missing["nasjonal"] / pop_total)) * unseen

    if "politisk_interesse" in agents:
        hazard *= agents["politisk_interesse"].map(cfg["interesse_faktor"]).fillna(1.0).to_numpy()
    return 1.0 - np.exp(-hazard)


def pull(agents: pd.DataFrame, debate: Debate, cfg: dict | None = None,
         tr: pd.DataFrame | None = None) -> np.ndarray:
    """Dragning langs debattens akse, −1 (mot −10-polen) … +1 (mot +10-polen)."""
    cfg = cfg or config.load("debates")
    tr = traits(agents) if tr is None else tr
    beta = cfg["beta"]
    num = np.zeros(len(agents))
    den = np.zeros(len(agents))
    for dim in DIMENSIONS:
        score = debate.arguments.get(dim, 0.0)
        z = sum(wt * tr[col].to_numpy() for col, wt in cfg["argumentverdier"][dim].items())
        recept = np.exp(beta * z)
        num += score * recept
        den += recept
    return num / (10.0 * den)


def react(agents: pd.DataFrame, debate: Debate, by: list[str], cfg: dict | None = None) -> pd.DataFrame:
    """Eksponering og dragning per gruppe, vektet med befolkningen."""
    cfg = cfg or config.load("debates")
    d = agents[by + ["vekt"]].copy()
    d["eksp"] = exposure(agents, debate, cfg)
    d["drag"] = pull(agents, debate, cfg)
    d["we"] = d["vekt"] * d["eksp"]            # personer som trolig har sett saken
    d["drag_w"] = d["we"] * d["drag"]
    g = d.groupby(by, observed=True).agg(
        agenter=("vekt", "size"), personer=("vekt", "sum"), eksponert=("we", "sum"), drag_w=("drag_w", "sum"))
    pop_drag = (d["drag_w"].sum() / d["we"].sum()) if d["we"].sum() > 0 else np.nan
    out = pd.DataFrame({
        "agenter": g["agenter"],
        "personer": g["personer"].round(),
        "andel_eksponert": g["eksponert"] / g["personer"],
        "eksponerte_personer": g["eksponert"].round(),
        "snitt_dragning": g["drag_w"] / g["eksponert"].where(g["eksponert"] > 0),
    }).reset_index()
    # 100 = trekkes like mye som befolkningen; 130 = 30 % sterkere enn snittet
    out["indeks_dragning"] = (100 * out["snitt_dragning"] / pop_drag) if pop_drag else np.nan
    return out.sort_values("eksponerte_personer", ascending=False)
