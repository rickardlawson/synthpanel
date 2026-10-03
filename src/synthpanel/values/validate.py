"""Testsett for verdilaget.

En andel av ESS runde 11 holdes helt utenfor. Panelet bygges med de øvrige
respondentene som donorer og skal deretter anslå hvordan undergrupper av de
holdte respondentene svarte. Resultatet sammenlignes med

  * grunnlinje: landssnittet uten demografi (det en «gjennomsnittsnordmann» gir),
  * støygulv: forventet feil bare fordi testgruppen er liten.

Panelet bør ligge under grunnlinjen og nær støygulvet.

Kjør:  python -m synthpanel.values.validate
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from synthpanel import config
from synthpanel.values import ess, match

INDICATORS = {
    "verdi_apenhet": "høy", "verdi_trygghet": "høy", "verdi_selvhevdelse": "høy",
    "verdi_fellesskap": "høy", "tillit": "høy", "risikovilje": "høy",
    "politisk_sted=venstre": None, "politisk_sted=høyre": None,
    "religiositet": "høy", "politisk_interesse": "høy", "klimabekymring": "høy",
}


def _indicator(df: pd.DataFrame, name: str) -> pd.Series:
    if "=" in name:
        col, val = name.split("=")
    else:
        col, val = name, INDICATORS[name]
    s = df[col]
    return pd.Series(np.where(s.isin(["ukjent"]) | s.isna(), np.nan, (s == val).astype(float)), index=df.index)


def _groups(f: pd.DataFrame) -> pd.DataFrame:
    g = pd.DataFrame(index=f.index)
    g["kjonn"] = f["kjonn"]
    g["alder"] = pd.cut(f["alder"], [17, 34, 59, 200], labels=["18-34", "35-59", "60+"]).astype(str)
    g["utdanning"] = pd.cut(f["edu"], [0, 2, 3.5, 5], labels=["lav", "middels", "høy"]).astype(str)
    g["region"] = f["region"]
    g["innvandrer"] = np.where(f["innv"] == "innvandrer", "ja", "nei")
    g["status"] = f["status"].where(f["status"].isin(["sysselsatt", "pensjonist", "student"]), "annet")
    g["kjonn×alder"] = g["kjonn"] + "|" + g["alder"]
    g["utdanning×alder"] = g["utdanning"] + "|" + g["alder"]
    return g


def _wshare(x: pd.Series, w: pd.Series) -> tuple[float, float]:
    ok = x.notna()
    x, w = x[ok], w[ok]
    if w.sum() == 0:
        return np.nan, 0.0
    return float((x * w).sum() / w.sum()), float(w.sum() ** 2 / (w ** 2).sum())


def run(seed: int = 7) -> dict:
    cfg = config.load("values")["validation"]
    agents = pd.read_parquet(config.PROCESSED_DIR / "agents.parquet")
    e = ess.load()
    e = e[e["agea"] >= 18].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    r11 = e.index[e["essround"] == cfg["holdout_round"]].to_numpy()
    hold_idx = rng.choice(r11, size=int(len(r11) * cfg["holdout_share"]), replace=False)
    hold = e.loc[hold_idx].reset_index(drop=True)
    train = e.drop(index=hold_idx).reset_index(drop=True)

    panel = match.attach(agents, train, np.random.default_rng(seed))

    # Klassifiser holdout med samme grenser som panelet (beregnet fra treningsdonorene).
    dtrain = match.derive(train)
    w_train = train["pspwght"].fillna(1.0)
    dh = match.derive(hold)
    for n in list(match.HIGHER_ORDER) + ["tillit"]:
        lo, hi = match.tertile_labels(dtrain[f"score_{n}"], w_train)
        sc = dh[f"score_{n}"]
        dh[f"verdi_{n}" if n != "tillit" else "tillit"] = np.select(
            [sc.isna(), sc < lo, sc < hi], ["ukjent", "lav", "middels"], "høy")
    for c in ["risikovilje", "politisk_sted", "religiositet", "politisk_interesse", "klimabekymring"]:
        dh[c] = dh[c].fillna("ukjent")
        dtrain[c] = dtrain[c].fillna("ukjent")
    for n in list(match.HIGHER_ORDER) + ["tillit"]:
        lo, hi = match.tertile_labels(dtrain[f"score_{n}"], w_train)
        sc = dtrain[f"score_{n}"]
        dtrain[f"verdi_{n}" if n != "tillit" else "tillit"] = np.select(
            [sc.isna(), sc < lo, sc < hi], ["ukjent", "lav", "middels"], "høy")

    gh = _groups(match.ess_features(hold).assign(region=hold["region"]))
    gp = _groups(match.agent_features(panel))
    wh = hold["pspwght"].fillna(1.0)
    wp = panel["vekt"]

    rows = []
    for ind in INDICATORS:
        yh, yp, yt = _indicator(dh, ind), _indicator(panel, ind), _indicator(dtrain, ind)
        base, _ = _wshare(yt, w_train)
        for dim in gh.columns:
            for val in sorted(set(gh[dim].dropna()) - {"nan"}):
                mh = gh[dim] == val
                truth, neff = _wshare(yh[mh], wh[mh])
                if neff < cfg["min_group_n"] or np.isnan(truth):
                    continue
                pred, _ = _wshare(yp[gp[dim] == val], wp[gp[dim] == val])
                se = np.sqrt(max(truth * (1 - truth), 1e-4) / neff)
                rows.append({"indikator": ind, "gruppe": f"{dim}={val}", "n": round(neff),
                             "fasit": truth, "panel": pred, "landssnitt": base,
                             "feil_panel": abs(pred - truth), "feil_landssnitt": abs(base - truth),
                             "stoygulv": se * np.sqrt(2 / np.pi)})
    res = pd.DataFrame(rows)
    pp = lambda s: round(float(s.mean() * 100), 2)
    summary = {
        "holdout_respondenter": int(len(hold)),
        "donorer": int(len(train)),
        "celler": int(len(res)),
        "snittfeil_panel_pp": pp(res["feil_panel"]),
        "snittfeil_landssnitt_pp": pp(res["feil_landssnitt"]),
        "stoygulv_pp": pp(res["stoygulv"]),
        "panel_bedre_enn_landssnitt_andel": round(float((res["feil_panel"] < res["feil_landssnitt"]).mean()), 3),
        "per_indikator": {
            ind: {"panel_pp": pp(g["feil_panel"]), "landssnitt_pp": pp(g["feil_landssnitt"]), "stoygulv_pp": pp(g["stoygulv"])}
            for ind, g in res.groupby("indikator")
        },
    }
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.PROCESSED_DIR / "validation_cells.csv", index=False)
    with open(config.PROCESSED_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
