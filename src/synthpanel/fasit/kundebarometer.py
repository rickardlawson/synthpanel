"""Merketest: rangerer panelet merkene omtrent som BI Norsk kundebarometer?

Hver persona i galleriet for hele befolkningen (de ti største grupperingene)
får lista over merker og anslår hvor fornøyd hen er (0–100), eller hopper over
merker hen ikke har erfaring med. Panelets score per merke er snittet vektet
med grupperingenes størrelse. Vi sammenligner med BI på tre måter:

- rangkorrelasjon (Spearman) på tvers av alle merker
- rangkorrelasjon *innen* bransje (det viktigste – «hvem slår hvem»)
- snittavvik etter at panelets nivå er justert (modellen bruker skalaen annerledes)

Krever ANTHROPIC_API_KEY.  Kjør:  python -m synthpanel.fasit.kundebarometer
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from synthpanel import config
from synthpanel.personas import voice

REPORT = config.PROCESSED_DIR / "fasit_kundebarometer.json"


def truth() -> pd.DataFrame:
    cfg = config.load("fasit/kundebarometer_2026")
    rows = [(b, m, s) for b, ms in cfg["merker"].items() for m, s in ms.items()]
    return pd.DataFrame(rows, columns=["bransje", "merke", "bi"])


def tool(brands: list[str]) -> dict:
    return {
        "name": "vurdering",
        "description": "Hvor fornøyd du er med hvert merke, 0–100. Utelat merker du ikke har erfaring med eller formening om.",
        "input_schema": {"type": "object", "properties": {
            b: {"type": "integer", "minimum": 0, "maximum": 100} for b in brands}},
    }


def rate(persona: dict, brands: list[str]) -> dict:
    prompt = ("Tenk på din egen erfaring som kunde, eller det inntrykket du har. Hvor fornøyd er du med disse "
              "selskapene og merkene, på en skala fra 0 (svært misfornøyd) til 100 (svært fornøyd)? "
              "Hopp over dem du ikke har noe forhold til. Svar med verktøyet `vurdering`.\n\n" + ", ".join(brands))
    resp = voice._call(voice.system_prompt(persona["brief"]), [{"role": "user", "content": prompt}],
                       max_tokens=900, tools=[tool(brands)], tool_choice={"type": "tool", "name": "vurdering"})
    data = next((b["input"] for b in resp.get("content", []) if b.get("type") == "tool_use"), {})
    return {"id": persona["id"], "score": {k: v for k, v in data.items() if k in brands and isinstance(v, (int, float))}}


def evaluate(personas: list[dict], ratings: list[dict], t: pd.DataFrame) -> dict:
    w = {p["id"]: p["andel"] for p in personas}
    panel = {}
    for m in t["merke"]:
        got = [(w[r["id"]], r["score"][m]) for r in ratings if m in r.get("score", {})]
        tw = sum(x for x, _ in got)
        panel[m] = sum(x * s for x, s in got) / tw if tw else None
    d = t.assign(panel=t["merke"].map(panel)).dropna()
    d["panel_justert"] = d["panel"] - d["panel"].mean() + d["bi"].mean()
    within = {b: g["panel"].corr(g["bi"], method="spearman") for b, g in d.groupby("bransje") if len(g) >= 3}
    return {
        "kilde": "BI Norsk kundebarometer 2026", "modell": voice._model(),
        "tidspunkt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "personas": len(personas), "merker": len(d),
        "spearman_alle": round(float(d["panel"].corr(d["bi"], method="spearman")), 3),
        "spearman_innen_bransje": {b: round(float(v), 3) for b, v in within.items()},
        "spearman_innen_bransje_snitt": round(float(pd.Series(within).mean()), 3),
        "snittavvik_justert": round(float((d["panel_justert"] - d["bi"]).abs().mean()), 2),
        "tabell": d.round(1).to_dict(orient="records"),
    }


def main() -> dict:
    from synthpanel.api.main import PersonaParams, _get_personas
    personas = _get_personas(PersonaParams())["personas"]
    t = truth()
    ratings = voice.run_all(personas, rate, list(t["merke"]))
    rep = evaluate(personas, ratings, t)
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merketest mot {rep['kilde']} ({rep['merker']} merker, {rep['personas']} personas, {rep['modell']})")
    print(f"  Rangkorrelasjon alle merker:      {rep['spearman_alle']:+.2f}")
    print(f"  Rangkorrelasjon innen bransje:    {rep['spearman_innen_bransje_snitt']:+.2f}  (1 = samme rekkefølge som BI)")
    for b, v in rep["spearman_innen_bransje"].items():
        print(f"     {b:<12} {v:+.2f}")
    print(f"  Snittavvik etter nivåjustering:   {rep['snittavvik_justert']:.1f} poeng")
    print(f"Rapport: {REPORT}")
    return rep


if __name__ == "__main__":
    main()
