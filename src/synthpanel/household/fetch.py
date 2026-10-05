"""Hent rådata for husholdningslaget fra SSB.

| Fil                              | Tabell        | Innhold                                                     |
|----------------------------------|---------------|-------------------------------------------------------------|
| hh_income_deciles_12558          | 12558         | Andel husholdninger i hver nasjonale desil per kommune, og desilgrensene (kr) |
| hh_income_median_06944           | 06944         | Median husholdningsinntekt per kommune og husholdningstype (kontroll) |
| hh_spend_type_quartile_14157     | 14157         | Utgift per husholdning etter vare-/tjenestegruppe, type × inntektskvartil (2022) |
| hh_spend_centrality_14161        | 14161         | … etter sentralitet                                         |
| hh_spend_alone_age_14227         | 14227         | … for aleneboende etter alder                               |
| hh_spend_total_14100             | 14100         | … landsgjennomsnitt (kontroll)                              |
| hh_kpi_14709                     | 14709         | KPI, årsgjennomsnitt (prisjustering 2022 → siste hele år)   |
| hh_cars_new_12906                | 12906         | Førstegangsregistrerte personbiler per kommune og drivstoff |
| hh_cars_fleet_13370              | 13370         | Personbiler per kommune etter eierform og drivstoff         |

Kjør:  python -m synthpanel.household.fetch
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from synthpanel import config, ssb

LATEST = {"variableCode": "Tid", "valueCodes": ["top(1)"]}
KOMMUNE = {"variableCode": "Region", "codelist": "vs_Kommun", "valueCodes": ["*"]}      # kjøretøytabellene
KOMMUNE_INNT = {"variableCode": "Region", "codelist": "vs_Kommune", "valueCodes": ["*"]}  # inntektstabellene
SPEND = {"variableCode": "ContentsCode", "valueCodes": ["Utgift", "Standardfeil"]}
GOODS = {"variableCode": "VareTjenesteGruppe", "valueCodes": ["*"]}


def _with_national(sel_kommune: list[dict], sel_national: list[dict], table: str):
    import pandas as pd
    k = ssb.get_table(table, sel_kommune)
    n = ssb.get_table(table, sel_national)
    return pd.concat([n, k], ignore_index=True)


JOBS = {
    "hh_income_deciles_12558": lambda: _with_national(
        [KOMMUNE_INNT, {"variableCode": "InntektSkatt", "valueCodes": ["00S"]},
         {"variableCode": "Desiler", "valueCodes": ["*"]},
         {"variableCode": "ContentsCode", "valueCodes": ["AndelHush", "VerdiDesil", "AntHush"]}, LATEST],
        [{"variableCode": "Region", "valueCodes": ["0"]}, {"variableCode": "InntektSkatt", "valueCodes": ["00S"]},
         {"variableCode": "Desiler", "valueCodes": ["*"]},
         {"variableCode": "ContentsCode", "valueCodes": ["AndelHush", "VerdiDesil", "AntHush"]}, LATEST], "12558"),
    "hh_income_median_06944": lambda: _with_national(
        [KOMMUNE_INNT, {"variableCode": "HusholdType", "valueCodes": ["*"]},
         {"variableCode": "ContentsCode", "valueCodes": ["InntSkatt", "AntallHushold"]}, LATEST],
        [{"variableCode": "Region", "valueCodes": ["0"]}, {"variableCode": "HusholdType", "valueCodes": ["*"]},
         {"variableCode": "ContentsCode", "valueCodes": ["InntSkatt", "AntallHushold"]}, LATEST], "06944"),
    "hh_spend_type_quartile_14157": lambda: ssb.get_table("14157", [
        GOODS, {"variableCode": "HusholdType", "valueCodes": ["*"]},
        {"variableCode": "Inntektskvartil", "valueCodes": ["*"]}, SPEND, LATEST]),
    "hh_spend_centrality_14161": lambda: ssb.get_table("14161", [
        GOODS, {"variableCode": "SentralitetKomm", "valueCodes": ["*"]}, SPEND, LATEST]),
    "hh_spend_alone_age_14227": lambda: ssb.get_table("14227", [
        GOODS, {"variableCode": "Alder", "valueCodes": ["*"]}, SPEND, LATEST]),
    "hh_spend_total_14100": lambda: ssb.get_table("14100", [GOODS, SPEND, LATEST]),
    "hh_kpi_14709": lambda: ssb.get_table("14709", [
        {"variableCode": "Maaned", "valueCodes": ["90"]},
        {"variableCode": "ContentsCode", "valueCodes": ["KpiIndMnd"]},
        {"variableCode": "Tid", "valueCodes": ["top(6)"]}]),
    "hh_cars_new_12906": lambda: ssb.get_table("12906", [
        KOMMUNE, {"variableCode": "TypeRegistrering", "valueCodes": ["N"]},
        {"variableCode": "EuroKlasser", "valueCodes": ["*"]},
        {"variableCode": "DrivstoffType", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Personbil1"]}, LATEST]),
    "hh_cars_fleet_13370": lambda: ssb.get_table("13370", [
        KOMMUNE, {"variableCode": "Eierform", "valueCodes": ["*"]},
        {"variableCode": "DrivstoffType", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Personbil1"]}, LATEST]),
}


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name, fn in JOBS.items():
        df = fn()
        df.to_parquet(config.RAW_DIR / f"{name}.parquet", index=False)
        period = sorted(df["Tid"].unique().tolist()) if "Tid" in df else None
        manifest[name] = {"rows": len(df), "period": period,
                          "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"{name}: {len(df):>7} rader  periode={period}")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
