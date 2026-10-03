"""Hent rådata for befolkningsrammen (L0) fra SSB og lagre som Parquet.

Kjør:  python -m synthpanel.frame.fetch
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from synthpanel import config, ssb

LATEST = {"variableCode": "Tid", "valueCodes": ["top(1)"]}


def fetch_population():
    return ssb.get_table(
        "07459",
        [
            {"variableCode": "Region", "codelist": "vs_Kommun", "valueCodes": ["*"]},
            {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
            {"variableCode": "Alder", "codelist": "vs_AlleAldre00B", "valueCodes": ["*"]},
            {"variableCode": "ContentsCode", "valueCodes": ["Personer1"]},
            LATEST,
        ],
    )


def fetch_education():
    return ssb.get_table(
        "08921",
        [
            {"variableCode": "Region", "valueCodes": ["*"]},
            {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
            {"variableCode": "Alder", "valueCodes": ["16-19", "20-24", "25-29", "30-39", "40-49", "50-59", "60-66", "067+"]},
            {"variableCode": "UtdanNivaa", "valueCodes": ["01", "02a", "11", "03a", "04a", "09a"]},
            {"variableCode": "ContentsCode", "valueCodes": ["Personer"]},
            LATEST,
        ],
    )


def fetch_background():
    return ssb.get_table(
        "07111",
        [
            {"variableCode": "Region", "valueCodes": ["*"]},
            {"variableCode": "Alder", "valueCodes": ["*"]},
            {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
            {"variableCode": "Landbakgrunn", "valueCodes": ["*"]},
            {"variableCode": "ContentsCode", "valueCodes": ["Personer"]},
            LATEST,
        ],
    )


def fetch_centrality():
    cfg = config.load("frame")["sources"]["centrality"]
    df = ssb.klass_correspondence(cfg["klass_source"], cfg["klass_target"], cfg["date"])
    return df.rename(
        columns={
            "sourceCode": "kommune",
            "sourceName": "kommune_navn",
            "targetCode": "sentralitet",
            "targetName": "sentralitet_tekst",
        }
    )[["kommune", "kommune_navn", "sentralitet", "sentralitet_tekst"]]


JOBS = {
    "population_07459": fetch_population,
    "education_08921": fetch_education,
    "background_07111": fetch_background,
    "centrality_klass128": fetch_centrality,
}


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, fn in JOBS.items():
        df = fn()
        path = config.RAW_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        period = sorted(df["Tid"].unique().tolist()) if "Tid" in df else None
        manifest[name] = {
            "rows": len(df),
            "period": period,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"{name}: {len(df):>7} rader  periode={period}")
    with open(config.RAW_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
