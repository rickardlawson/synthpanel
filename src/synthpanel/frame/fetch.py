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


# --- L0b -------------------------------------------------------------------
EDU_ALL = ["00", "1-2", "3-5a", "11", "6", "7-8", "9"]
INNV = ["B", "C", "Rest"]


def fetch_edu_by_immigrant_age():
    """Utdanning × innvandringskategori × alder × kjønn (hele landet)."""
    return ssb.get_table("09599", [
        {"variableCode": "UtdanNivaa", "valueCodes": EDU_ALL},
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "InnvandrKat", "valueCodes": INNV},
        {"variableCode": "Alder", "valueCodes": ["16-19", "20-24", "25-29", "30-34", "35-39",
                                                  "40-49", "50-59", "60-66", "067+"]},
        {"variableCode": "ContentsCode", "valueCodes": ["InnvNo"]},
        LATEST,
    ])


def fetch_edu_by_immigrant_fylke():
    """Utdanning × innvandringskategori × kjønn per fylke."""
    return ssb.get_table("12934", [
        {"variableCode": "Region", "valueCodes": ["*"]},
        {"variableCode": "UtdanNivaa", "valueCodes": EDU_ALL},
        {"variableCode": "InnvandrKat", "valueCodes": INNV},
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "ContentsCode", "valueCodes": ["InnvNo"]},
        LATEST,
    ])


EDU4 = ["1-2", "3-5", "6-8", "0_9"]


def fetch_labour_status():
    """Prioritert arbeidsstyrkestatus × kjønn × alder × utdanning × innvandrer (hele landet).
    12424: 15–19, 20–24, 25–29 · 12425: 30–54, 55–61 · 12426: 62–66, 67+.
    Alle statuskoder hentes; byggesteget plukker ut de gjensidig utelukkende."""
    import pandas as pd

    common = [
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "UtdNivaa", "valueCodes": EDU4},
        {"variableCode": "InnvandrKat", "valueCodes": ["A_C-G", "B"]},
        {"variableCode": "HovArbStyrkStatus", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Bosatte"]},
        LATEST,
    ]
    parts = []
    for table, ages in [("12424", ["15-19", "20-24", "25-29"]),
                        ("12425", ["30-54", "55-61"]),
                        ("12426", ["62-66", "67+"])]:
        df = ssb.get_table(table, common + [{"variableCode": "Alder", "valueCodes": ages}])
        parts.append(df.assign(tabell=table))
    return pd.concat(parts, ignore_index=True)


def fetch_activity_by_fylke():
    """Andel i arbeid/utdanning/tiltak per fylke × kjønn × alder (13678)."""
    return ssb.get_table("13678", [
        {"variableCode": "Region", "valueCodes": ["*"]},
        {"variableCode": "HovArbStyrkStatus", "valueCodes": ["TOT", "A.01xU.01xU.03"]},
        {"variableCode": "Alder", "valueCodes": ["15-19", "20-24", "25-29", "30-39", "40-49", "50-61", "62+"]},
        {"variableCode": "UtdNivaa", "valueCodes": ["0"]},
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Bosatte"]},
        LATEST,
    ])


def fetch_household_persons():
    """Personer etter kjønn, alder og husholdningstype (hele landet, 06071)."""
    return ssb.get_table("06071", [
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "Alder", "valueCodes": ["00-15", "16-29", "30-66", "067+"]},
        {"variableCode": "HusholdType", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Personer"]},
        LATEST,
    ])


def fetch_family_kids_by_age():
    """Andel voksne i familier med små barn / store barn / uten barn, per kjønn og alder (12836)."""
    return ssb.get_table("12836", [
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "Alder", "valueCodes": ["25-44", "45-61"]},
        {"variableCode": "InnvandrKat", "valueCodes": ["A-G"]},
        {"variableCode": "AntHjemBarnU18", "valueCodes": ["TO"]},
        {"variableCode": "FamilieType", "valueCodes": ["2.1+2.3", "2.2+2.4", "3.1+3.2+3.3"]},
        {"variableCode": "HovArbStyrkStatus", "valueCodes": ["TOT"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Bosatte"]},
        LATEST,
    ])


def fetch_employment_by_age():
    """Sysselsatte i prosent av befolkningen per ettårig alder og kjønn, hele landet (06161)."""
    return ssb.get_table("06161", [
        {"variableCode": "Region", "valueCodes": ["0"]},
        {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
        {"variableCode": "Alder", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Sysselsatte"]},
        LATEST,
    ])


def fetch_low_income_groups():
    """Andel med lavinntekt (EU-skala 60 %) for ulike grupper, inkl. studenthusholdninger (12599)."""
    return ssb.get_table("12599", [
        {"variableCode": "Forbruksenhet", "valueCodes": ["*"]},
        {"variableCode": "Populasjon", "valueCodes": ["93a"]},
        {"variableCode": "HovedInntYrkesinn", "valueCodes": ["Total"]},
        {"variableCode": "ContentsCode", "valueCodes": ["EUskala60", "AntPersoner"]},
        LATEST,
    ])


def fetch_low_income_education():
    """Vedvarende lavinntekt (EU-60) for 18–66 år etter utdanningsnivå, treårsperiode (09570)."""
    return ssb.get_table("09570", [
        {"variableCode": "UtdNivaa", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["EUskalaSeksti"]},
        LATEST,
    ])


FYLKER_2024 = ["0", "03", "11", "15", "18", "31", "32", "33", "34", "39", "40", "42", "46", "50", "55", "56"]


def fetch_housing_by_household():
    """Husholdninger etter eierstatus × husholdningstype × bygningstype per fylke (14901)."""
    return ssb.get_table("14901", [
        {"variableCode": "Region", "valueCodes": FYLKER_2024},
        {"variableCode": "EierStatus", "valueCodes": ["1", "2", "3"]},
        {"variableCode": "HusholdType", "valueCodes": ["*"]},
        {"variableCode": "BygnType", "valueCodes": ["11", "12", "13", "14b", "19"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Husholdning"]},
        LATEST,
    ])


def fetch_housing_by_income():
    """Eierstatus (14900) og bygningstype (14921) etter inntektskvartil per fylke."""
    import pandas as pd

    groups = ["0", "41", "42", "43", "44"]
    own = ssb.get_table("14900", [
        {"variableCode": "Region", "valueCodes": FYLKER_2024},
        {"variableCode": "EierStatus", "valueCodes": ["1", "2", "3"]},
        {"variableCode": "Inntekstgruppe", "valueCodes": groups},
        {"variableCode": "ContentsCode", "valueCodes": ["Husholdning"]},
        LATEST,
    ]).assign(dim="eierstatus").rename(columns={"EierStatus": "kode"})
    bld = ssb.get_table("14921", [
        {"variableCode": "Region", "valueCodes": FYLKER_2024},
        {"variableCode": "BygnType", "valueCodes": ["11", "12", "13", "14b", "19"]},
        {"variableCode": "Inntekstgruppe", "valueCodes": groups},
        {"variableCode": "ContentsCode", "valueCodes": ["Husholdning"]},
        LATEST,
    ]).assign(dim="boligtype").rename(columns={"BygnType": "kode"})
    cols = ["Region", "Inntekstgruppe", "dim", "kode", "value", "Tid"]
    return pd.concat([own[cols], bld[cols]], ignore_index=True)


def fetch_household_fylke():
    """Personer i privathusholdninger etter husholdningstype per fylke (10986)."""
    return ssb.get_table("10986", [
        {"variableCode": "Region", "valueCodes": ["*"]},
        {"variableCode": "HushType", "valueCodes": ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7",
                                                     "2.1", "2.2", "2.3"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Personer"]},
        LATEST,
    ])


def fetch_income_deciles():
    """Husholdninger etter inntekt etter skatt i nasjonale desiler, per husholdningstype og fylke (12563)."""
    return ssb.get_table("12563", [
        {"variableCode": "Region", "valueCodes": ["*"]},
        {"variableCode": "InntektSkatt", "valueCodes": ["00S"]},
        {"variableCode": "Desiler", "valueCodes": ["*"]},
        {"variableCode": "HushType", "valueCodes": ["*"]},
        {"variableCode": "ContentsCode", "valueCodes": ["Prosent"]},
        LATEST,
    ])


JOBS = {
    "population_07459": fetch_population,
    "education_08921": fetch_education,
    "background_07111": fetch_background,
    "centrality_klass128": fetch_centrality,
    "edu_immigrant_age_09599": fetch_edu_by_immigrant_age,
    "edu_immigrant_fylke_12934": fetch_edu_by_immigrant_fylke,
    "labour_status_1242x": fetch_labour_status,
    "activity_fylke_13678": fetch_activity_by_fylke,
    "household_persons_06071": fetch_household_persons,
    "household_fylke_10986": fetch_household_fylke,
    "family_kids_age_12836": fetch_family_kids_by_age,
    "employment_age_06161": fetch_employment_by_age,
    "low_income_groups_12599": fetch_low_income_groups,
    "low_income_education_09570": fetch_low_income_education,
    "income_deciles_12563": fetch_income_deciles,
    "housing_household_14901": fetch_housing_by_household,
    "housing_income_14900_14921": fetch_housing_by_income,
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
