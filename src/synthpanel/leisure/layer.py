"""Fritidslag: trening og friluftsliv siste 12 måneder (SSB, levekårsundersøkelsen
om idrett og friluftsliv, 2024).

Per aktivitet:
1. Grunnsannsynlighet fra kjønn × alder (13388 trening, 13372 friluftsliv).
2. Justering i log-odds for utdanning (13392 / 13376) og sentralitet (13389 / 13373),
   dempet med 0,6 fordi disse delvis forklares av alder, og begrenset til ±1 log-odds
   (smågrupper i utvalget gir ellers ekstreme utslag).
3. Forskyvning per kjønn × alder slik at andelene treffer SSB-tallene igjen.
4. Aktivitetene trekkes med en felles latent «aktivitetsfaktor» per agent
   (gaussisk kopula, rho = 0,45): den som løper, styrketrener oftere også, og
   den som jakter, fisker oftere. Treningsfrekvensen (13396) trekkes fra den
   samme faktoren, så de som trener hver uke er de som driver med flest ting.

Svakhet: sammenhengen mellom aktivitetene er satt skjønnsmessig (rho), ikke
estimert fra mikrodata.

Kjør henting:  python -m synthpanel.leisure.layer
"""
from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pandas as pd

from synthpanel import config, ssb

RAW = config.RAW_DIR
YEAR = {"variableCode": "Tid", "valueCodes": ["top(1)"]}
PCT = {"variableCode": "ContentsCode", "valueCodes": ["PersonerProsent"]}

# kolonne -> (kode, visningsnavn)
TRAINING = {"trening_lop": ("01", "Løping"), "trening_styrke": ("04", "Styrketrening"),
            "trening_langrenn": ("02", "Langrenn"), "trening_sykkel": ("03", "Sykling"),
            "trening_svom": ("05", "Svømming"), "trening_yoga": ("14", "Yoga/pilates"),
            "trening_fotball": ("06", "Fotball"), "trening_golf": ("12", "Golf")}
OUTDOOR = {"friluft_fottur": ("02.1", "Lang fottur (3 t+)"), "friluft_skitur": ("04", "Skitur"),
           "friluft_alpint": ("05", "Alpint/snowboard"), "friluft_fiske": ("06", "Fisking"),
           "friluft_jakt": ("11", "Jakt"), "friluft_baer": ("12", "Bær- og sopptur"),
           "friluft_overnatting": ("14", "Overnatting ute"), "friluft_bat": ("07", "Båttur")}
COLUMNS = {**{k: v[1] for k, v in TRAINING.items()}, **{k: v[1] for k, v in OUTDOOR.items()}}
FREQ_COL = "treningsfrekvens"   # ukentlig / av_og_til / sjelden

BANDS = [(16, 24, "16-24"), (25, 44, "25-44"), (45, 66, "45-66"), (67, 200, "067+")]
EDU = {"grunnskole": "1-2", "videregaende": "3-4-5", "fagskole": "3-4-5", "uh_kort": "6", "uh_lang": "7-8", "uoppgitt": "1-2"}
RHO = 0.45
DAMP = 0.6

JOBS = {
    "trening_kjonn_alder_13388": ("13388", "TreningsAkt", [{"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
                                                          {"variableCode": "Alder", "valueCodes": ["16-24", "25-44", "45-66", "067+"]}]),
    "trening_sentralitet_13389": ("13389", "TreningsAkt", [{"variableCode": "SentralitetKomm", "valueCodes": ["*"]}]),
    "trening_utdanning_13392": ("13392", "TreningsAkt", [{"variableCode": "UtdNivaa", "valueCodes": ["*"]}]),
    "friluft_kjonn_alder_13372": ("13372", "Friluftsaktiv", [{"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
                                                            {"variableCode": "Alder", "valueCodes": ["16-24", "25-44", "45-66", "067+"]}]),
    "friluft_sentralitet_13373": ("13373", "Friluftsaktiv", [{"variableCode": "SentralitetKomm", "valueCodes": ["*"]}]),
    "friluft_utdanning_13376": ("13376", "Friluftsaktiv", [{"variableCode": "UtdNivaa", "valueCodes": ["*"]}]),
}


def fetch() -> None:
    for name, (table, var, extra) in JOBS.items():
        d = ssb.get_table(table, [{"variableCode": var, "valueCodes": ["*"]}, *extra, PCT, YEAR])
        d.to_parquet(RAW / f"leisure_{name}.parquet", index=False)
        print(f"{table}: {len(d)} rader, år {sorted(d['Tid'].unique())}")
    f = ssb.get_table("13396", [{"variableCode": "TreningsAkt", "valueCodes": ["*"]},
                                {"variableCode": "Kjonn", "valueCodes": ["1", "2"]},
                                {"variableCode": "Alder", "valueCodes": ["16-24", "25-44", "45-66", "067+"]},
                                {"variableCode": "TreningHvorOfte", "valueCodes": ["*"]}, PCT, YEAR])
    f.to_parquet(RAW / "leisure_frekvens_13396.parquet", index=False)
    print(f"13396: {len(f)} rader")


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(RAW / f"leisure_{name}.parquet")


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _expit(z):
    return 1 / (1 + np.exp(-z))


_ncdf = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))


def _band(age: pd.Series) -> pd.Series:
    out = pd.Series("067+", index=age.index, dtype=object)
    for lo, hi, lab in BANDS:
        out[age.between(lo, hi)] = lab
    return out


def probabilities(a: pd.DataFrame) -> dict[str, np.ndarray]:
    """Sannsynlighet per agent og aktivitet, kalibrert til kjønn × alder."""
    band = _band(a["alder"])
    sex = a["kjonn"].map({"mann": "1", "kvinne": "2"})
    sent = "Se" + a["sentralitet"]            # 13389/13373 bruker Se01–Se06
    edu = a["utdanning"].map(EDU)
    w = a["vekt"].to_numpy()
    out = {}
    for group, var in (("trening", "TreningsAkt"), ("friluft", "Friluftsaktiv")):
        acts = TRAINING if group == "trening" else OUTDOOR
        ka = _read(f"{group}_kjonn_alder_{'13388' if group == 'trening' else '13372'}")
        se = _read(f"{group}_sentralitet_{'13389' if group == 'trening' else '13373'}")
        ed = _read(f"{group}_utdanning_{'13392' if group == 'trening' else '13376'}")
        for col, (code, _) in acts.items():
            base = ka[ka[var] == code].set_index(["Kjonn", "Alder"])["value"] / 100
            p0 = base.reindex(pd.MultiIndex.from_arrays([sex, band])).to_numpy()
            s = se[se[var] == code].set_index("SentralitetKomm")["value"] / 100
            e = ed[ed[var] == code].set_index("UtdNivaa")["value"] / 100
            tot = e.get("0", e.mean())  # «alle utdanningsnivå»
            s, e = s.where(s > 0), e.where(e > 0)   # 0 = for få svar (prikket), ikke «ingen»
            ls = np.clip(_logit(s.reindex(sent).to_numpy()) - _logit(tot), -1.0, 1.0)
            le = np.clip(_logit(e.reindex(edu).to_numpy()) - _logit(tot), -1.0, 1.0)
            z = _logit(p0) + DAMP * np.nan_to_num(ls) + DAMP * np.nan_to_num(le)
            # Forskyv per kjønn × alder så andelen treffer SSB igjen.
            for key, idx in pd.Series(range(len(a))).groupby([sex.to_numpy(), band.to_numpy()]).groups.items():
                ix = np.asarray(list(idx))
                target = float(base.get(key, np.nan))
                if np.isnan(target):
                    continue
                lo, hi = -6.0, 6.0
                for _ in range(50):
                    c = (lo + hi) / 2
                    if np.average(_expit(z[ix] + c), weights=w[ix]) < target:
                        lo = c
                    else:
                        hi = c
                z[ix] += (lo + hi) / 2
            out[col] = _expit(z)
    return out


def frequency_shares(a: pd.DataFrame) -> np.ndarray:
    """Andel per agent som trener (0) ukentlig og (1) minst månedlig, fra 13396 («Har trent eller mosjonert»)."""
    f = _read("frekvens_13396")
    f = f[f["TreningsAkt"] == "0B"]
    piv = f.pivot_table(index=["Kjonn", "Alder"], columns="TreningHvorOfte", values="value") / 100
    weekly = piv["04"] + piv["05"]
    monthly = weekly + piv["03"]
    key = pd.MultiIndex.from_arrays([a["kjonn"].map({"mann": "1", "kvinne": "2"}), _band(a["alder"])])
    return np.vstack([weekly.reindex(key).to_numpy(), monthly.reindex(key).to_numpy()]).T


def attach(a: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if not (RAW / "leisure_trening_kjonn_alder_13388.parquet").exists():
        return a
    probs = probabilities(a)
    n = len(a)
    z_train, z_out = rng.standard_normal(n), rng.standard_normal(n)
    out = a.copy()
    for col, p in probs.items():
        zf = z_train if col.startswith("trening_") else z_out
        x = RHO * zf + np.sqrt(1 - RHO**2) * rng.standard_normal(n)   # ~ N(0,1)
        out[col] = np.where(_ncdf(x) > 1 - p, "ja", "nei")             # høy faktor -> mer aktiv
    # Treningsfrekvens fra samme faktor (rangert): høy faktor -> ukentlig.
    fs = frequency_shares(a)
    u = _ncdf(z_train)
    out[FREQ_COL] = np.where(u > 1 - fs[:, 0], "ukentlig", np.where(u > 1 - fs[:, 1], "av_og_til", "sjelden"))
    return out


if __name__ == "__main__":
    fetch()
