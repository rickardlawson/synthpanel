"""Minimal klient for SSB PxWebApi v2 og Klass.

Returnerer alltid "tidy" pandas-tabeller: én kolonne per dimensjon (kode),
én `<dim>_tekst`-kolonne per dimensjon, og `value`.
"""
from __future__ import annotations

import itertools
import time
from typing import Any

import httpx
import pandas as pd

PXWEB = "https://data.ssb.no/api/pxwebapi/v2"
KLASS = "https://data.ssb.no/api/klass/v1"

_client = httpx.Client(timeout=60.0, headers={"User-Agent": "synthpanel/0.1"})


def _request(method: str, url: str, **kw: Any) -> httpx.Response:
    # SSB tillater ca. 30 kall per minutt; prøv igjen ved 429/5xx.
    for attempt in range(5):
        r = _client.request(method, url, **kw)
        if r.status_code < 400:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
    r.raise_for_status()
    return r


def jsonstat2_to_frame(d: dict) -> pd.DataFrame:
    """Gjør om et JSON-stat2-svar til en tidy DataFrame."""
    ids: list[str] = d["id"]
    dims = d["dimension"]
    codes_per_dim = []
    for dim in ids:
        cat = dims[dim]["category"]
        index = cat["index"]
        if isinstance(index, dict):
            ordered = sorted(index, key=index.get)
        else:
            ordered = list(index)
        codes_per_dim.append(ordered)

    values = d["value"]
    if isinstance(values, dict):  # sparse format
        n = 1
        for s in d["size"]:
            n *= s
        dense = [None] * n
        for k, v in values.items():
            dense[int(k)] = v
        values = dense

    rows = list(itertools.product(*codes_per_dim))
    df = pd.DataFrame(rows, columns=ids)
    df["value"] = values
    for dim in ids:
        labels = dims[dim]["category"].get("label", {})
        df[f"{dim}_tekst"] = df[dim].map(labels)
    return df


def get_table(table_id: str, selection: list[dict], lang: str = "no") -> pd.DataFrame:
    """Hent data fra en SSB-tabell.

    selection: [{"variableCode": "Region", "valueCodes": ["*"], "codelist": "vs_Kommun"}, ...]
    """
    url = f"{PXWEB}/tables/{table_id}/data"
    r = _request(
        "POST",
        url,
        params={"lang": lang, "outputFormat": "json-stat2"},
        json={"selection": selection},
    )
    return jsonstat2_to_frame(r.json())


def klass_correspondence(source_id: int, target_id: int, date: str) -> pd.DataFrame:
    url = f"{KLASS}/classifications/{source_id}/correspondsAt"
    r = _request(
        "GET",
        url,
        params={"targetClassificationId": target_id, "date": date},
        headers={"Accept": "application/json"},
    )
    items = r.json()["correspondenceItems"]
    return pd.DataFrame(items)


def klass_codes(classification_id: int, date: str) -> pd.DataFrame:
    url = f"{KLASS}/classifications/{classification_id}/codesAt"
    r = _request("GET", url, params={"date": date}, headers={"Accept": "application/json"})
    return pd.DataFrame(r.json()["codes"])
