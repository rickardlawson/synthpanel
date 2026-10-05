"""Geodata til kartet: kommune- og fylkesgrenser og befolkning på 1 km-rutenett.

Kilder
- Kommune- og fylkesgrenser 2024 (Kartverket, forenklet av github.com/robhop/fylker-og-kommuner,
  CC BY 4.0). Kommunenumrene er de samme som i SSB-tabellene fra 2024.
- Befolkning på rutenett 1 km, 1.1.2026 (SSB, OGC API Features på kart.ssb.no):
  antall bosatte, kvinner, menn og snittalder per rute.

Hver rute får kommunen midtpunktet ligger i (punkt-i-polygon). Ruter som havner
utenfor de forenklede grensene (kyst, øyer), får nærmeste kommune.

Lagres i data/raw:
  geo_kommuner.geojson   polygoner med kommune, navn og areal (km²)
  geo_fylker.geojson     fylkesgrenser (bare til visning)
  geo_grid_1km.parquet   lon, lat, bosatte, snittalder, kommune

Kjør:  python -m synthpanel.geo.fetch
"""
from __future__ import annotations

import json

import httpx
import numpy as np
import pandas as pd

from synthpanel import config

RAW = config.RAW_DIR
BORDERS = "https://raw.githubusercontent.com/robhop/fylker-og-kommuner/main/{name}"
GRID = ("https://kart.ssb.no/api/ogc/v1/befolkning_paa_rutenett_ogc_api/collections/"
        "069ba866-936d-7c74-8000-2a979362a039/items")   # «Befolkning 1km 2026»
GRID_YEAR = 2026
PAGE = 1000


def _get(url: str, headers: dict | None = None, **kw) -> httpx.Response:
    headers = {"User-Agent": "synthpanel/0.1", **(headers or {})}
    for attempt in range(5):
        r = httpx.get(url, timeout=120, follow_redirects=True, headers=headers, **kw)
        if r.status_code < 400:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            import time
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
    r.raise_for_status()
    return r


def fetch_borders() -> None:
    from pyproj import Geod
    from shapely.geometry import shape
    geod = Geod(ellps="GRS80")
    k = _get(BORDERS.format(name="Kommuner-S.geojson")).json()
    for f in k["features"]:
        p = f["properties"]
        area = abs(geod.geometry_area_perimeter(shape(f["geometry"]))[0]) / 1e6
        f["properties"] = {"kommune": p["kommunenummer"], "navn": p["kommunenavn"], "km2": round(area, 1)}
        f["geometry"] = _round(f["geometry"])
    (RAW / "geo_kommuner.geojson").write_text(json.dumps(k, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    fy = _get(BORDERS.format(name="Fylker-S.geojson")).json()
    for f in fy["features"]:
        p = f["properties"]
        f["properties"] = {"fylke": p.get("fylkesnummer") or p.get("id"), "navn": p.get("fylkesnavn") or p.get("name")}
        f["geometry"] = _round(f["geometry"])
    (RAW / "geo_fylker.geojson").write_text(json.dumps(fy, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Grenser: {len(k['features'])} kommuner, {len(fy['features'])} fylker")


def _round(geom: dict, nd: int = 4) -> dict:
    def r(c):
        return [r(x) for x in c] if isinstance(c[0], list) else [round(c[0], nd), round(c[1], nd)]
    return {"type": geom["type"], "coordinates": r(geom["coordinates"])}


def fetch_grid() -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        d = _get(GRID, params={"limit": PAGE, "offset": offset}, headers={"Accept": "application/geo+json"}).json()
        feats = d.get("features", [])
        for f in feats:
            p = f["properties"]
            rows.append((int(p["ssbid1000m"]), p.get("pop_tot") or 0, p.get("pop_ave")))
        offset += len(feats)
        if len(feats) < PAGE:
            break
    g = pd.DataFrame(rows, columns=["ssbid", "bosatte", "snittalder"]).drop_duplicates("ssbid")
    # SSB-id = (x + 2 000 000) * 10^7 + y for nedre venstre hjørne i UTM 33 (EPSG:25833).
    x = g["ssbid"] // 10**7 - 2_000_000 + 500
    y = g["ssbid"] % 10**7 + 500
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(25833, 4326, always_xy=True).transform(x.to_numpy(), y.to_numpy())
    g["lon"], g["lat"] = np.round(lon, 4), np.round(lat, 4)
    return g


def assign_kommune(g: pd.DataFrame) -> pd.Series:
    from shapely import STRtree, points
    from shapely.geometry import shape
    k = json.loads((RAW / "geo_kommuner.geojson").read_text(encoding="utf-8"))
    geoms = [shape(f["geometry"]) for f in k["features"]]
    codes = np.array([f["properties"]["kommune"] for f in k["features"]])
    tree = STRtree(geoms)
    pts = points(g["lon"].to_numpy(), g["lat"].to_numpy())
    pi, gi = tree.query(pts, predicate="within")
    out = np.full(len(g), None, dtype=object)
    out[pi] = codes[gi]
    miss = np.flatnonzero(pd.isna(out))
    if len(miss):
        out[miss] = codes[tree.nearest(pts[miss])]
    return pd.Series(out, index=g.index)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    fetch_borders()
    g = fetch_grid()
    g["kommune"] = assign_kommune(g)
    g.to_parquet(RAW / "geo_grid_1km.parquet", index=False)
    print(f"Rutenett {GRID_YEAR}: {len(g)} ruter, {int(g['bosatte'].sum()):,} bosatte".replace(",", " "))


if __name__ == "__main__":
    main()
