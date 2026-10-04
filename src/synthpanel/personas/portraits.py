"""Portrettbiblioteket: AI-genererte ansikter per kjønn × aldersgruppe × landbakgrunn.

Bildene lages av scripts/generate_portraits.py og ligger i web/portraits/.
Ingen av dem forestiller virkelige personer.
"""
from __future__ import annotations

import zlib
from functools import lru_cache
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "web" / "portraits"

# Aldersgrupper for portretter (alder i prompten = midtpunkt)
AGES = {"18-24": 21, "25-34": 30, "35-44": 40, "45-54": 50, "55-66": 61, "67-79": 73, "80+": 84}

# Antall varianter per kjønn × aldersgruppe og hvilke opphav som brukes i prompt (og navn).
BACKGROUNDS = {
    "norsk": (10, ["Norwegian"]),
    "europa": (3, ["Polish", "Lithuanian", "Swedish", "German", "Bosnian"]),
    "asia": (4, ["Pakistani", "Iraqi", "Filipino", "Vietnamese", "Afghan", "Syrian", "Thai", "Indian"]),
    "afrika": (3, ["Somali", "Eritrean", "Ethiopian", "Nigerian"]),
    "annet": (2, ["Chilean", "Colombian", "Brazilian", "American"]),
}


def age_band(age: int) -> str:
    for band in AGES:
        lo, _, hi = band.replace("+", "-200").partition("-")
        if int(lo) <= age <= int(hi):
            return band
    return "80+"


def nationality(bg: str, band: str, variant: int) -> str:
    nats = BACKGROUNDS[bg][1]
    return nats[(variant + list(AGES).index(band)) % len(nats)]


@lru_cache(maxsize=1)
def _available_cached(stamp: float) -> dict[tuple[str, str, str], list[int]]:
    out: dict[tuple[str, str, str], list[int]] = {}
    for p in DIR.glob("*.webp"):
        bg, sex, band, v = p.stem.split("_")
        out.setdefault((bg, sex, band), []).append(int(v))
    return {k: sorted(v) for k, v in out.items()}


def available() -> dict[tuple[str, str, str], list[int]]:
    """Hvilke portretter finnes (leses på nytt når mappen endres)."""
    stamp = DIR.stat().st_mtime if DIR.exists() else 0.0
    return _available_cached(stamp)


def pick(bg: str, sex: str, age: int, key: str, taken: set[str]) -> dict | None:
    """Velg et portrett som passer kjønn, alder og bakgrunn, og som ikke er brukt i galleriet.
    Faller tilbake til nabo-aldersgruppen (og til slutt norsk) hvis cellen mangler bilder."""
    have = available()
    bands = list(AGES)
    want = age_band(age)
    # Bare samme eller nabo-aldersgruppe – heller ingen portrett enn et ansikt med feil alder.
    order = [b for b in sorted(bands, key=lambda b: abs(bands.index(b) - bands.index(want)))
             if abs(bands.index(b) - bands.index(want)) <= 1]
    for bg_try in ([bg, "norsk"] if bg != "norsk" else ["norsk"]):
        bg_try = bg_try if bg_try in BACKGROUNDS else "norsk"
        for band in order:
            variants = have.get((bg_try, sex, band), [])
            if not variants:
                continue
            start = zlib.crc32(key.encode()) % len(variants)
            for i in range(len(variants)):
                v = variants[(start + i) % len(variants)]
                f = f"{bg_try}_{sex}_{band}_{v}.webp"
                if f not in taken:
                    taken.add(f)
                    return {"fil": f"/portraits/{f}", "opphav": nationality(bg_try, band, v), "bakgrunn": bg_try}
    return None
