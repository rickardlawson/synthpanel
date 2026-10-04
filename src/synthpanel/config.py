from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Les KEY=verdi fra .env (uten å overstyre miljøet). Nøkler holdes utenfor git."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")
CONFIG_DIR = Path(os.environ.get("SYNTHPANEL_CONFIG_DIR", ROOT / "configs"))
DATA_DIR = Path(os.environ.get("SYNTHPANEL_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


@lru_cache
def load(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def age_band_label(lo: int, hi: int) -> str:
    return f"{lo}+" if hi >= 120 else f"{lo}-{hi}"
