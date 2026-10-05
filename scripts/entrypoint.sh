#!/bin/sh
# Bygger populasjonen ved første oppstart (data ligger i et volum og overlever omstart).
set -e
if [ ! -f "$SYNTHPANEL_DATA_DIR/processed/agents.parquet" ]; then
  echo "Ingen populasjon funnet – henter fra SSB og bygger (tar under ett minutt)..."
  python -m synthpanel.frame.fetch
  python -m synthpanel.politics.fetch
  python -m synthpanel.media.layer
  python -m synthpanel.leisure.layer
  python -m synthpanel.household.fetch
  if [ -n "$ESS_USER_ID" ]; then python -m synthpanel.values.ess; else echo "ESS_USER_ID ikke satt – bygger uten verdilag"; fi
  python -m synthpanel.frame.build
fi
# Husholdningslaget (inntekt i kr, forbruk, kjøp) kom etter første versjon: hent og bygg på nytt ved behov.
if [ ! -f "$SYNTHPANEL_DATA_DIR/raw/hh_income_deciles_12558.parquet" ]; then
  echo "Henter husholdningsdata fra SSB og bygger populasjonen på nytt..."
  python -m synthpanel.household.fetch && python -m synthpanel.frame.build
fi
# Geodata til kartet (kommunegrenser og 1 km-rutenett, ~2 min første gang)
if [ ! -f "$SYNTHPANEL_DATA_DIR/raw/geo_grid_1km.parquet" ]; then
  python -m synthpanel.geo.fetch || echo "Fikk ikke hentet geodata – kartfanen virker ikke før geodata er hentet"
fi
if [ ! -f "$SYNTHPANEL_DATA_DIR/raw/names_first_10467.parquet" ]; then
  python -m synthpanel.personas.names || echo "Fikk ikke hentet navnestatistikk – personas bruker reserveliste"
fi
exec uvicorn synthpanel.api.main:app --host 0.0.0.0 --port "${PORT:-8090}"
