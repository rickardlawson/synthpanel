#!/bin/sh
# Bygger populasjonen ved første oppstart (data ligger i et volum og overlever omstart).
set -e
if [ ! -f "$SYNTHPANEL_DATA_DIR/processed/agents.parquet" ]; then
  echo "Ingen populasjon funnet – henter fra SSB og bygger (tar under ett minutt)..."
  python -m synthpanel.frame.fetch
  python -m synthpanel.politics.fetch
  python -m synthpanel.media.layer
  python -m synthpanel.leisure.layer
  if [ -n "$ESS_USER_ID" ]; then python -m synthpanel.values.ess; else echo "ESS_USER_ID ikke satt – bygger uten verdilag"; fi
  python -m synthpanel.frame.build
fi
if [ ! -f "$SYNTHPANEL_DATA_DIR/raw/names_first_10467.parquet" ]; then
  python -m synthpanel.personas.names || echo "Fikk ikke hentet navnestatistikk – personas bruker reserveliste"
fi
exec uvicorn synthpanel.api.main:app --host 0.0.0.0 --port "${PORT:-8090}"
