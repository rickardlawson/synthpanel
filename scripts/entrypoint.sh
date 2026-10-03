#!/bin/sh
# Bygger populasjonen ved første oppstart (data ligger i et volum og overlever omstart).
set -e
if [ ! -f "$SYNTHPANEL_DATA_DIR/processed/agents.parquet" ]; then
  echo "Ingen populasjon funnet – henter fra SSB og bygger (tar under ett minutt)..."
  python -m synthpanel.frame.fetch
  python -m synthpanel.frame.build
fi
exec uvicorn synthpanel.api.main:app --host 0.0.0.0 --port "${PORT:-8090}"
