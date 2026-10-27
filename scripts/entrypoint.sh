#!/bin/sh
# Bygger populasjonen ved første oppstart (data ligger i et volum og overlever omstart).
#
# Hvert hentesteg merkes som ferdig (raw/.ok_<steg>), så en omstart henter ikke alt
# på nytt. Feiler en ekstern kilde, fortsetter oppstarten uten det laget – bare
# befolkningsrammen (SSB) er påkrevd.
RAW="$SYNTHPANEL_DATA_DIR/raw"
AGENTS="$SYNTHPANEL_DATA_DIR/processed/agents.parquet"
mkdir -p "$RAW"
CHANGED=0

step() {  # step <navn> <påkrevd:ja|nei> <kommando...>
  name=$1; required=$2; shift 2
  [ -f "$RAW/.ok_$name" ] && return 0
  echo "Henter: $name"
  if "$@"; then
    touch "$RAW/.ok_$name"; CHANGED=1
  elif [ "$required" = "ja" ]; then
    echo "FEIL: $name feilet og er påkrevd – avbryter"; exit 1
  else
    echo "ADVARSEL: $name feilet – fortsetter uten (prøves igjen ved neste oppstart)"
  fi
}

step ramme      ja  python -m synthpanel.frame.fetch
step politikk   nei python -m synthpanel.politics.fetch
step medier     nei python -m synthpanel.media.layer
step fritid     nei python -m synthpanel.leisure.layer
step husholdning nei python -m synthpanel.household.fetch
if [ -n "$ESS_USER_ID" ]; then
  step ess nei python -m synthpanel.values.ess
else
  echo "ESS_USER_ID ikke satt – bygger uten verdilag"
fi
step geo        nei python -m synthpanel.geo.fetch
step navn       nei python -m synthpanel.personas.names

if [ ! -f "$AGENTS" ] || [ "$CHANGED" = "1" ]; then
  echo "Bygger populasjonen..."
  python -m synthpanel.frame.build || { echo "FEIL: byggingen feilet"; exit 1; }
fi

exec uvicorn synthpanel.api.main:app --host 0.0.0.0 --port "${PORT:-8090}"
