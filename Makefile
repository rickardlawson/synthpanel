.PHONY: install fetch fetch-ess build data test validate serve up down rebuild-data

install:
	pip install -e ".[dev]"

fetch:          ## Hent rådata fra SSB
	python -m synthpanel.frame.fetch

build:          ## Bygg syntetisk populasjon fra rådata
	python -m synthpanel.frame.build

fetch-ess:      ## Hent ESS (verdier) – krever ESS_USER_ID, f.eks. i .env
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	if [ -n "$$ESS_USER_ID" ]; then python -m synthpanel.values.ess; else echo "Hopper over ESS: ESS_USER_ID er ikke satt (se README)"; fi

data: fetch fetch-ess build

validate:       ## Testsett: mål verdilaget mot holdte ESS-respondenter
	python -m synthpanel.values.validate

test:
	pytest -q

serve:          ## Kjør API lokalt på http://localhost:8090/docs
	uvicorn synthpanel.api.main:app --reload --port 8090

up:             ## Docker: bygg og start
	docker compose up -d --build

down:
	docker compose down

rebuild-data:   ## Docker: hent ferske SSB-tall og bygg på nytt
	docker compose exec synthpanel sh -c "python -m synthpanel.frame.fetch && python -m synthpanel.frame.build"
	docker compose restart synthpanel
