.PHONY: install fetch build data test serve up down rebuild-data

install:
	pip install -e ".[dev]"

fetch:          ## Hent rådata fra SSB
	python -m synthpanel.frame.fetch

build:          ## Bygg syntetisk populasjon fra rådata
	python -m synthpanel.frame.build

data: fetch build

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
