# Laboratorio 03 - Reglas de asociacion
#
# Los objetivos corren tanto dentro del contenedor (donde estan las dependencias)
# como desde el host via `make d-<objetivo>`, que los delega a docker compose.

COMPOSE := docker compose -f docker/docker-compose.yml
PY      := python

.PHONY: help build up jupyter test parse etl etl-force eda dow rules bench \
         notebook report-tables report all clean d-test d-parse d-etl d-etl-force d-eda d-dow \
         d-rules d-bench d-notebook d-report d-all

help:
	@echo "Dentro del contenedor:  make test | parse | etl | eda | dow | rules | bench | notebook | report | all"
	@echo "Desde el host (macOS):  make build | up | jupyter | d-test | d-etl | d-etl-force | d-eda | d-dow | d-rules | d-bench | d-notebook | d-report | d-all"

# ----------------------------------------------------------------- host / docker

build:
	$(COMPOSE) build shell

up:
	$(COMPOSE) run --rm shell bash

jupyter:
	@echo "JupyterLab en http://localhost:8888  (Ctrl-C para parar)"
	$(COMPOSE) up jupyter

d-test:     ; $(COMPOSE) run --rm shell make test
d-parse:    ; $(COMPOSE) run --rm shell make parse
d-etl:      ; $(COMPOSE) run --rm shell make etl
d-etl-force:; $(COMPOSE) run --rm shell make etl-force
d-eda:      ; $(COMPOSE) run --rm shell make eda
d-dow:      ; $(COMPOSE) run --rm shell make dow
d-rules:    ; $(COMPOSE) run --rm shell make rules
d-bench:    ; $(COMPOSE) run --rm bench
d-notebook: ; $(COMPOSE) run --rm shell make notebook
d-report:   ; $(COMPOSE) run --rm shell make report
d-all:      ; $(COMPOSE) run --rm shell make all

# -------------------------------------------------------------- dentro del lab

test:
	$(PY) -m unittest discover -s tests -v

parse:
	$(PY) -m src.parse

etl:
	$(PY) -m src.etl.pipeline

etl-force:
	$(PY) -m src.etl.pipeline --force

eda:
	$(PY) -m src.eda

dow:
	$(PY) -m src.dow

rules:
	$(PY) -m src.rules

bench:
	$(PY) -m src.bench

notebook:
	jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=3600 notebooks/lab03.ipynb

report-tables:
	$(PY) -m src.report_tables

report: report-tables
	cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null \
		&& pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null
	@pdfinfo report/report.pdf | grep -E '^Pages' || true
	@echo "Fuentes embebidas:" && pdffonts report/report.pdf | head -5

all: test etl eda dow rules bench notebook report

clean:
	rm -rf outputs/etl
	rm -f outputs/*.parquet outputs/figs/*.png outputs/tables/*.csv
	rm -f report/report.aux report/report.log report/report.out report/report.pdf
