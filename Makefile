PY ?= .venv/bin/python
AAQA = $(PY) -m aaqa

.PHONY: install data train forecast update report all test lint api app dashboard monitor mascot notebooks docker clean

install:            ## Create the environment and install the project
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

data:               ## Download and clean the data (about 1 min)
	$(AAQA) fetch process

report:             ## Exploration statistics and figures
	$(AAQA) report

train:              ## Train, evaluate and save the models
	$(AAQA) train

forecast:           ## D+1..D+3 forecast from the latest data
	$(AAQA) forecast

update:             ## New data + new forecast, without retraining (run daily)
	$(AAQA) fetch process calibrate forecast

all:                ## Full pipeline
	$(AAQA) all

test:
	$(PY) -m pytest --cov=aaqa --cov-report=term-missing

lint:
	$(PY) -m ruff check .

api:                ## http://localhost:8000/docs
	$(PY) -m uvicorn aaqa.api.main:app --reload

app:                ## Weather & air quality app: http://localhost:8501
	$(PY) -m streamlit run dashboard/app.py

dashboard: app

monitor:            ## Model monitoring dashboard: http://localhost:8502
	$(PY) -m streamlit run dashboard/monitoring.py --server.port 8502

mascot:             ## Mascot gallery: docs/mascot/index.html
	$(PY) scripts/build_mascot_gallery.py

notebooks:          ## Re-run the notebooks on the current data
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

docker:
	docker compose up --build

clean:
	rm -rf data/raw/*.csv data/raw/*.json data/processed/*.csv models/*.joblib .pytest_cache .ruff_cache .coverage
