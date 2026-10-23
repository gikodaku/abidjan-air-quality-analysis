# Developer guide

How AAQA is built and how to change it safely. For running the apps, see the
[user guide](user_guide.md); for the science, the [technical documentation](technical_docs.md).

## Architecture in one picture

```
config/config.yaml ─┐
                    ▼
aaqa fetch ──► data/raw/ (CAMS, ERA5, grid points)
aaqa process ──► data/processed/ (hourly, daily)
aaqa calibrate ──► data/raw/embassy_pm25.csv, reports/calibration.json   (needs AIRNOW_API_KEY)
aaqa report ──► reports/statistics.json, reports/figures/
aaqa train ──► models/forecaster.joblib, reports/metrics.json, reports/test_predictions_*.csv
aaqa forecast ──► reports/latest_forecast.json, data/monitoring/forecast_log.csv
                    │
                    ├── dashboard/app.py         public app (reads the files + live Open-Meteo)
                    ├── dashboard/monitoring.py  team dashboard (reads the files + checks services)
                    └── src/aaqa/api/main.py     REST API (reads the files)
```

Each step reads the files of the previous one, so any step can be re-run alone. The apps never
train or download history themselves, except through their explicit refresh buttons.

| Module | Responsibility |
|---|---|
| `pollutants.py` | One `Pollutant` per pollutant: forecast statistic, WHO guideline, AQI breakpoints. **Single source of truth**: everything loops over `POLLUTANTS`. |
| `aqi.py` | US EPA index, categories, advice, overall index. |
| `data/fetch.py` | Open-Meteo downloads (history), grid points. `_get` is the only HTTP helper (retries, timeouts). |
| `data/air_now.py`, `data/weather_forecast.py` | Live values for the apps (short timeouts). |
| `data/embassy.py` | AirNow download of the US Embassy monitor. |
| `data/clean.py`, `features/build.py` | Cleaning, daily aggregation, features. No feature may use the future (tested). |
| `models/train.py` | Training, evaluation, model selection; `models/predictor.py` loads and predicts. |
| `calibration.py` | CAMS PM2.5 corrected on the embassy monitor. |
| `monitoring.py` | Forecast log, scoring, health rules, model comparison, drift, service checks. |
| `assistant.py`, `gamification.py` | Kos's answers, and the eco journey. |
| `cities.py` | Cities in the selector; only Abidjan has the full pipeline. |
| `dashboard/mascot.py` | Kos the leopard: one state per AQI level (`LEVELS`, `MOTION`). `mascot(level, size, backdrop=False)` draws it without a scene background, to blend into a card (the app); `backdrop=True` draws the full scene (the gallery). Edit there, then `make mascot` to refresh `docs/mascot/`. |

## Everyday tasks

**Refresh data and forecast (daily).** `make update`. To run it automatically every morning at
07:00 on a Mac or Linux machine, add a cron job (`crontab -e`):

```
0 7 * * * cd ~/git/AAQA && make update >> data/update.log 2>&1
```

**Retrain (monthly, or when the monitoring shows Watch/Alert).** `make train`, or the button in
the monitoring dashboard. The API picks up the new model automatically.

**Add a pollutant.** Add a `Pollutant` to `POLLUTANTS` in `pollutants.py`, with its Open-Meteo
variable added to `AIR_QUALITY_VARS` in `data/fetch.py` and its daily statistic in `_DAILY_AGG`
in `features/build.py`. Add it to `model.pollutants` in `config/config.yaml`, run `make all`, and
check it in the monitoring dashboard.

**Change the location.** Edit `location` in `config/config.yaml`, then `make clean all`. The
districts on the map (`dashboard/maps.py`) and the embassy bounding box (`data/embassy.py`) are
specific to Abidjan.

**Add a city.** Add a `City` to `CITIES` in `cities.py` (and a landmark to `_LANDMARKS` in
`dashboard/illustrations.py` if you like). It gets live CAMS values and the CAMS forecast; the
AAQA models stay Abidjan's.

**Change the app's look.** Colours and layout are in `style()` in `dashboard/app.py`; the mascot
in `dashboard/mascot.py`; the skylines in `dashboard/illustrations.py`. The KOSCON logo (`dashboard/assets/koscon-logo.png`) must
not be modified (brand guidelines) and stays on a white background.

## Keys and secrets

| Key | Used for | Where |
|---|---|---|
| `AIRNOW_API_KEY` | US Embassy PM2.5 (calibration) | `.env` at the project root, or the environment; GitHub secret for CI |
| `ANTHROPIC_API_KEY` | Kos AI mode (optional) | environment |

- `.env` is git-ignored. Never commit a key, and never paste one into an issue or a chat.
- Error messages from AirNow are rewritten so that the key never reaches logs (tested).
- To rotate a key, replace it in `.env` (and in the GitHub secret).

## Quality gates

```bash
make lint    # ruff
make test    # 168 tests on synthetic data, no network; about 3 minutes
```

- The tests mock every network call. A new data source needs a fake in `tests/`.
- The pipeline tests (`tests/test_pipeline.py`) run the full pipeline once and render both apps
  with Streamlit's AppTest: a broken page fails the suite.
- GitHub Actions runs lint and tests on every push (`.github/workflows/ci.yml`).

## Releasing

1. Update `CHANGELOG.md` and the version in `pyproject.toml` and `src/aaqa/__init__.py`.
2. `make lint test`, then `make all` to retrain and refresh the reports.
3. Commit, tag (`git tag v1.2.0`) and push.

The model file records the package version and training date (`/health` and the monitoring
dashboard show them).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Monitoring shows a service in Alert | Check the detail column; Open-Meteo or AirNow may be down or slow; the apps keep working on the last data. |
| Data pipeline in Alert | `make update` has not run; check `data/update.log` if you use cron. |
| `bad interpreter` from `.venv/bin/...` | The folder moved: `rm -rf .venv && make install`. |
| Streamlit shows an old version of a module | Restart the app: Streamlit reloads the page script, not the package. |
