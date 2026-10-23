# User guide

## 1. Installation

Requirements: Python 3.11 or later and an internet connection. The data comes from Open-Meteo,
which is free and needs no API key.

```bash
make install
# or, without make:
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## 2. The pipeline

Each step reads the files of the previous step and writes its own:

| Command | What it does | Output |
|---|---|---|
| `aaqa fetch` | downloads hourly data since August 2022 | `data/raw/*.csv` |
| `aaqa process` | cleans, merges and aggregates by day | `data/processed/hourly.csv`, `daily.csv` |
| `aaqa report` | exploration statistics and figures | `reports/statistics.json`, `reports/figures/` |
| `aaqa train` | evaluates, then trains, the D+1..D+3 models for six pollutants | `models/forecaster.joblib`, `reports/metrics.json` |
| `aaqa forecast` | forecast from the last complete day, plus the weather | `reports/latest_forecast.json` |
| `aaqa all` | all of the above, in order | |

Run the commands with the environment activated (`source .venv/bin/activate`) or with the
prefix: `.venv/bin/aaqa all`. To update the forecast each day, run `aaqa all` again (about
4 minutes).

Settings (location, dates, pollutants, horizons, test-set size) are in
[config/config.yaml](../config/config.yaml).

## 3. The app

```bash
make app     # http://localhost:8501
```

On first open, a KOSCON loading page appears for about three seconds. The header shows the
KOSCON logo and your eco-journey status (level, points, streak); under it, Kos gives a one-line
summary of the day. The app has four tabs:

- **🌤️ Today**
  - Date and local time (refreshed every minute), the temperature right now, the sky, and the
    day's high, low and "feels like".
  - **Next 24 hours**: temperature, icon and chance of rain hour by hour (scroll sideways).
  - **Forecast**: today and the next three days, with weather and the air-quality category. Air
    quality is forecast three days ahead, so the fourth day shows weather only.
  - **Air quality**: today's overall index on a ring, the main pollutant and the health advice.
  - **Pollutants** and **Details** tiles: each pollutant against its WHO guideline; humidity,
    wind, UV, rain and whether it is the harmattan season.
  - The background follows the sky (clear, cloudy, rain, storm, night) and turns sandy and hazy
    when the air is polluted (index above 100).
- **Kos the leopard** sits in the main card. Its fur colour, health, surroundings and movement
  follow the AQI and the health advice:

  | AQI level | Kos |
  |---|---|
  | Good | golden, in the sun, jumping for joy: enjoy the outdoors |
  | Moderate | calm, light haze |
  | Unhealthy for sensitive groups | worried, runny nose, sweating |
  | Unhealthy | sick and coughing in an FFP2 mask: wear one outside |
  | Very unhealthy | at home behind a window, with a blanket: stay indoors |
  | Hazardous | a zombie, with flies: health emergency |

  Pollution particles around Kos get denser as the index rises. Animations stop when the system
  setting "reduce motion" is on. All six states: `docs/mascot/index.html`.
- **City**: the buttons under the header switch between Abidjan, Paris, New York and New Delhi.
  Abidjan uses AAQA's trained models (and the calibration); the other cities show live CAMS
  values and the CAMS 3-day forecast. Comparing them is the quickest way to see Kos the leopard
  in its different states; the full set is in `docs/mascot/index.html`.
- **💬 Ask Kos**: type a question or tap a suggestion. Kos answers from the live forecast:
  weather for a day, air quality, whether to exercise outside, the best time to go out, and
  what PM2.5, PM10, ozone, NO2, SO2, CO, the AQI, the WHO guidelines or the harmattan are.
  - **Offline mode** (default) works without internet access to an AI service.
  - **AI mode** uses Claude (`claude-opus-5-5`) when `ANTHROPIC_API_KEY` is set before starting
    the app; it receives the same forecast data and answers in a few sentences. If the service
    is unreachable, Kos switches back to offline mode automatically.
- **🌱 Eco journey**
  - **Points**: +5 for opening the app each day (plus a streak bonus), +10 for the daily quiz,
    +15 for the daily eco-challenge, +2 per question to Kos (up to 5 a day).
  - **Levels**: Seed (0) → Sprout (50) → Sapling (150) → Tree (350) → Forest (700).
  - **Badges**: first visit, 3- and 7-day streaks, 5 quiz answers right, 5 challenges done,
    10 questions asked, a harmattan check-in, reaching Sapling.
  - Progress is saved on this computer in `data/user/progress.json`; delete that file to start
    over.
- **📈 Insights**: history and seasonality of any pollutant, forecast accuracy, the index scale
  and a button to refresh the data and retrain.

### Calibration on the US Embassy monitor

With an AirNow key in `.env` (`AIRNOW_API_KEY=...`, free at https://docs.airnowapi.org/),
`make all` and `make update` download the embassy's PM2.5 and correct the model with it. The app
then shows calibrated PM2.5, and the overall index computed from it, by default. The switch
**Calibrated on the US Embassy monitor**, under the header, shows the raw model instead. The
Particulate matter card shows the model value next to the calibrated one, and the monitor's last
reading. The monitor has not reported since May 2025.

### Model monitoring

```bash
make update    # once a day: new data + new forecast (about 1 min, no retraining)
make monitor   # http://localhost:8502
```

- **Source**: *Live* scores the forecasts the app has issued, once the real values are in;
  *Backtest* shows the test year (all four models, every day). Live becomes the default once
  about a week of forecasts has been scored.
- **Filters**: pollutant, days ahead (D+1 to D+3), window (14 days to a year) and **metric**:
  MAE (average miss), RMSE (big misses count more), R² (share of the variation explained),
  bias (average over- or under-forecast) and right AQI category.
- **Health of the 18 models**: ✅ Healthy, ⚠️ Watch or ⛔ Alert. Alert means the model is more
  than 10% worse than "tomorrow = today", its error is more than twice its test-year error, or
  its 80% range covers less than half of the real values. Hover a cell for the reason.
- **Model comparison**: leaderboard (every metric, ★ = best), a pollutant × model table for
  the chosen metric, and a chart by day ahead.
- **Forecast vs observed**, rolling error and interval coverage for the selected model, with
  the models of your choice on the chart.
- **Data drift**: the last 30 days against the same period in past years.
- Buttons to update the data and forecast, or retrain all models.

### Reading the index

The index follows the US EPA scale (2024 revision). The overall index is that of the worst
pollutant. For PM2.5 (24-hour mean):

| PM2.5 (µg/m³) | Index | Category | Advice |
|---|---|---|---|
| 0 – 9.0 | 0 – 50 | Good | no restrictions |
| 9.1 – 35.4 | 51 – 100 | Moderate | unusually sensitive people: limit prolonged exertion |
| 35.5 – 55.4 | 101 – 150 | Unhealthy for sensitive groups | children, older adults, people with asthma, pregnant women: reduce exertion |
| 55.5 – 125.4 | 151 – 200 | Unhealthy | everyone: avoid outdoor sport |
| 125.5 – 225.4 | 201 – 300 | Very unhealthy | stay indoors |
| above 225.4 | 301 – 500 | Hazardous | health emergency |

The WHO guideline (15 µg/m³ over 24 hours) is stricter than the "Moderate" ceiling: a day can be
"Moderate" and still exceed the WHO recommendation.

| Pollutant | Forecast statistic | WHO guideline |
|---|---|---|
| PM2.5 | 24-h mean | 15 µg/m³ (24-h) |
| PM10 | 24-h mean | 45 µg/m³ (24-h) |
| O3 | max 8-h mean | 100 µg/m³ (8-h peak) |
| NO2 | max hourly | 25 µg/m³ (24-h mean) |
| SO2 | max hourly | 40 µg/m³ (24-h mean) |
| CO | max 8-h mean | 10 mg/m³ (8-h) |

## 4. The API

```bash
make api     # interactive docs: http://localhost:8000/docs
```

| Method | Route | Description |
|---|---|---|
| GET | `/health` | service status and model version |
| GET | `/forecast` | D+1..D+3 forecast of every pollutant, overall index and weather |
| GET | `/weather` | weather for today and the next three days |
| POST | `/predict` | forecast one pollutant from supplied features |
| GET | `/history?days=30&pollutant=pm10` | observed daily values |
| GET | `/aqi?value=40&pollutant=pm2_5` | concentration → index and category |
| GET | `/pollutants` | tracked pollutants, statistics and WHO guidelines |

Examples:

```bash
curl -s localhost:8000/forecast
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"day": "2026-01-10", "horizon": 1, "pollutant": "pm2_5",
       "features": {"pm2_5_mean": 45, "dust_mean": 80}}'
```

For `/predict`, only the pollutant's own statistic (for example `pm2_5_mean` or `ozone_max8h`)
is required. Absent features are treated as missing, which lowers accuracy. The full feature
list is in [technical_docs.md](technical_docs.md#features).

## 5. Docker

```bash
docker compose up --build    # API on :8000, app on :8501
```

On first start the container downloads the data and trains the models (about 4 minutes). Data
and models are kept in Docker volumes.

## 6. Tests and quality

```bash
make test    # synthetic data, no network access
make lint    # ruff
```

GitHub Actions runs the tests on every push ([ci.yml](../.github/workflows/ci.yml)) and
produces a forecast every day at 06:00 ([daily-forecast.yml](../.github/workflows/daily-forecast.yml)),
downloadable from the repository's *Actions* tab.

## 7. Notebooks

The three notebooks in [notebooks/](../notebooks/) are executed and saved with their results.
To re-run them on today's data: `make notebooks`.

1. `data_exploration.ipynb`: data quality, trends, seasons, hourly cycle, wind, correlations, all
   pollutants
2. `feature_engineering.ipynb`: how the features are built, leakage check
3. `modeling.ipynb`: model comparison for each pollutant, errors by season, feature importance

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError … run the previous step first` | run `aaqa all` |
| API: `503 No model` | run `aaqa train` |
| `fetch` fails | check the connection; Open-Meteo allows about 10,000 calls a day (the pipeline makes about ten) |
| No weather in the app | the weather forecast needs internet; the air-quality forecast still works offline |
| `bad interpreter` when running `.venv/bin/...` | the folder was moved after `.venv` was created: `rm -rf .venv && make install` |
