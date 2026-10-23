# Technical documentation

## Architecture

```
Open-Meteo (CAMS, ERA5)
      │  aaqa fetch         src/aaqa/data/fetch.py
      ▼
data/raw/*.csv
      │  aaqa process       data/clean.py, features/build.py::aggregate_daily
      ▼
data/processed/{hourly,daily}.csv ──► aaqa report   visualization/plots.py → reports/figures/
      │  aaqa train         features/build.py, models/train.py
      ▼
models/forecaster.joblib, reports/metrics.json
      │  aaqa forecast      models/predictor.py  +  data/weather_forecast.py (official weather)
      ▼
API (api/main.py)  ·  app (dashboard/app.py)  ·  reports/latest_forecast.json
```

The code is an installable Python package with a *src layout* (`pyproject.toml`). Steps only
communicate through files, so each can be re-run on its own. Every pollutant is described once in
[pollutants.py](../src/aaqa/pollutants.py) (forecast statistic, WHO guideline, AQI breakpoints,
main sources), and the rest of the code loops over that registry.

## Data

| Source | Variables | Resolution | Availability |
|---|---|---|---|
| [Copernicus CAMS global](https://open-meteo.com/en/docs/air-quality-api) | PM2.5, PM10, NO2, O3, CO, SO2, dust | hourly, ~40 km grid | from August 2022 |
| [ERA5](https://open-meteo.com/en/docs/historical-weather-api) | temperature, humidity, rain, wind, pressure | hourly, ~25 km | up to D-6 |
| [Open-Meteo forecast](https://open-meteo.com/en/docs) | same weather variables | hourly | fills the last 6 days |
| [Open-Meteo forecast](https://open-meteo.com/en/docs), daily | sky, high/low, feels-like, rain, wind, UV | daily | the weather shown in the app |

Location: 5.36° N, 4.01° W (central Abidjan), local time (GMT).

**Cleaning** (`clean_hourly`): complete hourly grid, duplicates removed, values outside physical
ranges set to NaN, time interpolation of gaps up to 6 hours (wind direction, a circular variable,
is carried forward instead of interpolated), hours without PM2.5 dropped. On the current data no
missing values remain after cleaning.

**Daily aggregation** (`aggregate_daily`): mean, minimum, maximum or sum depending on the variable,
plus the AQI statistics: daily maximum of the rolling 8-hour mean for O3 and CO, daily hourly
maximum for NO2 and SO2. A day with fewer than 18 hours of PM2.5 is dropped. Wind is averaged as a
vector (u and v components).

**Weather.** The weather itself is not forecast by AAQA: the numerical models behind Open-Meteo
(ECMWF, GFS, ICON…) are far more accurate than anything that could be learned from four years of
local data. The app and the `/forecast` endpoint show that official forecast next to the
pollution forecast.

## Forecasting problem

- **Target**: for each pollutant, its daily statistic on day D+h, h ∈ {1, 2, 3}.
- **Available information**: everything known at the end of day D, plus the calendar of the target
  day (known in advance). No weather forecast is used as an input, to keep the evaluation strictly
  honest.

### Features

| Family | Features |
|---|---|
| Day-D levels | `pm2_5_mean`, `pm2_5_max`, `pm10_mean`, `nitrogen_dioxide_{mean,max}`, `ozone_{mean,max,max8h}`, `carbon_monoxide_{mean,max8h}`, `sulphur_dioxide_{mean,max}`, `dust_{mean,max}` |
| Day-D weather | `temperature_2m_{mean,min,max}`, `temp_range`, `relative_humidity_2m_mean`, `precipitation_sum`, `rain_roll3`, `dry_days`, `wind_speed_10m_mean`, `wind_u_mean`, `wind_v_mean`, `surface_pressure_mean` |
| PM2.5 history | `pm2_5_lag{1,2,3,7}`, `pm2_5_roll{3,7,14,30}`, `pm2_5_std7`, `pm2_5_diff1` |
| Other pollutants' history | `{pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide}_{lag1,roll3,roll7}` |
| Dust | `dust_lag1`, `dust_roll3` |
| Day-D calendar | `doy_sin`, `doy_cos`, `dow`, `is_weekend`, `is_harmattan`, `is_dry_season` |
| Target-day calendar | the same, prefixed `target_` |

The absence of leakage is checked by an automated test: changing the data after a date changes no
feature before that date (`tests/test_features.py`).

## Model

`LogRatioRegressor(HistGradientBoostingRegressor)` in [models/train.py](../src/aaqa/models/train.py),
one per pollutant and horizon:

- **Transformed target**: `log(1 + C[D+h]) − log(1 + C[D])`. The mean PM2.5 level dropped from
  17.8 µg/m³ (2023) to 10.0 µg/m³ (2025). A tree model predicting the absolute level cannot
  extrapolate below the levels seen in training, and did *worse* than persistence (D+3 PM2.5 MAE
  3.29 vs 2.83). Predicting the relative change follows the current level automatically.
- **Absolute-error loss** (robust to harmattan peaks) and **strong regularisation** (8 leaves, at
  least 40 samples per leaf, L2 = 2), suited to about 1,200 training days.
- **Prediction interval**: two quantile models (10% and 90%) on the same target. Because the
  transform is monotonic, their quantiles map directly to µg/m³.
- **Model selection**: if the model does not beat persistence in cross-validation for a pollutant
  and horizon, persistence is served instead. The choice uses training data only. Today this
  applies to SO2 at every horizon.
- The shipped models are **retrained on all data** after evaluation.

scikit-learn's gradient boosting was chosen over XGBoost or LightGBM: equivalent accuracy on this
amount of data, no system dependency (OpenMP), native handling of missing values. An LSTM is not
justified with about 1,500 daily observations.

## Evaluation

- **Test set**: the last 365 days (21 Oct 2025 → 20 Oct 2026), never used in training; a full year
  covers every season.
- **Cross-validation**: chronological (`TimeSeriesSplit`, 5 folds) on the training period.
- **Baselines**: persistence (D+h = D), monthly climatology, Ridge regression on the same features.

### Results (test set, mean absolute error in µg/m³)

| Pollutant | D+1 model / persistence | D+2 | D+3 | 80% interval coverage |
|---|---|---|---|---|
| PM2.5 | **1.70** / 1.81 | **2.19** / 2.52 | **2.42** / 2.83 | 70–73% |
| PM10 | **3.76** / 4.12 | **5.12** / 5.92 | **5.83** / 6.86 | 71–72% |
| O3 | **4.08** / 4.45 | **5.08** / 5.73 | **5.42** / 6.20 | 76–78% |
| NO2 | **0.61** / 0.67 | **0.70** / 0.93 | **0.70** / 0.93 | 64–71% |
| SO2 (persistence served) | 0.14 / 0.15 | 0.16 / 0.19 | 0.16 / 0.19 | 71–73% |
| CO | **24.8** / 25.7 | **30.1** / 32.9 | **32.4** / 34.3 | 74–78% |

PM2.5 in detail:

| | D+1 | D+2 | D+3 |
|---|---|---|---|
| R² | 0.70 | 0.51 | 0.38 |
| Right AQI category, model | **81%** | **77%** | **75%** |
| Right AQI category, persistence | 81% | 74% | 72% |
| WHO exceedance precision / recall | 0.61 / 0.56 | | |

For PM10, the WHO exceedance (45 µg/m³) is detected with precision 0.86 and recall 0.80 at D+1.
For the other pollutants, "right category" scores of 96–100% mean little: those pollutants are
almost always in the "Good" category.

### Cross-validation (training period)

Gain in mean absolute error over persistence:

| Pollutant | D+1 | D+2 | D+3 |
|---|---|---|---|
| PM2.5 | 3% | 5% | 7% |
| PM10 | 5% | 6% | 8% |
| O3 | 3% | 5% | 4% |
| NO2 | 1% | 8% | 16% |
| SO2 | −4% | −10% | −4% |
| CO | 2% | 5% | 5% |

The gain is smaller in cross-validation than on the test set. The hyperparameters were chosen
while also looking at the PM2.5 test set, so **the cross-validation figure is the more cautious
one**. Absolute errors are higher in cross-validation because the training period contains the
strong harmattan episodes of 2023 and 2024.

### Feature importance

Computed by permutation on the test set (`reports/feature_importance.csv`,
`reports/figures/feature_importance.png`). Each pollutant's own day-D value dominates by
construction, since the model predicts a change relative to it, so the figure leaves it out.

## Main findings of the exploration

Computed by `aaqa report` (`reports/statistics.json`) and detailed in
`notebooks/data_exploration.ipynb`:

- PM2.5 mean 14.1 µg/m³, 2.8× the WHO annual guideline; 31% of days above 15 µg/m³.
- PM10 mean 25.4 µg/m³, 1.7× the WHO annual guideline; 9% of days above 45 µg/m³, 35% during the
  harmattan, when its mean (45.3 µg/m³) is right at the guideline.
- Ozone 8-hour peak mean 63.8 µg/m³; 2.7% of days above 100 µg/m³, 12% during the harmattan.
- Harmattan (Dec–Feb): PM2.5 23.7 µg/m³ on average and 64% of days above the WHO guideline; rainy
  season (Apr–Jul): 11.1 µg/m³ and 20%.
- Northerly winds: 26.7 µg/m³ PM2.5 on average; southerly winds, which dominate the rest of the
  year: 13.6 µg/m³.
- Spearman correlation of daily PM2.5 with CO: 0.56 (combustion); with dust: 0.26; with
  humidity: ≈ 0; with rain: −0.15.
- No visible weekly cycle (13.7 to 14.5 µg/m³ depending on the day).
- The hourly profile jumps at 12:00 all year round: an artefact of stitching two CAMS forecast
  cycles, not a real phenomenon.

## Calibration on the US Embassy monitor

[calibration.py](../src/aaqa/calibration.py), [data/embassy.py](../src/aaqa/data/embassy.py),
pipeline step `aaqa calibrate`.

**Data.** Hourly PM2.5 from the US Embassy's Met One BAM-1022 reference monitor, from the
AirNow API (free key in `AIRNOW_API_KEY`, environment or `.env`). Monthly requests run 8 at a
time and the file is saved after every batch. Available: 39,890 hours, 3 February 2020 → 1 May
2025. **The monitor has not reported since 1 May 2025.** Daily means need at least 18 valid hours.

**What the monitor shows.**
- Embassy PM2.5 averages 25.1 µg/m³: 73% of days are above 15 µg/m³, the harmattan averages
  42.0 µg/m³, and yearly means show no trend (19.5–29.8 µg/m³).
- CAMS was close to the monitor in 2022–2023 (embassy ÷ CAMS ≈ 0.8–1.4), then fell from 2024
  while the monitor did not (ratio 1.7–3.0 in 2024–2025). The drop in the model data after 2023
  is a model artefact, not an improvement.

**Method.**
- Candidate corrections:
  - ratio: k · CAMS;
  - linear: a + b · CAMS;
  - seasonal: log(1 + y) = a + b·log(1 + CAMS) + harmattan + rainy-season terms, with a
    smearing correction.
- Each is learned on all past days, the last 365 days or the last 180 days.
- The comparison is walk-forward: every month is corrected with what was learned before it,
  after a 6-month warm-up. The best combination is refitted on its window and applied to the
  live value, the 1-, 8- and 24-hour averages, and the forecasts (the 80% range included).

**Result** (776 paired days, August 2022 → April 2025, walk-forward):

| | MAE (µg/m³) | Bias |
|---|---|---|
| Raw CAMS | 9.0 | −5.7 |
| Seasonal, all past days | 7.2 | −3.0 |
| **Seasonal, last 365 days (chosen)** | **6.9** | **−2.5** |
| Ratio, last 180 days | 7.1 | −2.0 |

Only PM2.5 is calibrated. After May 2025 the last correction is applied without being checkable.

## Limitations

1. **Few ground measurements.** The models forecast CAMS output. Only PM2.5 is corrected, with a
   monitor that stopped in May 2025, so the correction can't be checked after that date. CAMS
   represents local sources poorly (traffic, dumps, waste burning, fish smoking). Its NO2 mean of
   1.7 µg/m³ is implausibly low for a city of 6 million people, so NO2, SO2 and CO are almost
   certainly underestimated.
2. **~40 km resolution.** One value for the whole city; districts such as Yopougon and Cocody
   cannot be compared with this source.
3. **Model change in 2024.** The drop in CAMS PM2.5 from 2024 is not seen by the embassy monitor:
   it is a model artefact. The forecasting models learn the CAMS level, and the calibration
   learns from the most recent year to follow it.
4. **Intervals too narrow**: 64–78% coverage for a nominal 80%. Conformal prediction would fix
   this.
5. **Lag on sudden episodes**: when a dust episode starts, the forecast follows the observation
   about a day late (see `reports/figures/forecast_test.png`). Adding next-day CAMS dust forecasts
   as a feature is the most promising improvement.

## App features

### Kos assistant ([assistant.py](../src/aaqa/assistant.py))

`build_context` turns the forecast, the daily weather and the current conditions into one compact
JSON object; both modes answer from it.

- **Offline mode**: regular-expression intents (day, weather, air, activity, best time,
  pollutant or topic definition) and a built-in glossary. No network needed.
- **Claude mode**: active when `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` is set. One
  `messages.create` call on `claude-opus-5-5` with `effort: low`, a fixed system prompt (so it
  can be cached), the last six chat turns, and the JSON context in the latest user message. The
  server-side refusal fallback (`fallbacks: "default"`) is enabled. Refusals get a polite reply,
  and any error (network, credentials, rate limit) falls back to the offline answer.

### Eco journey ([gamification.py](../src/aaqa/gamification.py))

A `Progress` object saved as JSON in `data/user/progress.json` (one user per installation, not
versioned). Every action is idempotent per day, so reloading the page never awards points twice.
The daily quiz and challenge are chosen from the date, so they are the same all day; the
challenge depends on the day's rain and air quality two days out of three.

### Branding

The KOSCON logo ([dashboard/assets/koscon-logo.png](../dashboard/assets/koscon-logo.png)) comes
from the official brand files, with only the empty margin trimmed. Per the brand guidelines it is
never modified, always sits on white, and is shown at least 3.5 cm (about 140 px) wide. Colours
are the brand palette (#074930, #3FAB3D, #006079, #0EA7DC). The brand font is Gotham, a licensed
font, so the app uses Montserrat, its closest free equivalent; replace it in `style()` if a
Gotham web licence is available.

## Reproducibility

- `random_state` is fixed (config); the tests use deterministic synthetic data.
- Tested versions are in `requirements.txt`; minimum versions in `pyproject.toml`.
- Downloaded data and models are not versioned: `make all` regenerates them. Recent CAMS values
  can be revised, so figures may vary slightly from one run to the next.
