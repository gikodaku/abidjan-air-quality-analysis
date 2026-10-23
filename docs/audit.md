# AAQA audit — 23 October 2026

Scope: the code, documentation, security, data collection, models and operations of AAQA 1.2.0.
Every check below was run, not assumed. ✅ solid · ⚠️ works, with a risk to manage · ⛔ must be
fixed before relying on the product.

## Summary

The technical foundation is solid:
- the pipeline is clear and modular, each step can be re-run alone, and one registry
  (`pollutants.py`) drives every pollutant;
- 168 tests cover 96% of the code, and lint is clean;
- data collection has no gaps, and no dependency has a known vulnerability.

The two most urgent risks are not in the code:
1. **Nothing is committed to git.** The whole product exists only on this computer.
2. **The free Open-Meteo API does not allow commercial use.** A KOSCON product would need a paid
   subscription.

| Area | Status | In one line |
|---|---|---|
| Architecture and code | ✅ | Modular package, single source of truth for pollutants, no unused code |
| Tests | ✅ | 168 tests, 96% coverage, no network needed, both apps rendered in tests |
| Documentation | ✅ | README, user guide, technical docs, developer guide, changelog, this audit |
| Data collection | ✅ | 0 missing hours in 4 years; live data with short timeouts and fallbacks |
| Security | ⚠️ | Keys protected; API has no authentication (fine locally, not for public exposure) |
| Ground truth | ⚠️ | US Embassy monitor offline since 1 May 2025 |
| Models | ⚠️ | All 18 healthy; gains over the baseline are modest; 80% ranges too narrow |
| Operations | ⚠️ | Daily update is manual; Docker and CI not yet run |
| Version control | ⛔ | No commit since the original skeleton |
| Licensing | ⛔ for commercial use | Open-Meteo free tier is non-commercial only |

## 1. Architecture and code ✅

- **Pipeline.** `fetch → process → calibrate → report → train → forecast`. Steps communicate only
  through files, so each can be re-run or debugged alone.
- **Two apps and an API** read the same files; none of them trains or downloads history except
  through explicit buttons.
- **One registry.** Adding a pollutant means one `Pollutant` entry and two config lines (see the
  developer guide).
- **Clean code.** Lint (ruff) passes; a scan found no unused function or constant (two were
  removed).

## 2. Tests ✅

- **168 tests, 96% line coverage.** The lowest is `data/fetch.py` (69%): its network paths are
  covered by simulated responses only.
- **No network in tests.** Every external call is mocked, so the suite is deterministic.
- **End-to-end.** The full pipeline runs once on synthetic data, then both Streamlit apps are
  rendered and clicked (tabs, switch, chat, filters) with Streamlit's AppTest.
- **Notebooks.** The three notebooks run end to end on the current code (checked with nbconvert).
- **Fixed during the audit.** The suite took over 10 minutes; it now takes about 3, because the
  pipeline tests share one training run.

## 3. Documentation ✅

| Document | For whom |
|---|---|
| [README](../README.md) | Everyone: what it does, how to run it, results, limits |
| [User guide](user_guide.md) | Users: apps, index, API, monitoring, calibration |
| [Technical documentation](technical_docs.md) | Data scientists: data, method, evaluation, limitations |
| [Developer guide](developer_guide.md) | Developers: architecture, everyday changes, keys, releases |
| [Recommendations](../reports/recommendations.md) | Authorities |
| [Changelog](../CHANGELOG.md) | Everyone: what changed between versions |

Added during the audit: the developer guide, the changelog and this audit.

## 4. Security ⚠️

**Checked and fine:**

- **No secrets in files.** No key in the project files: only `.env`, which git ignores, holds
  one.
- **No known vulnerabilities.** `pip-audit` found none in the installed dependencies.
- **No injected HTML.** The apps build HTML only from the project's own strings and numbers.
  Chat messages and AI answers are rendered as Markdown, never as raw HTML.
- **Validated input.** API input is checked by Pydantic (types, ranges, known pollutants).

**Fixed during the audit:**

- ⛔ → ✅ **API key in error messages.** A rejected AirNow request produced an error message
  containing the full URL, API key included, which was written to the terminal. Messages are now
  rewritten without the key (tested).
- ⚠️ → ✅ **Undeclared dependency.** `python-dotenv`, used to read `.env`, was not declared and
  only worked because another package installs it.
- ⚠️ → ✅ **Docker ran as root.** The image now runs as a non-root user, with a health check.
- ⚠️ → ✅ **Privacy.** The Ask Kos tab now says that, in AI mode, questions are sent to
  Anthropic's API.

**Remaining:**

- ⚠️ **No authentication or rate limit on the API.** Fine on a laptop; required before exposing
  it on the internet (an API key header, rate limiting, HTTPS behind a reverse proxy).
  `docker compose` publishes ports 8000, 8501 and 8502 on all interfaces.
- ⚠️ **Model files are pickles.** `joblib` files can run code when loaded: only load models this
  project produced.

## 5. Data collection ✅

| Source | Coverage | State |
|---|---|---|
| CAMS pollution (Open-Meteo) | 37,056 hours, Aug 2022 → yesterday | 0 missing hours; only the first 3 days have empty values (before CAMS starts) |
| ERA5 / forecast weather (Open-Meteo) | same | 0 missing hours, no empty values |
| Daily data | 1,541 days | no missing PM2.5 day |
| US Embassy PM2.5 (AirNow) | 39,890 of 45,931 hours (87%) | ⚠️ last reading 1 May 2025 |
| Forecast log | since 21 Oct 2026 | 2 issue days, growing daily with `make update` |

- **Robust downloads.** Retries with back-off, monthly AirNow requests run in parallel and are
  saved after each batch, and live requests use short timeouts.
- **Graceful failures.** The apps keep working when a source is down (live values fall back to
  forecasts, the weather is optional).
- **Monitored.** The monitoring dashboard now shows the health of every service and the age of
  every pipeline output, refreshed every minute.

## 6. Models ⚠️

- **Health.** All 18 models (6 pollutants × 3 days ahead) are healthy on the backtest. 15 use
  gradient boosting; SO2 at D+1 and D+2 serves the "tomorrow = today" baseline, which the model
  did not beat in cross-validation.
- **Modest gains.** The models beat the baseline by 3–15% in cross-validation. The weakest was NO2
  at D+1 (+1.8%); a longer NO2 history, given to the NO2 models only, raised it to +3.0% during
  the audit.
- **Calibration.** PM2.5 is calibrated on the US Embassy monitor: the error drops by 24% and the
  bias from −5.7 to −2.5 µg/m³. Since May 2025 the correction can no longer be checked.
- **Uncertainty.** The "80%" ranges contain the real value only 64–78% of the time: they are
  too narrow.
- **Fixed during the audit.** The API kept a model in memory after retraining; it now reloads it
  when the file changes (tested).

## 7. Operations ⚠️

- **Manual daily update.** `make update` must be run by hand; the developer guide gives the cron
  line to automate it. The GitHub daily workflow forecasts, but its outputs are only kept as
  14-day artifacts, so it cannot feed the live monitoring.
- **Not run yet.** The Docker image and the GitHub Actions CI have never run (Docker is not
  installed here; nothing has been pushed).
- **Alerts on screen only.** Watch/Alert statuses are not sent anywhere (no email or SMS).

## 8. Licensing ⛔ for commercial use

- **Open-Meteo.** The free API is for **non-commercial use only** (under 10,000 calls a day). A
  commercial KOSCON service needs a paid API subscription, or a self-hosted Open-Meteo. The data
  is CC BY 4.0, so keep the attribution.
- **Copernicus CAMS and ERA5.** Free use under the Copernicus licence, with attribution.
- **AirNow (US Embassy).** Free, but preliminary and not fully validated; attributed in the app.
- **Map tiles.** © OpenStreetMap contributors © CARTO, attributed on the map.

## Action plan

| Priority | Action | Effort |
|---|---|---|
| 1 | **Commit and push** the project, with `.env` excluded | 5 min |
| 2 | Decide the commercial status; if commercial, subscribe to Open-Meteo and add the key | 1 h |
| 3 | Schedule `make update` daily (cron) so live monitoring builds up | 10 min |
| 4 | Find a new ground-truth source: OpenAQ / AirQo sensors, or a KOSCON reference sensor | days |
| 5 | Before any public deployment: API key, rate limit and HTTPS on the API | 1 day |
| 6 | Push alerts (email or SMS) when a model or the pipeline turns Alert | ½ day |
| 7 | Recalibrate the 80% ranges (conformal prediction) | ½ day |
| 8 | Run the Docker image and the CI once, after pushing | 1 h |

## Sources

- Open-Meteo [terms](https://open-meteo.com/en/terms) and [pricing](https://open-meteo.com/en/pricing)
- [AirNow — US Embassies and Consulates](https://www.airnow.gov/international/us-embassies-and-consulates/)
- [Abidjan embassy monitor in the literature (RSC, 2024)](https://pubs.rsc.org/en/content/articlehtml/2024/ea/d4ea00012a)
