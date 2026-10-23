# Changelog

## 1.3.0 — 2026-10-23

- Kos is now a leopard: fur colour, health, scene and animation follow the AQI and the health
  advice (outdoors, mask, at home, zombie at Hazardous). Gallery in `docs/mascot/` (`make mascot`).
- City selector: Abidjan (AAQA models), Paris, New York and New Delhi (live CAMS values and CAMS
  3-day forecast), each with its own skyline.
- In the app, Kos blends into the card behind it (no backdrop, shoulders fade out); the gallery
  keeps the full scenes.

## 1.2.0 — 2026-10-23

- PM2.5 calibrated on the US Embassy monitor (AirNow): walk-forward selection of the correction
  and its training window; ground-truth section in the monitoring dashboard.
- Finding: the drop in CAMS PM2.5 after 2023 is a model artefact; reports corrected.
- Monitoring dashboard: light theme, simpler model cells, model comparison (MAE, RMSE, R², bias,
  right AQI category), system health of services and data pipeline, sampling map.
- App: mobile-style layout, live AQI, Kos changes colour and face with each AQI level, sampling
  map, calibration switch.
- NO2 models get a full NO2 history (D+1 cross-validation gain 1.8% → 3.0%).
- API reloads the model after retraining. AirNow errors never expose the API key.
- Docker image runs as a non-root user and has a health check.

## 1.1.0 — 2026-10-22

- Six pollutants forecast three days ahead, weather forecast, Kos assistant, eco journey,
  KOSCON branding, English throughout.

## 1.0.0 — 2026-10-22

- First pipeline: Open-Meteo download, PM2.5 forecast, API, dashboard, tests, documentation.
