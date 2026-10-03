# Predictive Thermal Memory System (Software-Only Version)

A smart-home mini-project that learns time-of-day and occupancy patterns
from historical indoor sensor data and predicts the ideal temperature
setting ahead of time — instead of just reacting to the current reading.

This version runs entirely on your computer: no ESP32 or physical sensors
required. A realistic synthetic dataset stands in for the hardware layer.

## Architecture

```
simulate_data.py  -->  thermal_memory.db (SQLite)  -->  train_model.py  -->  thermal_model.joblib
                                                                                    |
                                                                                    v
                                                                    app.py (Flask) --> dashboard (index.html)
```

1. **simulate_data.py** — generates 60 days of realistic synthetic
   temperature, humidity, and occupancy readings (15-minute intervals)
   and stores them in SQLite. This is the "memory" data source.
2. **train_model.py** — loads the data, engineers time/occupancy features,
   trains a Decision Tree (explainable baseline) and a Random Forest
   (used for real predictions), and saves the trained model.
3. **app.py** + **templates/index.html** — a Flask dashboard showing
   actual vs predicted temperature over the last 48 hours, the model's
   prediction for the next hour, and a simulated control action
   (Cooling ON/OFF) based on a comfort threshold.

## Pages

- **`/`** — Dashboard: live gauge, 48-hour chart, weekly memory heatmap
- **`/controls`** — Adjust the comfort threshold, strategy mode, comfort corridor, and pre-cool toggle. Changes here immediately affect the Dashboard's Cooling ON/OFF decision — both pages read from the same underlying setting.
  - **Strategy modes actually change the decision logic**: `predictive_auto` compares the model's forecast directly to the threshold; `eco` (±1.5°C) and `strict` (±0.5°C) apply real hysteresis, so cooling won't flicker on/off right at the threshold — once it's on, it stays on until the predicted temperature drops well below the threshold, and vice versa; `override` ignores the forecast entirely and reacts to the current sensor reading instead.
  - **The corridor bar is a real draggable control** — drag the teal (min) and orange (max) dots to set the comfort range; the white dot shows the live room temperature moving between them. Click "Apply & Sync to Model" to persist your changes.
  - The status badge and live temperature dot auto-refresh every 8 seconds, same as the Dashboard.
- **`/history`** — Raw readings table (actual vs predicted vs delta) with a CSV export
- **`/how-it-works`** — Architecture walkthrough and live model metrics (feature importances, MAE, training set size), all pulled from the trained model rather than hardcoded

## Note on app_backup.py

`app_backup.py` is a snapshot of an earlier backend version, kept for reference only — it is not imported or run by anything. Safe to delete if you don't need it.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python3 simulate_data.py    # 1. generate the dataset (creates thermal_memory.db)
python3 train_model.py      # 2. train the model (creates thermal_model.joblib)
python3 app.py              # 3. start the dashboard
```

Then open **http://127.0.0.1:5000** in your browser. The dashboard now
auto-refreshes every 8 seconds on its own.

## Making it look "live" for a presentation

By default the dashboard just replays historical data, which looks static.
To make it visibly active while presenting, open a **second terminal**
(keep `app.py` running in the first one) and run:

```bash
python3 live_feed.py --interval 5
```

This inserts one new sensor reading every 5 seconds using the real current
time. Because the dashboard auto-refreshes every 8 seconds, you'll see:
- the chart's right edge extending with new points as you talk
- the "Predicted next hour" and "Cooling ON/OFF" cards updating on their own
- the "Live — last updated ..." timestamp at the bottom of the page ticking forward

Stop it any time with `Ctrl+C` in that second terminal.

**Note:** `live_feed.py` only adds new readings — it does not retrain the
model. The model was trained once on the historical dataset, so live
readings are being *predicted against*, not *learned from*, in real time.
If asked in a viva, this is the correct thing to say: real online learning
(retraining as new data arrives) is a natural extension, but out of scope
for this version.

## Regenerating the dataset

`simulate_data.py` now uses a **random seed by default**, so re-running it
gives you a fresh (but still realistically patterned) dataset each time —
useful if you want to demonstrate the pipeline working more than once.
For a reproducible dataset (e.g. to match figures already in your report),
pass a fixed seed:

```bash
python3 simulate_data.py --seed 42
```

## What to put in your report

- **Problem statement**: reactive thermostats only respond after discomfort;
  this system predicts ahead of time using historical patterns ("memory").
- **Dataset**: synthetic data generated to mimic realistic daily temperature
  cycles and occupancy patterns (see `simulate_data.py` for the exact logic:
  sinusoidal daily temperature cycle, time-dependent occupancy probability,
  humidity inversely related to temperature, all with added noise).
- **Model**: run `train_model.py` and copy the printed Decision Tree
  structure and MAE/RMSE metrics directly into your report — the tree text
  output is a genuinely good explainability artifact for a viva.
- **Feature importance**: also printed by `train_model.py` — shows which
  factors (hour of day, occupancy, etc.) drive predictions most.
- **Results**: screenshot the dashboard chart (actual vs predicted line
  overlay) as your key results figure.
- **Future scope**: swap the synthetic data source for a real
  ESP32 + DHT22 + PIR pipeline without changing any downstream code, since
  the database schema stays the same either way.

## Pages

- **`/` — Dashboard**: live gauge, 48-hour chart, weekly memory heatmap
- **`/controls`**: adjust the comfort target, HVAC strategy mode, and pre-cool toggle. Saved to SQLite (`hvac_controls` table) and survives a restart. The target temperature here is the *same* value the Dashboard uses to decide Cooling ON/OFF — the two pages are wired together, not independent.
- **`/history`**: table of raw readings with actual vs predicted temperature and delta, plus a CSV export
- **`/how-it-works`**: architecture walkthrough and live model metrics (feature importances, MAE, RMSE) pulled directly from the trained model, not hardcoded

## Extra API endpoints

- `POST /api/settings/update` — update `target_temp`, `strategy_mode`, `precool_enabled`
- `GET /api/live` — current temperature, prediction, and HVAC status in one call
- `POST /api/record` — accepts `{"temp": .., "rh": .., "pir": ..}`, the same shape a real ESP32 would send, so this is the drop-in point for real hardware later
- `GET /api/export/csv` — downloads the full readings table as CSV



- Re-run `simulate_data.py` any time to regenerate a fresh dataset (it wipes
  and repopulates the `readings` table).
- The comfort threshold for the control action (27.5°C) is set in `app.py`
  (`COMFORT_THRESHOLD`) — adjust it to whatever feels realistic for your
  scenario.
- `app.py` runs in Flask debug mode by default, which is fine for local
  development/demo but should be turned off (`debug=False`) if you ever
  deploy it anywhere public.
