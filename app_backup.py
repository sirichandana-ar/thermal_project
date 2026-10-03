"""
Predictive Thermal Memory System
Flask Backend
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
import csv
import io

from datetime import datetime
from flask import Flask, jsonify, render_template, request, Response


DB_PATH = "thermal_memory.db"
MODEL_PATH = "thermal_model.joblib"

app = Flask(__name__)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
MODEL_MAE = bundle.get("mae")
MODEL_RMSE = bundle.get("rmse")
TRAINED_ON_ROWS = bundle.get("trained_on_rows")

HVAC_SETTINGS = {
    "target_temp": 24.5,
    "min_comfort": 21.0,
    "max_comfort": 27.5,
    "strategy_mode": "predictive_auto",
    "precool_enabled": True,
    "precool_minutes": 30,
    "status": "Cooling Standby",
}

FEATURE_LABELS = {
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend flag",
    "occupancy": "Occupancy now",
    "occupancy_rolling": "Occupancy trend",
}

FEATURE_IMPORTANCES = sorted(
    [
        {"label": FEATURE_LABELS.get(feature, feature), "value": round(float(value), 4)}
        for feature, value in zip(FEATURES, model.feature_importances_)
    ],
    key=lambda item: item["value"],
    reverse=True,
)


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_hvac_table_and_load():
    """Create the hvac_controls table if missing, and load any previously
    saved settings into HVAC_SETTINGS so they survive a server restart."""
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hvac_controls (
            id INTEGER PRIMARY KEY,
            target_temp REAL,
            min_comfort REAL,
            max_comfort REAL,
            strategy_mode TEXT,
            precool_enabled INTEGER,
            updated_at TEXT
        )
    """)
    conn.commit()
    row = conn.execute("SELECT * FROM hvac_controls WHERE id = 1").fetchone()
    if row:
        HVAC_SETTINGS["target_temp"] = row["target_temp"]
        HVAC_SETTINGS["min_comfort"] = row["min_comfort"]
        HVAC_SETTINGS["max_comfort"] = row["max_comfort"]
        HVAC_SETTINGS["strategy_mode"] = row["strategy_mode"]
        HVAC_SETTINGS["precool_enabled"] = bool(row["precool_enabled"])
    conn.close()


ensure_hvac_table_and_load()


def load_recent_data(hours=48):
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM readings ORDER BY timestamp", conn)
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["occupancy_rolling"] = df["occupancy"].rolling(window=8, min_periods=1).mean()
    return df.tail(hours * 4)


@app.route("/")
def index():
    return render_template("index.html", active_page="dashboard")


@app.route("/api/history")
def api_history():
    df = load_recent_data(hours=48)
    if df.empty:
        return jsonify({"timestamps": [], "actual": [], "predicted": [], "occupancy": []})
    X = df[FEATURES]
    df["predicted"] = model.predict(X)
    return jsonify({
        "timestamps": df["timestamp"].dt.strftime("%m-%d %H:%M").tolist(),
        "actual": df["temperature"].round(2).tolist(),
        "predicted": df["predicted"].round(2).tolist(),
        "occupancy": df["occupancy"].tolist(),
    })


@app.route("/api/next_hour")
def api_next_hour():
    now = datetime.now()
    next_hour = (now.hour + now.minute / 60 + 1) % 24
    day_of_week = now.weekday()
    is_weekend = int(day_of_week >= 5)

    df = load_recent_data(hours=2)
    if df.empty:
        return jsonify({"error": "No sensor readings available."}), 404

    last_row = df.iloc[-1]
    next_row = pd.DataFrame([{
        "hour": next_hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "occupancy": last_row["occupancy"],
        "occupancy_rolling": last_row["occupancy_rolling"],
    }])[FEATURES]

    predicted_temp = float(model.predict(next_row)[0])
    # Uses the SAME threshold the Controls page edits -> the two pages agree.
    threshold = HVAC_SETTINGS["target_temp"]
    action = "Cooling ON" if predicted_temp > threshold else "Cooling OFF"

    return jsonify({
        "predicted_temp": round(predicted_temp, 2),
        "threshold": threshold,
        "action": action,
        "occupancy_now": bool(last_row["occupancy"]),
        "occupancy_rolling_pct": round(float(last_row["occupancy_rolling"]) * 100, 1),
        "feature_importances": FEATURE_IMPORTANCES,
        "model_mae": MODEL_MAE,
        "model_rmse": MODEL_RMSE,
        "trained_on_rows": TRAINED_ON_ROWS,
    })


@app.route("/api/weekly_pattern")
def api_weekly_pattern():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM readings", conn)
    conn.close()

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_names_full = ["Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays"]

    if df.empty:
        return jsonify({"cells": [], "day_labels": day_names, "insight": "No historical data available yet.", "temp_min": 0, "temp_max": 0})

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_bucket"] = df["timestamp"].dt.hour

    grouped = (
        df.groupby(["day_of_week", "hour_bucket"])
        .agg(avg_temp=("temperature", "mean"), avg_occupancy=("occupancy", "mean"))
        .reset_index()
    )

    cells = []
    for _, row in grouped.iterrows():
        cells.append({
            "day": int(row["day_of_week"]),
            "day_label": day_names[int(row["day_of_week"])],
            "hour": int(row["hour_bucket"]),
            "avg_temp": round(float(row["avg_temp"]), 2),
            "avg_occupancy": round(float(row["avg_occupancy"]), 2),
        })

    hottest = grouped.loc[grouped["avg_temp"].idxmax()]
    coolest = grouped.loc[grouped["avg_temp"].idxmin()]
    occ_temp = df.groupby("occupancy")["temperature"].mean()
    occ_diff = None
    if 1 in occ_temp.index and 0 in occ_temp.index:
        occ_diff = round(float(occ_temp[1] - occ_temp[0]), 2)

    insight = (
        f"Around {day_names_full[int(hottest['day_of_week'])]}, this room typically peaks near "
        f"{int(hottest['hour_bucket']):02d}:00 ({hottest['avg_temp']:.1f}\u00b0C) and is coolest near "
        f"{int(coolest['hour_bucket']):02d}:00 ({coolest['avg_temp']:.1f}\u00b0C)."
    )
    if occ_diff is not None:
        insight += f" It also runs about {abs(occ_diff):.1f}\u00b0C {'warmer' if occ_diff > 0 else 'cooler'} when occupied."

    return jsonify({
        "cells": cells,
        "day_labels": day_names,
        "insight": insight,
        "temp_min": round(float(grouped["avg_temp"].min()), 1),
        "temp_max": round(float(grouped["avg_temp"].max()), 1),
    })


@app.route("/controls")
def controls_page():
    settings = dict(HVAC_SETTINGS)
    settings["corridor_min"] = settings["min_comfort"]
    settings["corridor_max"] = settings["max_comfort"]
    settings["corridor_span"] = round(settings["max_comfort"] - settings["min_comfort"], 1)
    settings["status"] = f"{settings['status']} ({settings['target_temp']}\u00b0C)"
    return render_template("controls.html", settings=settings)


@app.route("/history")
def history_page():
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT timestamp, temperature, humidity, occupancy, day_of_week, is_weekend "
        "FROM readings ORDER BY timestamp DESC LIMIT 500",
        conn,
    )
    conn.close()

    DISPLAY_LIMIT = 50
    readings = []
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")  # ascending, so rolling window is correct
        df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
        df["occupancy_rolling"] = df["occupancy"].rolling(window=8, min_periods=1).mean()
        X = df[FEATURES]
        df["predicted_temp"] = model.predict(X)
        df["delta"] = df["temperature"] - df["predicted_temp"]
        df = df.sort_values("timestamp", ascending=False).head(DISPLAY_LIMIT)

        for _, row in df.iterrows():
            readings.append({
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "temperature": round(float(row["temperature"]), 2),
                "predicted_temp": round(float(row["predicted_temp"]), 2),
                "delta": round(float(row["delta"]), 2),
                "occupancy": int(row["occupancy"]),
            })

    if readings:
        mae = round(float(df["delta"].abs().mean()), 2)
        conn = get_db_connection()
        total_records = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        conn.close()
    else:
        mae = MODEL_MAE if MODEL_MAE is not None else 0
        total_records = 0

    summary = {"total_records": total_records, "mae": mae}
    return render_template("history.html", readings=readings, summary=summary)


@app.route("/how-it-works")
def how_it_works_page():
    feature_map = {f["label"]: f["value"] for f in FEATURE_IMPORTANCES}
    model_metadata = {
        "checkpoint": MODEL_PATH,
        "estimators": getattr(model, "n_estimators", "N/A"),
        "mae": MODEL_MAE,
        "rmse": MODEL_RMSE,
        "trained_on_rows": TRAINED_ON_ROWS,
        "features": FEATURE_IMPORTANCES,
        "feature_map": feature_map,
    }
    return render_template("how_it_works.html", meta=model_metadata)


@app.route("/api/live")
def api_live():
    now = datetime.now()
    df = load_recent_data(hours=2)

    if df.empty:
        current_temp = 0
        occupancy = 0
    else:
        last_row = df.iloc[-1]
        current_temp = float(last_row["temperature"])
        occupancy = int(last_row["occupancy"])

    if not df.empty:
        current_row = pd.DataFrame([{
            "hour": now.hour + now.minute / 60,
            "day_of_week": now.weekday(),
            "is_weekend": int(now.weekday() >= 5),
            "occupancy": occupancy,
            "occupancy_rolling": float(df.iloc[-1]["occupancy_rolling"]),
        }])[FEATURES]
        predicted = float(model.predict(current_row)[0])
    else:
        predicted = current_temp

    if current_temp > HVAC_SETTINGS["max_comfort"]:
        status = "Active Pre-Cooling" if HVAC_SETTINGS["precool_enabled"] else "Cooling Active"
    elif current_temp < HVAC_SETTINGS["min_comfort"]:
        status = "Below Comfort Corridor"
    else:
        status = "Comfort Corridor Holding"

    HVAC_SETTINGS["status"] = status

    return jsonify({
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": round(current_temp, 2),
        "predicted": round(predicted, 2),
        "delta": round(current_temp - predicted, 2),
        "occupancy": bool(occupancy),
        "hvac_status": status,
        "corridor": {
            "target": HVAC_SETTINGS["target_temp"],
            "min": HVAC_SETTINGS["min_comfort"],
            "max": HVAC_SETTINGS["max_comfort"],
        }
    })


@app.route("/api/record", methods=["POST"])
def api_record_sensor():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received."}), 400

    try:
        temperature = float(data.get("temp"))
        humidity = float(data.get("rh", 0))
        occupancy = int(data.get("pir", 0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid sensor values."}), 400

    zone = data.get("zone", "Main Living Area")
    now = datetime.now()
    day_of_week = now.weekday()
    is_weekend = int(day_of_week >= 5)

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO readings (timestamp, temperature, humidity, occupancy, day_of_week, is_weekend) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (now.strftime("%Y-%m-%d %H:%M:%S"), temperature, humidity, occupancy, day_of_week, is_weekend),
    )
    conn.commit()
    conn.close()

    return jsonify({
        "status": "success",
        "logged_temp": temperature,
        "humidity": humidity,
        "occupancy": occupancy,
        "zone": zone,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
    }), 201


@app.route("/api/settings/update", methods=["POST"])
def api_update_settings():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "No settings received."}), 400

    try:
        if "target_temp" in data:
            HVAC_SETTINGS["target_temp"] = round(float(data["target_temp"]), 1)
        if "min_comfort" in data:
            HVAC_SETTINGS["min_comfort"] = round(float(data["min_comfort"]), 1)
        if "max_comfort" in data:
            HVAC_SETTINGS["max_comfort"] = round(float(data["max_comfort"]), 1)
        if "strategy_mode" in data:
            allowed_modes = ["predictive_auto", "eco", "strict", "override"]
            if data["strategy_mode"] in allowed_modes:
                HVAC_SETTINGS["strategy_mode"] = data["strategy_mode"]
        if "precool_enabled" in data:
            HVAC_SETTINGS["precool_enabled"] = bool(data["precool_enabled"])
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid control value."}), 400

    conn = get_db_connection()
    conn.execute("""
        INSERT OR REPLACE INTO hvac_controls
        (id, target_temp, min_comfort, max_comfort, strategy_mode, precool_enabled, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        1,
        HVAC_SETTINGS["target_temp"],
        HVAC_SETTINGS["min_comfort"],
        HVAC_SETTINGS["max_comfort"],
        HVAC_SETTINGS["strategy_mode"],
        1 if HVAC_SETTINGS["precool_enabled"] else 0,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()

    return jsonify({"status": "synchronized", "settings": HVAC_SETTINGS})


@app.route("/api/export/csv")
def export_telemetry_csv():
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT timestamp, temperature, humidity, occupancy, day_of_week, is_weekend "
        "FROM readings ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Temperature (\u00b0C)", "Humidity (%)", "Occupancy", "Day of Week", "Weekend"])
    for row in rows:
        writer.writerow([row["timestamp"], row["temperature"], row["humidity"], row["occupancy"], row["day_of_week"], row["is_weekend"]])
    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=thermal_telemetry.csv"},
    )


if __name__ == "__main__":
    print("Predictive Thermal Memory Server")
    print("Open: http://127.0.0.1:5000/")
    app.run(debug=True)
