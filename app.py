"""
Predictive Thermal Memory System
Flask Backend

Features:
    /                      -> Main dashboard
    /api/history           -> Actual vs predicted history
    /api/next_hour         -> Next-hour prediction
    /api/weekly_pattern    -> Weekly thermal memory

    /controls              -> Adjustable thermal controls
    /history               -> Sensor history table
    /how-it-works          -> Project architecture

    /api/live              -> Live telemetry
    /api/record            -> Receive sensor data
    /api/settings/update   -> Save control settings
    /api/export/csv        -> Export telemetry

Important:
    The ML model was trained using 15-minute thermal data.

    Model features:
        hour
        day_of_week
        is_weekend
        occupancy
        occupancy_rolling
        temp_lag_1
        temp_lag_4
        temp_lag_8
        temp_rolling_4
        temp_rolling_8

    The live feed may insert readings faster than every 15 minutes
    for presentation purposes. Therefore, model features are built
    from a 15-minute resampled thermal history.
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
import csv
import io

from datetime import datetime, timedelta

from flask import Flask, jsonify, render_template, request, Response


# ============================================================
# BASIC CONFIGURATION
# ============================================================

DB_PATH = "thermal_memory.db"
MODEL_PATH = "thermal_model.joblib"

app = Flask(__name__)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
FEATURES = bundle["features"]

MODEL_MAE = bundle.get("mae")
MODEL_RMSE = bundle.get("rmse")
TRAINED_ON_ROWS = bundle.get("trained_on_rows")

BASELINE_MAE = bundle.get("baseline_mae")
BASELINE_RMSE = bundle.get("baseline_rmse")


# ============================================================
# HVAC / THERMAL CONTROL SETTINGS
# ============================================================

HVAC_SETTINGS = {
    "target_temp": 24.5,
    "min_comfort": 21.0,
    "max_comfort": 27.5,

    "strategy_mode": "predictive_auto",

    "precool_enabled": True,
    "precool_minutes": 30,

    "status": "Cooling Standby",
    "last_action": "Cooling OFF",
}


# Wider buffer = fewer ON/OFF changes
STRATEGY_BUFFERS = {
    "eco": 1.5,
    "strict": 0.5,
}


ALLOWED_MODES = [
    "predictive_auto",
    "eco",
    "strict",
    "override",
]


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# HVAC DECISION LOGIC
# ============================================================

def decide_action(
    predicted_temp,
    current_actual_temp,
    threshold,
    strategy_mode
):
    """
    Decide whether cooling should be ON or OFF.

    predictive_auto:
        Uses predicted temperature directly.

    eco:
        Uses a wider hysteresis buffer.

    strict:
        Uses a smaller hysteresis buffer.

    override:
        Ignores the model and reacts to current actual temperature.
    """

    # --------------------------------------------------------
    # MANUAL OVERRIDE
    # --------------------------------------------------------

    if strategy_mode == "override":
        action = (
            "Cooling ON"
            if current_actual_temp > threshold
            else "Cooling OFF"
        )

        HVAC_SETTINGS["last_action"] = action

        return action


    # --------------------------------------------------------
    # BUFFER
    # --------------------------------------------------------

    buffer = STRATEGY_BUFFERS.get(
        strategy_mode,
        0
    )


    # --------------------------------------------------------
    # PREDICTIVE AUTO
    # --------------------------------------------------------

    if buffer == 0:

        action = (
            "Cooling ON"
            if predicted_temp > threshold
            else "Cooling OFF"
        )

        HVAC_SETTINGS["last_action"] = action

        return action


    # --------------------------------------------------------
    # HYSTERESIS
    # --------------------------------------------------------

    was_on = (
        HVAC_SETTINGS.get("last_action")
        == "Cooling ON"
    )

    if was_on:

        if predicted_temp < (
            threshold - buffer
        ):
            action = "Cooling OFF"
        else:
            action = "Cooling ON"

    else:

        if predicted_temp > (
            threshold + buffer
        ):
            action = "Cooling ON"
        else:
            action = "Cooling OFF"


    HVAC_SETTINGS["last_action"] = action

    return action


# ============================================================
# PERSISTED HVAC SETTINGS
# ============================================================

def load_persisted_settings():

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

    row = conn.execute(
        "SELECT * FROM hvac_controls WHERE id = 1"
    ).fetchone()

    conn.close()

    if row:

        HVAC_SETTINGS["target_temp"] = float(
            row["target_temp"]
        )

        HVAC_SETTINGS["min_comfort"] = float(
            row["min_comfort"]
        )

        HVAC_SETTINGS["max_comfort"] = float(
            row["max_comfort"]
        )

        HVAC_SETTINGS["strategy_mode"] = (
            row["strategy_mode"]
        )

        HVAC_SETTINGS["precool_enabled"] = bool(
            row["precool_enabled"]
        )


load_persisted_settings()


# ============================================================
# FEATURE LABELS
# ============================================================

FEATURE_LABELS = {
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend flag",
    "occupancy": "Occupancy now",
    "occupancy_rolling": "Occupancy trend",

    "temp_lag_1": "Previous 15-min temperature",
    "temp_lag_4": "Previous 1-hour temperature",
    "temp_lag_8": "Previous 2-hour temperature",

    "temp_rolling_4": "Recent 1-hour temperature average",
    "temp_rolling_8": "Recent 2-hour temperature average",
}


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

FEATURE_IMPORTANCES = sorted(
    [
        {
            "label": FEATURE_LABELS.get(
                feature,
                feature
            ),

            "feature": feature,

            "value": round(
                float(value),
                4
            ),
        }

        for feature, value in zip(
            FEATURES,
            model.feature_importances_
        )
    ],

    key=lambda item: item["value"],

    reverse=True,
)


# ============================================================
# LOAD RAW SENSOR DATA
# ============================================================

def load_all_sensor_data():

    conn = get_db_connection()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            timestamp,
            temperature,
            humidity,
            occupancy,
            day_of_week,
            is_weekend
        FROM readings
        ORDER BY timestamp
        """,
        conn
    )

    conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["temperature"] = pd.to_numeric(
        df["temperature"],
        errors="coerce"
    )

    df["humidity"] = pd.to_numeric(
        df["humidity"],
        errors="coerce"
    )

    df["occupancy"] = pd.to_numeric(
        df["occupancy"],
        errors="coerce"
    ).fillna(0)

    df = df.dropna(
        subset=[
            "timestamp",
            "temperature"
        ]
    )

    return df


# ============================================================
# RESAMPLE TO MODEL SAMPLING RATE
# ============================================================

def resample_for_model(df):

    if df.empty:
        return df

    work = df.copy()

    work = work.sort_values(
        "timestamp"
    )

    work = work.set_index(
        "timestamp"
    )


    # --------------------------------------------------------
    # 15-MINUTE MODEL GRID
    # --------------------------------------------------------

    model_df = work.resample(
        "15min"
    ).agg({
        "temperature": "mean",
        "humidity": "mean",
        "occupancy": "last",
    })


    model_df = model_df.dropna(
        subset=["temperature"]
    )


    # Occupancy can be missing in an empty interval.
    model_df["occupancy"] = (
        model_df["occupancy"]
        .ffill()
        .fillna(0)
    )


    # Recalculate calendar information
    # from the resampled timestamp.
    model_df["day_of_week"] = (
        model_df.index.dayofweek
    )

    model_df["is_weekend"] = (
        model_df["day_of_week"] >= 5
    ).astype(int)


    return model_df.reset_index()


# ============================================================
# CREATE MODEL FEATURES
# ============================================================

def create_model_features(df):

    """
    Creates exactly the same feature structure used
    during model training.

    Thermal memory features:

        temp_lag_1
            Previous 15-minute temperature

        temp_lag_4
            Previous 1-hour temperature

        temp_lag_8
            Previous 2-hour temperature

        temp_rolling_4
            Previous 1-hour average

        temp_rolling_8
            Previous 2-hour average
    """

    if df.empty:
        return df

    result = df.copy()

    result = result.sort_values(
        "timestamp"
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    result["hour"] = (
        result["timestamp"].dt.hour
        + result["timestamp"].dt.minute / 60
    )


    result["day_of_week"] = (
        result["timestamp"].dt.dayofweek
    )


    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)


    # --------------------------------------------------------
    # OCCUPANCY FEATURES
    # --------------------------------------------------------

    result["occupancy_rolling"] = (
        result["occupancy"]
        .rolling(
            window=8,
            min_periods=1
        )
        .mean()
    )


    # --------------------------------------------------------
    # THERMAL MEMORY
    # --------------------------------------------------------

    result["temp_lag_1"] = (
        result["temperature"].shift(1)
    )

    result["temp_lag_4"] = (
        result["temperature"].shift(4)
    )

    result["temp_lag_8"] = (
        result["temperature"].shift(8)
    )


    # --------------------------------------------------------
    # TEMPERATURE ROLLING AVERAGES
    #
    # shift(1) prevents the current temperature from
    # leaking into its own prediction.
    # --------------------------------------------------------

    previous_temperature = (
        result["temperature"].shift(1)
    )


    result["temp_rolling_4"] = (
        previous_temperature
        .rolling(
            window=4,
            min_periods=4
        )
        .mean()
    )


    result["temp_rolling_8"] = (
        previous_temperature
        .rolling(
            window=8,
            min_periods=8
        )
        .mean()
    )


    return result


# ============================================================
# BUILD MODEL DATA
# ============================================================

def get_model_dataframe():

    raw_df = load_all_sensor_data()

    if raw_df.empty:
        return raw_df

    model_df = resample_for_model(
        raw_df
    )

    if model_df.empty:
        return model_df

    model_df = create_model_features(
        model_df
    )

    return model_df


# ============================================================
# RECENT MODEL DATA
# ============================================================

def load_recent_data(hours=48):

    df = get_model_dataframe()

    if df.empty:
        return df

    cutoff = (
        df["timestamp"].max()
        - pd.Timedelta(hours=hours)
    )

    return df[
        df["timestamp"] >= cutoff
    ].copy()


# ============================================================
# MAIN DASHBOARD
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active_page="dashboard"
    )


# ============================================================
# API - HISTORY
# ============================================================

@app.route("/api/history")
def api_history():

    df = load_recent_data(
        hours=48
    )


    if df.empty:

        return jsonify({
            "timestamps": [],
            "actual": [],
            "predicted": [],
            "occupancy": []
        })


    # Only rows containing every required
    # thermal-memory feature can be predicted.
    usable = df.dropna(
        subset=FEATURES
    ).copy()


    if usable.empty:

        return jsonify({
            "timestamps": [],
            "actual": [],
            "predicted": [],
            "occupancy": []
        })


    X = usable[FEATURES]


    usable["predicted"] = (
        model.predict(X)
    )


    return jsonify({

        "timestamps":
            usable["timestamp"]
            .dt.strftime(
                "%m-%d %H:%M"
            )
            .tolist(),

        "actual":
            usable["temperature"]
            .round(2)
            .tolist(),

        "predicted":
            usable["predicted"]
            .round(2)
            .tolist(),

        "occupancy":
            usable["occupancy"]
            .astype(int)
            .tolist(),
    })


# ============================================================
# BUILD ONE MODEL FEATURE ROW
# ============================================================

def build_feature_row(
    history_df,
    timestamp,
    temperature,
    occupancy
):

    """
    Adds one temperature point to a temporary model history
    and creates the feature row required by the Random Forest.
    """

    if history_df.empty:

        temp_history = pd.DataFrame([
            {
                "timestamp": timestamp,
                "temperature": temperature,
                "humidity": np.nan,
                "occupancy": occupancy,
            }
        ])

    else:

        temp_history = history_df[
            [
                "timestamp",
                "temperature",
                "humidity",
                "occupancy",
            ]
        ].copy()

        temp_history = pd.concat(
            [
                temp_history,

                pd.DataFrame([
                    {
                        "timestamp": timestamp,
                        "temperature": temperature,
                        "humidity": np.nan,
                        "occupancy": occupancy,
                    }
                ])
            ],
            ignore_index=True
        )


    temp_history = (
        temp_history
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .reset_index(drop=True)
    )


    featured = create_model_features(
        temp_history
    )


    row = featured[
        featured["timestamp"] == timestamp
    ]


    if row.empty:
        return None


    row = row.iloc[-1]


    if row[FEATURES].isna().any():
        return None


    return pd.DataFrame(
        [row[FEATURES].to_dict()]
    )[FEATURES]


# ============================================================
# PREDICT FUTURE TEMPERATURE
# ============================================================

def predict_future_temperature(
    history_df,
    future_timestamp,
    occupancy
):

    """
    Predict one future 15-minute temperature.

    The model uses the existing thermal state
    and the predicted temperature is appended to
    the temporary history.

    This allows the next-hour prediction to be
    generated recursively.
    """

    if history_df.empty:
        return None


    history_df = history_df.copy()

    history_df = history_df.sort_values(
        "timestamp"
    )


    latest_temperature = float(
        history_df.iloc[-1]["temperature"]
    )


    # Build features using the latest known temperature.
    feature_row = build_feature_row(
        history_df=history_df,
        timestamp=future_timestamp,
        temperature=latest_temperature,
        occupancy=occupancy
    )


    if feature_row is None:
        return None


    predicted = float(
        model.predict(feature_row)[0]
    )


    return predicted


# ============================================================
# API - NEXT HOUR
# ============================================================

@app.route("/api/next_hour")
def api_next_hour():

    model_df = get_model_dataframe()


    if model_df.empty:

        return jsonify({
            "error":
                "No sensor readings available."
        }), 404


    model_df = model_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


    # Need enough history for the 2-hour features.
    usable_history = model_df.dropna(
        subset=FEATURES
    )


    if usable_history.empty:

        return jsonify({
            "error":
                "Not enough thermal history for prediction."
        }), 404


    history = model_df[
        model_df["timestamp"]
        >= model_df["timestamp"].max()
        - pd.Timedelta(hours=3)
    ].copy()


    last_row = model_df.iloc[-1]


    current_temperature = float(
        last_row["temperature"]
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The model operates on a 15-minute grid.
    # We recursively predict four 15-minute steps:
    #
    # +15 min
    # +30 min
    # +45 min
    # +60 min
    #
    # The final value becomes the next-hour prediction.
    # --------------------------------------------------------

    future_history = history.copy()

    future_temperature = current_temperature


    for step in range(1, 5):

        future_timestamp = (
            last_row["timestamp"]
            + pd.Timedelta(
                minutes=15 * step
            )
        )


        # We don't have a separate occupancy prediction
        # model yet, so use the latest observed occupancy
        # as the short-horizon assumption.
        future_occupancy = int(
            last_row["occupancy"]
        )


        feature_history = future_history.copy()


        # Use the predicted temperature from the previous
        # step as the latest thermal state.
        feature_row = build_feature_row(
            history_df=feature_history,
            timestamp=future_timestamp,
            temperature=future_temperature,
            occupancy=future_occupancy
        )


        if feature_row is None:

            return jsonify({
                "error":
                    "Unable to build thermal-memory features."
            }), 500


        future_temperature = float(
            model.predict(feature_row)[0]
        )


        # Add the predicted point to the recursive history.
        future_history = pd.concat(
            [
                future_history,

                pd.DataFrame([
                    {
                        "timestamp":
                            future_timestamp,

                        "temperature":
                            future_temperature,

                        "humidity":
                            np.nan,

                        "occupancy":
                            future_occupancy,

                        "day_of_week":
                            future_timestamp.dayofweek,

                        "is_weekend":
                            int(
                                future_timestamp.weekday()
                                >= 5
                            ),
                    }
                ])
            ],
            ignore_index=True
        )


    predicted_temp = future_temperature


    # --------------------------------------------------------
    # CONTROL DECISION
    # --------------------------------------------------------

    threshold = float(
        HVAC_SETTINGS["target_temp"]
    )


    action = decide_action(
        predicted_temp=predicted_temp,
        current_actual_temp=current_temperature,
        threshold=threshold,
        strategy_mode=HVAC_SETTINGS[
            "strategy_mode"
        ]
    )


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return jsonify({

        "predicted_temp":
            round(
                predicted_temp,
                2
            ),

        "threshold":
            threshold,

        "action":
            action,

        "strategy_mode":
            HVAC_SETTINGS[
                "strategy_mode"
            ],

        "occupancy_now":
            bool(
                last_row["occupancy"]
            ),

        "occupancy_rolling_pct":
            round(
                float(
                    last_row[
                        "occupancy_rolling"
                    ]
                ) * 100,
                1
            ),

        "feature_importances":
            FEATURE_IMPORTANCES,

        "model_mae":
            MODEL_MAE,

        "model_rmse":
            MODEL_RMSE,

        "baseline_mae":
            BASELINE_MAE,

        "trained_on_rows":
            TRAINED_ON_ROWS,

        "prediction_horizon":
            "60 minutes",

        "model_interval":
            "15 minutes",
    })


# ============================================================
# API - WEEKLY THERMAL MEMORY
# ============================================================

@app.route("/api/weekly_pattern")
def api_weekly_pattern():

    conn = get_db_connection()

    df = pd.read_sql_query(
        "SELECT * FROM readings",
        conn
    )

    conn.close()


    if df.empty:

        return jsonify({

            "cells": [],

            "day_labels": [
                "Mon",
                "Tue",
                "Wed",
                "Thu",
                "Fri",
                "Sat",
                "Sun"
            ],

            "insight":
                "No historical data available yet.",

            "temp_min": 0,
            "temp_max": 0,
        })


    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )


    df["hour_bucket"] = (
        df["timestamp"].dt.hour
    )


    grouped = (
        df.groupby(
            [
                "day_of_week",
                "hour_bucket"
            ]
        )
        .agg(
            avg_temp=(
                "temperature",
                "mean"
            ),

            avg_occupancy=(
                "occupancy",
                "mean"
            )
        )
        .reset_index()
    )


    day_names = [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun"
    ]


    day_names_full = [
        "Mondays",
        "Tuesdays",
        "Wednesdays",
        "Thursdays",
        "Fridays",
        "Saturdays",
        "Sundays"
    ]


    cells = []


    for _, row in grouped.iterrows():

        cells.append({

            "day":
                int(
                    row["day_of_week"]
                ),

            "day_label":
                day_names[
                    int(
                        row["day_of_week"]
                    )
                ],

            "hour":
                int(
                    row["hour_bucket"]
                ),

            "avg_temp":
                round(
                    float(
                        row["avg_temp"]
                    ),
                    2
                ),

            "avg_occupancy":
                round(
                    float(
                        row["avg_occupancy"]
                    ),
                    2
                ),
        })


    hottest = grouped.loc[
        grouped["avg_temp"].idxmax()
    ]


    coolest = grouped.loc[
        grouped["avg_temp"].idxmin()
    ]


    occ_temp = (
        df.groupby(
            "occupancy"
        )["temperature"]
        .mean()
    )


    occ_diff = None


    if (
        1 in occ_temp.index
        and 0 in occ_temp.index
    ):

        occ_diff = round(
            float(
                occ_temp[1]
                - occ_temp[0]
            ),
            2
        )


    insight = (
        f"Around "
        f"{day_names_full[int(hottest['day_of_week'])]}, "
        f"this room typically peaks near "
        f"{int(hottest['hour_bucket']):02d}:00 "
        f"({hottest['avg_temp']:.1f}°C) "
        f"and is coolest near "
        f"{int(coolest['hour_bucket']):02d}:00 "
        f"({coolest['avg_temp']:.1f}°C)."
    )


    if occ_diff is not None:

        insight += (
            f" It also runs about "
            f"{abs(occ_diff):.1f}°C "
            f"{'warmer' if occ_diff > 0 else 'cooler'} "
            f"when occupied."
        )


    return jsonify({

        "cells":
            cells,

        "day_labels":
            day_names,

        "insight":
            insight,

        "temp_min":
            round(
                float(
                    grouped[
                        "avg_temp"
                    ].min()
                ),
                1
            ),

        "temp_max":
            round(
                float(
                    grouped[
                        "avg_temp"
                    ].max()
                ),
                1
            ),
    })


# ============================================================
# PAGE - CONTROLS
# ============================================================

@app.route("/controls")
def controls_page():

    status_label = (
        "Cooling Active"
        if HVAC_SETTINGS["status"]
        not in (
            "Cooling Standby",
            "Comfort Corridor Holding"
        )
        else HVAC_SETTINGS["status"]
    )


    view_settings = dict(
        HVAC_SETTINGS
    )


    # Names expected by the current controls template.
    view_settings["corridor_min"] = (
        HVAC_SETTINGS["min_comfort"]
    )

    view_settings["corridor_max"] = (
        HVAC_SETTINGS["max_comfort"]
    )

    view_settings["corridor_span"] = round(
        HVAC_SETTINGS["max_comfort"]
        - HVAC_SETTINGS["min_comfort"],
        1
    )


    view_settings["status"] = (
        f"{status_label} "
        f"({HVAC_SETTINGS['target_temp']}°C)"
    )


    return render_template(
        "controls.html",
        settings=view_settings
    )


# ============================================================
# HISTORY PAGE DATA
# ============================================================

def get_history_data(limit=50):

    model_df = get_model_dataframe()


    if model_df.empty:

        return [], {
            "total_records": 0,
            "mae": MODEL_MAE
            if MODEL_MAE is not None
            else 0
        }


    model_df = model_df.sort_values(
        "timestamp"
    )


    # Keep the latest model-compatible records.
    usable = model_df.dropna(
        subset=FEATURES
    ).copy()


    usable = usable.tail(
        limit
    )


    if not usable.empty:

        usable["predicted_temp"] = (
            model.predict(
                usable[FEATURES]
            )
        )

        usable["delta"] = (
            usable["temperature"]
            - usable["predicted_temp"]
        )


    readings = []


    for _, row in usable.sort_values(
        "timestamp",
        ascending=False
    ).iterrows():

        readings.append({

            "timestamp":
                row["timestamp"].strftime(
                    "%Y-%m-%d %H:%M"
                ),

            "temperature":
                round(
                    float(
                        row["temperature"]
                    ),
                    2
                ),

            "predicted_temp":
                round(
                    float(
                        row["predicted_temp"]
                    ),
                    2
                ),

            "delta":
                round(
                    float(
                        row["delta"]
                    ),
                    2
                ),

            "occupancy":
                int(
                    row["occupancy"]
                ),
        })


    if readings:

        recent_mae = round(
            float(
                usable["delta"]
                .abs()
                .mean()
            ),
            3
        )

    else:

        recent_mae = (
            MODEL_MAE
            if MODEL_MAE is not None
            else 0
        )


    conn = get_db_connection()

    total_records = conn.execute(
        "SELECT COUNT(*) FROM readings"
    ).fetchone()[0]

    conn.close()


    summary = {

        "total_records":
            total_records,

        "mae":
            recent_mae,

        "model_mae":
            MODEL_MAE,

        "model_rmse":
            MODEL_RMSE,
    }


    return readings, summary


# ============================================================
# HISTORY PAGE
# ============================================================

@app.route("/history")
def history_page():

    readings, summary = (
        get_history_data()
    )


    return render_template(
        "history.html",
        readings=readings,
        summary=summary,
        active_page="history"
    )


# ============================================================
# API - HISTORY TABLE
# ============================================================

@app.route("/api/readings")
def api_readings():

    readings, summary = (
        get_history_data()
    )


    return jsonify({

        "readings":
            readings,

        "summary":
            summary
    })


# ============================================================
# HOW IT WORKS
# ============================================================

@app.route("/how-it-works")
def how_it_works_page():

    model_metadata = {

        "checkpoint":
            MODEL_PATH,

        "estimators":
            getattr(
                model,
                "n_estimators",
                "N/A"
            ),

        "mae":
            MODEL_MAE,

        "rmse":
            MODEL_RMSE,

        "baseline_mae":
            BASELINE_MAE,

        "trained_on_rows":
            TRAINED_ON_ROWS,

        "features":
            FEATURE_IMPORTANCES,
    }


    return render_template(
        "how_it_works.html",
        meta=model_metadata,
        active_page="how-it-works"
    )


# ============================================================
# API - LIVE DATA
# ============================================================

@app.route("/api/live")
def api_live():

    raw_df = load_all_sensor_data()


    if raw_df.empty:

        return jsonify({

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "temperature":
                0,

            "predicted":
                0,

            "delta":
                0,

            "occupancy":
                False,

            "hvac_status":
                "No Sensor Data",

            "corridor": {

                "target":
                    HVAC_SETTINGS[
                        "target_temp"
                    ],

                "min":
                    HVAC_SETTINGS[
                        "min_comfort"
                    ],

                "max":
                    HVAC_SETTINGS[
                        "max_comfort"
                    ],
            }
        })


    raw_df = raw_df.sort_values(
        "timestamp"
    )


    last_row = raw_df.iloc[-1]


    current_timestamp = (
        last_row["timestamp"]
    )


    current_temp = float(
        last_row["temperature"]
    )


    occupancy = int(
        last_row["occupancy"]
    )


    # --------------------------------------------------------
    # MODEL PREDICTION FOR CURRENT THERMAL STATE
    # --------------------------------------------------------

    model_df = resample_for_model(
        raw_df
    )


    featured = create_model_features(
        model_df
    )


    usable = featured.dropna(
        subset=FEATURES
    )


    if usable.empty:

        predicted = current_temp

    else:

        current_model_row = (
            usable.iloc[-1]
        )

        predicted = float(
            model.predict(
                pd.DataFrame(
                    [current_model_row[FEATURES]]
                )
            )[0]
        )


    # --------------------------------------------------------
    # CONTROL DECISION
    # --------------------------------------------------------

    threshold = float(
        HVAC_SETTINGS[
            "target_temp"
        ]
    )


    action = decide_action(
        predicted_temp=predicted,
        current_actual_temp=current_temp,
        threshold=threshold,
        strategy_mode=HVAC_SETTINGS[
            "strategy_mode"
        ]
    )


    if action == "Cooling ON":

        if HVAC_SETTINGS[
            "precool_enabled"
        ]:

            status = "Cooling Active"

        else:

            status = "Cooling Active"

    else:

        status = "Cooling Standby"


    HVAC_SETTINGS["status"] = status


    # --------------------------------------------------------
    # SENSOR AGE
    # --------------------------------------------------------

    now = pd.Timestamp.now()

    sensor_age_seconds = max(
        0,
        int(
            (
                now
                - current_timestamp
            ).total_seconds()
        )
    )


    # --------------------------------------------------------
    # OCCUPANCY TREND
    # --------------------------------------------------------

    recent_model = load_recent_data(
        hours=2
    )


    if recent_model.empty:

        occupancy_rolling_pct = 0

    else:

        occupancy_rolling_pct = round(
            float(
                recent_model[
                    "occupancy"
                ]
                .tail(8)
                .mean()
            ) * 100,
            1
        )


    return jsonify({

        "timestamp":
            current_timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "temperature":
            round(
                current_temp,
                2
            ),

        "predicted":
            round(
                predicted,
                2
            ),

        "delta":
            round(
                current_temp
                - predicted,
                2
            ),

        "occupancy":
            bool(
                occupancy
            ),

        "occupancy_rolling_pct":
            occupancy_rolling_pct,

        "hvac_status":
            status,

        "action":
            action,

        "strategy_mode":
            HVAC_SETTINGS[
                "strategy_mode"
            ],

        "sensor_age_seconds":
            sensor_age_seconds,

        "corridor": {

            "target":
                HVAC_SETTINGS[
                    "target_temp"
                ],

            "min":
                HVAC_SETTINGS[
                    "min_comfort"
                ],

            "max":
                HVAC_SETTINGS[
                    "max_comfort"
                ],
        }
    })


# ============================================================
# API - RECEIVE SENSOR DATA
# ============================================================

@app.route(
    "/api/record",
    methods=["POST"]
)
def api_record_sensor():

    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({
            "status":
                "error",

            "message":
                "No JSON data received."
        }), 400


    try:

        temperature = float(
            data.get("temp")
        )

        humidity = float(
            data.get("rh", 0)
        )

        occupancy = int(
            data.get("pir", 0)
        )

    except (
        TypeError,
        ValueError
    ):

        return jsonify({

            "status":
                "error",

            "message":
                "Invalid sensor values."
        }), 400


    # --------------------------------------------------------
    # BASIC SENSOR VALIDATION
    # --------------------------------------------------------

    if not (
        0 <= temperature <= 60
    ):

        return jsonify({

            "status":
                "error",

            "message":
                "Temperature must be between 0°C and 60°C."
        }), 400


    if not (
        0 <= humidity <= 100
    ):

        return jsonify({

            "status":
                "error",

            "message":
                "Humidity must be between 0% and 100%."
        }), 400


    if occupancy not in (
        0,
        1
    ):

        return jsonify({

            "status":
                "error",

            "message":
                "Occupancy must be 0 or 1."
        }), 400


    zone = data.get(
        "zone",
        "Main Living Area"
    )


    now = datetime.now()

    day_of_week = (
        now.weekday()
    )

    is_weekend = int(
        day_of_week >= 5
    )


    conn = get_db_connection()

    cursor = conn.cursor()


    cursor.execute(
        """
        INSERT INTO readings
        (
            timestamp,
            temperature,
            humidity,
            occupancy,
            day_of_week,
            is_weekend
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,

        (
            now.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            temperature,
            humidity,
            occupancy,
            day_of_week,
            is_weekend,
        )
    )


    conn.commit()

    conn.close()


    return jsonify({

        "status":
            "success",

        "logged_temp":
            temperature,

        "humidity":
            humidity,

        "occupancy":
            occupancy,

        "zone":
            zone,

        "timestamp":
            now.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
    }), 201


# ============================================================
# API - UPDATE HVAC SETTINGS
# ============================================================

@app.route(
    "/api/settings/update",
    methods=["POST"]
)
def api_update_settings():

    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({

            "status":
                "error",

            "message":
                "No settings received."
        }), 400


    # Work on temporary values first.
    # This prevents partially updating settings
    # when one value is invalid.

    new_target = float(
        HVAC_SETTINGS[
            "target_temp"
        ]
    )

    new_min = float(
        HVAC_SETTINGS[
            "min_comfort"
        ]
    )

    new_max = float(
        HVAC_SETTINGS[
            "max_comfort"
        ]
    )

    new_mode = HVAC_SETTINGS[
        "strategy_mode"
    ]

    new_precool = HVAC_SETTINGS[
        "precool_enabled"
    ]


    try:

        if "target_temp" in data:

            new_target = round(
                float(
                    data[
                        "target_temp"
                    ]
                ),
                1
            )


        if "min_comfort" in data:

            new_min = round(
                float(
                    data[
                        "min_comfort"
                    ]
                ),
                1
            )


        if "max_comfort" in data:

            new_max = round(
                float(
                    data[
                        "max_comfort"
                    ]
                ),
                1
            )


        if "strategy_mode" in data:

            if (
                data[
                    "strategy_mode"
                ]
                not in ALLOWED_MODES
            ):

                return jsonify({

                    "status":
                        "error",

                    "message":
                        "Invalid strategy mode."
                }), 400


            new_mode = data[
                "strategy_mode"
            ]


        if "precool_enabled" in data:

            value = data[
                "precool_enabled"
            ]

            if isinstance(
                value,
                bool
            ):

                new_precool = value

            else:

                new_precool = (
                    str(value).lower()
                    in (
                        "true",
                        "1",
                        "yes",
                        "on"
                    )
                )


    except (
        TypeError,
        ValueError
    ):

        return jsonify({

            "status":
                "error",

            "message":
                "Invalid control value."
        }), 400


    # --------------------------------------------------------
    # RANGE VALIDATION
    # --------------------------------------------------------

    if not (
        18.0
        <= new_min
        <= 32.0
    ):

        return jsonify({

            "status":
                "error",

            "message":
                "Minimum comfort temperature must be between 18°C and 32°C."
        }), 400


    if not (
        18.0
        <= new_target
        <= 32.0
    ):

        return jsonify({

            "status":
                "error",

            "message":
                "Target temperature must be between 18°C and 32°C."
        }), 400


    if not (
        18.0
        <= new_max
        <= 32.0
    ):

        return jsonify({

            "status":
                "error",

            "message":
                "Maximum comfort temperature must be between 18°C and 32°C."
        }), 400


    if new_min >= new_max:

        return jsonify({

            "status":
                "error",

            "message":
                "Minimum comfort temperature must be lower than maximum comfort temperature."
        }), 400


    if not (
        new_min
        <= new_target
        <= new_max
    ):

        return jsonify({

            "status":
                "error",

            "message":
                "Target temperature must be inside the comfort corridor."
        }), 400


    # --------------------------------------------------------
    # APPLY SETTINGS
    # --------------------------------------------------------

    HVAC_SETTINGS[
        "target_temp"
    ] = new_target

    HVAC_SETTINGS[
        "min_comfort"
    ] = new_min

    HVAC_SETTINGS[
        "max_comfort"
    ] = new_max

    HVAC_SETTINGS[
        "strategy_mode"
    ] = new_mode

    HVAC_SETTINGS[
        "precool_enabled"
    ] = new_precool


    # --------------------------------------------------------
    # SAVE TO SQLITE
    # --------------------------------------------------------

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


    conn.execute(
        """
        INSERT OR REPLACE INTO hvac_controls
        (
            id,
            target_temp,
            min_comfort,
            max_comfort,
            strategy_mode,
            precool_enabled,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,

        (
            1,

            HVAC_SETTINGS[
                "target_temp"
            ],

            HVAC_SETTINGS[
                "min_comfort"
            ],

            HVAC_SETTINGS[
                "max_comfort"
            ],

            HVAC_SETTINGS[
                "strategy_mode"
            ],

            1
            if HVAC_SETTINGS[
                "precool_enabled"
            ]
            else 0,

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        )
    )


    conn.commit()

    conn.close()


    return jsonify({

        "status":
            "synchronized",

        "settings":
            HVAC_SETTINGS
    })


# ============================================================
# API - CSV EXPORT
# ============================================================

@app.route("/api/export/csv")
def export_telemetry_csv():

    conn = get_db_connection()


    rows = conn.execute(
        """
        SELECT
            timestamp,
            temperature,
            humidity,
            occupancy,
            day_of_week,
            is_weekend
        FROM readings
        ORDER BY timestamp DESC
        """
    ).fetchall()


    conn.close()


    output = io.StringIO()

    writer = csv.writer(
        output
    )


    writer.writerow([
        "Timestamp",
        "Temperature (°C)",
        "Humidity (%)",
        "Occupancy",
        "Day of Week",
        "Weekend"
    ])


    for row in rows:

        writer.writerow([

            row["timestamp"],

            row["temperature"],

            row["humidity"],

            row["occupancy"],

            row["day_of_week"],

            row["is_weekend"],
        ])


    output.seek(0)


    return Response(

        output.getvalue(),

        mimetype="text/csv",

        headers={
            "Content-Disposition":
                "attachment; "
                "filename=thermal_telemetry.csv"
        }
    )


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print(
        "Predictive Thermal Memory Server"
    )

    print(
        "Open: http://127.0.0.1:5000/"
    )

    app.run(
        debug=True
    )