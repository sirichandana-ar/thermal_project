"""
train_model.py

Trains the thermal prediction model for the
Predictive Thermal Memory System.

The model uses:
    - Time of day
    - Day of week
    - Weekend indicator
    - Current occupancy
    - Recent occupancy history
    - Previous temperatures
    - Temperature rolling averages

A time-ordered train/test split is used so that
future readings are never used to train the model.

A persistence baseline is also calculated:

    predicted temperature = current temperature

This provides a simple reference against which the
Random Forest model can be evaluated.
"""

import sqlite3
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DB_PATH = "thermal_memory.db"

MODEL_PATH = "thermal_model.joblib"

TREE_PATH = "decision_tree.txt"

TEST_RATIO = 0.20


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def create_features(df):
    """
    Create all features used by the prediction model.

    IMPORTANT:
    This function will later also be used by app.py so
    training and live prediction use the same feature
    definitions.
    """

    df = df.copy()

    # -----------------------------------------------------
    # Timestamp
    # -----------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # -----------------------------------------------------
    # Time features
    # -----------------------------------------------------

    df["hour"] = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # -----------------------------------------------------
    # Occupancy history
    # -----------------------------------------------------

    # Four readings = approximately one hour because
    # the historical simulator produces readings every
    # 15 minutes.

    df["occupancy_rolling"] = (
        df["occupancy"]
        .rolling(
            window=8,
            min_periods=1
        )
        .mean()
    )

    # -----------------------------------------------------
    # Temperature memory
    # -----------------------------------------------------

    # Previous reading = 15 minutes ago

    df["temp_lag_1"] = (
        df["temperature"].shift(1)
    )

    # Four readings = 1 hour ago

    df["temp_lag_4"] = (
        df["temperature"].shift(4)
    )

    # Eight readings = 2 hours ago

    df["temp_lag_8"] = (
        df["temperature"].shift(8)
    )

    # -----------------------------------------------------
    # Temperature rolling averages
    # -----------------------------------------------------

    df["temp_rolling_4"] = (
        df["temperature"]
        .shift(1)
        .rolling(
            window=4,
            min_periods=4
        )
        .mean()
    )

    df["temp_rolling_8"] = (
        df["temperature"]
        .shift(1)
        .rolling(
            window=8,
            min_periods=8
        )
        .mean()
    )

    return df


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(actual, predicted):
    """
    Calculate MAE and RMSE.
    """

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    return float(mae), float(rmse)


# ---------------------------------------------------------
# Main training process
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("Predictive Thermal Memory System")
    print("Model Training")
    print("=" * 60)

    # -----------------------------------------------------
    # Load database
    # -----------------------------------------------------

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        """
        SELECT
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
        raise RuntimeError(
            "No readings found in thermal_memory.db. "
            "Run simulate_data.py first."
        )

    print(
        f"\nLoaded {len(df)} sensor readings."
    )

    # -----------------------------------------------------
    # Create features
    # -----------------------------------------------------

    df = create_features(df)

    # -----------------------------------------------------
    # Define model features
    # -----------------------------------------------------

    features = [
        "hour",
        "day_of_week",
        "is_weekend",
        "occupancy",
        "occupancy_rolling",
        "temp_lag_1",
        "temp_lag_4",
        "temp_lag_8",
        "temp_rolling_4",
        "temp_rolling_8"
    ]

    target = "temperature"

    # -----------------------------------------------------
    # Remove rows without sufficient history
    # -----------------------------------------------------

    model_df = df.dropna(
        subset=features + [target]
    ).copy()

    print(
        f"Usable rows after feature engineering: "
        f"{len(model_df)}"
    )

    # -----------------------------------------------------
    # Time-ordered train/test split
    # -----------------------------------------------------

    split_index = int(
        len(model_df)
        * (1 - TEST_RATIO)
    )

    train_df = model_df.iloc[
        :split_index
    ]

    test_df = model_df.iloc[
        split_index:
    ]

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    print(
        f"\nTraining rows: {len(train_df)}"
    )

    print(
        f"Testing rows:  {len(test_df)}"
    )

    print(
        f"Training period: "
        f"{train_df['timestamp'].min()} "
        f"-> "
        f"{train_df['timestamp'].max()}"
    )

    print(
        f"Testing period:  "
        f"{test_df['timestamp'].min()} "
        f"-> "
        f"{test_df['timestamp'].max()}"
    )

    # -----------------------------------------------------
    # Persistence baseline
    # -----------------------------------------------------

    # Simple forecasting rule:
    #
    # "The next temperature will be approximately
    #  the most recent temperature."

    persistence_predictions = (
        test_df["temp_lag_1"]
    )

    persistence_mae, persistence_rmse = (
        calculate_metrics(
            y_test,
            persistence_predictions
        )
    )

    print("\n" + "-" * 60)
    print("PERSISTENCE BASELINE")
    print("-" * 60)

    print(
        f"MAE :  {persistence_mae:.4f} °C"
    )

    print(
        f"RMSE:  {persistence_rmse:.4f} °C"
    )

    # -----------------------------------------------------
    # Decision Tree
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("DECISION TREE")
    print("-" * 60)

    tree_model = DecisionTreeRegressor(
        max_depth=6,
        random_state=42
    )

    tree_model.fit(
        X_train,
        y_train
    )

    tree_predictions = tree_model.predict(
        X_test
    )

    tree_mae, tree_rmse = calculate_metrics(
        y_test,
        tree_predictions
    )

    print(
        f"MAE :  {tree_mae:.4f} °C"
    )

    print(
        f"RMSE:  {tree_rmse:.4f} °C"
    )

    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("RANDOM FOREST")
    print("-" * 60)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    rf_mae, rf_rmse = calculate_metrics(
        y_test,
        predictions
    )

    print(
        f"MAE :  {rf_mae:.4f} °C"
    )

    print(
        f"RMSE:  {rf_rmse:.4f} °C"
    )

    # -----------------------------------------------------
    # Compare models
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        f"Persistence MAE : {persistence_mae:.4f} °C"
    )

    print(
        f"Decision Tree MAE: {tree_mae:.4f} °C"
    )

    print(
        f"Random Forest MAE: {rf_mae:.4f} °C"
    )

    # -----------------------------------------------------
    # Improvement over persistence
    # -----------------------------------------------------

    if persistence_mae > 0:

        improvement = (
            (
                persistence_mae
                - rf_mae
            )
            / persistence_mae
        ) * 100

    else:

        improvement = 0.0

    print(
        f"\nRandom Forest improvement "
        f"over persistence: "
        f"{improvement:.2f}%"
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n" + "-" * 60)
    print("FEATURE IMPORTANCE")
    print("-" * 60)

    for _, row in importance_df.iterrows():

        print(
            f"{row['feature']:20s} "
            f"{row['importance']:.4f}"
        )

    # -----------------------------------------------------
    # Save Decision Tree text
    # -----------------------------------------------------

    try:

        from sklearn.tree import export_text

        tree_text = export_text(
            tree_model,
            feature_names=features
        )

        with open(
            TREE_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(tree_text)

        print(
            f"\nDecision tree exported to "
            f"{TREE_PATH}"
        )

    except Exception as exc:

        print(
            f"\nWarning: could not export "
            f"decision tree: {exc}"
        )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_bundle = {
        "model": model,
        "features": features,

        "mae": rf_mae,
        "rmse": rf_rmse,

        "baseline_mae": persistence_mae,
        "baseline_rmse": persistence_rmse,

        "tree_mae": tree_mae,
        "tree_rmse": tree_rmse,

        "improvement_over_baseline_percent":
            float(improvement),

        "feature_importance":
            importance_df.to_dict(
                orient="records"
            ),

        "trained_on_rows":
            int(len(train_df)),

        "tested_on_rows":
            int(len(test_df)),

        "training_period": [
            str(train_df["timestamp"].min()),
            str(train_df["timestamp"].max())
        ],

        "testing_period": [
            str(test_df["timestamp"].min()),
            str(test_df["timestamp"].max())
        ],

        "sampling_interval_minutes":
            15
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    print(
        f"\nModel saved to "
        f"{MODEL_PATH}"
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Random Forest MAE : "
        f"{rf_mae:.4f} °C"
    )

    print(
        f"Random Forest RMSE: "
        f"{rf_rmse:.4f} °C"
    )

    print(
        f"Baseline MAE     : "
        f"{persistence_mae:.4f} °C"
    )

    print(
        f"Improvement       : "
        f"{improvement:.2f}%"
    )

    print(
        f"Features          : "
        f"{len(features)}"
    )


if __name__ == "__main__":
    main()