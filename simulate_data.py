"""
simulate_data.py

Generates synthetic historical sensor data for the
Predictive Thermal Memory System.

The simulator produces 60 days of readings at
15-minute intervals.

Each reading uses:
    - Time of day
    - Weekday/weekend
    - Occupancy
    - Day-specific weather variation
    - Previous room temperature
    - Environmental noise

The previous temperature creates thermal inertia,
which gives the dataset genuine temporal/thermal memory.
"""

import argparse
import sqlite3
from datetime import datetime, timedelta

import numpy as np

from sensor_model import (
    generate_reading,
    daily_weather_offset
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DB_PATH = "thermal_memory.db"

DAYS_TO_SIMULATE = 60

INTERVAL_MINUTES = 15


# ---------------------------------------------------------
# Generate historical readings
# ---------------------------------------------------------

def generate_readings(rng):
    """
    Generate historical sensor readings.

    Returns
    -------
    list
        List of tuples ready to be inserted into SQLite.
    """

    readings = []

    # Start 60 days in the past.

    start_time = (
        datetime.now()
        - timedelta(days=DAYS_TO_SIMULATE)
    )

    current_time = start_time

    # This stores the previous indoor temperature.

    previous_temperature = None

    # Number of readings required.

    total_steps = int(
        DAYS_TO_SIMULATE
        * 24
        * 60
        / INTERVAL_MINUTES
    )

    for _ in range(total_steps):

        # -------------------------------------------------
        # Current time information
        # -------------------------------------------------

        hour = (
            current_time.hour
            + current_time.minute / 60
        )

        day_of_week = current_time.weekday()

        is_weekend = day_of_week >= 5

        # -------------------------------------------------
        # Day-specific weather condition
        # -------------------------------------------------

        weather_offset = daily_weather_offset(
            current_time.date()
        )

        # -------------------------------------------------
        # Generate sensor reading
        # -------------------------------------------------

        temp, humidity, occupancy = generate_reading(
            hour=hour,
            is_weekend=is_weekend,
            rng=rng,
            previous_temperature=previous_temperature,
            weather_offset=weather_offset
        )

        # -------------------------------------------------
        # Store reading
        # -------------------------------------------------

        readings.append(
            (
                current_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                temp,
                humidity,
                occupancy,
                day_of_week,
                int(is_weekend)
            )
        )

        # -------------------------------------------------
        # Thermal memory
        # -------------------------------------------------

        # The current temperature becomes the previous
        # temperature for the next 15-minute reading.

        previous_temperature = temp

        # -------------------------------------------------
        # Move to next sensor reading
        # -------------------------------------------------

        current_time += timedelta(
            minutes=INTERVAL_MINUTES
        )

    return readings


# ---------------------------------------------------------
# Create database
# ---------------------------------------------------------

def create_database(conn):
    """
    Create the readings table if it does not exist.
    """

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            occupancy INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL,
            is_weekend INTEGER NOT NULL
        )
    """)

    conn.commit()


# ---------------------------------------------------------
# Insert generated data
# ---------------------------------------------------------

def insert_readings(conn, readings):
    """
    Replace existing readings with the newly generated
    historical dataset.
    """

    cur = conn.cursor()

    # Remove previous simulation data.

    cur.execute("DELETE FROM readings")

    # Insert new data.

    cur.executemany("""
        INSERT INTO readings (
            timestamp,
            temperature,
            humidity,
            occupancy,
            day_of_week,
            is_weekend
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, readings)

    conn.commit()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate historical thermal sensor data."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for reproducible "
            "sensor noise and occupancy."
        )
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Random generator
    # -----------------------------------------------------

    if args.seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(args.seed)

    print(
        f"Generating {DAYS_TO_SIMULATE} days "
        f"of thermal sensor data..."
    )

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    readings = generate_readings(rng)

    print(
        f"Generated {len(readings)} readings."
    )

    # -----------------------------------------------------
    # Database
    # -----------------------------------------------------

    conn = sqlite3.connect(DB_PATH)

    try:

        create_database(conn)

        insert_readings(
            conn,
            readings
        )

    finally:

        conn.close()

    # -----------------------------------------------------
    # Verification
    # -----------------------------------------------------

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM readings"
    )

    count = cur.fetchone()[0]

    cur.execute("""
        SELECT
            timestamp,
            temperature,
            humidity,
            occupancy
        FROM readings
        ORDER BY id DESC
        LIMIT 5
    """)

    latest_rows = cur.fetchall()

    conn.close()

    print(
        f"Database now contains {count} readings."
    )

    print("\nLatest readings:")

    for row in latest_rows:
        print(
            f"  {row[0]} | "
            f"temp={row[1]}°C | "
            f"humidity={row[2]}% | "
            f"occupancy={row[3]}"
        )

    print("\nHistorical simulation completed successfully.")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()