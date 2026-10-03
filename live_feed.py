"""
live_feed.py

Simulates a live sensor feed for the
Predictive Thermal Memory System.

A new sensor reading is inserted into SQLite
at a configurable interval.

The live feed continues from the latest
temperature already stored in the database,
so the simulated room maintains thermal
continuity.

Usage:

    python live_feed.py

    python live_feed.py --interval 5

Press Ctrl+C to stop.
"""

import argparse
import sqlite3
import time
from datetime import datetime

import numpy as np

from sensor_model import (
    generate_reading,
    daily_weather_offset
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DB_PATH = "thermal_memory.db"


# ---------------------------------------------------------
# Get latest temperature
# ---------------------------------------------------------

def get_last_temperature():
    """
    Get the latest temperature stored in the database.

    This allows the live feed to continue from the
    historical simulation instead of starting from
    a fresh temperature state.
    """

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute("""
        SELECT temperature
        FROM readings
        ORDER BY id DESC
        LIMIT 1
    """)

    row = cur.fetchone()

    conn.close()

    if row is None:
        return None

    return float(row[0])


# ---------------------------------------------------------
# Insert one live reading
# ---------------------------------------------------------

def insert_live_reading(rng, previous_temperature):
    """
    Generate and insert one live sensor reading.

    Returns
    -------
    float
        The newly generated temperature.
    """

    now = datetime.now()

    # -----------------------------------------------------
    # Current time information
    # -----------------------------------------------------

    hour = (
        now.hour
        + now.minute / 60
    )

    day_of_week = now.weekday()

    is_weekend = day_of_week >= 5

    # -----------------------------------------------------
    # Current day's environmental condition
    # -----------------------------------------------------

    weather_offset = daily_weather_offset(
        now.date()
    )

    # -----------------------------------------------------
    # Generate sensor reading
    # -----------------------------------------------------

    temp, humidity, occupancy = generate_reading(
        hour=hour,
        is_weekend=is_weekend,
        rng=rng,
        previous_temperature=previous_temperature,
        weather_offset=weather_offset
    )

    # -----------------------------------------------------
    # Store in SQLite
    # -----------------------------------------------------

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute("""
        INSERT INTO readings (
            timestamp,
            temperature,
            humidity,
            occupancy,
            day_of_week,
            is_weekend
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        now.strftime("%Y-%m-%d %H:%M:%S"),
        temp,
        humidity,
        occupancy,
        day_of_week,
        int(is_weekend)
    ))

    conn.commit()

    conn.close()

    # -----------------------------------------------------
    # Terminal output
    # -----------------------------------------------------

    print(
        f"[{now.strftime('%H:%M:%S')}] "
        f"New reading -> "
        f"temp={temp}°C  "
        f"humidity={humidity}%  "
        f"occupancy={occupancy}"
    )

    return temp


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Start the live thermal sensor feed."
        )
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help=(
            "Seconds between readings "
            "(default: 10)"
        )
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Validate interval
    # -----------------------------------------------------

    if args.interval <= 0:

        print(
            "Error: interval must be greater than 0."
        )

        return

    # -----------------------------------------------------
    # Random generator
    # -----------------------------------------------------

    rng = np.random.default_rng()

    # -----------------------------------------------------
    # Continue from database state
    # -----------------------------------------------------

    previous_temperature = get_last_temperature()

    if previous_temperature is None:

        print(
            "No previous temperature found. "
            "The first reading will initialize "
            "the thermal state."
        )

    else:

        print(
            f"Continuing from last temperature: "
            f"{previous_temperature:.2f}°C"
        )

    # -----------------------------------------------------
    # Start feed
    # -----------------------------------------------------

    print(
        f"Starting live feed - "
        f"one reading every {args.interval} seconds."
    )

    print(
        "Press Ctrl+C to stop."
    )

    try:

        while True:

            previous_temperature = insert_live_reading(
                rng,
                previous_temperature
            )

            time.sleep(
                args.interval
            )

    except KeyboardInterrupt:

        print(
            "\nLive feed stopped."
        )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()