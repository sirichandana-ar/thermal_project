"""
sensor_model.py

Thermal environment simulator for the
Predictive Thermal Memory System.

The simulator models:
    - Daily temperature cycle
    - Day-to-day weather variation
    - Thermal inertia
    - Occupancy heat
    - Sensor/environment noise

The purpose is to create synthetic data that behaves
more like an indoor thermal environment rather than
being determined only by the clock.
"""

import numpy as np


# ---------------------------------------------------------
# Occupancy model
# ---------------------------------------------------------

def occupancy_probability(hour, is_weekend):
    """
    Estimate the probability that the room is occupied.
    """

    if is_weekend:

        # Weekend occupancy is generally higher
        # throughout the day.

        if 1 <= hour <= 6:
            return 0.90

        elif 7 <= hour <= 22:
            return 0.75

        else:
            return 0.60

    else:

        # Weekday pattern

        if 1 <= hour <= 6:
            return 0.90

        elif 7 <= hour <= 17:
            return 0.15

        elif 18 <= hour <= 22:
            return 0.85

        else:
            return 0.50


# ---------------------------------------------------------
# Outdoor/environment temperature
# ---------------------------------------------------------

def outdoor_temperature(hour, weather_offset=0.0):
    """
    Generate a daily outdoor temperature pattern.

    weather_offset represents the fact that one day can
    be slightly warmer or cooler than another day.

    Typical range remains compatible with the project's
    existing 18-32°C dashboard scale.
    """

    daily_cycle = 3.0 * np.sin(
        (hour - 7) / 24 * 2 * np.pi
    )

    return 27.0 + daily_cycle + weather_offset


# ---------------------------------------------------------
# Generate one sensor reading
# ---------------------------------------------------------

def generate_reading(
    hour,
    is_weekend,
    rng,
    previous_temperature=None,
    weather_offset=0.0
):
    """
    Generate one simulated indoor sensor reading.

    Parameters
    ----------
    hour : float
        Current hour of the day.

    is_weekend : bool
        True for Saturday/Sunday.

    rng : numpy random generator
        Random number generator used for occupancy,
        temperature noise and humidity noise.

    previous_temperature : float or None
        Temperature from the previous sensor reading.

    weather_offset : float
        Persistent temperature offset for the current day.

    Returns
    -------
    temperature, humidity, occupancy
    """

    # -----------------------------------------------------
    # 1. Determine occupancy
    # -----------------------------------------------------

    occ_prob = occupancy_probability(
        hour,
        is_weekend
    )

    occupancy = (
        1
        if rng.random() < occ_prob
        else 0
    )

    # -----------------------------------------------------
    # 2. Calculate environmental temperature
    # -----------------------------------------------------

    outdoor_temp = outdoor_temperature(
        hour,
        weather_offset
    )

    # -----------------------------------------------------
    # 3. Thermal inertia
    # -----------------------------------------------------

    if previous_temperature is None:

        # Initial indoor temperature.
        #
        # Start slightly cooler than the outdoor
        # environment because indoor temperature normally
        # does not instantly equal outdoor temperature.

        temperature = outdoor_temp - 1.5

    else:

        # The room gradually moves toward the surrounding
        # environmental temperature.
        #
        # Only 10% of the difference is transferred during
        # each 15-minute simulated interval.

        thermal_change = (
            outdoor_temp - previous_temperature
        ) * 0.10

        temperature = (
            previous_temperature
            + thermal_change
        )

    # -----------------------------------------------------
    # 4. Occupancy heat
    # -----------------------------------------------------

    if occupancy:
        temperature += 0.35

    # -----------------------------------------------------
    # 5. Environmental / sensor noise
    # -----------------------------------------------------

    temperature += rng.normal(
        0,
        0.20
    )

    # -----------------------------------------------------
    # 6. Keep temperature inside a sensible range
    # -----------------------------------------------------

    temperature = float(
        np.clip(
            temperature,
            18.0,
            32.0
        )
    )

    # -----------------------------------------------------
    # 7. Humidity
    # -----------------------------------------------------

    humidity = (
        55
        - 0.4 * (temperature - 26)
        + rng.normal(0, 2.0)
    )

    humidity = float(
        np.clip(
            humidity,
            30.0,
            85.0
        )
    )

    # -----------------------------------------------------
    # 8. Return sensor values
    # -----------------------------------------------------

    return (
        round(temperature, 2),
        round(humidity, 2),
        occupancy
    )


# ---------------------------------------------------------
# Generate a persistent daily weather variation
# ---------------------------------------------------------

def daily_weather_offset(date):
    """
    Generate a deterministic weather variation for a
    particular calendar day.

    The same date always receives the same offset, even
    if the simulator is restarted.

    This gives the dataset day-to-day variation that
    cannot be explained by hour alone.
    """

    # Use the calendar date to create a stable seed.

    seed = (
        date.toordinal() * 1009
        + 7919
    )

    rng = np.random.default_rng(seed)

    offset = rng.normal(
        0,
        0.7
    )

    # Prevent unrealistic extreme weather differences.

    offset = float(
        np.clip(
            offset,
            -1.5,
            1.5
        )
    )

    return offset