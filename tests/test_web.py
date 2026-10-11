import copy
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import app as thermal_app


class ThermalWebSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        thermal_app.app.config["TESTING"] = True
        cls.client = thermal_app.app.test_client()

    def setUp(self):
        self.original_settings = copy.deepcopy(
            thermal_app.HVAC_SETTINGS
        )
        self.database_directory = tempfile.TemporaryDirectory()
        self.database_path = str(
            Path(self.database_directory.name) / "test_thermal.db"
        )
        self.database_patch = patch.object(
            thermal_app,
            "DB_PATH",
            self.database_path,
        )
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)
        self.addCleanup(self.database_directory.cleanup)
        self.addCleanup(self.restore_hvac_settings)
        self.create_readings_table()

    def restore_hvac_settings(self):
        thermal_app.HVAC_SETTINGS.clear()
        thermal_app.HVAC_SETTINGS.update(self.original_settings)

    def create_readings_table(self):
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                """
                CREATE TABLE readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    humidity REAL NOT NULL,
                    occupancy INTEGER NOT NULL,
                    day_of_week INTEGER NOT NULL,
                    is_weekend INTEGER NOT NULL
                )
                """
            )
            now = datetime.now().replace(second=0, microsecond=0)
            start = now - timedelta(minutes=15 * 20)
            connection.executemany(
                """
                INSERT INTO readings (
                    timestamp, temperature, humidity, occupancy,
                    day_of_week, is_weekend
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        (start + timedelta(minutes=15 * index)).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        24.0 + index * 0.02,
                        45.0,
                        index % 2,
                        (start + timedelta(minutes=15 * index)).weekday(),
                        int(
                            (start + timedelta(minutes=15 * index)).weekday()
                            >= 5
                        ),
                    )
                    for index in range(20)
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def test_pages_load_with_shared_navigation(self):
        pages = (
            ("/", b"Dashboard"),
            ("/controls", b"Controls"),
            ("/history", b"History"),
            ("/how-it-works", b"Model Evaluation"),
        )

        for path, expected_content in pages:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(expected_content, response.data)
                self.assertIn(b"Primary navigation", response.data)
                self.assertIn(b'aria-current="page"', response.data)
                if path == "/":
                    self.assertIn(b"System health", response.data)

    def test_prediction_and_history_apis_return_data(self):
        history_response = self.client.get("/api/history")
        prediction_response = self.client.get("/api/next_hour")

        self.assertEqual(history_response.status_code, 200)
        self.assertEqual(prediction_response.status_code, 200)

        history = history_response.get_json()
        prediction = prediction_response.get_json()

        self.assertEqual(
            len(history["timestamps"]),
            len(history["actual"]),
        )
        self.assertEqual(
            len(history["actual"]),
            len(history["predicted"]),
        )
        self.assertIn("predicted_temp", prediction)
        self.assertIn("action", prediction)

    def test_model_page_uses_current_metrics(self):
        response = self.client.get("/how-it-works")
        self.assertEqual(response.status_code, 200)
        expected_mae = f"{thermal_app.MODEL_MAE:.4f}".encode()
        expected_baseline = (
            f"{thermal_app.BASELINE_MAE:.4f}".encode()
        )
        self.assertIn(expected_mae, response.data)
        self.assertIn(expected_baseline, response.data)
        self.assertIn(
            b"not a claim of real-home accuracy",
            response.data,
        )
        self.assertIn(b"held-out test set", response.data)

    def test_health_api_reports_dependencies_and_data_source(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        health = response.get_json()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["database"], "ready")
        self.assertEqual(health["model"], "ready")
        self.assertEqual(health["reading_count"], 20)
        self.assertIsNotNone(health["latest_reading"])
        self.assertEqual(health["data_source"], "simulated")

    def test_health_api_reports_database_failure(self):
        with patch.object(
            thermal_app,
            "get_db_connection",
            side_effect=sqlite3.OperationalError("database unavailable"),
        ):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "degraded")
        self.assertEqual(response.get_json()["database"], "unavailable")

    def test_record_api_rejects_invalid_sensor_values(self):
        invalid_payloads = (
            [],
            {"rh": 50, "pir": 0},
            {"temp": float("nan"), "rh": 50, "pir": 0},
            {"temp": 25, "rh": float("inf"), "pir": 0},
            {"temp": 25, "rh": 50, "pir": 1.5},
            {"temp": 25, "rh": 50, "pir": 0, "zone": " "},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post(
                    "/api/record",
                    json=payload,
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(
                    response.get_json()["status"],
                    "error",
                )

        connection = sqlite3.connect(self.database_path)
        try:
            count = connection.execute(
                "SELECT COUNT(*) FROM readings"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(count, 20)

    def test_record_api_accepts_valid_reading(self):
        response = self.client.post(
            "/api/record",
            json={
                "temp": 24.5,
                "rh": 46,
                "pir": 1,
                "zone": "  Living Room  ",
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["zone"], "Living Room")

        health = self.client.get("/api/health").get_json()
        self.assertEqual(health["reading_count"], 21)

    def test_settings_are_persisted_and_invalid_values_rejected(self):
        response = self.client.post(
            "/api/settings/update",
            json={
                "target_temp": 24.5,
                "min_comfort": 21,
                "max_comfort": 27,
                "strategy_mode": "eco",
                "precool_enabled": False,
            },
        )
        self.assertEqual(response.status_code, 200)

        connection = sqlite3.connect(self.database_path)
        try:
            saved = connection.execute(
                """
                SELECT target_temp, strategy_mode, precool_enabled
                FROM hvac_controls
                WHERE id = 1
                """
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(saved, (24.5, "eco", 0))

        invalid = self.client.post(
            "/api/settings/update",
            json={"target_temp": float("nan")},
        )
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(thermal_app.HVAC_SETTINGS["target_temp"], 24.5)

    def test_csv_export_has_expected_headers(self):
        response = self.client.get("/api/export/csv")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn(
            b"Timestamp,Temperature (",
            response.data,
        )


if __name__ == "__main__":
    unittest.main()
