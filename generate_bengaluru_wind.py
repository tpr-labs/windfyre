#!/usr/bin/env python3
"""
Generate a self-contained Bengaluru wind and weather dashboard.

The script downloads historical hourly wind/weather data from Open-Meteo,
trains compact regression models, predicts every hour of 2026, and writes a
polished single-page HTML file with the prediction embedded.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np
import requests
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error


LATITUDE = 13.0143
LONGITUDE = 77.5355
TIMEZONE = "Asia/Kolkata"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TARGET_YEAR = 2026
TRAIN_START_YEAR = 2014
TRAIN_END_YEAR = 2025
OUTPUT_HTML = "bengaluru-wind-direction.html"


@dataclass(frozen=True)
class WindRow:
    timestamp: datetime
    speed_kmh: float
    direction_deg: float
    temperature_c: float
    humidity_pct: float
    precipitation_mm: float
    weather_code: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=TRAIN_START_YEAR)
    parser.add_argument("--end-year", type=int, default=TRAIN_END_YEAR)
    parser.add_argument("--target-year", type=int, default=TARGET_YEAR)
    parser.add_argument("--output", default=OUTPUT_HTML)
    parser.add_argument("--model", default="era5", help="Open-Meteo archive model; retries without it if rejected.")
    return parser.parse_args()


def fetch_archive_year(year: int, model: str | None) -> list[WindRow]:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": "wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,precipitation,weather_code",
        "timezone": TIMEZONE,
    }
    if model:
        params["models"] = model

    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    if response.status_code >= 400 and model:
        fallback_params = dict(params)
        fallback_params.pop("models", None)
        response = requests.get(ARCHIVE_URL, params=fallback_params, timeout=60)
    response.raise_for_status()

    payload = response.json()
    if "hourly" not in payload:
        raise RuntimeError(f"Open-Meteo response for {year} had no hourly data: {payload}")

    hourly = payload["hourly"]
    times = hourly.get("time", [])
    speeds = hourly.get("wind_speed_10m", [])
    directions = hourly.get("wind_direction_10m", [])
    temperatures = hourly.get("temperature_2m", [])
    humidities = hourly.get("relative_humidity_2m", [])
    precipitation = hourly.get("precipitation", [])
    weather_codes = hourly.get("weather_code", [])
    rows: list[WindRow] = []

    for idx, (stamp, speed, direction) in enumerate(zip(times, speeds, directions)):
        temperature = temperatures[idx] if idx < len(temperatures) else None
        humidity = humidities[idx] if idx < len(humidities) else None
        rain = precipitation[idx] if idx < len(precipitation) else None
        code = weather_codes[idx] if idx < len(weather_codes) else None
        if speed is None or direction is None or temperature is None or humidity is None:
            continue
        rain_value = float(rain or 0)
        humidity_value = float(humidity)
        code_value = int(code) if code is not None else infer_weather_code(float(temperature), humidity_value, rain_value)
        rows.append(
            WindRow(
                timestamp=datetime.fromisoformat(stamp),
                speed_kmh=float(speed),
                direction_deg=float(direction) % 360,
                temperature_c=float(temperature),
                humidity_pct=humidity_value,
                precipitation_mm=rain_value,
                weather_code=code_value,
            )
        )
    return rows


def load_historical_rows(start_year: int, end_year: int, model: str | None) -> list[WindRow]:
    rows: list[WindRow] = []
    for year in range(start_year, end_year + 1):
        year_rows = fetch_archive_year(year, model)
        print(f"{year}: loaded {len(year_rows):,} hourly rows")
        rows.extend(year_rows)
    if len(rows) < 20_000:
        raise RuntimeError(f"Too few historical rows loaded for a stable model: {len(rows):,}")
    return rows


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def year_hours(year: int) -> list[datetime]:
    current = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    hours: list[datetime] = []
    while current < end:
        hours.append(current)
        current += timedelta(hours=1)
    return hours


def circular_day_of_year(stamp: datetime) -> float:
    days = 366 if is_leap_year(stamp.year) else 365
    return ((stamp.timetuple().tm_yday - 1) + stamp.hour / 24) / days


def features(stamps: Iterable[datetime]) -> np.ndarray:
    matrix: list[list[float]] = []
    for stamp in stamps:
        day = circular_day_of_year(stamp)
        hour = (stamp.hour + stamp.minute / 60) / 24
        month = (stamp.month - 1) / 12
        monsoon = 1.0 if stamp.month in (6, 7, 8, 9) else 0.0
        transition = 1.0 if stamp.month in (3, 4, 5, 10, 11) else 0.0
        matrix.append(
            [
                math.sin(2 * math.pi * day),
                math.cos(2 * math.pi * day),
                math.sin(4 * math.pi * day),
                math.cos(4 * math.pi * day),
                math.sin(2 * math.pi * hour),
                math.cos(2 * math.pi * hour),
                math.sin(4 * math.pi * hour),
                math.cos(4 * math.pi * hour),
                math.sin(2 * math.pi * month),
                math.cos(2 * math.pi * month),
                monsoon,
                transition,
            ]
        )
    return np.asarray(matrix, dtype=float)


def vector_targets(rows: list[WindRow]) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        radians = math.radians(row.direction_deg)
        speed = max(row.speed_kmh, 0.0)
        values.append([math.sin(radians) * speed, math.cos(radians) * speed, speed])
    return np.asarray(values, dtype=float)


def weather_targets(rows: list[WindRow]) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        values.append(
            [
                row.temperature_c,
                row.humidity_pct,
                math.log1p(max(row.precipitation_mm, 0.0)),
            ]
        )
    return np.asarray(values, dtype=float)


def infer_weather_code(temperature_c: float, humidity_pct: float, precipitation_mm: float) -> int:
    if precipitation_mm >= 7.6:
        return 65
    if precipitation_mm >= 2.5:
        return 63
    if precipitation_mm >= 0.2:
        return 61
    if humidity_pct >= 96:
        return 45
    if humidity_pct >= 86:
        return 3
    if humidity_pct >= 68:
        return 2
    return 1 if temperature_c < 31 else 0


def direction_from_vector(eastish: np.ndarray, northish: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(eastish, northish)) + 360) % 360


def circular_abs_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs(((predicted - actual + 180) % 360) - 180)


def sector_name(degrees: float) -> str:
    sectors = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return sectors[int((degrees + 11.25) // 22.5) % len(sectors)]


def eight_sector(degrees: np.ndarray) -> np.ndarray:
    return ((degrees + 22.5) // 45).astype(int) % 8


def train_and_predict(rows: list[WindRow], target_year: int) -> tuple[dict, list[dict]]:
    rows = sorted(rows, key=lambda item: item.timestamp)
    stamps = [row.timestamp for row in rows]
    X = features(stamps)
    y = vector_targets(rows)
    weather_y = weather_targets(rows)
    directions = np.asarray([row.direction_deg for row in rows])
    speeds = np.asarray([row.speed_kmh for row in rows])
    temperatures = np.asarray([row.temperature_c for row in rows])
    humidities = np.asarray([row.humidity_pct for row in rows])
    precipitation = np.asarray([row.precipitation_mm for row in rows])

    validation_year = max(row.timestamp.year for row in rows)
    train_mask = np.asarray([row.timestamp.year < validation_year for row in rows])
    validation_mask = ~train_mask

    validation_model = ExtraTreesRegressor(
        n_estimators=260,
        min_samples_leaf=9,
        max_features=0.85,
        random_state=42,
        n_jobs=-1,
    )
    validation_model.fit(X[train_mask], y[train_mask])
    validation_pred = validation_model.predict(X[validation_mask])
    validation_direction = direction_from_vector(validation_pred[:, 0], validation_pred[:, 1])
    validation_speed = np.maximum(validation_pred[:, 2], 0)

    actual_direction = directions[validation_mask]
    actual_speed = speeds[validation_mask]
    direction_mae = float(np.mean(circular_abs_error(actual_direction, validation_direction)))
    speed_mae = float(mean_absolute_error(actual_speed, validation_speed))
    sector_accuracy = float(np.mean(eight_sector(actual_direction) == eight_sector(validation_direction)))

    weather_validation_model = ExtraTreesRegressor(
        n_estimators=260,
        min_samples_leaf=11,
        max_features=0.85,
        random_state=126,
        n_jobs=-1,
    )
    weather_validation_model.fit(X[train_mask], weather_y[train_mask])
    weather_validation_pred = weather_validation_model.predict(X[validation_mask])
    validation_temp = weather_validation_pred[:, 0]
    validation_humidity = np.clip(weather_validation_pred[:, 1], 0, 100)
    validation_precip = np.maximum(np.expm1(weather_validation_pred[:, 2]), 0)

    actual_temp = temperatures[validation_mask]
    actual_humidity = humidities[validation_mask]
    actual_precip = precipitation[validation_mask]
    temperature_mae = float(mean_absolute_error(actual_temp, validation_temp))
    humidity_mae = float(mean_absolute_error(actual_humidity, validation_humidity))
    precipitation_mae = float(mean_absolute_error(actual_precip, validation_precip))
    wet_hour_accuracy = float(np.mean((actual_precip >= 0.2) == (validation_precip >= 0.2)))

    full_model = ExtraTreesRegressor(
        n_estimators=360,
        min_samples_leaf=8,
        max_features=0.85,
        random_state=86,
        n_jobs=-1,
    )
    backup_model = RandomForestRegressor(
        n_estimators=180,
        min_samples_leaf=14,
        max_features=0.85,
        random_state=17,
        n_jobs=-1,
    )
    full_model.fit(X, y)
    backup_model.fit(X, y)

    weather_full_model = ExtraTreesRegressor(
        n_estimators=360,
        min_samples_leaf=9,
        max_features=0.85,
        random_state=224,
        n_jobs=-1,
    )
    weather_backup_model = RandomForestRegressor(
        n_estimators=180,
        min_samples_leaf=16,
        max_features=0.85,
        random_state=331,
        n_jobs=-1,
    )
    weather_full_model.fit(X, weather_y)
    weather_backup_model.fit(X, weather_y)

    target_stamps = year_hours(target_year)
    target_X = features(target_stamps)
    full_pred = full_model.predict(target_X)
    backup_pred = backup_model.predict(target_X)
    blended = (full_pred * 0.72) + (backup_pred * 0.28)

    pred_direction = direction_from_vector(blended[:, 0], blended[:, 1])
    pred_speed = np.maximum(blended[:, 2], 0)
    backup_direction = direction_from_vector(backup_pred[:, 0], backup_pred[:, 1])
    agreement = 1 - np.minimum(circular_abs_error(pred_direction, backup_direction) / 90, 1)

    weather_full_pred = weather_full_model.predict(target_X)
    weather_backup_pred = weather_backup_model.predict(target_X)
    weather_blended = (weather_full_pred * 0.72) + (weather_backup_pred * 0.28)
    pred_temp = np.clip(weather_blended[:, 0], 5, 45)
    pred_humidity = np.clip(weather_blended[:, 1], 10, 100)
    pred_precip = np.maximum(np.expm1(weather_blended[:, 2]), 0)

    backup_temp = np.clip(weather_backup_pred[:, 0], 5, 45)
    backup_humidity = np.clip(weather_backup_pred[:, 1], 10, 100)
    backup_precip = np.maximum(np.expm1(weather_backup_pred[:, 2]), 0)
    temp_agreement = 1 - np.minimum(np.abs(pred_temp - backup_temp) / 8, 1)
    humidity_agreement = 1 - np.minimum(np.abs(pred_humidity - backup_humidity) / 28, 1)
    precip_agreement = 1 - np.minimum(np.abs(pred_precip - backup_precip) / 6, 1)
    weather_agreement = np.clip((temp_agreement * 0.45) + (humidity_agreement * 0.25) + (precip_agreement * 0.30), 0, 1)

    records: list[dict] = []
    for stamp, direction, speed, confidence, temp, humidity, rain, weather_confidence in zip(
        target_stamps,
        pred_direction,
        pred_speed,
        agreement,
        pred_temp,
        pred_humidity,
        pred_precip,
        weather_agreement,
    ):
        weather_code = infer_weather_code(float(temp), float(humidity), float(rain))
        records.append(
            {
                "t": stamp.strftime("%Y-%m-%dT%H:00"),
                "d": int(round(float(direction))) % 360,
                "s": round(float(speed), 1),
                "c": int(round(55 + 40 * float(confidence))),
                "sector": sector_name(float(direction)),
                "tc": round(float(temp), 1),
                "rh": int(round(float(humidity))),
                "pr": round(float(rain), 2),
                "wx": weather_code,
                "q": int(round(55 + 40 * float(weather_confidence))),
            }
        )

    metrics = {
        "training_rows": len(rows),
        "validation_year": validation_year,
        "direction_mae_deg": round(direction_mae, 1),
        "speed_mae_kmh": round(speed_mae, 2),
        "eight_sector_accuracy_pct": round(sector_accuracy * 100, 1),
        "temperature_mae_c": round(temperature_mae, 2),
        "humidity_mae_pct": round(humidity_mae, 1),
        "precipitation_mae_mm": round(precipitation_mae, 2),
        "wet_hour_accuracy_pct": round(wet_hour_accuracy * 100, 1),
        "target_year": target_year,
        "predicted_rows": len(records),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return metrics, records


def monthly_summary(records: list[dict]) -> list[dict]:
    months: list[dict] = []
    for month in range(1, 13):
        bucket = [item for item in records if int(item["t"][5:7]) == month]
        radians = [math.radians(item["d"]) for item in bucket]
        mean_east = mean(math.sin(angle) for angle in radians)
        mean_north = mean(math.cos(angle) for angle in radians)
        avg_direction = (math.degrees(math.atan2(mean_east, mean_north)) + 360) % 360
        avg_speed = mean(item["s"] for item in bucket)
        months.append(
            {
                "m": month,
                "d": int(round(avg_direction)) % 360,
                "s": round(avg_speed, 1),
                "sector": sector_name(avg_direction),
            }
        )
    return months


def build_html(metrics: dict, records: list[dict]) -> str:
    payload = {
        "location": "Bengaluru, Karnataka",
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "forecastUrl": FORECAST_URL,
        "metrics": metrics,
        "monthly": monthly_summary(records),
        "records": records,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    return r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Windfyre // The Winds of Bengaluru</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        slate: {
                            950: '#060a13',
                            900: '#0b111e',
                            800: '#172033',
                            700: '#24324f',
                        },
                        cyan: {
                            400: '#22d3ee',
                            500: '#06b6d4',
                            600: '#0891b2',
                        },
                        amber: {
                            400: '#fbbf24',
                            500: '#f59e0b',
                            600: '#d97706',
                        }
                    }
                }
            }
        }
    </script>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Font Awesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #060a13;
        }
        .font-mono {
            font-family: 'JetBrains Mono', monospace;
        }
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #060a13;
        }
        ::-webkit-scrollbar-thumb {
            background: #172033;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #24324f;
        }
        @keyframes compassNeedleLive {
            0%, 100% { transform: rotate(-0.7deg) translateX(-0.4px); }
            28% { transform: rotate(0.9deg) translateX(0.6px); }
            58% { transform: rotate(-0.35deg) translateX(-0.2px); }
            78% { transform: rotate(0.55deg) translateX(0.3px); }
        }
        @keyframes compassNeedlePredicted {
            0%, 100% { transform: rotate(0.45deg) translateX(0.25px); }
            34% { transform: rotate(-0.65deg) translateX(-0.45px); }
            63% { transform: rotate(0.25deg) translateX(0.2px); }
            84% { transform: rotate(-0.35deg) translateX(-0.2px); }
        }
        .needle-wobble-live,
        .needle-wobble-predicted {
            transform-origin: bottom center;
            will-change: transform;
        }
        .needle-wobble-live {
            animation: compassNeedleLive 1.85s ease-in-out infinite;
        }
        .needle-wobble-predicted {
            animation: compassNeedlePredicted 2.2s ease-in-out infinite;
        }
    </style>
</head>
<body class="text-slate-100 min-h-screen flex flex-col selection:bg-cyan-500/30 selection:text-cyan-400 overflow-x-hidden">

    <!-- Top Loading / Offline Status bar -->
    <div id="offline-banner" class="hidden bg-amber-600/20 border-b border-amber-500/30 text-amber-400 px-4 py-2 text-xs md:text-sm text-center font-mono flex items-center justify-center gap-2 transition-all duration-300">
        <span class="relative flex h-2 w-2">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
        </span>
        Live telemetry is unavailable. Showing embedded 2026 ML prediction data until the feed reconnects.
    </div>

    <!-- Main Navigation Bar -->
    <header class="border-b border-slate-800/60 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40 transition-colors">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="relative">
                    <div class="absolute -inset-1.5 bg-gradient-to-r from-cyan-500 to-amber-500 rounded-lg blur opacity-45 group-hover:opacity-100 transition duration-1000 group-hover:duration-200 animate-tilt"></div>
                    <div class="relative bg-slate-950 px-3 py-1.5 rounded-md border border-slate-800 flex items-center gap-2">
                        <i class="fa-solid fa-wind text-cyan-400 animate-pulse"></i>
                        <span class="font-extrabold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-white via-slate-100 to-slate-400 text-sm md:text-base">WINDFYRE</span>
                    </div>
                </div>
                <span class="hidden sm:inline-block px-2 py-0.5 text-[10px] uppercase font-mono font-bold tracking-widest bg-cyan-950 text-cyan-400 border border-cyan-800/60 rounded">v2.5</span>
            </div>

            <!-- Refresh State -->
            <div class="flex items-center gap-4">
                <button onclick="refreshData()" class="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-100 transition-all focus:outline-none focus:ring-1 focus:ring-cyan-500" title="Sync live telemetry">
                    <i id="refresh-icon" class="fa-solid fa-arrows-rotate"></i>
                </button>
            </div>
        </div>
    </header>

    <!-- App Wrapper -->
    <main class="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 md:py-10 space-y-8">
        
        <!-- Hero Title Section -->
        <div class="border-b border-slate-900 pb-6">
            <div class="space-y-2">
                <h1 class="text-3xl md:text-5xl font-black tracking-tight text-white flex items-center gap-3">
                    The Winds of Bengaluru
                </h1>
                <p class="text-slate-400 text-sm md:text-base max-w-2xl">
                    Adaptive 2026 wind modeling matched with real-time Open-Meteo telemetry. Comparing historical machine learning baselines against live thermal flows.
                </p>
            </div>
        </div>

        <!-- Notification Message Box -->
        <div id="toast-message" class="hidden transform scale-95 opacity-0 transition-all duration-300 bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center justify-between gap-3 shadow-2xl">
            <div class="flex items-center gap-3">
                <div class="p-2 rounded-lg bg-cyan-950 text-cyan-400"><i class="fa-solid fa-circle-info"></i></div>
                <div>
                    <h4 id="toast-title" class="font-bold text-sm text-white">System Update</h4>
                    <p id="toast-desc" class="text-xs text-slate-400">Loading new telemetry charts...</p>
                </div>
            </div>
            <button onclick="dismissToast()" class="text-slate-500 hover:text-white transition-all"><i class="fa-solid fa-xmark"></i></button>
        </div>

        <!-- Telemetry Highlights Grid (4-cards) -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <!-- Card 1: Local Time & Telemetry Status -->
            <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 backdrop-blur-md flex flex-col justify-between h-32 relative overflow-hidden group hover:border-slate-700/80 transition-all duration-300">
                <div class="absolute top-0 right-0 p-3 text-slate-800 group-hover:text-cyan-500/10 transition-colors"><i class="fa-solid fa-clock text-4xl"></i></div>
                <div>
                    <span class="text-xs text-slate-500 uppercase tracking-widest font-mono block">Local Clock</span>
                    <span id="live-time" class="text-2xl font-black text-white font-mono block tracking-tight mt-1">--:--:--</span>
                </div>
                <div class="flex items-center justify-between mt-2">
                    <span class="text-xs text-slate-400 flex items-center gap-1.5">
                        <span class="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping"></span>
                        Asia/Kolkata
                    </span>
                    <span class="text-[10px] font-mono text-slate-500">Bengaluru</span>
                </div>
            </div>

            <!-- Card 2: Live Weather Telemetry -->
            <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 backdrop-blur-md flex flex-col justify-between h-32 relative overflow-hidden group hover:border-slate-700/80 transition-all duration-300">
                <div class="absolute top-0 right-0 p-3 text-slate-800 group-hover:text-cyan-500/10 transition-colors"><i class="fa-solid fa-cloud-sun text-4xl"></i></div>
                <div>
                    <span class="text-xs text-slate-500 uppercase tracking-widest font-mono block">Live Conditions</span>
                    <span id="live-weather-desc" class="text-xl font-bold text-white block tracking-tight mt-1 truncate">Checking...</span>
                </div>
                <div class="flex justify-between items-center mt-2">
                    <span id="live-temp" class="text-sm font-mono text-cyan-400 font-semibold">--°C</span>
                    <span id="live-humidity" class="text-xs text-slate-400">Humidity: --%</span>
                </div>
            </div>

            <!-- Card 3: Live Wind Speed / Direction -->
            <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 backdrop-blur-md flex flex-col justify-between h-32 relative overflow-hidden group hover:border-slate-700/80 transition-all duration-300">
                <div class="absolute top-0 right-0 p-3 text-slate-800 group-hover:text-cyan-500/10 transition-colors"><i class="fa-solid fa-gauge-simple-high text-4xl"></i></div>
                <div>
                    <span class="text-xs text-slate-500 uppercase tracking-widest font-mono block">Live Wind Telemetry</span>
                    <div class="flex items-baseline gap-1 mt-1">
                        <span id="live-wind-speed" class="text-2xl font-black text-white font-mono">--.-</span>
                        <span id="wind-unit-label" class="text-xs font-mono text-slate-400">km/h</span>
                    </div>
                </div>
                <div class="flex justify-between items-center mt-2">
                    <span id="live-wind-dir" class="text-xs font-mono text-cyan-400 font-medium">---° --</span>
                    <span id="live-wind-gusts" class="text-[10px] text-slate-500 uppercase font-mono">Gusts: -- km/h</span>
                </div>
            </div>

            <!-- Card 4: Forecast Baseline Alignment -->
            <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 backdrop-blur-md flex flex-col justify-between h-32 relative overflow-hidden group hover:border-slate-700/80 transition-all duration-300">
                <div class="absolute top-0 right-0 p-3 text-slate-800 group-hover:text-amber-500/10 transition-colors"><i class="fa-solid fa-network-wired text-4xl"></i></div>
                <div>
                    <span class="text-xs text-slate-500 uppercase tracking-widest font-mono block">Model Evaluation</span>
                    <div class="flex items-baseline gap-1 mt-1">
                        <span id="live-match-score" class="text-2xl font-black text-amber-400 font-mono">--%</span>
                        <span class="text-xs font-mono text-slate-400">Match</span>
                    </div>
                </div>
                <div class="flex justify-between items-center mt-2">
                    <span id="predicted-wind-baseline" class="text-xs font-mono text-amber-500">Predicted: -- km/h</span>
                    <span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-amber-950 text-amber-400 border border-amber-900/60 uppercase">ML 2026</span>
                </div>
            </div>
        </div>

        <!-- Main Workspace: Wind Compass -->
        <div class="grid grid-cols-1 gap-8">
            
            <!-- Animated Compass and Interactive Selector -->
            <div class="bg-slate-900/25 border border-slate-800/60 rounded-3xl p-6 backdrop-blur-md flex flex-col justify-between gap-6 relative">
                
                <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                        <h2 class="text-xl font-bold text-white flex items-center gap-2">
                            <i class="fa-solid fa-compass text-cyan-400"></i> Wind Compass
                        </h2>
                        <p class="text-xs text-slate-400 mt-1">Live directional vectors compared with embedded 2026 ML predictions.</p>
                    </div>
                    
                    <!-- Interactive Clock Time Scrubber -->
                    <div class="bg-slate-950 p-2.5 rounded-xl border border-slate-800 flex items-center gap-3 w-full sm:w-auto">
                        <span class="text-xs font-mono text-slate-400 whitespace-nowrap"><i class="fa-solid fa-hourglass-half text-amber-500 mr-1 animate-spin-slow"></i> Timeline Scrub:</span>
                        <input id="hour-scrubber" type="range" min="0" max="23" value="12" oninput="scrubHour(this.value)" class="w-full sm:w-32 accent-cyan-500 bg-slate-800 rounded-lg cursor-pointer h-1.5">
                        <span id="scrub-hour-label" class="text-xs font-mono font-bold text-cyan-400">12:00</span>
                    </div>
                </div>

                <!-- Animated Canvas Frame & Compass Grid -->
                <div class="relative w-full flex flex-col md:flex-row items-center justify-around gap-8 py-4">
                    
                    <!-- CANVAS COMPASS CONTAINER -->
                    <div class="relative w-64 h-64 md:w-96 md:h-96 lg:w-[26rem] lg:h-[26rem] flex-shrink-0 flex items-center justify-center">
                        
                        <!-- Embedded Flow Canvas inside the Compass Frame -->
                        <canvas id="windCanvas" class="absolute inset-0 rounded-full w-full h-full opacity-60 pointer-events-none"></canvas>
                        
                        <!-- Compass Background Dial Rings -->
                        <div class="absolute inset-0 rounded-full border border-slate-800/80 flex items-center justify-center shadow-inner">
                            <div class="w-[85%] h-[85%] rounded-full border border-dashed border-slate-800/60 flex items-center justify-center">
                                <div class="w-[70%] h-[70%] rounded-full border border-slate-800/80 bg-slate-950/40"></div>
                            </div>
                        </div>

                        <!-- Cardinal guide lines -->
                        <div class="absolute left-6 right-6 top-1/2 h-px -translate-y-1/2 bg-slate-700/45 pointer-events-none z-10"></div>
                        <div class="absolute top-6 bottom-6 left-1/2 w-px -translate-x-1/2 bg-slate-700/45 pointer-events-none z-10"></div>

                        <!-- Card Points (N, S, E, W, etc.) -->
                        <div class="absolute inset-0 font-mono text-xs font-bold text-slate-500 p-2 pointer-events-none select-none">
                            <span class="absolute top-2 left-1/2 -translate-x-1/2 text-cyan-400/90 font-extrabold">N</span>
                            <span class="absolute right-3 top-1/2 -translate-y-1/2">E</span>
                            <span class="absolute bottom-2 left-1/2 -translate-x-1/2">S</span>
                            <span class="absolute left-3 top-1/2 -translate-y-1/2">W</span>
                            
                        </div>

                        <!-- Outer degree markers -->
                        <div class="absolute inset-0 font-mono text-[9px] md:text-[10px] text-slate-600 pointer-events-none select-none">
                            <span class="absolute top-7 left-1/2 -translate-x-1/2">0°</span>
                            <span class="absolute top-[15%] right-[15%]">45°</span>
                            <span class="absolute right-7 top-1/2 -translate-y-1/2">90°</span>
                            <span class="absolute bottom-[15%] right-[15%]">135°</span>
                            <span class="absolute bottom-7 left-1/2 -translate-x-1/2">180°</span>
                            <span class="absolute bottom-[15%] left-[15%]">225°</span>
                            <span class="absolute left-7 top-1/2 -translate-y-1/2">270°</span>
                            <span class="absolute top-[15%] left-[15%]">315°</span>
                        </div>

                        <!-- CENTERED ROTATION ANCHOR BOX (NEEDLES) -->
                        <div class="absolute inset-0 flex items-center justify-center pointer-events-none z-20">
                            
                            <!-- LIVE NEEDLE (Cyan Glow) - Pivots precisely around its bottom center -->
                            <div id="needle-live" class="absolute bottom-1/2 w-2.5 h-32 md:h-40 origin-bottom transition-transform duration-1000 ease-out" style="transform: rotate(0deg);">
                                <div class="needle-wobble-live relative w-full h-full">
                                    <div class="w-full h-full bg-gradient-to-t from-cyan-500/20 via-cyan-500 to-cyan-400 rounded-full shadow-[0_0_15px_#06b6d4]"></div>
                                    <!-- Live indicator arrowhead -->
                                    <div class="absolute -top-2 left-1/2 -translate-x-1/2 w-0 h-0 border-l-[8px] border-l-transparent border-r-[8px] border-r-transparent border-b-[14px] border-b-cyan-400"></div>
                                </div>
                            </div>

                            <!-- PREDICTED NEEDLE (Amber Dash/Line) - Pivots precisely around its bottom center -->
                            <div id="needle-predicted" class="absolute bottom-1/2 w-2.5 h-32 md:h-40 origin-bottom transition-transform duration-1000 ease-out opacity-45" style="transform: rotate(0deg);">
                                <div class="needle-wobble-predicted relative w-full h-full">
                                    <div class="w-full h-full bg-gradient-to-t from-amber-500/10 via-amber-500/60 to-amber-300 rounded-full shadow-[0_0_8px_rgba(245,158,11,0.55)]"></div>
                                    <!-- Predicted indicator arrowhead -->
                                    <div class="absolute -top-2 left-1/2 -translate-x-1/2 w-0 h-0 border-l-[8px] border-l-transparent border-r-[8px] border-r-transparent border-b-[14px] border-b-amber-300"></div>
                                </div>
                            </div>

                        </div>

                        <!-- Center Telemetry Overlay Hub -->
                        <div class="absolute w-28 h-28 md:w-32 md:h-32 bg-slate-950 border border-slate-800/80 rounded-full flex flex-col items-center justify-center text-center p-2.5 shadow-2xl z-30">
                            <span class="text-[8px] uppercase tracking-widest text-slate-500 font-mono">Telemetry</span>
                            <span id="center-heading" class="text-base font-bold font-mono tracking-tight text-white my-0.5">--° --</span>
                            <span id="center-speed" class="text-lg font-extrabold text-cyan-400 font-mono">--.-</span>
                            <span id="center-speed-unit" class="text-[9px] font-mono text-slate-400">km/h</span>
                        </div>
                    </div>

                    <!-- Compass Meta Readout side-panel -->
                    <div class="flex-grow space-y-4 w-full md:max-w-sm">
                        <div class="bg-slate-950/60 rounded-xl p-4 border border-slate-800/60 space-y-3">
                            <div class="flex items-center justify-between border-b border-slate-900 pb-2">
                                <span class="text-xs text-slate-400 flex items-center gap-1.5 font-medium">
                                    <span class="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_6px_#06b6d4]"></span>
                                    Live Telemetry
                                </span>
                                <span id="comp-live-val" class="text-xs font-mono font-bold text-white">--</span>
                            </div>
                            <div class="flex items-center justify-between border-b border-slate-900 pb-2">
                                <span class="text-xs text-slate-400 flex items-center gap-1.5 font-medium">
                                    <span class="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_6px_#f59e0b]"></span>
                                    Model Baseline
                                </span>
                                <span id="comp-pred-val" class="text-xs font-mono font-bold text-white">--</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-xs text-slate-400 font-medium">Direction Divergence</span>
                                <span id="comp-divergence-val" class="text-xs font-mono font-bold text-rose-400">--°</span>
                            </div>
                        </div>

                        <!-- Fast Information card -->
                        <div class="bg-gradient-to-br from-slate-900 to-slate-950 p-4 rounded-xl border border-slate-800/50">
                            <h4 class="text-xs font-bold uppercase tracking-wider text-slate-300 font-mono mb-1.5">Seasonality Signature</h4>
                            <p id="seasonal-desc" class="text-xs text-slate-400 leading-relaxed">
                                Loading local Bengaluru seasonal signature variables based on 2026 climate tracking matrices.
                            </p>
                        </div>
                    </div>
                </div>
                <p class="text-[11px] font-mono text-slate-500 flex items-center justify-end gap-2 text-right">
                    <span class="w-1.5 h-1.5 rounded-full bg-slate-500"></span>
                    <span id="last-updated-detail">Last updated: loading local prediction data</span>
                </p>
            </div>
        </div>

        <!-- Charts Suite Container (Next 12 Hours & Next 7 Days) -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            <!-- Hourly Forecast Comparison Chart Card -->
            <div class="bg-slate-900/45 border border-slate-800/70 rounded-3xl p-6 backdrop-blur-md">
                <div class="flex items-center justify-between mb-6">
                    <div>
                        <h3 class="text-lg font-bold text-white flex items-center gap-2">
                            <i class="fa-solid fa-chart-line text-cyan-400"></i> Today's Predicted Wind Speed Curve
                        </h3>
                        <p class="text-xs text-slate-400 mt-1">Comparing today's 24-hour prediction baseline vs live Open-Meteo forecasts.</p>
                    </div>
                    <span class="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950/60 text-cyan-400 border border-cyan-900/40 uppercase font-bold">Hourly</span>
                </div>

                <div class="relative h-64 w-full">
                    <canvas id="hourlyChart"></canvas>
                </div>
                <div class="mt-3 flex justify-end">
                    <span id="anomaly-scan-status" class="text-[10px] font-mono text-slate-500 text-right">Anomaly scan: waiting for live data</span>
                </div>
            </div>

            <!-- Weekly Max Mixed Trends Card -->
            <div class="bg-slate-900/45 border border-slate-800/70 rounded-3xl p-6 backdrop-blur-md">
                <div class="flex items-center justify-between mb-6">
                    <div>
                        <h3 class="text-lg font-bold text-white flex items-center gap-2">
                            <i class="fa-solid fa-chart-line text-amber-400"></i> Next 7 Days Velocity & Predictions
                        </h3>
                        <p class="text-xs text-slate-400 mt-1">Comparing maximum daily wind speed forecasts (Bars) with historical model predictions (Line).</p>
                    </div>
                    <span class="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-950/60 text-amber-400 border border-amber-900/40 uppercase font-bold">7-Day Outlines</span>
                </div>

                <div class="relative h-64 w-full">
                    <canvas id="weeklyChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Rain Outlook Section -->
        <div id="rain-outlook-section" class="bg-slate-900/45 border border-slate-800/70 rounded-3xl p-6 backdrop-blur-md">
            <div class="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-5 mb-6">
                <div>
                    <h3 class="text-lg font-bold text-white flex items-center gap-2">
                        <i class="fa-solid fa-cloud-rain text-emerald-400"></i> Rainfall Forecast
                    </h3>
                    <p class="text-xs text-slate-400 mt-1">Live Open-Meteo rainfall compared with the embedded ML rain prediction for Bengaluru.</p>
                </div>
                <span class="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950/60 text-emerald-400 border border-emerald-900/40 uppercase font-bold self-start">Rain Models</span>
            </div>

            <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <div class="rounded-2xl border border-slate-800/80 bg-slate-950/45 p-4">
                    <div class="flex items-center justify-between gap-3 mb-3">
                        <h4 class="text-sm font-bold text-white">Today Rainfall</h4>
                        <span id="rain-hourly-anomaly-status" class="text-[10px] font-mono text-slate-500 text-right">Rain anomaly: waiting for live data</span>
                    </div>
                    <div class="relative h-60 w-full">
                        <canvas id="hourlyRainChart"></canvas>
                    </div>
                </div>
                <div class="rounded-2xl border border-slate-800/80 bg-slate-950/45 p-4">
                    <div class="flex items-center justify-between gap-3 mb-3">
                        <h4 class="text-sm font-bold text-white">Next 7 Days Rainfall</h4>
                        <span id="weather-rain-status" class="text-[10px] font-mono text-slate-500 text-right">Waiting for live forecast</span>
                    </div>
                    <div class="relative h-60 w-full">
                        <canvas id="weatherRainChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- Live Weather Outlook, shown only when Open-Meteo provides weather forecast data -->
        <div id="weather-outlook-section" class="hidden bg-slate-900/45 border border-slate-800/70 rounded-3xl p-6 backdrop-blur-md">
            <div class="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-5 mb-6">
                <div>
                    <h3 class="text-lg font-bold text-white flex items-center gap-2">
                        <i class="fa-solid fa-temperature-half text-cyan-400"></i> 7-Day Temperature Outlook
                    </h3>
                    <p class="text-xs text-slate-400 mt-1">Live Open-Meteo temperature forecast for Bengaluru, shown with the daily condition strip.</p>
                </div>
                <div id="weather-condition-strip" class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-xs">
                    <!-- Filled from live API weather data -->
                </div>
            </div>

            <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <div class="rounded-2xl border border-slate-800/80 bg-slate-950/45 p-4">
                    <div class="flex items-center justify-between gap-3 mb-3">
                        <h4 class="text-sm font-bold text-white">Max Temperature</h4>
                        <span id="weather-max-status" class="text-[10px] font-mono text-slate-500 text-right">Waiting for live forecast</span>
                    </div>
                    <div class="relative h-60 w-full">
                        <canvas id="weatherMaxChart"></canvas>
                    </div>
                </div>
                <div class="rounded-2xl border border-slate-800/80 bg-slate-950/45 p-4">
                    <div class="flex items-center justify-between gap-3 mb-3">
                        <h4 class="text-sm font-bold text-white">Min Temperature</h4>
                        <span id="weather-min-status" class="text-[10px] font-mono text-slate-500 text-right">Waiting for live forecast</span>
                    </div>
                    <div class="relative h-60 w-full">
                        <canvas id="weatherMinChart"></canvas>
                    </div>
                </div>
            </div>
            <div class="mt-3 flex justify-end">
                <span id="weather-graph-status" class="text-[10px] font-mono text-slate-500 text-right">Weather graph: waiting for live forecast</span>
            </div>
        </div>

        <!-- Monthly Wind Characters Segment -->
        <div class="bg-slate-900/20 border border-slate-800/60 rounded-3xl p-6 backdrop-blur-md space-y-6">
            <div>
                <h3 class="text-xl font-bold text-white flex items-center gap-2">
                    <i class="fa-solid fa-calendar-days text-cyan-400"></i> Bengaluru Monthly Wind Character
                </h3>
                <p class="text-xs text-slate-400 mt-1">Continuous climatological profiles. Select any card to expand deep analytical and atmospheric metrics.</p>
            </div>

            <div class="grid grid-cols-1 xl:grid-cols-4 gap-6">
                <!-- 12-Month Always-Visible Grid -->
                <div id="monthly-grid" class="xl:col-span-3 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                    <!-- Dynamically populated monthly selection cards -->
                </div>

                <!-- Focus Month Detailed Expansion Card -->
                <div id="monthly-character-card" class="xl:col-span-1 bg-slate-950/60 rounded-2xl p-5 border border-slate-800/80 flex flex-col justify-between gap-5 h-full min-h-[350px] relative overflow-hidden">
                    <div class="absolute top-5 right-5 w-14 h-14 rounded-full bg-slate-950 border border-slate-700/70 flex items-center justify-center text-slate-300 shadow-[0_0_14px_rgba(148,163,184,0.08)]" title="Dominant monthly direction">
                        <i id="mon-detail-arrow" class="fa-solid fa-arrow-up text-xl transition-transform duration-300" style="transform: rotate(0deg)"></i>
                    </div>
                    <div class="space-y-3">
                        <span id="mon-subtitle" class="text-[10px] font-mono text-cyan-400 uppercase tracking-widest block">Season segment</span>
                        <h4 id="mon-title" class="text-2xl font-extrabold text-white pr-16">January</h4>
                        <p id="mon-tagline" class="text-xs font-mono text-slate-400 italic">"Cool northerly trade winds"</p>
                        
                        <div class="border-t border-slate-800/60 pt-3 space-y-2 text-xs">
                            <div class="flex justify-between py-1 border-b border-slate-900/60">
                                <span class="text-slate-500">Speed Range</span>
                                <span id="mon-speed" class="font-mono text-white font-bold">--</span>
                            </div>
                            <div class="flex justify-between py-1 border-b border-slate-900/60">
                                <span class="text-slate-500">Dominant Dir</span>
                                <span id="mon-direction" class="font-mono text-white font-bold">--</span>
                            </div>
                            <div class="flex justify-between py-1 border-b border-slate-900/60">
                                <span class="text-slate-500">Prediction Trust</span>
                                <span id="mon-moisture" class="font-mono text-cyan-400 font-bold">--</span>
                            </div>
                            <div class="flex justify-between py-1">
                                <span class="text-slate-500">Peak Hour</span>
                                <span id="mon-activity" class="font-mono text-amber-400 font-bold">--</span>
                            </div>
                        </div>
                    </div>

                    <div class="bg-slate-900/40 border border-slate-800/40 rounded-xl p-4">
                        <span class="text-[9px] text-slate-500 font-mono uppercase tracking-wider block mb-1">Climatology Analysis</span>
                        <p id="mon-analysis" class="text-xs text-slate-400 leading-relaxed">
                            Loading climate matrices...
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- How Reliable Is This Section -->
        <div class="bg-slate-900/45 border border-slate-800/70 rounded-3xl p-6 backdrop-blur-md">
            <h3 class="text-xl font-bold text-white mb-2"><i class="fa-solid fa-circle-question text-cyan-400"></i> Dynamic Model Reliability Check</h3>
            <p class="text-sm text-slate-400 mb-6">Transparency in wind and weather prediction modeling, based on the 2025 back-test against historical Open-Meteo archive data.</p>
            
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div class="bg-slate-950 p-4 rounded-2xl border border-slate-900 flex flex-col justify-between">
                    <span class="text-xs text-slate-500 font-mono uppercase tracking-wider">Learned From</span>
                    <span id="stats-learned" class="text-lg font-bold text-white mt-1">--</span>
                    <span id="stats-learned-note" class="text-[10px] text-slate-400 mt-1">Open-Meteo archive rows</span>
                </div>
                <div class="bg-slate-950 p-4 rounded-2xl border border-slate-900 flex flex-col justify-between">
                    <span class="text-xs text-slate-500 font-mono uppercase tracking-wider">Typical Direction Miss</span>
                    <span id="stats-dir-miss" class="text-lg font-bold text-cyan-400 mt-1">--</span>
                    <span class="text-[10px] text-slate-400 mt-1">2025 validation average</span>
                </div>
                <div class="bg-slate-950 p-4 rounded-2xl border border-slate-900 flex flex-col justify-between">
                    <span class="text-xs text-slate-500 font-mono uppercase tracking-wider">Typical Speed Miss</span>
                    <span id="stats-speed-miss" class="text-lg font-bold text-cyan-400 mt-1">--</span>
                    <span class="text-[10px] text-slate-400 mt-1">2025 validation average</span>
                </div>
                <div class="bg-slate-950 p-4 rounded-2xl border border-slate-900 flex flex-col justify-between">
                    <span class="text-xs text-slate-500 font-mono uppercase tracking-wider">Prediction Trust index</span>
                    <span id="stats-trust" class="text-lg font-bold text-emerald-400 mt-1">--</span>
                    <span class="text-[10px] text-slate-400 mt-1">Current hourly model confidence</span>
                </div>
            </div>

            <div class="mt-6 pt-6 border-t border-slate-800/70">
                <div class="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-2 mb-4">
                    <div>
                        <span class="text-[10px] font-mono text-emerald-400 uppercase tracking-widest">Weather Prediction Details</span>
                        <h4 class="text-base font-bold text-white mt-1">How the weather model is performing</h4>
                    </div>
                    <span id="weather-reliability-note" class="text-[10px] font-mono text-slate-500">Current hourly weather baseline</span>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-6 gap-4">
                    <div class="bg-slate-950 p-4 rounded-2xl border border-slate-900 flex flex-col justify-between">
                        <span class="text-xs text-slate-500 font-mono uppercase tracking-wider">Weather Trust</span>
                        <span id="weather-stats-trust" class="text-lg font-bold text-emerald-400 mt-1">--</span>
                        <span class="text-[10px] text-slate-400 mt-1">Current hour confidence</span>
                    </div>
                    <div class="bg-slate-950 p-4 rounded-2xl border border-slate-900 flex flex-col justify-between">
                        <span class="text-xs text-slate-500 font-mono uppercase tracking-wider">Predicted Now</span>
                        <span id="weather-stats-now" class="text-lg font-bold text-white mt-1">--</span>
                        <span id="weather-stats-now-note" class="text-[10px] text-slate-400 mt-1">Condition baseline</span>
                    </div>
                    <div class="bg-slate-950 p-4 rounded-2xl border border-slate-900 flex flex-col justify-between">
                        <span class="text-xs text-slate-500 font-mono uppercase tracking-wider">Temp Miss</span>
                        <span id="weather-stats-temp" class="text-lg font-bold text-amber-400 mt-1">--</span>
                        <span class="text-[10px] text-slate-400 mt-1">2025 validation average</span>
                    </div>
                    <div class="bg-slate-950 p-4 rounded-2xl border border-slate-900 flex flex-col justify-between">
                        <span class="text-xs text-slate-500 font-mono uppercase tracking-wider">Humidity Miss</span>
                        <span id="weather-stats-humidity" class="text-lg font-bold text-cyan-400 mt-1">--</span>
                        <span class="text-[10px] text-slate-400 mt-1">2025 validation average</span>
                    </div>
                    <div class="bg-slate-950 p-4 rounded-2xl border border-slate-900 flex flex-col justify-between">
                        <span class="text-xs text-slate-500 font-mono uppercase tracking-wider">Rain Miss</span>
                        <span id="weather-stats-rain" class="text-lg font-bold text-emerald-400 mt-1">--</span>
                        <span class="text-[10px] text-slate-400 mt-1">Hourly precipitation error</span>
                    </div>
                    <div class="bg-slate-950 p-4 rounded-2xl border border-slate-900 flex flex-col justify-between">
                        <span class="text-xs text-slate-500 font-mono uppercase tracking-wider">Wet Hour Hit</span>
                        <span id="weather-stats-wet" class="text-lg font-bold text-emerald-400 mt-1">--</span>
                        <span class="text-[10px] text-slate-400 mt-1">Rain/no-rain classification</span>
                    </div>
                </div>
            </div>
        </div>

    </main>

    <!-- Footer Segment -->
    <footer class="border-t border-slate-900 bg-slate-950 py-8 mt-12 text-center text-slate-600 text-xs">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-3">
            <div class="flex items-center justify-center gap-2">
                <i class="fa-solid fa-wind text-cyan-600"></i>
                <span class="font-extrabold tracking-wider text-slate-400">WINDFYRE</span>
            </div>
            <p>Designed and generated for real-time microclimate research on Bengaluru winds.</p>
            <p class="font-mono text-[10px] text-slate-700">Open-Meteo Free API License Model Integration // Client-side prediction model</p>
        </div>
    </footer>

    <script id="wind-data" type="application/json">__PAYLOAD_JSON__</script>

    <!-- Core Javascript Code -->
    <script>
        // UI State Variables
        let selectedHour = new Date().getHours();
        let telemetryData = null; 
        let hourlyChart = null;
        let hourlyRainChart = null;
        let weeklyChart = null;
        let weatherCharts = {};

        const payload = JSON.parse(document.getElementById('wind-data').textContent);
        const BENGALURU_LOCATION = {
            lat: payload.latitude,
            lon: payload.longitude,
            name: payload.location,
            short: "Bengaluru",
            timezone: payload.timezone
        };
        const UNIT_LABEL = "km/h";
        const records = payload.records;
        const recordMap = new Map(records.map((record, index) => [record.t, { record, index }]));
        const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
        const shortMonthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        const bengaluruFormatter = new Intl.DateTimeFormat('en-GB', {
            timeZone: payload.timezone,
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            hour12: false
        });

        function bengaluruParts(date = new Date()) {
            const parts = Object.fromEntries(bengaluruFormatter.formatToParts(date).map(part => [part.type, part.value]));
            return {
                year: Number(parts.year),
                month: Number(parts.month),
                day: Number(parts.day),
                hour: Number(parts.hour),
                minute: Number(parts.minute)
            };
        }

        function recordKeyFor(dateObj = new Date(), hourValue = null) {
            const parts = bengaluruParts(dateObj);
            const hour = hourValue === null ? parts.hour : Number(hourValue);
            return `${payload.metrics.target_year}-${String(parts.month).padStart(2, '0')}-${String(parts.day).padStart(2, '0')}T${String(hour).padStart(2, '0')}:00`;
        }

        function recordForDateHour(dateObj = new Date(), hourValue = null) {
            const key = recordKeyFor(dateObj, hourValue);
            const found = recordMap.get(key);
            if (found) return found.record;
            const parts = bengaluruParts(dateObj);
            const fallbackIndex = Math.max(0, Math.min(records.length - 1, (((parts.month - 1) * 31 + (parts.day - 1)) * 24) + (hourValue ?? parts.hour)));
            return records[fallbackIndex] || records[0];
        }

        function monthRecords(monthIndex) {
            const month = String(monthIndex + 1).padStart(2, '0');
            return records.filter(record => record.t.slice(5, 7) === month);
        }

        function average(values) {
            return values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0;
        }

        function peakHourForMonth(monthIndex) {
            const bucket = monthRecords(monthIndex);
            const hourly = Array.from({ length: 24 }, (_, hour) => {
                const values = bucket.filter(record => Number(record.t.slice(11, 13)) === hour).map(record => record.s);
                return average(values);
            });
            return hourly.indexOf(Math.max(...hourly));
        }

        const MONTHLY_PROFILES = payload.monthly.map((month, index) => {
            const bucket = monthRecords(index);
            const speeds = bucket.map(record => record.s);
            const confidences = bucket.map(record => record.c);
            const minSpeed = Math.min(...speeds);
            const maxSpeed = Math.max(...speeds);
            const peakHour = peakHourForMonth(index);
            return {
                id: index,
                name: monthNames[index],
                subtitle: `${payload.metrics.target_year} prediction`,
                tagline: `${month.sector} dominant wind, ${month.s.toFixed(1)} km/h average`,
                speed: `${minSpeed.toFixed(1)} - ${maxSpeed.toFixed(1)} km/h`,
                direction: `${month.sector} (${month.d}°)`,
                confidence: `${Math.round(average(confidences))}%`,
                activity: `${String(peakHour).padStart(2, '0')}:00 peak average`,
                analysis: `${monthNames[index]} is predicted from ${bucket.length.toLocaleString('en-IN')} hourly model records. The expected mean wind is ${month.s.toFixed(1)} km/h from ${month.sector} (${month.d}°), with observed predicted hourly speeds ranging from ${minSpeed.toFixed(1)} to ${maxSpeed.toFixed(1)} km/h.`
            };
        });

        let selectedMonthId = new Date().getMonth();

        function getHistoricalPrediction(dateObj, hourValue) {
            const record = recordForDateHour(dateObj, hourValue);
            return {
                speed: record.s,
                direction: record.d,
                confidence: record.c,
                sector: record.sector,
                temperature: typeof record.tc === 'number' ? record.tc : null,
                humidity: typeof record.rh === 'number' ? record.rh : null,
                precipitation: typeof record.pr === 'number' ? record.pr : null,
                weatherCode: typeof record.wx === 'number' ? record.wx : null,
                weatherConfidence: typeof record.q === 'number' ? record.q : null,
                stamp: record.t
            };
        }

        // Display all wind speeds in km/h to match the trained prediction payload.
        function formatSpeed(speedKmh) {
            if (speedKmh === undefined || speedKmh === null) return "--.-";
            return Math.round(speedKmh * 10) / 10;
        }

        // Get cardinal direction label
        function getCardinal(deg) {
            if (deg === undefined || deg === null) return "--";
            const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
            const idx = Math.round(((deg % 360) / 45)) % 8;
            return directions[idx];
        }

        function matchScoreClasses(score) {
            if (score >= 88) return 'text-emerald-400';
            if (score >= 72) return 'text-lime-400';
            if (score >= 55) return 'text-amber-400';
            return 'text-rose-400';
        }

        // Keep a compact status line instead of a large telemetry log panel.
        function logToTerminal(message, type = 'info') {
            const status = document.getElementById('last-updated-detail');
            if (!status) return;
            const timeStr = new Date().toLocaleTimeString('en-US', { hour12: false });
            status.className = 'text-slate-500';
            status.textContent = `Last updated ${timeStr}: ${message}`;
        }

        // Toast Messages handler
        function showToast(title, desc) {
            const toast = document.getElementById('toast-message');
            const tTitle = document.getElementById('toast-title');
            const tDesc = document.getElementById('toast-desc');
            
            tTitle.innerText = title;
            tDesc.innerText = desc;
            
            toast.classList.remove('hidden');
            setTimeout(() => {
                toast.classList.remove('scale-95', 'opacity-0');
                toast.classList.add('scale-100', 'opacity-100');
            }, 50);

            // auto-dismiss
            setTimeout(() => {
                dismissToast();
            }, 6000);
        }

        function dismissToast() {
            const toast = document.getElementById('toast-message');
            if (toast) {
                toast.classList.remove('scale-100', 'opacity-100');
                toast.classList.add('scale-95', 'opacity-0');
                setTimeout(() => {
                    toast.classList.add('hidden');
                }, 300);
            }
        }

        // -------------------------------------------------------------------------
        // Dynamic Wind Flow Canvas System (Real-time particle rendering)
        // -------------------------------------------------------------------------
        const canvas = document.getElementById('windCanvas');
        const ctx = canvas.getContext('2d');
        let particles = [];
        let particleAnimationId = null;
        let activeWindFlow = { speed: 10, direction: 90 };

        function resizeCanvas() {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
        }

        function clamp(value, min, max) {
            return Math.max(min, Math.min(max, value));
        }

        function needleLengthForSpeed(speedKmh) {
            const rect = canvas.getBoundingClientRect();
            const radius = Math.max(120, Math.min(rect.width, rect.height) / 2);
            const minLength = radius * 0.56;
            const maxLength = radius * 0.88;
            const speedRatio = clamp((Number(speedKmh) || 0) / 24, 0, 1);
            return Math.round(minLength + ((maxLength - minLength) * speedRatio));
        }

        function setNeedleLength(needleId, speedKmh) {
            const needle = document.getElementById(needleId);
            if (!needle) return;
            const length = needleLengthForSpeed(speedKmh);
            needle.style.height = `${length}px`;
            needle.dataset.windSpeed = Number.isFinite(speedKmh) ? speedKmh.toFixed(1) : '';
            needle.dataset.needleLength = String(length);
        }

        function setNeedleLengths(liveSpeedKmh, predictedSpeedKmh) {
            setNeedleLength('needle-live', liveSpeedKmh);
            setNeedleLength('needle-predicted', predictedSpeedKmh);
        }

        function syncNeedleLengthsForCurrentState() {
            const prediction = getHistoricalPrediction(new Date(), selectedHour);
            setNeedleLengths(activeWindFlow.speed, prediction.speed);
        }

        function particleCountForSpeed(speed) {
            return Math.round(clamp(42 + (speed * 3.6), 48, 135));
        }

        function setWindFlow(speed, direction) {
            activeWindFlow = {
                speed: Number.isFinite(speed) ? speed : 10,
                direction: Number.isFinite(direction) ? direction : 90
            };
            canvas.dataset.flowSpeed = activeWindFlow.speed.toFixed(1);
            canvas.dataset.flowDirection = Math.round(activeWindFlow.direction);
            syncParticleDensity();
        }

        class WindParticle {
            constructor(w, h) {
                this.w = w;
                this.h = h;
                this.reset();
            }

            reset() {
                this.x = Math.random() * this.w;
                this.y = Math.random() * this.h;
                this.life = Math.random() * 80 + 40;
                this.maxLife = this.life;
                this.alpha = 0;
            }

            update(speed, angle) {
                this.life--;
                const angleRad = (angle - 90) * Math.PI / 180;
                const speedStep = clamp(speed * 0.24, 0.7, 7.5);
                this.x += Math.cos(angleRad) * speedStep;
                this.y += Math.sin(angleRad) * speedStep;

                if (this.x < 0) this.x = this.w;
                if (this.x > this.w) this.x = 0;
                if (this.y < 0) this.y = this.h;
                if (this.y > this.h) this.y = 0;

                if (this.life > this.maxLife * 0.8) {
                    this.alpha += 0.06;
                    if (this.alpha > 0.58) this.alpha = 0.58;
                } else if (this.life < this.maxLife * 0.2) {
                    this.alpha -= 0.045;
                    if (this.alpha < 0) this.alpha = 0;
                }

                if (this.life <= 0) {
                    this.reset();
                }
            }

            draw(ctx) {
                const speedGlow = clamp(activeWindFlow.speed / 28, 0.35, 1);
                ctx.strokeStyle = `rgba(103, 232, 249, ${this.alpha * speedGlow})`;
                ctx.lineWidth = 1.15 + speedGlow;
                ctx.beginPath();
                ctx.arc(this.x, this.y, 1, 0, Math.PI * 2);
                ctx.stroke();
            }
        }

        function syncParticleDensity() {
            const rect = canvas.getBoundingClientRect();
            const targetCount = particleCountForSpeed(activeWindFlow.speed);
            while (particles.length < targetCount) {
                particles.push(new WindParticle(rect.width, rect.height));
            }
            if (particles.length > targetCount) {
                particles.splice(targetCount);
            }
            canvas.dataset.particleCount = String(particles.length);
        }

        function initParticles() {
            const rect = canvas.getBoundingClientRect();
            particles = [];
            const targetCount = particleCountForSpeed(activeWindFlow.speed);
            for (let i = 0; i < targetCount; i++) {
                particles.push(new WindParticle(rect.width, rect.height));
            }
            canvas.dataset.particleCount = String(particles.length);
        }

        function animateParticles() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            syncParticleDensity();

            particles.forEach(p => {
                p.update(activeWindFlow.speed, activeWindFlow.direction);
                p.draw(ctx);
            });

            particleAnimationId = requestAnimationFrame(animateParticles);
        }

        // -------------------------------------------------------------------------
        // Core Open-Meteo Integration & Logic Hydration (Progressive Background Sync)
        // -------------------------------------------------------------------------
        async function syncLiveTelemetry() {
            // Streamlined URL requesting ONLY necessary rendering parameters
            const url = `${payload.forecastUrl}?latitude=${BENGALURU_LOCATION.lat}&longitude=${BENGALURU_LOCATION.lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m&hourly=wind_speed_10m,wind_direction_10m,precipitation&daily=wind_speed_10m_max,weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum&timezone=${encodeURIComponent(BENGALURU_LOCATION.timezone)}`;
            
            logToTerminal(`Initiating background sync with Open-Meteo [${BENGALURU_LOCATION.short}]...`, "system");
            
            let retries = 3;
            let delay = 1000;
            
            while (retries > 0) {
                try {
                    const response = await fetch(url);
                    if (!response.ok) throw new Error("HTTP " + response.status);
                    
                    const data = await response.json();
                    if (!data || !data.current || !data.hourly || !data.daily) {
                        throw new Error("Malformed JSON payload structure");
                    }
                    
                    // Success! Hydrate telemetry and update interface
                    telemetryData = data;
                    document.getElementById('offline-banner').classList.add('hidden');
                    updateUI();
                    
                    logToTerminal(`Live weather telemetry successfully synchronized for ${BENGALURU_LOCATION.name}.`, "success");
                    showToast(`Live Feed Connected`, `Live meteorological metrics loaded for ${BENGALURU_LOCATION.short}.`);
                    return;
                } catch (error) {
                    retries--;
                    logToTerminal(`Background synchronization attempt failed (Error: ${error.message}). Retries remaining: ${retries}`, "warn");
                    
                    if (retries === 0) {
                        logToTerminal("All synchronization retries offline. Retaining predictive baseline models.", "error");
                        document.getElementById('offline-banner').classList.remove('hidden');
                        return;
                    }
                    await new Promise(res => setTimeout(res, delay));
                    delay *= 1.5;
                }
            }
        }

        // Prediction-only fallback uses the embedded ML records and does not invent live telemetry.
        function buildPredictionOnlyData() {
            const now = new Date();
            const currentPrediction = getHistoricalPrediction(now, now.getHours());
            
            const hourlyTime = [];
            const hourlySpeed = [];
            const hourlyDirection = [];
            const hourlyPrecipitation = [];
            
            for (let i = 0; i < 24; i++) {
                const targetTime = new Date(now);
                targetTime.setHours(i);
                const hourlyPrediction = getHistoricalPrediction(targetTime, i);
                
                hourlyTime.push(targetTime.toISOString().substring(0, 16));
                hourlySpeed.push(hourlyPrediction.speed);
                hourlyDirection.push(hourlyPrediction.direction);
                hourlyPrecipitation.push(hourlyPrediction.precipitation);
            }

            const dailyTime = [];
            const dailySpeedMax = [];
            const dailyTempMax = [];
            const dailyTempMin = [];
            const dailyPrecipSum = [];
            const dailyWeatherCode = [];
            for (let i = 0; i < 7; i++) {
                const targetDate = new Date(now);
                targetDate.setDate(now.getDate() + i);
                const dayPredictions = Array.from({ length: 24 }, (_, hour) => getHistoricalPrediction(targetDate, hour));
                const daySpeeds = dayPredictions.map(item => item.speed);
                const dayTemps = dayPredictions.map(item => item.temperature).filter(value => typeof value === 'number');
                const dayRain = dayPredictions.map(item => item.precipitation).filter(value => typeof value === 'number');
                const dayCodes = dayPredictions.map(item => item.weatherCode).filter(value => typeof value === 'number');
                dailyTime.push(targetDate.toISOString().substring(0, 10));
                dailySpeedMax.push(Math.max(...daySpeeds));
                dailyTempMax.push(dayTemps.length ? Math.max(...dayTemps) : null);
                dailyTempMin.push(dayTemps.length ? Math.min(...dayTemps) : null);
                dailyPrecipSum.push(dayRain.length ? Math.round(dayRain.reduce((total, value) => total + value, 0) * 10) / 10 : null);
                dailyWeatherCode.push(dayCodes.length ? dominantWeatherCode(dayCodes) : null);
            }

            return {
                predictionOnly: true,
                current: {
                    temperature_2m: currentPrediction.temperature,
                    relative_humidity_2m: currentPrediction.humidity,
                    weather_code: currentPrediction.weatherCode,
                    wind_speed_10m: currentPrediction.speed,
                    wind_direction_10m: currentPrediction.direction,
                    wind_gusts_10m: null
                },
                hourly: {
                    time: hourlyTime,
                    wind_speed_10m: hourlySpeed,
                    wind_direction_10m: hourlyDirection,
                    precipitation: hourlyPrecipitation
                },
                daily: {
                    time: dailyTime,
                    wind_speed_10m_max: dailySpeedMax,
                    weather_code: dailyWeatherCode,
                    temperature_2m_max: dailyTempMax,
                    temperature_2m_min: dailyTempMin,
                    precipitation_probability_max: Array(7).fill(null),
                    precipitation_sum: dailyPrecipSum
                }
            };
        }

        // Hydrate GUI with telemetry data
        function updateUI() {
            if (!telemetryData) return;

            const current = telemetryData.current;
            const hasLiveTelemetry = !telemetryData.predictionOnly && current && typeof current.wind_speed_10m === 'number';

            // Set Live Weather Cards
            document.getElementById('live-weather-desc').innerText = hasLiveTelemetry ? interpretWeatherCode(current.weather_code) : 'Awaiting live feed';
            document.getElementById('live-temp').innerText = hasLiveTelemetry && typeof current.temperature_2m === 'number' ? `${Math.round(current.temperature_2m * 10) / 10}°C` : '--°C';
            document.getElementById('live-humidity').innerText = hasLiveTelemetry && typeof current.relative_humidity_2m === 'number' ? `Humidity: ${current.relative_humidity_2m}%` : 'Humidity: --%';

            // Set Live Wind Speeds
            const liveSpeedConverted = formatSpeed(current.wind_speed_10m);
            const gustSpeedConverted = hasLiveTelemetry ? formatSpeed(current.wind_gusts_10m) : '--';
            const unitLabel = UNIT_LABEL;

            document.getElementById('live-wind-speed').innerText = hasLiveTelemetry ? liveSpeedConverted : '--.-';
            document.getElementById('wind-unit-label').innerText = unitLabel;
            document.getElementById('live-wind-dir').innerText = hasLiveTelemetry ? `${Math.round(current.wind_direction_10m)}° ${getCardinal(current.wind_direction_10m)}` : 'Live direction pending';
            document.getElementById('live-wind-gusts').innerText = `Gusts: ${gustSpeedConverted} ${unitLabel}`;

            // Calculate historical prediction comparisons
            const now = new Date();
            const prediction = getHistoricalPrediction(now, now.getHours());

            // Alignment metrics
            const comparisonSpeed = hasLiveTelemetry ? current.wind_speed_10m : prediction.speed;
            const comparisonDirection = hasLiveTelemetry ? current.wind_direction_10m : prediction.direction;
            const rawSpeedDelta = Math.abs(comparisonSpeed - prediction.speed);
            let dirDelta = Math.abs(comparisonDirection - prediction.direction) % 360;
            if (dirDelta > 180) dirDelta = 360 - dirDelta;

            const speedMatch = Math.max(0, 100 - (rawSpeedDelta * 6));
            const directionMatch = Math.max(0, 100 - (dirDelta * 0.5));
            const matchPercentage = Math.round((speedMatch + directionMatch) / 2);

            const matchScoreEl = document.getElementById('live-match-score');
            matchScoreEl.innerText = hasLiveTelemetry ? `${matchPercentage}%` : '--%';
            matchScoreEl.className = `text-2xl font-black font-mono ${hasLiveTelemetry ? matchScoreClasses(matchPercentage) : 'text-slate-500'}`;
            document.getElementById('predicted-wind-baseline').innerText = `Predicted: ${formatSpeed(prediction.speed)} ${unitLabel}`;

            // Update Compass Needle Vectors
            const needleLive = document.getElementById('needle-live');
            const needlePred = document.getElementById('needle-predicted');

            setNeedleLengths(comparisonSpeed, prediction.speed);
            needleLive.style.transform = `rotate(${comparisonDirection}deg)`;
            needlePred.style.transform = `rotate(${prediction.direction}deg)`;
            setWindFlow(comparisonSpeed, comparisonDirection);

            // Update Compass Text Overlays
            document.getElementById('center-heading').innerText = `${Math.round(comparisonDirection)}° ${getCardinal(comparisonDirection)}`;
            document.getElementById('center-speed').innerText = hasLiveTelemetry ? liveSpeedConverted : formatSpeed(prediction.speed);
            document.getElementById('center-speed-unit').innerText = unitLabel;

            document.getElementById('comp-live-val').innerText = hasLiveTelemetry ? `${liveSpeedConverted} ${unitLabel} @ ${Math.round(current.wind_direction_10m)}°` : 'Live feed pending';
            document.getElementById('comp-pred-val').innerText = `${formatSpeed(prediction.speed)} ${unitLabel} @ ${Math.round(prediction.direction)}°`;
            document.getElementById('comp-divergence-val').innerText = hasLiveTelemetry ? `${Math.round(dirDelta)}°` : '--°';

            // Setup dynamic Seasonal Description
            const curMonth = now.getMonth();
            const prof = MONTHLY_PROFILES[curMonth];
            document.getElementById('seasonal-desc').innerText = `${prof.name} uses ${monthRecords(curMonth).length.toLocaleString('en-IN')} embedded hourly prediction records. Current profile: ${prof.tagline.toLowerCase()}.`;

            // Reliability panel dynamic updates
            document.getElementById('stats-learned').innerText = Number(payload.metrics.training_rows).toLocaleString('en-IN');
            document.getElementById('stats-learned-note').innerText = `${payload.metrics.validation_year} back-test validation`;
            document.getElementById('stats-dir-miss').innerText = `±${payload.metrics.direction_mae_deg}°`;
            document.getElementById('stats-speed-miss').innerText = `±${payload.metrics.speed_mae_kmh} km/h`;
            document.getElementById('stats-trust').innerText = `${prediction.confidence}%`;
            document.getElementById('weather-stats-trust').innerText = typeof prediction.weatherConfidence === 'number' ? `${prediction.weatherConfidence}%` : '--%';
            document.getElementById('weather-stats-trust').className = `text-lg font-bold mt-1 ${typeof prediction.weatherConfidence === 'number' ? matchScoreClasses(prediction.weatherConfidence) : 'text-slate-500'}`;
            document.getElementById('weather-stats-now').innerText = typeof prediction.temperature === 'number' ? `${prediction.temperature}°C` : '--';
            document.getElementById('weather-stats-now-note').innerText = `${interpretWeatherCode(prediction.weatherCode)} · ${typeof prediction.humidity === 'number' ? prediction.humidity : '--'}% humidity`;
            document.getElementById('weather-stats-temp').innerText = `±${payload.metrics.temperature_mae_c}°C`;
            document.getElementById('weather-stats-humidity').innerText = `±${payload.metrics.humidity_mae_pct}%`;
            document.getElementById('weather-stats-rain').innerText = `±${payload.metrics.precipitation_mae_mm} mm`;
            document.getElementById('weather-stats-wet').innerText = `${payload.metrics.wet_hour_accuracy_pct}%`;
            document.getElementById('weather-stats-wet').className = `text-lg font-bold mt-1 ${matchScoreClasses(payload.metrics.wet_hour_accuracy_pct)}`;
            document.getElementById('weather-reliability-note').innerText = `${prediction.stamp} embedded weather prediction`;

            // Render Charts
            buildHourlyChart();
            buildHourlyRainChart();
            buildWeeklyChart();
            buildWeatherChart();
        }

        // Handle dynamic scrubbing across the 24 hour timeline
        function scrubHour(hourVal) {
            selectedHour = parseInt(hourVal);
            
            const padHour = selectedHour.toString().padStart(2, '0');
            document.getElementById('scrub-hour-label').innerText = `${padHour}:00`;

            const targetTime = new Date();
            targetTime.setHours(selectedHour);
            
            const prediction = getHistoricalPrediction(targetTime, selectedHour);

            let targetSpeed = prediction.speed;
            let targetDir = prediction.direction;

            if (telemetryData && !telemetryData.predictionOnly && telemetryData.hourly) {
                targetSpeed = telemetryData.hourly.wind_speed_10m[selectedHour];
                targetDir = telemetryData.hourly.wind_direction_10m[selectedHour];
            }

            const unitLabel = UNIT_LABEL;

            document.getElementById('needle-live').style.transform = `rotate(${targetDir}deg)`;
            document.getElementById('needle-predicted').style.transform = `rotate(${prediction.direction}deg)`;
            setNeedleLengths(targetSpeed, prediction.speed);
            setWindFlow(targetSpeed, targetDir);

            document.getElementById('center-heading').innerText = `${Math.round(targetDir)}° ${getCardinal(targetDir)}`;
            document.getElementById('center-speed').innerText = formatSpeed(targetSpeed);
            document.getElementById('center-speed-unit').innerText = unitLabel;

            document.getElementById('comp-live-val').innerText = `${formatSpeed(targetSpeed)} ${unitLabel} @ ${Math.round(targetDir)}°`;
            document.getElementById('comp-pred-val').innerText = `${formatSpeed(prediction.speed)} ${unitLabel} @ ${Math.round(prediction.direction)}°`;

            let deltaD = Math.abs(targetDir - prediction.direction) % 360;
            if (deltaD > 180) deltaD = 360 - deltaD;
            document.getElementById('comp-divergence-val').innerText = `${Math.round(deltaD)}°`;

            logToTerminal(`Scrubbing clock vector timeline: hour=${padHour}:00 | Predicted direction=${prediction.direction}° | Forecasted direction=${targetDir}°`, 'info');
        }

        // Manual refresh trigger
        async function refreshData() {
            const icon = document.getElementById('refresh-icon');
            icon.classList.add('animate-spin');
            
            logToTerminal("Initiating manual telemetry refresh...", "system");
            await syncLiveTelemetry();
            
            setTimeout(() => {
                icon.classList.remove('animate-spin');
            }, 600);
        }

        // -------------------------------------------------------------------------
        // Chart.js Building Block Methods
        // -------------------------------------------------------------------------
        function seededRandom(seed) {
            let value = seed >>> 0;
            return () => {
                value += 0x6D2B79F5;
                let mixed = value;
                mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
                mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
                return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
            };
        }

        function averagePathLength(size) {
            if (size <= 1) return 0;
            if (size === 2) return 1;
            return 2 * (Math.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size);
        }

        function buildIsolationTree(rows, depth, maxDepth, random) {
            if (depth >= maxDepth || rows.length <= 1) {
                return { external: true, size: rows.length };
            }

            const featureCount = rows[0].features.length;
            const featureIndex = Math.floor(random() * featureCount);
            const values = rows.map(row => row.features[featureIndex]);
            const min = Math.min(...values);
            const max = Math.max(...values);

            if (min === max) {
                return { external: true, size: rows.length };
            }

            const split = min + random() * (max - min);
            const left = rows.filter(row => row.features[featureIndex] < split);
            const right = rows.filter(row => row.features[featureIndex] >= split);

            if (!left.length || !right.length) {
                return { external: true, size: rows.length };
            }

            return {
                external: false,
                featureIndex,
                split,
                left: buildIsolationTree(left, depth + 1, maxDepth, random),
                right: buildIsolationTree(right, depth + 1, maxDepth, random)
            };
        }

        function isolationPathLength(row, node, depth = 0) {
            if (node.external) {
                return depth + averagePathLength(node.size);
            }
            return row.features[node.featureIndex] < node.split
                ? isolationPathLength(row, node.left, depth + 1)
                : isolationPathLength(row, node.right, depth + 1);
        }

        function detectHourlyWindAnomalies(rows) {
            if (rows.length < 8) {
                return { byHour: new Map(), count: 0, message: "Anomaly scan: waiting for live data" };
            }

            const random = seededRandom(20260522);
            const treeCount = 72;
            const sampleSize = Math.min(16, rows.length);
            const maxDepth = Math.ceil(Math.log2(sampleSize));
            const trees = [];

            for (let i = 0; i < treeCount; i++) {
                const sample = [];
                const used = new Set();
                while (sample.length < sampleSize) {
                    const pick = Math.floor(random() * rows.length);
                    if (!used.has(pick)) {
                        used.add(pick);
                        sample.push(rows[pick]);
                    }
                }
                trees.push(buildIsolationTree(sample, 0, maxDepth, random));
            }

            const normalization = averagePathLength(sampleSize) || 1;
            const scores = rows.map(row => {
                const path = trees.reduce((total, tree) => total + isolationPathLength(row, tree), 0) / treeCount;
                return { ...row, score: Math.pow(2, -path / normalization) };
            });

            const flagged = scores
                .filter(row => row.score >= 0.62 || (row.speedDelta >= 8 && row.directionDelta >= 55))
                .sort((a, b) => b.score - a.score)
                .slice(0, 4);

            const byHour = new Map(flagged.map(row => [row.hour, row]));
            const message = flagged.length
                ? `Anomaly scan: ${flagged.length} unusual hour${flagged.length > 1 ? "s" : ""} flagged`
                : "Anomaly scan: no unusual live hours";

            return { byHour, count: flagged.length, message };
        }

        function detectHourlyRainAnomalies(rows) {
            if (rows.length < 8) {
                return { byHour: new Map(), count: 0, message: "Rain anomaly: waiting for live data" };
            }

            const random = seededRandom(20260523);
            const treeCount = 72;
            const sampleSize = Math.min(16, rows.length);
            const maxDepth = Math.ceil(Math.log2(sampleSize));
            const trees = [];

            for (let i = 0; i < treeCount; i++) {
                const sample = [];
                const used = new Set();
                while (sample.length < sampleSize) {
                    const pick = Math.floor(random() * rows.length);
                    if (!used.has(pick)) {
                        used.add(pick);
                        sample.push(rows[pick]);
                    }
                }
                trees.push(buildIsolationTree(sample, 0, maxDepth, random));
            }

            const normalization = averagePathLength(sampleSize) || 1;
            const scores = rows.map(row => {
                const path = trees.reduce((total, tree) => total + isolationPathLength(row, tree), 0) / treeCount;
                return { ...row, score: Math.pow(2, -path / normalization) };
            });

            const flagged = scores
                .filter(row => row.score >= 0.64 || row.rainDelta >= 2 || (row.liveRain >= 1.5 && row.predictedRain <= 0.15))
                .sort((a, b) => b.score - a.score)
                .slice(0, 4);

            const byHour = new Map(flagged.map(row => [row.hour, row]));
            const message = flagged.length
                ? `Rain anomaly: ${flagged.length} unusual hour${flagged.length > 1 ? "s" : ""} flagged`
                : "Rain anomaly: no unusual live hours";

            return { byHour, count: flagged.length, message };
        }

        function buildHourlyChart() {
            const ctx = document.getElementById('hourlyChart').getContext('2d');
            
            if (hourlyChart) {
                hourlyChart.destroy();
            }

            const now = new Date();
            const predictionDataset = [];
            const liveDataset = [];
            const anomalyDataset = [];
            const anomalyRows = [];
            const labels = [];

            for (let i = 0; i < 24; i++) {
                const targetTime = new Date(now);
                targetTime.setHours(i);
                
                const predictedObj = getHistoricalPrediction(targetTime, i);
                const predictedSpeed = formatSpeed(predictedObj.speed);
                predictionDataset.push(predictedSpeed);

                if (telemetryData && !telemetryData.predictionOnly && telemetryData.hourly && telemetryData.hourly.wind_speed_10m) {
                    const liveSpeedRaw = telemetryData.hourly.wind_speed_10m[i];
                    const liveDirection = telemetryData.hourly.wind_direction_10m?.[i];
                    const liveSpeed = formatSpeed(liveSpeedRaw);
                    liveDataset.push(liveSpeed);

                    if (typeof liveSpeedRaw === 'number' && typeof liveDirection === 'number') {
                        let directionDelta = Math.abs(liveDirection - predictedObj.direction) % 360;
                        if (directionDelta > 180) directionDelta = 360 - directionDelta;
                        const speedDelta = Math.abs(liveSpeedRaw - predictedObj.speed);
                        const directionRad = liveDirection * Math.PI / 180;

                        anomalyRows.push({
                            hour: i,
                            speed: liveSpeed,
                            speedDelta,
                            directionDelta,
                            features: [
                                liveSpeedRaw / 35,
                                Math.sin(directionRad),
                                Math.cos(directionRad),
                                speedDelta / 18,
                                directionDelta / 180,
                                i / 23
                            ]
                        });
                    }
                } else {
                    liveDataset.push(null);
                }

                labels.push(`${i.toString().padStart(2, '0')}:00`);
            }

            const anomalyScan = detectHourlyWindAnomalies(anomalyRows);
            for (let i = 0; i < 24; i++) {
                anomalyDataset.push(anomalyScan.byHour.get(i)?.speed ?? null);
            }

            const anomalyStatus = document.getElementById('anomaly-scan-status');
            if (anomalyStatus) {
                anomalyStatus.textContent = anomalyScan.message;
                anomalyStatus.className = anomalyScan.count
                    ? 'text-[10px] font-mono text-rose-400 text-right'
                    : 'text-[10px] font-mono text-slate-500 text-right';
            }

            const unitStr = UNIT_LABEL;

            hourlyChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: `ML Prediction (${unitStr})`,
                            data: predictionDataset,
                            borderColor: '#fbbf24',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            pointStyle: 'circle',
                            pointRadius: 0,
                            fill: false,
                            tension: 0.4
                        },
                        {
                            label: `Live Forecast (${unitStr})`,
                            data: liveDataset,
                            borderColor: '#22d3ee',
                            borderWidth: 3,
                            backgroundColor: 'rgba(34, 211, 238, 0.05)',
                            fill: true,
                            tension: 0.4,
                            pointStyle: 'line',
                            pointBackgroundColor: '#22d3ee',
                            pointHoverRadius: 6
                        },
                        {
                            label: 'Anomaly',
                            data: anomalyDataset,
                            borderColor: '#fb7185',
                            backgroundColor: '#fb7185',
                            pointBorderColor: '#fecdd3',
                            pointBorderWidth: 2,
                            pointRadius: 6,
                            pointHoverRadius: 8,
                            pointStyle: 'circle',
                            showLine: false,
                            fill: false,
                            order: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: {
                                color: '#94a3b8',
                                usePointStyle: true,
                                pointStyle: 'line',
                                boxWidth: 28,
                                boxHeight: 6,
                                font: { family: 'Plus Jakarta Sans', size: 11 }
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: '#0f172a',
                            borderColor: '#1e293b',
                            borderWidth: 1,
                            titleColor: '#f1f5f9',
                            bodyColor: '#cbd5e1'
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(255,255,255,0.03)' },
                            ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
                        }
                    }
                }
            });
        }

        function buildHourlyRainChart() {
            const canvasEl = document.getElementById('hourlyRainChart');
            if (!canvasEl) return;

            if (hourlyRainChart) {
                hourlyRainChart.destroy();
            }

            const now = new Date();
            const labels = [];
            const predictionDataset = [];
            const liveDataset = [];
            const anomalyRows = [];
            const anomalyDataset = [];

            for (let i = 0; i < 24; i++) {
                const targetTime = new Date(now);
                targetTime.setHours(i);

                const predictedObj = getHistoricalPrediction(targetTime, i);
                const predictedRain = typeof predictedObj.precipitation === 'number' ? Math.round(predictedObj.precipitation * 100) / 100 : null;
                predictionDataset.push(predictedRain);

                if (telemetryData && !telemetryData.predictionOnly && telemetryData.hourly && Array.isArray(telemetryData.hourly.precipitation)) {
                    const liveRainRaw = telemetryData.hourly.precipitation[i];
                    const liveRain = typeof liveRainRaw === 'number' ? Math.round(liveRainRaw * 100) / 100 : null;
                    liveDataset.push(liveRain);

                    if (typeof liveRainRaw === 'number' && typeof predictedRain === 'number') {
                        const rainDelta = Math.abs(liveRainRaw - predictedRain);
                        anomalyRows.push({
                            hour: i,
                            rain: liveRain,
                            liveRain: liveRainRaw,
                            predictedRain,
                            rainDelta,
                            features: [
                                Math.min(liveRainRaw / 12, 1),
                                Math.min(predictedRain / 12, 1),
                                Math.min(rainDelta / 8, 1),
                                i / 23,
                                liveRainRaw > 0 ? 1 : 0,
                                predictedRain > 0 ? 1 : 0
                            ]
                        });
                    }
                } else {
                    liveDataset.push(null);
                }

                labels.push(`${i.toString().padStart(2, '0')}:00`);
            }

            const anomalyScan = detectHourlyRainAnomalies(anomalyRows);
            for (let i = 0; i < 24; i++) {
                anomalyDataset.push(anomalyScan.byHour.get(i)?.rain ?? null);
            }

            const anomalyStatus = document.getElementById('rain-hourly-anomaly-status');
            if (anomalyStatus) {
                anomalyStatus.textContent = anomalyScan.message;
                anomalyStatus.className = anomalyScan.count
                    ? 'text-[10px] font-mono text-rose-400 text-right'
                    : 'text-[10px] font-mono text-slate-500 text-right';
            }

            hourlyRainChart = new Chart(canvasEl.getContext('2d'), {
                type: 'line',
                data: {
                    labels,
                    datasets: [
                        {
                            label: 'ML Predicted Rainfall (mm)',
                            data: predictionDataset,
                            borderColor: 'rgba(74, 222, 128, 0.76)',
                            backgroundColor: 'rgba(74, 222, 128, 0.04)',
                            pointBackgroundColor: '#4ade80',
                            pointBorderColor: '#0f172a',
                            pointBorderWidth: 2,
                            pointRadius: 3,
                            pointHoverRadius: 7,
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.35,
                            order: 2
                        },
                        {
                            label: 'Live Rainfall (mm)',
                            data: liveDataset,
                            borderColor: '#22c55e',
                            backgroundColor: 'rgba(34, 197, 94, 0.08)',
                            pointBackgroundColor: '#22c55e',
                            pointBorderColor: '#0f172a',
                            pointBorderWidth: 2,
                            pointRadius: 4,
                            pointHoverRadius: 7,
                            borderWidth: 3,
                            fill: true,
                            tension: 0.35,
                            order: 1
                        },
                        {
                            label: 'Anomaly',
                            data: anomalyDataset,
                            borderColor: '#fb7185',
                            backgroundColor: '#fb7185',
                            pointBorderColor: '#fecdd3',
                            pointBorderWidth: 2,
                            pointRadius: 6,
                            pointHoverRadius: 8,
                            pointStyle: 'circle',
                            showLine: false,
                            fill: false,
                            order: 0
                        }
                    ]
                },
                options: weatherChartOptions('Rainfall mm', true)
            });
        }

        function buildWeeklyChart() {
            const ctx = document.getElementById('weeklyChart').getContext('2d');
            
            if (weeklyChart) {
                weeklyChart.destroy();
            }

            const labels = [];
            const liveSpeedDataset = [];
            const predictedSpeedDataset = [];
            const unitStr = UNIT_LABEL;
            const now = new Date();

            for (let i = 0; i < 7; i++) {
                const targetDate = new Date();
                targetDate.setDate(now.getDate() + i);
                
                const dayName = targetDate.toLocaleDateString('en-US', { weekday: 'short' });
                const dateStr = targetDate.getDate().toString().padStart(2, '0');
                labels.push(`${dayName} ${dateStr}`);
                
                // Live Max Forecast
                if (telemetryData && !telemetryData.predictionOnly && telemetryData.daily && telemetryData.daily.wind_speed_10m_max) {
                    liveSpeedDataset.push(formatSpeed(telemetryData.daily.wind_speed_10m_max[i]));
                } else {
                    liveSpeedDataset.push(null);
                }

                const predictedDaySpeeds = Array.from({ length: 24 }, (_, hour) => getHistoricalPrediction(targetDate, hour).speed);
                predictedSpeedDataset.push(formatSpeed(Math.max(...predictedDaySpeeds))); 
            }

            weeklyChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            type: 'line',
                            label: `ML Predicted Max (${unitStr})`,
                            data: predictedSpeedDataset,
                            borderColor: '#fbbf24', 
                            borderWidth: 2,
                            borderDash: [5, 5],
                            pointStyle: 'circle',
                            pointBackgroundColor: '#fbbf24',
                            pointHoverRadius: 8,
                            pointRadius: 7,
                            pointBorderColor: '#0f172a',
                            pointBorderWidth: 3,
                            pointHitRadius: 12,
                            fill: false,
                            tension: 0.3,
                            order: 1 
                        },
                        {
                            type: 'bar',
                            label: `Live Forecast Max (${unitStr})`,
                            data: liveSpeedDataset,
                            backgroundColor: 'rgba(34, 211, 238, 0.25)', 
                            borderColor: '#06b6d4',
                            hoverBackgroundColor: 'rgba(34, 211, 238, 0.55)',
                            borderWidth: 1.5,
                            borderRadius: 6,
                            pointStyle: 'line',
                            order: 2 
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { 
                            display: true,
                            position: 'top',
                            labels: {
                                color: '#94a3b8',
                                usePointStyle: true,
                                pointStyle: 'line',
                                boxWidth: 28,
                                boxHeight: 6,
                                font: { family: 'Plus Jakarta Sans', size: 11 }
                            }
                        },
                        tooltip: {
                            backgroundColor: '#0f172a',
                            borderColor: '#1e293b',
                            borderWidth: 1,
                            titleColor: '#f1f5f9',
                            bodyColor: '#cbd5e1'
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { color: '#64748b', font: { family: 'Plus Jakarta Sans', size: 11 } }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
                        }
                    }
                }
            });
        }

        function hasDailyWeatherData() {
            return Boolean(
                telemetryData &&
                telemetryData.daily &&
                Array.isArray(telemetryData.daily.temperature_2m_max) &&
                telemetryData.daily.temperature_2m_max.some(value => typeof value === 'number') &&
                getDailyWeatherPrediction(new Date()).hasWeather
            );
        }

        function getDailyWeatherPrediction(dateObj) {
            const hourly = Array.from({ length: 24 }, (_, hour) => getHistoricalPrediction(dateObj, hour));
            const temps = hourly.map(item => item.temperature).filter(value => typeof value === 'number');
            const rain = hourly.map(item => item.precipitation).filter(value => typeof value === 'number');
            const codes = hourly.map(item => item.weatherCode).filter(value => typeof value === 'number');
            const confidences = hourly.map(item => item.weatherConfidence).filter(value => typeof value === 'number');

            return {
                hasWeather: temps.length > 0,
                maxTemp: temps.length ? Math.round(Math.max(...temps) * 10) / 10 : null,
                minTemp: temps.length ? Math.round(Math.min(...temps) * 10) / 10 : null,
                rain: rain.length ? Math.round(rain.reduce((total, value) => total + value, 0) * 10) / 10 : null,
                weatherCode: codes.length ? dominantWeatherCode(codes) : null,
                confidence: confidences.length ? Math.round(confidences.reduce((total, value) => total + value, 0) / confidences.length) : null
            };
        }

        function detectWeatherParameterAnomalies(rows, options) {
            const flagged = rows
                .filter(row => {
                    if (!row.hasLive || typeof row.live !== 'number' || typeof row.predicted !== 'number') return false;
                    const delta = Math.abs(row.live - row.predicted);
                    return delta >= options.deltaThreshold || (
                        typeof options.heavyLiveThreshold === 'number' &&
                        row.live >= options.heavyLiveThreshold &&
                        row.predicted <= (options.lowPredictionThreshold || 0)
                    );
                })
                .map(row => ({ ...row, delta: Math.abs(row.live - row.predicted) }))
                .sort((a, b) => b.delta - a.delta)
                .slice(0, 3);

            const byIndex = new Map(flagged.map(row => [row.index, row]));
            const message = flagged.length
                ? `${options.label}: ${flagged.length} unusual day${flagged.length > 1 ? "s" : ""}`
                : rows.some(row => row.hasLive) ? `${options.label}: no unusual days` : `${options.label}: waiting for live forecast`;

            return { byIndex, count: flagged.length, message };
        }

        function weatherChartOptions(yTitle, beginAtZero = false) {
            return {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#94a3b8',
                            usePointStyle: true,
                            pointStyle: 'line',
                            boxWidth: 26,
                            boxHeight: 6,
                            font: { family: 'Plus Jakarta Sans', size: 11 }
                        }
                    },
                    tooltip: {
                        backgroundColor: '#0f172a',
                        borderColor: '#1e293b',
                        borderWidth: 1,
                        titleColor: '#f1f5f9',
                        bodyColor: '#cbd5e1'
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#64748b', font: { family: 'Plus Jakarta Sans', size: 11 } }
                    },
                    y: {
                        beginAtZero,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        title: { display: true, text: yTitle, color: '#64748b', font: { family: 'Plus Jakarta Sans', size: 11 } },
                        ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
                    }
                }
            };
        }

        function renderWeatherParameterChart(config) {
            const canvasEl = document.getElementById(config.canvasId);
            const status = document.getElementById(config.statusId);
            if (!canvasEl) return { count: 0, message: `${config.anomalyLabel}: chart unavailable` };

            if (weatherCharts[config.key]) {
                weatherCharts[config.key].destroy();
            }

            const anomalyScan = detectWeatherParameterAnomalies(config.rows, {
                label: config.anomalyLabel,
                deltaThreshold: config.deltaThreshold,
                heavyLiveThreshold: config.heavyLiveThreshold,
                lowPredictionThreshold: config.lowPredictionThreshold
            });
            const anomalyPoints = config.rows.map(row => anomalyScan.byIndex.get(row.index)?.marker ?? null);

            weatherCharts[config.key] = new Chart(canvasEl.getContext('2d'), {
                type: 'line',
                data: {
                    labels: config.labels,
                    datasets: [
                        {
                            type: 'line',
                            label: config.liveLabel,
                            data: config.liveData,
                            borderColor: config.liveColor,
                            backgroundColor: config.liveFill,
                            pointBackgroundColor: config.liveColor,
                            pointBorderColor: '#0f172a',
                            pointBorderWidth: 2,
                            pointRadius: 5,
                            pointHoverRadius: 7,
                            borderWidth: 3,
                            fill: config.fillLive,
                            tension: 0.35,
                            order: 1
                        },
                        {
                            type: 'line',
                            label: config.predictedLabel,
                            data: config.predictedData,
                            borderColor: config.predictedColor,
                            backgroundColor: config.predictedFill,
                            pointBackgroundColor: config.predictedPointColor,
                            pointBorderColor: '#0f172a',
                            pointBorderWidth: 2,
                            pointRadius: 4,
                            pointHoverRadius: 7,
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.35,
                            order: 2
                        },
                        {
                            type: 'line',
                            label: 'Anomaly',
                            data: anomalyPoints,
                            borderColor: '#fb7185',
                            backgroundColor: '#fb7185',
                            pointBorderColor: '#fecdd3',
                            pointBorderWidth: 2,
                            pointRadius: 6,
                            pointHoverRadius: 8,
                            pointStyle: 'circle',
                            showLine: false,
                            fill: false,
                            order: 0
                        }
                    ]
                },
                options: weatherChartOptions(config.yTitle, config.beginAtZero)
            });

            if (status) {
                status.textContent = anomalyScan.message;
                status.className = anomalyScan.count
                    ? 'text-[10px] font-mono text-rose-400 text-right'
                    : 'text-[10px] font-mono text-slate-500 text-right';
            }

            return anomalyScan;
        }

        function buildWeatherChart() {
            const section = document.getElementById('weather-outlook-section');
            const status = document.getElementById('weather-graph-status');
            const strip = document.getElementById('weather-condition-strip');
            const canvases = ['weatherMaxChart', 'weatherMinChart', 'weatherRainChart'];
            
            if (!section) return;

            if (!hasDailyWeatherData()) {
                section.classList.add('hidden');
                Object.values(weatherCharts).forEach(chart => chart.destroy());
                weatherCharts = {};
                if (status) status.textContent = 'Temperature graph: waiting for live forecast';
                canvases.forEach(id => {
                    const canvasStatus = document.getElementById(id.replace('Chart', '-status').replace('weatherMax', 'weather-max').replace('weatherMin', 'weather-min').replace('weatherRain', 'weather-rain'));
                    if (canvasStatus) canvasStatus.textContent = 'Waiting for live forecast';
                });
                return;
            }

            section.classList.remove('hidden');

            const daily = telemetryData.daily;
            const labels = [];
            const predictedMaxTemps = [];
            const predictedMinTemps = [];
            const predictedRain = [];
            const liveMaxTemps = [];
            const liveMinTemps = [];
            const liveRain = [];
            const maxRows = [];
            const minRows = [];
            const rainRows = [];

            for (let index = 0; index < 7; index++) {
                const dateStr = daily.time[index];
                const date = dateStr ? new Date(`${dateStr}T00:00`) : new Date();
                if (!dateStr) date.setDate(new Date().getDate() + index);
                labels.push(date.toLocaleDateString('en-US', { weekday: 'short', day: '2-digit' }));

                const predicted = getDailyWeatherPrediction(date);
                const liveMax = typeof daily.temperature_2m_max?.[index] === 'number' ? Math.round(daily.temperature_2m_max[index] * 10) / 10 : null;
                const liveMin = typeof daily.temperature_2m_min?.[index] === 'number' ? Math.round(daily.temperature_2m_min[index] * 10) / 10 : null;
                const liveRainSum = typeof daily.precipitation_sum?.[index] === 'number' ? Math.round(daily.precipitation_sum[index] * 10) / 10 : null;
                const hasLiveMax = !telemetryData.predictionOnly && typeof liveMax === 'number' && typeof predicted.maxTemp === 'number';
                const hasLiveMin = !telemetryData.predictionOnly && typeof liveMin === 'number' && typeof predicted.minTemp === 'number';
                const hasLiveRain = !telemetryData.predictionOnly && typeof liveRainSum === 'number' && typeof predicted.rain === 'number';

                predictedMaxTemps.push(predicted.maxTemp);
                predictedMinTemps.push(predicted.minTemp);
                predictedRain.push(predicted.rain);
                liveMaxTemps.push(telemetryData.predictionOnly ? null : liveMax);
                liveMinTemps.push(telemetryData.predictionOnly ? null : liveMin);
                liveRain.push(telemetryData.predictionOnly ? null : liveRainSum);

                maxRows.push({ index, hasLive: hasLiveMax, live: liveMax, predicted: predicted.maxTemp, marker: liveMax });
                minRows.push({ index, hasLive: hasLiveMin, live: liveMin, predicted: predicted.minTemp, marker: liveMin });
                rainRows.push({ index, hasLive: hasLiveRain, live: liveRainSum, predicted: predicted.rain, marker: liveRainSum });
            }

            if (strip) {
                strip.innerHTML = labels.map((label, index) => {
                    const predicted = getDailyWeatherPrediction(new Date(`${daily.time[index]}T00:00`));
                    const code = daily.weather_code?.[index] ?? predicted.weatherCode;
                    const rainChance = daily.precipitation_probability_max?.[index];
                    const rainValue = liveRain[index] ?? predicted.rain;
                    const rainText = typeof rainChance === 'number'
                        ? `${rainChance}% rain`
                        : typeof rainValue === 'number' ? `${rainValue} mm`
                        : '--';
                    const confidenceText = typeof predicted.confidence === 'number' ? `${predicted.confidence}% ML` : 'ML pending';
                    return `
                        <div class="rounded-xl border border-slate-800/80 bg-slate-950/55 px-3 py-2 min-w-0">
                            <div class="flex items-center justify-between gap-2">
                                <span class="font-mono text-[10px] text-slate-500">${label}</span>
                                <i class="${weatherIconForCode(code)} text-slate-300"></i>
                            </div>
                            <div class="mt-1 truncate text-[11px] font-semibold text-slate-200">${interpretWeatherCode(code)}</div>
                            <div class="mt-0.5 flex items-center justify-between gap-2 text-[10px] font-mono">
                                <span class="text-cyan-400">${rainText}</span>
                                <span class="text-slate-500">${confidenceText}</span>
                            </div>
                        </div>
                    `;
                }).join('');
            }

            const maxScan = renderWeatherParameterChart({
                key: 'max',
                canvasId: 'weatherMaxChart',
                statusId: 'weather-max-status',
                labels,
                rows: maxRows,
                liveData: liveMaxTemps,
                predictedData: predictedMaxTemps,
                liveLabel: 'Live Max (°C)',
                predictedLabel: 'ML Predicted Max (°C)',
                yTitle: 'Max temperature °C',
                anomalyLabel: 'Max temp anomaly',
                deltaThreshold: 4.5,
                liveColor: '#fbbf24',
                liveFill: 'rgba(251, 191, 36, 0.08)',
                predictedColor: 'rgba(251, 191, 36, 0.72)',
                predictedFill: 'rgba(251, 191, 36, 0.03)',
                predictedPointColor: '#fbbf24',
                fillLive: true,
                beginAtZero: false
            });

            const minScan = renderWeatherParameterChart({
                key: 'min',
                canvasId: 'weatherMinChart',
                statusId: 'weather-min-status',
                labels,
                rows: minRows,
                liveData: liveMinTemps,
                predictedData: predictedMinTemps,
                liveLabel: 'Live Min (°C)',
                predictedLabel: 'ML Predicted Min (°C)',
                yTitle: 'Min temperature °C',
                anomalyLabel: 'Min temp anomaly',
                deltaThreshold: 4.5,
                liveColor: '#22d3ee',
                liveFill: 'rgba(34, 211, 238, 0.06)',
                predictedColor: 'rgba(34, 211, 238, 0.70)',
                predictedFill: 'rgba(34, 211, 238, 0.03)',
                predictedPointColor: '#22d3ee',
                fillLive: false,
                beginAtZero: false
            });

            const rainScan = renderWeatherParameterChart({
                key: 'rain',
                canvasId: 'weatherRainChart',
                statusId: 'weather-rain-status',
                labels,
                rows: rainRows,
                liveData: liveRain,
                predictedData: predictedRain,
                liveLabel: 'Live Rainfall (mm)',
                predictedLabel: 'ML Predicted Rainfall (mm)',
                yTitle: 'Rainfall mm',
                anomalyLabel: 'Rainfall anomaly',
                deltaThreshold: 8,
                heavyLiveThreshold: 12,
                lowPredictionThreshold: 1.5,
                liveColor: '#22c55e',
                liveFill: 'rgba(34, 197, 94, 0.08)',
                predictedColor: 'rgba(74, 222, 128, 0.74)',
                predictedFill: 'rgba(74, 222, 128, 0.05)',
                predictedPointColor: '#4ade80',
                fillLive: false,
                beginAtZero: true
            });

            const totalAnomalies = maxScan.count + minScan.count;
            if (status) {
                status.textContent = totalAnomalies
                    ? `Temperature anomaly: ${totalAnomalies} parameter alert${totalAnomalies > 1 ? 's' : ''} across 7 days`
                    : 'Temperature anomaly: no unusual parameter days';
                status.className = totalAnomalies
                    ? 'text-[10px] font-mono text-rose-400 text-right'
                    : 'text-[10px] font-mono text-slate-500 text-right';
            }
        }

        // Initialize Month Cards Grid (Continuous display of core metrics)
        function buildMonthlyCards() {
            const gridContainer = document.getElementById('monthly-grid');
            if (!gridContainer) return;
            
            gridContainer.innerHTML = '';

            MONTHLY_PROFILES.forEach((profile) => {
                const card = document.createElement('div');
                card.id = `mon-card-${profile.id}`;
                card.onclick = () => focusMonth(profile.id);

                const degMatch = profile.direction.match(/\((\d+)°\)/);
                const deg = degMatch ? parseInt(degMatch[1]) : 0;

                card.className = "cursor-pointer bg-slate-900/40 p-4 rounded-2xl border border-slate-800/80 hover:border-slate-700/80 hover:bg-slate-900/60 transition-all duration-300 flex flex-col justify-between min-h-[175px]";
                
                card.innerHTML = `
                    <div>
                        <span class="text-[9px] font-mono text-slate-500 uppercase tracking-wider block">${profile.subtitle.split(' ')[0]} Season</span>
                        <h4 class="text-sm font-extrabold text-white mt-0.5">${profile.name}</h4>
                    </div>

                    <div class="w-full flex items-center justify-center py-3">
                        <div class="w-14 h-14 rounded-full bg-slate-950 border border-slate-700/70 flex items-center justify-center text-slate-300 shadow-[0_0_14px_rgba(148,163,184,0.08)] transition-transform duration-300 hover:scale-110" title="Dominant direction: ${profile.direction}">
                            <i class="fa-solid fa-arrow-up text-xl" style="transform: rotate(${deg}deg)"></i>
                        </div>
                    </div>
                    
                    <div class="flex items-center justify-between">
                        <div class="flex flex-col">
                            <span class="text-[9px] text-slate-500 font-mono">AVG SPEED</span>
                            <span class="text-xs font-black font-mono text-cyan-400">${profile.speed}</span>
                        </div>
                    </div>
                    
                    <div class="border-t border-slate-800/30 pt-2 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                        <span>Dir: ${profile.direction.split(' ')[0]}</span>
                        <span class="text-slate-600">${profile.direction}</span>
                    </div>
                `;
                gridContainer.appendChild(card);
            });

            focusMonth(selectedMonthId);
        }

        // Handle focused climatology expansion display
        function focusMonth(monthId) {
            selectedMonthId = monthId;
            const profile = MONTHLY_PROFILES[monthId];

            MONTHLY_PROFILES.forEach((p) => {
                const card = document.getElementById(`mon-card-${p.id}`);
                if (card) {
                    if (p.id === monthId) {
                        card.className = "cursor-pointer bg-slate-950 p-4 rounded-2xl border-2 border-cyan-500/80 shadow-[0_0_15px_rgba(6,182,212,0.15)] transition-all duration-300 flex flex-col justify-between min-h-[175px]";
                    } else {
                        card.className = "cursor-pointer bg-slate-900/40 p-4 rounded-2xl border border-slate-800/80 hover:border-slate-700/80 hover:bg-slate-900/60 transition-all duration-300 flex flex-col justify-between min-h-[175px]";
                    }
                }
            });

            document.getElementById('mon-subtitle').innerText = profile.subtitle;
            document.getElementById('mon-title').innerText = profile.name;
            document.getElementById('mon-tagline').innerText = `"${profile.tagline}"`;
            document.getElementById('mon-speed').innerText = profile.speed;
            document.getElementById('mon-direction').innerText = profile.direction;
            document.getElementById('mon-moisture').innerText = profile.confidence;
            document.getElementById('mon-activity').innerText = profile.activity;
            document.getElementById('mon-analysis').innerText = profile.analysis;

            const degMatch = profile.direction.match(/\((\d+)°\)/);
            const deg = degMatch ? parseInt(degMatch[1]) : 0;
            const detailArrow = document.getElementById('mon-detail-arrow');
            if (detailArrow) {
                detailArrow.style.transform = `rotate(${deg}deg)`;
            }
        }

        function dominantWeatherCode(codes) {
            const priority = [99, 96, 95, 82, 81, 80, 65, 63, 61, 55, 53, 51, 48, 45, 3, 2, 1, 0];
            const counts = codes.reduce((map, code) => {
                map.set(code, (map.get(code) || 0) + 1);
                return map;
            }, new Map());
            return priority.find(code => counts.has(code)) ?? codes[0] ?? null;
        }

        function interpretWeatherCode(code) {
            const codes = {
                0: "Clear Sky",
                1: "Mainly Clear",
                2: "Partly Cloudy",
                3: "Overcast",
                45: "Fog",
                48: "Depositing Rime Fog",
                51: "Light Drizzle",
                53: "Moderate Drizzle",
                55: "Dense Drizzle",
                61: "Slight Rain",
                63: "Moderate Rain",
                65: "Heavy Rain",
                71: "Slight Snow",
                73: "Moderate Snow",
                75: "Heavy Snow",
                77: "Snow Grains",
                80: "Slight Showers",
                81: "Moderate Showers",
                82: "Violent Showers",
                95: "Thunderstorm",
                96: "Thunderstorm + Slight Hail",
                99: "Thunderstorm + Heavy Hail"
            };
            return codes[code] || "Stable Atmosphere";
        }

        function weatherIconForCode(code) {
            if (code === 0) return "fa-solid fa-sun text-amber-300";
            if ([1, 2].includes(code)) return "fa-solid fa-cloud-sun text-amber-200";
            if (code === 3) return "fa-solid fa-cloud text-slate-300";
            if ([45, 48].includes(code)) return "fa-solid fa-smog text-slate-400";
            if ([51, 53, 55, 61, 63, 65, 80, 81, 82].includes(code)) return "fa-solid fa-cloud-rain text-cyan-300";
            if ([95, 96, 99].includes(code)) return "fa-solid fa-cloud-bolt text-amber-300";
            return "fa-solid fa-cloud-sun text-slate-300";
        }

        function initTickingClock() {
            setInterval(() => {
                const now = new Date();
                const clockStr = now.toLocaleTimeString('en-US', { hour12: false, timeZone: 'Asia/Kolkata' });
                document.getElementById('live-time').innerText = clockStr;
            }, 1000);
        }

        // -------------------------------------------------------------------------
        // Application Startup Initialization
        // -------------------------------------------------------------------------
        window.onload = function() {
            // Setup canvas size hooks
            resizeCanvas();
            window.addEventListener('resize', () => {
                resizeCanvas();
                syncNeedleLengthsForCurrentState();
                initParticles();
            });

            // Start particle animations
            initParticles();
            animateParticles();

            // Set system clocks
            initTickingClock();

            // Setup Monthly Grid selections
            buildMonthlyCards();

            // INSTANT HYDRATION: Populates predictions immediately to give a flawless <100ms load
            logToTerminal("Embedded ML prediction engine online. Hydrating prediction charts...", "system");
            telemetryData = buildPredictionOnlyData();
            updateUI();

            // Sync with Open-Meteo in the background silently
            syncLiveTelemetry();
        };
    </script>
</body>
</html>
""".replace("__PAYLOAD_JSON__", payload_json)


def main() -> None:
    args = parse_args()
    rows = load_historical_rows(args.start_year, args.end_year, args.model)
    metrics, records = train_and_predict(rows, args.target_year)
    if len(records) != (8784 if is_leap_year(args.target_year) else 8760):
        raise RuntimeError(f"Unexpected record count for {args.target_year}: {len(records)}")

    html = build_html(metrics, records)
    output = Path(args.output)
    output.write_text(html, encoding="utf-8")

    print("\nModel validation")
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {output.resolve()}")


if __name__ == "__main__":
    main()
