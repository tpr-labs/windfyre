#!/usr/bin/env python3
"""Generate the GitHub Pages AI weather analysis from Open-Meteo and Gemma."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "predictions"
CONFIG_PATH = PREDICTIONS_DIR / "ai-analysis-config.json"
CITY_INDEX_PATH = PREDICTIONS_DIR / "city-index.json"
OUTPUT_PATH = PREDICTIONS_DIR / "ai-weather-analysis.json"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
GEMINI_MODEL = "gemma-4-26b-a4b-it"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

ANALYSIS_SCHEMA = {
    "type": "object",
    "required": [
        "status", "summary", "current_conditions", "baseline_comparison",
        "next_24h_outlook", "next_7d_outlook", "anomalies", "watch_items",
        "confidence", "data_quality_note",
    ],
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "summary": {"type": "string", "maxLength": 280},
        "current_conditions": {"type": "string", "maxLength": 500},
        "baseline_comparison": {"type": "string", "maxLength": 500},
        "next_24h_outlook": {"type": "string", "maxLength": 500},
        "next_7d_outlook": {"type": "string", "maxLength": 700},
        "anomalies": {
            "type": "array", "maxItems": 6,
            "items": {
                "type": "object",
                "required": ["type", "severity", "timeframe", "description"],
                "properties": {
                    "type": {"type": "string", "enum": ["wind", "temperature", "humidity", "precipitation", "weather"]},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "timeframe": {"type": "string", "maxLength": 80},
                    "description": {"type": "string", "maxLength": 280},
                },
            },
        },
        "watch_items": {"type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 180}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "data_quality_note": {"type": "string", "maxLength": 280},
    },
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def request_json(url: str, *, params: dict | None = None, payload: dict | None = None, timeout: int = 60) -> dict:
    if params:
        url = f"{url}?{urlencode(params)}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"HTTP {error.code} from {url.split('?')[0]}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Network error for {url.split('?')[0]}: {error.reason}") from error


def mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if isinstance(value, (int, float))]
    return round(sum(values) / len(values), 1) if values else None


def total(values: list[float]) -> float | None:
    values = [float(value) for value in values if isinstance(value, (int, float))]
    return round(sum(values), 1) if values else None


def circular_mean(values: list[float]) -> int | None:
    values = [float(value) for value in values if isinstance(value, (int, float))]
    if not values:
        return None
    east = sum(math.sin(math.radians(value)) for value in values)
    north = sum(math.cos(math.radians(value)) for value in values)
    return int(round((math.degrees(math.atan2(east, north)) + 360) % 360))


def weather_context(city: dict) -> dict:
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "timezone": city.get("timezone", "auto"),
        "past_days": 7,
        "forecast_days": 7,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation",
        "hourly": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation",
    }
    return request_json(OPEN_METEO_URL, params=params)


def daily_context(telemetry: dict, baseline_records: dict[str, dict], timezone: str) -> list[dict]:
    hourly = telemetry["hourly"]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, timestamp in enumerate(hourly["time"]):
        grouped[timestamp[:10]].append(index)

    current_local_date = datetime.now(ZoneInfo(timezone)).date().isoformat()
    days = []
    for date_key, indices in grouped.items():
        observed = {field: [hourly.get(field, [None] * len(hourly["time"]))[index] for index in indices] for field in (
            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "temperature_2m",
            "relative_humidity_2m", "precipitation", "weather_code",
        )}
        baseline = [baseline_records.get(hourly["time"][index]) for index in indices]
        baseline = [item for item in baseline if item]
        days.append({
            "date": date_key,
            "phase": "recent_observation" if date_key < current_local_date else "forecast",
            "observed": {
                "wind_avg_kmh": mean(observed["wind_speed_10m"]),
                "wind_max_kmh": round(max(value for value in observed["wind_speed_10m"] if isinstance(value, (int, float))), 1) if any(isinstance(value, (int, float)) for value in observed["wind_speed_10m"]) else None,
                "wind_direction_deg": circular_mean(observed["wind_direction_10m"]),
                "gust_max_kmh": round(max(value for value in observed["wind_gusts_10m"] if isinstance(value, (int, float))), 1) if any(isinstance(value, (int, float)) for value in observed["wind_gusts_10m"]) else None,
                "temperature_avg_c": mean(observed["temperature_2m"]),
                "humidity_avg_pct": mean(observed["relative_humidity_2m"]),
                "precipitation_total_mm": total(observed["precipitation"]),
                "weather_codes": dict(Counter(str(value) for value in observed["weather_code"] if isinstance(value, (int, float)))),
            },
            "ml_baseline": {
                "wind_avg_kmh": mean([item.get("s") for item in baseline]),
                "wind_direction_deg": circular_mean([item.get("d") for item in baseline]),
                "temperature_avg_c": mean([item.get("tc") for item in baseline]),
                "humidity_avg_pct": mean([item.get("rh") for item in baseline]),
                "precipitation_total_mm": total([item.get("pr") for item in baseline]),
            },
        })
    return days


def build_context(city: dict, prediction: dict, telemetry: dict) -> dict:
    records = {item["t"]: item for item in prediction.get("records", [])}
    current = telemetry.get("current", {})
    timestamp = current.get("time")
    baseline_current = records.get(timestamp, {}) if timestamp else {}
    return {
        "city": {key: city[key] for key in ("slug", "displayName", "location", "timezone")},
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "current_conditions": current,
        "current_ml_baseline": {
            "wind_speed_kmh": baseline_current.get("s"), "wind_direction_deg": baseline_current.get("d"),
            "temperature_c": baseline_current.get("tc"), "humidity_pct": baseline_current.get("rh"),
            "precipitation_mm": baseline_current.get("pr"), "weather_code": baseline_current.get("wx"),
        },
        "model_validation": {key: prediction.get("metrics", {}).get(key) for key in (
            "validation_year", "direction_mae_deg", "speed_mae_kmh", "temperature_mae_c",
            "humidity_mae_pct", "precipitation_mae_mm", "wet_hour_accuracy_pct",
        )},
        "daily_comparison": daily_context(telemetry, records, city.get("timezone", "Asia/Kolkata")),
    }


def analysis_prompt(context: dict) -> str:
    return """You are an informational weather-data analyst. Analyze only the supplied JSON context.
Compare recent Open-Meteo observations and the 7-day forecast with the city ML baseline.
Identify material deviations in wind speed/direction, gusts, temperature, humidity, precipitation, and weather codes.
Describe current conditions, the next 24 hours, the next 7 days, and noteworthy anomalies.
Do not invent observations, claim a causal explanation, provide travel advice, or issue official weather or safety warnings.
Forecasts are uncertain: reflect this in confidence and data_quality_note.
Return only a JSON object matching the supplied response schema. Be compact: use one sentence of at most 180 characters for each narrative field, at most 3 anomalies, and at most 2 watch items. Keep the summary at or below 180 characters.

CONTEXT:
""" + json.dumps(context, separators=(",", ":"))


def gemini_analysis(context: dict, api_key: str) -> dict:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": analysis_prompt(context)}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1536,
            "thinkingConfig": {"thinkingLevel": "minimal"},
            "responseMimeType": "application/json",
            "responseJsonSchema": ANALYSIS_SCHEMA,
        },
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = request_json(
                f"{GEMINI_URL}?{urlencode({'key': api_key})}",
                payload=payload,
                timeout=180,
            )
            text = response["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (TimeoutError, OSError, IndexError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError("Gemini did not return a valid analysis after 3 attempts") from last_error


def validate_analysis(analysis: dict) -> None:
    required = set(ANALYSIS_SCHEMA["required"])
    missing = required - set(analysis)
    if missing or analysis.get("status") != "ok":
        raise RuntimeError(f"Gemini analysis did not satisfy required fields: {sorted(missing)}")
    for field in ("summary", "current_conditions", "baseline_comparison", "next_24h_outlook", "next_7d_outlook", "data_quality_note"):
        if not isinstance(analysis[field], str) or not analysis[field].strip():
            raise RuntimeError(f"Gemini analysis field {field!r} must be a non-empty string")
    if analysis["confidence"] not in {"low", "medium", "high"}:
        raise RuntimeError("Gemini confidence is invalid")
    if not isinstance(analysis["watch_items"], list) or not isinstance(analysis["anomalies"], list):
        raise RuntimeError("Gemini list fields are invalid")
    for anomaly in analysis["anomalies"]:
        if not isinstance(anomaly, dict) or anomaly.get("type") not in {"wind", "temperature", "humidity", "precipitation", "weather"} or anomaly.get("severity") not in {"low", "medium", "high"}:
            raise RuntimeError("Gemini anomaly is invalid")


def write_output(payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=OUTPUT_PATH.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(OUTPUT_PATH)


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY must be configured as a GitHub Actions secret")

    config = read_json(CONFIG_PATH)
    city_index = read_json(CITY_INDEX_PATH)
    config_cities = config.get("cities", {})
    index_cities = {city["slug"]: city for city in city_index.get("cities", [])}
    if set(config_cities) != set(index_cities):
        raise RuntimeError("AI analysis config cities must exactly match city-index.json")

    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = {}
    for slug, settings in config_cities.items():
        if not settings.get("enabled", False):
            continue
        city = index_cities[slug]
        prediction = read_json(ROOT / city["predictionPath"])
        telemetry = weather_context(city)
        analysis = gemini_analysis(build_context(city, prediction, telemetry), api_key)
        validate_analysis(analysis)
        analysis["updated_at"] = generated_at
        results[slug] = analysis
        print(f"Generated analysis for {slug}")

    if not results:
        raise RuntimeError("No cities are enabled in ai-analysis-config.json")
    write_output({"generated_at": generated_at, "model": GEMINI_MODEL, "source": "Open-Meteo live telemetry + Windfyre ML baseline", "cities": results})


if __name__ == "__main__":
    main()
