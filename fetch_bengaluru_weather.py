
#!/usr/bin/env python3
"""
fetch_bengaluru_weather.py
--------------------------
Collects 20 years of daily historical weather data for Bengaluru, India
(12.9716° N, 77.5946° E) using the free Open-Meteo Archive & Forecast APIs
(no API key required) and maintains `bengaluru_weather.json`.

Behavior:
1. Initial Run (when `bengaluru_weather.json` does not exist):
   Fetches the full 20-year daily record up to today and writes `bengaluru_weather.json`.
2. Delta Update Run (invoked daily by GitHub Actions):
   Reads the latest date in `bengaluru_weather.json`, fetches only the delta days
   (plus a 3-day lookback to finalize recent observations) up to today, and appends/updates
   the JSON file cleanly.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Any

# Bengaluru City Observatory Coordinates
LATITUDE = 12.9716
LONGITUDE = 77.5946
TIMEZONE = "Asia/Kolkata"
OUTPUT_FILE = "bengaluru_weather.json"
DEFAULT_HISTORY_YEARS = 20

# Open-Meteo Daily Variables
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "relative_humidity_2m_mean",
    "pressure_msl_mean",
    "shortwave_radiation_sum",
    "cloud_cover_mean",
    "et0_fao_evapotranspiration",
]


def fetch_json_with_retry(url: str, retries: int = 3, backoff: float = 2.0) -> Dict[str, Any]:
    """Fetches JSON from a URL with exponential backoff retry handling."""
    headers = {"User-Agent": "BengaluruWeatherGitHubAction/1.0"}
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed fetching {url} after {retries} attempts: {exc}") from exc
            sleep_sec = backoff ** attempt
            print(f"[WARN] Attempt {attempt} failed ({exc}). Retrying in {sleep_sec:.1f}s...")
            time.sleep(sleep_sec)
    return {}


def parse_open_meteo_response(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalizes Open-Meteo daily arrays into clean dictionary records."""
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    records: List[Dict[str, Any]] = []

    for i, d_str in enumerate(dates):
        t_max = daily.get("temperature_2m_max", [None])[i]
        t_min = daily.get("temperature_2m_min", [None])[i]

        # Skip future or unrecorded dates where primary temperature is null
        if t_max is None or t_min is None:
            continue

        t_mean = daily.get("temperature_2m_mean", [None])[i]
        if t_mean is None:
            t_mean = round((t_max + t_min) / 2.0, 1)

        records.append({
            "date": d_str,
            "temp_max": round(float(t_max), 1),
            "temp_min": round(float(t_min), 1),
            "temp_mean": round(float(t_mean), 1),
            "precipitation_mm": round(float(daily.get("precipitation_sum", [0.0])[i] or 0.0), 1),
            "wind_speed_max": round(float(daily.get("wind_speed_10m_max", [0.0])[i] or 0.0), 1),
            "wind_gusts_max": round(float(daily.get("wind_gusts_10m_max", [0.0])[i] or 0.0), 1),
            "wind_direction_deg": int(round(float(daily.get("wind_direction_10m_dominant", [0])[i] or 0))),
            "humidity_mean": int(round(float(daily.get("relative_humidity_2m_mean", [65])[i] or 65))),
            "pressure_msl": round(float(daily.get("pressure_msl_mean", [1010.0])[i] or 1010.0), 1),
            "solar_radiation": round(float(daily.get("shortwave_radiation_sum", [18.0])[i] or 18.0), 1),
            "cloud_cover": int(round(float(daily.get("cloud_cover_mean", [50])[i] or 50))),
            "evapotranspiration": round(float(daily.get("et0_fao_evapotranspiration", [4.0])[i] or 4.0), 2),
        })

    return records


def fetch_historical_range(start_dt: date, end_dt: date) -> List[Dict[str, Any]]:
    """Fetches historical daily data from the Open-Meteo Archive API."""
    if start_dt > end_dt:
        return []

    params = urllib.parse.urlencode({
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_dt.isoformat(),
        "end_date": end_dt.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
    })
    url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
    print(f"[INFO] Fetching archive range: {start_dt.isoformat()} -> {end_dt.isoformat()}")
    payload = fetch_json_with_retry(url)
    return parse_open_meteo_response(payload)


def fetch_recent_and_today(past_days: int = 5) -> List[Dict[str, Any]]:
    """
    Fetches the most recent days including Today from the Open-Meteo Forecast API
    (since ERA5 archive has a 2-5 day reanalysis lag).
    """
    params = urllib.parse.urlencode({
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "past_days": past_days,
        "forecast_days": 1,
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
    })
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    print(f"[INFO] Fetching recent/today delta via Forecast API (past_days={past_days})")
    payload = fetch_json_with_retry(url)
    return parse_open_meteo_response(payload)


def load_existing_json(filepath: str) -> Dict[str, Any]:
    """Loads existing `bengaluru_weather.json` if present and valid."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("daily"), list):
                return data
    except Exception as exc:
        print(f"[WARN] Could not parse existing {filepath}: {exc}")
    return {}


def main() -> int:
    # Determine current date in IST (UTC+5:30)
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    today_ist = datetime.now(ist_tz).date()

    existing_data = load_existing_json(OUTPUT_FILE)
    existing_records = existing_data.get("daily", [])
    records_by_date: Dict[str, Dict[str, Any]] = {
        r["date"]: r for r in existing_records if "date" in r
    }

    # If the JSON doesn't exist yet OR only has the starter seed (< 5,000 days),
    # perform a full 20-year historical backfill from Open-Meteo Archive API.
    if len(records_by_date) < 5000:
        # INITIAL 20-YEAR BACKFILL
        start_date = date(today_ist.year - DEFAULT_HISTORY_YEARS, 1, 1)
        archive_cutoff = today_ist - timedelta(days=5)
        print(f"[INIT] Dataset has {len(records_by_date)} days (<5000). Backfilling full 20-year history from {start_date} to {today_ist}...")

        # Fetch in 5-year chunks to be polite to the API
        cursor = start_date
        while cursor <= archive_cutoff:
            chunk_end = min(archive_cutoff, date(cursor.year + 4, 12, 31))
            chunk_records = fetch_historical_range(cursor, chunk_end)
            for rec in chunk_records:
                records_by_date[rec["date"]] = rec
            cursor = chunk_end + timedelta(days=1)
            time.sleep(0.5)

        # Fetch the last 5 days + Today
        recent_records = fetch_recent_and_today(past_days=7)
        for rec in recent_records:
            records_by_date[rec["date"]] = rec

    else:
        # DELTA UPDATE MODE
        sorted_dates = sorted(records_by_date.keys())
        last_recorded_str = sorted_dates[-1]
        last_recorded_dt = date.fromisoformat(last_recorded_str)
        print(f"[DELTA] Existing dataset has {len(sorted_dates)} days (latest: {last_recorded_str}).")

        # Refresh 3 days prior to last_recorded_dt up to today to ensure finalized metrics
        delta_start = last_recorded_dt - timedelta(days=3)
        days_gap = (today_ist - delta_start).days

        if days_gap > 7:
            archive_end = today_ist - timedelta(days=5)
            if delta_start <= archive_end:
                archive_delta = fetch_historical_range(delta_start, archive_end)
                for rec in archive_delta:
                    records_by_date[rec["date"]] = rec

        recent_delta = fetch_recent_and_today(past_days=min(14, max(3, days_gap + 1)))
        for rec in recent_delta:
            records_by_date[rec["date"]] = rec

    # Sort chronologically and write output
    final_daily_list = [records_by_date[k] for k in sorted(records_by_date.keys())]

    output_payload = {
        "metadata": {
            "city": "Bengaluru, Karnataka, India",
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "elevation_m": 920,
            "timezone": TIMEZONE,
            "last_updated": datetime.now(ist_tz).isoformat(),
            "total_days": len(final_daily_list),
            "source": "Open-Meteo Historical Archive & Forecast API",
        },
        "daily": final_daily_list,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"[SUCCESS] Wrote {len(final_daily_list)} daily records to {OUTPUT_FILE} (up to {final_daily_list[-1]['date']}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())