#!/usr/bin/env python3
"""
Generate GitHub Pages-ready multi-city wind/weather prediction payloads.

Reads cities.txt, fetches Open-Meteo historical hourly data, trains the same
compact wind/weather models used by the single-city dashboard, and writes one
prediction JSON file per city plus a lightweight city-index.json.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

import generate_bengaluru_wind as base


DEFAULT_CITIES_FILE = "cities.txt"
DEFAULT_PREDICTIONS_DIR = "predictions"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

CITY_ALIASES = {
    "bengaluru (nandini layout default)": {
        "slug": "bengaluru-nandini-layout",
        "displayName": "Bengaluru",
        "location": "Bengaluru, Karnataka",
        "latitude": 13.0143,
        "longitude": 77.5355,
        "timezone": "Asia/Kolkata",
    },
    "bangalore (nandini layout default)": {
        "slug": "bengaluru-nandini-layout",
        "displayName": "Bengaluru",
        "location": "Bengaluru, Karnataka",
        "latitude": 13.0143,
        "longitude": 77.5355,
        "timezone": "Asia/Kolkata",
    },
    "nagamangala": {
        "slug": "nagamangala",
        "displayName": "Nagamangala",
        "location": "Nagamangala, Karnataka",
        "latitude": 12.81939,
        "longitude": 76.75456,
        "timezone": "Asia/Kolkata",
    },
    "sakleshpura": {
        "slug": "sakleshpur",
        "displayName": "Sakleshpur",
        "location": "Sakleshpur, Karnataka",
        "latitude": 12.94119,
        "longitude": 75.78467,
        "timezone": "Asia/Kolkata",
    },
    "sakleshpur": {
        "slug": "sakleshpur",
        "displayName": "Sakleshpur",
        "location": "Sakleshpur, Karnataka",
        "latitude": 12.94119,
        "longitude": 75.78467,
        "timezone": "Asia/Kolkata",
    },
    "mangalore": {
        "slug": "mangalore",
        "displayName": "Mangalore",
        "location": "Mangaluru, Karnataka",
        "latitude": 12.91723,
        "longitude": 74.85603,
        "timezone": "Asia/Kolkata",
    },
    "mangaluru": {
        "slug": "mangalore",
        "displayName": "Mangalore",
        "location": "Mangaluru, Karnataka",
        "latitude": 12.91723,
        "longitude": 74.85603,
        "timezone": "Asia/Kolkata",
    },
    "ooty": {
        "slug": "ooty",
        "displayName": "Ooty",
        "location": "Udhagamandalam, Tamil Nadu",
        "latitude": 11.4134,
        "longitude": 76.69521,
        "timezone": "Asia/Kolkata",
    },
    "waynad": {
        "slug": "wayanad",
        "displayName": "Wayanad",
        "location": "Wayanad, Kerala",
        "latitude": 11.8032,
        "longitude": 76.00451,
        "timezone": "Asia/Kolkata",
    },
    "wayanad": {
        "slug": "wayanad",
        "displayName": "Wayanad",
        "location": "Wayanad, Kerala",
        "latitude": 11.8032,
        "longitude": 76.00451,
        "timezone": "Asia/Kolkata",
    },
    "munnay": {
        "slug": "munnar",
        "displayName": "Munnar",
        "location": "Munnar, Kerala",
        "latitude": 10.08818,
        "longitude": 77.06239,
        "timezone": "Asia/Kolkata",
    },
    "munnar": {
        "slug": "munnar",
        "displayName": "Munnar",
        "location": "Munnar, Kerala",
        "latitude": 10.08818,
        "longitude": 77.06239,
        "timezone": "Asia/Kolkata",
    },
    "shimla": {
        "slug": "shimla",
        "displayName": "Shimla",
        "location": "Shimla, Himachal Pradesh",
        "latitude": 31.10442,
        "longitude": 77.16662,
        "timezone": "Asia/Kolkata",
    },
    "meghalaya": {
        "slug": "meghalaya",
        "displayName": "Meghalaya",
        "location": "Shillong, Meghalaya",
        "latitude": 25.56892,
        "longitude": 91.88313,
        "timezone": "Asia/Kolkata",
    },
    "jaipur": {
        "slug": "jaipur",
        "displayName": "Jaipur",
        "location": "Jaipur, Rajasthan",
        "latitude": 26.91962,
        "longitude": 75.78781,
        "timezone": "Asia/Kolkata",
    },
    "mysuru": {
        "slug": "mysuru",
        "displayName": "Mysuru",
        "location": "Mysuru, Karnataka",
        "latitude": 12.29791,
        "longitude": 76.63925,
        "timezone": "Asia/Kolkata",
    },
    "chennai": {
        "slug": "chennai",
        "displayName": "Chennai",
        "location": "Chennai, Tamil Nadu",
        "latitude": 13.08784,
        "longitude": 80.27847,
        "timezone": "Asia/Kolkata",
    },
    "allepy": {
        "slug": "alappuzha",
        "displayName": "Alappuzha",
        "location": "Alappuzha, Kerala",
        "latitude": 9.49004,
        "longitude": 76.3264,
        "timezone": "Asia/Kolkata",
    },
    "alappuzha": {
        "slug": "alappuzha",
        "displayName": "Alappuzha",
        "location": "Alappuzha, Kerala",
        "latitude": 9.49004,
        "longitude": 76.3264,
        "timezone": "Asia/Kolkata",
    },
    "sirsi": {
        "slug": "sirsi",
        "displayName": "Sirsi",
        "location": "Sirsi, Karnataka",
        "latitude": 14.62072,
        "longitude": 74.83554,
        "timezone": "Asia/Kolkata",
    },
    "kodikanel": {
        "slug": "kodaikanal",
        "displayName": "Kodaikanal",
        "location": "Kodaikanal, Tamil Nadu",
        "latitude": 10.23925,
        "longitude": 77.48932,
        "timezone": "Asia/Kolkata",
    },
    "kodaikanal": {
        "slug": "kodaikanal",
        "displayName": "Kodaikanal",
        "location": "Kodaikanal, Tamil Nadu",
        "latitude": 10.23925,
        "longitude": 77.48932,
        "timezone": "Asia/Kolkata",
    },
}


@dataclass(frozen=True)
class City:
    slug: str
    displayName: str
    location: str
    latitude: float
    longitude: float
    timezone: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", default=DEFAULT_CITIES_FILE)
    parser.add_argument("--predictions-dir", default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--start-year", type=int, default=base.TRAIN_START_YEAR)
    parser.add_argument("--end-year", type=int, default=base.TRAIN_END_YEAR)
    parser.add_argument("--target-year", type=int, default=base.TARGET_YEAR)
    parser.add_argument("--model", default="era5")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing city JSON payloads when present and only train missing cities.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized.lower()).strip("-")
    return slug or "city"


def location_label(result: dict) -> str:
    parts = [
        result.get("name"),
        result.get("admin1"),
        result.get("country"),
    ]
    return ", ".join(part for part in parts if part)


def geocode_city(raw_name: str) -> City:
    normalized = raw_name.strip().lower()
    alias = CITY_ALIASES.get(normalized)
    if isinstance(alias, dict):
        return City(**alias)

    query = alias if isinstance(alias, str) else raw_name
    response = requests.get(
        GEOCODING_URL,
        params={"name": query, "count": 1, "language": "en", "format": "json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results") or []
    if not results:
        raise RuntimeError(f"No Open-Meteo geocoding result for city line: {raw_name!r}")

    result = results[0]
    display_name = raw_name.strip()
    if normalized in CITY_ALIASES and isinstance(alias, str):
        display_name = raw_name.strip().title()

    return City(
        slug=slugify(display_name),
        displayName=display_name,
        location=location_label(result),
        latitude=float(result["latitude"]),
        longitude=float(result["longitude"]),
        timezone=result.get("timezone") or base.TIMEZONE,
    )


def parse_pipe_city(line: str) -> City:
    parts = [part.strip() for part in line.split("|")]
    if len(parts) != 6:
        raise ValueError(
            "Pipe-delimited city rows must use: slug | display name | location | latitude | longitude | timezone"
        )
    slug, display_name, location, latitude, longitude, timezone = parts
    return City(
        slug=slugify(slug),
        displayName=display_name,
        location=location,
        latitude=float(latitude),
        longitude=float(longitude),
        timezone=timezone,
    )


def load_cities(path: Path) -> list[City]:
    if not path.exists():
        raise FileNotFoundError(f"Missing city source file: {path}")

    cities: list[City] = []
    seen: set[str] = set()
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        city = parse_pipe_city(line) if "|" in line else geocode_city(line)
        if city.slug in seen:
            raise ValueError(f"Duplicate city slug {city.slug!r} at {path}:{line_no}")
        seen.add(city.slug)
        cities.append(city)

    if not cities:
        raise ValueError(f"No cities found in {path}")
    return cities


def prediction_payload(city: City, metrics: dict, records: list[dict]) -> dict:
    return {
        "slug": city.slug,
        "displayName": city.displayName,
        "location": city.location,
        "latitude": city.latitude,
        "longitude": city.longitude,
        "timezone": city.timezone,
        "forecastUrl": base.FORECAST_URL,
        "metrics": metrics,
        "monthly": base.monthly_summary(records),
        "records": records,
    }


def train_city(city: City, args: argparse.Namespace, output_dir: Path) -> dict:
    print(f"\n=== {city.displayName} [{city.latitude:.4f}, {city.longitude:.4f}] ===")

    base.LATITUDE = city.latitude
    base.LONGITUDE = city.longitude
    base.TIMEZONE = city.timezone

    rows = base.load_historical_rows(args.start_year, args.end_year, args.model)
    metrics, records = base.train_and_predict(rows, args.target_year)
    expected = 8784 if base.is_leap_year(args.target_year) else 8760
    if len(records) != expected:
        raise RuntimeError(f"{city.slug}: expected {expected} records, got {len(records)}")

    payload = prediction_payload(city, metrics, records)
    city_file = output_dir / "cities" / f"{city.slug}.json"
    city_file.parent.mkdir(parents=True, exist_ok=True)
    city_file.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {city_file} ({len(records):,} predicted rows)")

    return {
        **asdict(city),
        "predictionPath": f"predictions/cities/{city.slug}.json",
        "metrics": {
            "training_rows": metrics["training_rows"],
            "validation_year": metrics["validation_year"],
            "direction_mae_deg": metrics["direction_mae_deg"],
            "speed_mae_kmh": metrics["speed_mae_kmh"],
            "temperature_mae_c": metrics["temperature_mae_c"],
            "wet_hour_accuracy_pct": metrics["wet_hour_accuracy_pct"],
            "predicted_rows": metrics["predicted_rows"],
            "generated_at": metrics["generated_at"],
        },
    }


def index_entry_from_payload(city: City, city_file: Path) -> dict:
    payload = json.loads(city_file.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    return {
        **asdict(city),
        "predictionPath": f"predictions/cities/{city.slug}.json",
        "metrics": {
            "training_rows": metrics["training_rows"],
            "validation_year": metrics["validation_year"],
            "direction_mae_deg": metrics["direction_mae_deg"],
            "speed_mae_kmh": metrics["speed_mae_kmh"],
            "temperature_mae_c": metrics["temperature_mae_c"],
            "wet_hour_accuracy_pct": metrics["wet_hour_accuracy_pct"],
            "predicted_rows": metrics["predicted_rows"],
            "generated_at": metrics["generated_at"],
        },
    }


def build_city_index_entry(city: City, args: argparse.Namespace, output_dir: Path) -> dict:
    city_file = output_dir / "cities" / f"{city.slug}.json"
    if args.reuse_existing and city_file.exists():
        print(f"Reusing {city_file}")
        return index_entry_from_payload(city, city_file)
    return train_city(city, args, output_dir)


def main() -> None:
    args = parse_args()
    cities_path = Path(args.cities)
    output_dir = Path(args.predictions_dir)
    cities = load_cities(cities_path)

    index_cities = [build_city_index_entry(city, args, output_dir) for city in cities]
    index = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_year": args.target_year,
        "source": str(cities_path),
        "cities": index_cities,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "city-index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nWrote {index_path} with {len(index_cities)} cities")


if __name__ == "__main__":
    main()
