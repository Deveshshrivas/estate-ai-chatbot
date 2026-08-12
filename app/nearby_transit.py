"""Nearest metro lookup backed by OpenStreetMap and cached on property rows."""

from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .database import properties


OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
CACHE_TTL = timedelta(days=30)
SEARCH_RADIUS_METRES = 10_000
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_nominatim_lock = asyncio.Lock()
_last_nominatim_request = 0.0


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance between two WGS84 coordinates."""
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def _station(element: dict[str, Any], latitude: float, longitude: float) -> dict[str, Any] | None:
    tags = element.get("tags") or {}
    name = tags.get("name:en") or tags.get("name")
    station_lat = element.get("lat") or (element.get("center") or {}).get("lat")
    station_lon = element.get("lon") or (element.get("center") or {}).get("lon")
    if not name or not isinstance(station_lat, (int, float)) or not isinstance(station_lon, (int, float)):
        return None
    distance = distance_km(latitude, longitude, float(station_lat), float(station_lon))
    osm_type = str(element.get("type") or "node")
    osm_id = element.get("id")
    return {
        "status": "found",
        "name": str(name),
        "distance_km": round(distance, 2),
        "latitude": float(station_lat),
        "longitude": float(station_lon),
        "network": tags.get("network") or tags.get("operator"),
        "source": "OpenStreetMap",
        "source_url": f"https://www.openstreetmap.org/{osm_type}/{osm_id}" if osm_id else None,
        "distance_type": "straight_line",
        "fetched_at": datetime.now(timezone.utc),
    }


def nearest_from_elements(
    elements: list[dict[str, Any]], latitude: float, longitude: float
) -> dict[str, Any] | None:
    stations = [
        station for element in elements
        if (station := _station(element, latitude, longitude)) is not None
    ]
    return min(stations, key=lambda item: item["distance_km"], default=None)


async def fetch_nearest_metro(latitude: float, longitude: float) -> dict[str, Any] | None:
    """Search nearby metro stations while respecting the public service limit."""
    global _last_nominatim_request
    radius_km = SEARCH_RADIUS_METRES / 1000
    latitude_delta = radius_km / 111.32
    longitude_delta = radius_km / max(1.0, 111.32 * math.cos(math.radians(latitude)))
    params = {
        "q": "metro station",
        "format": "jsonv2",
        "viewbox": (
            f"{longitude-longitude_delta:.7f},{latitude+latitude_delta:.7f},"
            f"{longitude+longitude_delta:.7f},{latitude-latitude_delta:.7f}"
        ),
        "bounded": 1,
        "limit": 20,
        "addressdetails": 1,
        "namedetails": 1,
        "extratags": 1,
    }
    headers = {
        "User-Agent": "EstateAI/1.0 (property assistant; admin@pratap.ai)",
        "Referer": "https://pratap.ai/",
    }
    nominatim_error: Exception | None = None
    try:
        async with _nominatim_lock:
            delay = 1.05 - (time.monotonic() - _last_nominatim_request)
            if delay > 0:
                await asyncio.sleep(delay)
            async with httpx.AsyncClient(timeout=10, headers=headers) as client:
                response = await client.get(NOMINATIM_URL, params=params)
            _last_nominatim_request = time.monotonic()
        response.raise_for_status()
        elements = []
        for row in response.json():
            if row.get("type") not in {"station", "halt", "stop"}:
                continue
            namedetails = row.get("namedetails") or {}
            extratags = row.get("extratags") or {}
            name = (
                namedetails.get("name:en")
                or namedetails.get("name")
                or str(row.get("display_name") or "").split(",", 1)[0]
            )
            try:
                station_lat = float(row["lat"])
                station_lon = float(row["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            elements.append({
                "type": row.get("osm_type") or "node",
                "id": row.get("osm_id"),
                "lat": station_lat,
                "lon": station_lon,
                "tags": {
                    "name": name,
                    "network": extratags.get("network") or extratags.get("operator"),
                },
            })
        return nearest_from_elements(elements, latitude, longitude)
    except (httpx.HTTPError, ValueError) as exc:
        nominatim_error = exc

    # Short fallback for temporary Nominatim failures. A successful empty
    # Nominatim result is authoritative and never triggers this slower query.
    query = (
        f'[out:json][timeout:7];node(around:{SEARCH_RADIUS_METRES},'
        f'{latitude:.7f},{longitude:.7f})["railway"="station"]'
        '["station"="subway"];out body;'
    )
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=9, headers=headers) as client:
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                response = await client.get(endpoint, params={"data": query})
                response.raise_for_status()
                return nearest_from_elements(
                    response.json().get("elements") or [], latitude, longitude
                )
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
    if last_error or nominatim_error:
        raise RuntimeError("Metro lookup service is temporarily unavailable") from last_error
    return None


def _fresh(cache: dict[str, Any]) -> bool:
    fetched = cache.get("fetched_at")
    if isinstance(fetched, str):
        try:
            fetched = datetime.fromisoformat(fetched.replace("Z", "+00:00"))
        except ValueError:
            return False
    if not isinstance(fetched, datetime):
        return False
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return fetched >= datetime.now(timezone.utc) - CACHE_TTL


async def nearest_metro_for_property(listing: dict[str, Any]) -> dict[str, Any]:
    cached = listing.get("nearby_metro")
    if isinstance(cached, dict) and _fresh(cached):
        return cached

    coordinates = listing.get("coordinates") or {}
    latitude = coordinates.get("latitude")
    longitude = coordinates.get("longitude")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return {"status": "coordinates_missing"}

    result = await fetch_nearest_metro(float(latitude), float(longitude))
    if result is None:
        result = {
            "status": "not_found",
            "source": "OpenStreetMap",
            "search_radius_km": SEARCH_RADIUS_METRES / 1000,
            "fetched_at": datetime.now(timezone.utc),
        }
    title = str(listing.get("title") or "").strip()
    if title:
        await properties.update_one(
            {"title": title},
            {"$set": {"nearby_metro": result, "updated_at": datetime.now(timezone.utc)}},
        )
    return result
