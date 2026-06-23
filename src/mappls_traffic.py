"""
Live Bengaluru traffic snapshot via Mappls Route ADV.
Used by the command-center map corridor delay panel.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from src.mappls_feature_engine import MapplsFeatureEngine

# Key BTP corridors (from_lat, from_lon, to_lat, to_lon)
BENGALURU_CORRIDORS: List[Dict[str, Any]] = [
    {
        "id": "silk_marathahalli",
        "name": "Silk Board → Marathahalli",
        "from": (12.9177, 77.6228),
        "to": (12.9592, 77.6974),
    },
    {
        "id": "hebbal_mekhri",
        "name": "Hebbal → Mekhri Circle",
        "from": (13.0354, 77.5910),
        "to": (13.0090, 77.5770),
    },
    {
        "id": "krpuram_marathahalli",
        "name": "KR Puram → Marathahalli",
        "from": (13.0053, 77.6946),
        "to": (12.9592, 77.6974),
    },
    {
        "id": "mg_indiranagar",
        "name": "MG Road → Indiranagar",
        "from": (12.9757, 77.6013),
        "to": (12.9718, 77.6412),
    },
    {
        "id": "yeshwanthpur_hebbal",
        "name": "Yeshwanthpur → Hebbal",
        "from": (13.0267, 77.5361),
        "to": (13.0354, 77.5910),
    },
    {
        "id": "ecity_silkboard",
        "name": "Electronic City → Silk Board",
        "from": (12.8458, 77.6601),
        "to": (12.9177, 77.6228),
    },
]

CACHE_TTL_SEC = 120
_snapshot_cache: Dict[str, Any] = {"key": None, "expires": 0.0, "payload": None}


def _congestion_level(ratio: float) -> str:
    if ratio < 1.12:
        return "LOW"
    if ratio < 1.35:
        return "MEDIUM"
    return "HIGH"


@lru_cache(maxsize=1)
def _engine() -> MapplsFeatureEngine:
    return MapplsFeatureEngine()


def _probe_corridor(
    engine: MapplsFeatureEngine,
    corridor_id: str,
    name: str,
    origin: Tuple[float, float],
    dest: Tuple[float, float],
) -> Dict[str, Any]:
    from_lat, from_lon = origin
    to_lat, to_lon = dest
    probe = engine.probe_route_delay(from_lat, from_lon, to_lat, to_lon, cache_prefix=corridor_id)
    return {
        "id": corridor_id,
        "name": name,
        **probe,
    }


def get_traffic_snapshot(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> Dict[str, Any]:
    """Return live corridor delays + optional event-location probe."""
    cache_key = f"{round(lat or 0, 3)}:{round(lon or 0, 3)}"
    now = time.time()
    if (
        _snapshot_cache["payload"]
        and _snapshot_cache["key"] == cache_key
        and now < _snapshot_cache["expires"]
    ):
        return _snapshot_cache["payload"]

    engine = _engine()
    corridors: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                _probe_corridor,
                engine,
                c["id"],
                c["name"],
                c["from"],
                c["to"],
            ): c["id"]
            for c in BENGALURU_CORRIDORS
        }
        for fut in as_completed(futures):
            try:
                corridors.append(fut.result())
            except Exception:
                cid = futures[fut]
                match = next((c for c in BENGALURU_CORRIDORS if c["id"] == cid), None)
                if match:
                    corridors.append({
                        "id": cid,
                        "name": match["name"],
                        "available": False,
                        "delay_mins": None,
                        "congestion_level": "UNKNOWN",
                        "source": "error",
                    })

    corridors.sort(key=lambda x: BENGALURU_CORRIDORS.index(next(c for c in BENGALURU_CORRIDORS if c["id"] == x["id"])))

    event_probe = None
    if lat is not None and lon is not None:
        # 2 km east probe from event pin — local congestion context
        event_probe = engine.probe_route_delay(
            lat, lon, lat, lon + 0.018,
            cache_prefix="event_probe",
        )
        event_probe["label"] = "Event location (2 km probe)"

    available_count = sum(1 for c in corridors if c.get("available"))
    worst = max(
        (c for c in corridors if c.get("available")),
        key=lambda x: x.get("delay_mins") or 0,
        default=None,
    )

    payload = {
        "available": available_count > 0,
        "city": "Bengaluru",
        "updated_at": int(now),
        "refresh_sec": CACHE_TTL_SEC,
        "corridors": corridors,
        "event_probe": event_probe,
        "summary": {
            "corridors_live": available_count,
            "worst_corridor": worst["name"] if worst else None,
            "worst_delay_mins": worst.get("delay_mins") if worst else None,
            "worst_level": worst.get("congestion_level") if worst else None,
        },
        "legend": {
            "LOW": "Free flow",
            "MEDIUM": "Moderate delay",
            "HIGH": "Heavy congestion",
        },
        "map_traffic_hint": "Enable traffic overlay on map for live road colours (green/orange/red).",
    }

    _snapshot_cache["key"] = cache_key
    _snapshot_cache["expires"] = now + CACHE_TTL_SEC
    _snapshot_cache["payload"] = payload
    return payload