"""
Phase 3 — Prescriptive Recommendation Engine (PuLP + What-If)
============================================================
Standalone module. Consumes V12-style model outputs + Phase 2 graph data.
Does NOT modify inference.py, V12 training scripts, or production models.

Inputs (per event):
  - predicted duration, impact_score, quantiles P35/P50/P65
  - MoE regime (short / medium / long)
  - affected subgraph / graph features (Phase 2)
  - active_overlap (concurrent load)

Outputs:
  - Structured JSON: manpower allocation, barricade/diversion plan,
    what-if improvement estimate, rationale, confidence.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
_PULP_TMP = BASE_DIR / ".pulp_tmp"
_PULP_TMP.mkdir(exist_ok=True)
# CBC on Windows breaks when user profile path contains spaces (e.g. "Nisha kumari")
os.environ.setdefault("TMP", str(_PULP_TMP))
os.environ.setdefault("TEMP", str(_PULP_TMP))

try:
    from pulp import HiGHS_CMD, LpInteger, LpMinimize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum, value
    from pulp.apis.core import PulpSolverError

    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    PulpSolverError = Exception  # type: ignore
    logger.warning("PuLP not installed — manpower allocation will use greedy fallback.")

try:
    from src.mappls_feature_engine import BTP_STATIONS, ISEC_CONGESTION
except Exception:
    BTP_STATIONS = {
        "Yeshwanthpur": (13.0253, 77.5397),
        "Hebbal": (13.0350, 77.5970),
        "Marathahalli": (12.9592, 77.6974),
        "Electronic City": (12.8399, 77.6770),
        "Whitefield": (12.9698, 77.7499),
        "Indiranagar": (12.9718, 77.6412),
        "Koramangala": (12.9352, 77.6245),
    }
    ISEC_CONGESTION = {h: 1.0 + 0.5 * (1 if 8 <= h <= 10 or 17 <= h <= 20 else 0) for h in range(24)}

REGIME_MULTIPLIER = {"short": 0.85, "medium": 1.0, "long": 1.35}
REGIME_MIN_OFFICERS = {"short": 2, "medium": 4, "long": 6}
DEFAULT_STATION_CAPACITY = 8
MAX_RESPONSE_MINS = 28
AVG_DISPATCH_SPEED_KMH = 22.0
MAX_OFFICERS_PER_EVENT = 14


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def response_mins(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_km(lat1, lon1, lat2, lon2) / AVG_DISPATCH_SPEED_KMH * 60.0


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def normalize_v12_event(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize heterogeneous V12 / API payloads into a canonical event bundle.
    Accepts flat dicts or nested {'ml_prediction': ..., 'stis': ...} shapes.
    """
    ml = raw.get("ml_prediction") or raw.get("v12") or raw
    quant = raw.get("quantiles") or ml.get("quantiles") or raw

    p35 = _safe_float(raw.get("p35_hours") or quant.get("p35") or quant.get("p35_hours") or ml.get("p10_hours"), 0.0)
    p50 = _safe_float(
        raw.get("p50_hours")
        or raw.get("predicted_hours")
        or quant.get("p50")
        or quant.get("p50_hours")
        or ml.get("predicted_hours"),
        2.0,
    )
    p65 = _safe_float(raw.get("p65_hours") or quant.get("p65") or quant.get("p65_hours") or ml.get("p90_hours"), p50 * 1.5)

    if p35 <= 0:
        p35 = max(0.1, p50 * 0.65)
    if p65 <= 0:
        p65 = max(p50, p50 * 1.45)

    regime = str(raw.get("regime") or raw.get("moe_regime") or ml.get("regime") or "medium").lower()
    if regime not in REGIME_MULTIPLIER:
        if p50 < 1.5:
            regime = "short"
        elif p50 < 4.0:
            regime = "medium"
        else:
            regime = "long"

    stis_raw = raw.get("stis")
    if isinstance(stis_raw, dict):
        stis_raw = stis_raw.get("stis")
    impact = _safe_float(raw.get("impact_score") or stis_raw or raw.get("impact"), p50 * 1.2)

    return {
        "event_id": str(raw.get("event_id") or raw.get("id") or "event_0"),
        "event_cause": str(raw.get("event_cause") or "vehicle_breakdown"),
        "corridor": str(raw.get("corridor") or "Non-corridor"),
        "zone": str(raw.get("zone") or "Central"),
        "police_station": str(raw.get("police_station") or "unknown"),
        "hour": int(raw.get("hour") or 12),
        "lat": _safe_float(raw.get("lat") or raw.get("latitude"), 13.0),
        "lon": _safe_float(raw.get("lon") or raw.get("longitude"), 77.6),
        "requires_road_closure": bool(raw.get("requires_road_closure", False)),
        "p35_hours": p35,
        "p50_hours": p50,
        "p65_hours": p65,
        "impact_score": impact,
        "regime": regime,
        "active_overlap": int(raw.get("active_overlap") or raw.get("active_overlap_2km") or 0),
        "graph_features": raw.get("graph_features") or {},
        "affected_subgraph": raw.get("affected_subgraph") or {},
    }


@dataclass
class ManpowerResult:
    allocation: Dict[str, Dict[str, Any]]
    total_deployed: int
    triage_active: bool
    solver: str
    station_assignments: Dict[str, Dict[str, int]] = field(default_factory=dict)


class ManpowerAllocator:
    """
    PuLP integer program: assign officers across concurrent events to minimize
    weighted residual impact under zone/station capacity and response-time limits.
    """

    def __init__(
        self,
        station_capacity: Optional[Dict[str, int]] = None,
        max_response_mins: float = MAX_RESPONSE_MINS,
    ):
        self.station_capacity = station_capacity or {s: DEFAULT_STATION_CAPACITY for s in BTP_STATIONS}
        self.max_response_mins = max_response_mins

    def _min_officers(self, ev: Dict[str, Any]) -> int:
        base = REGIME_MIN_OFFICERS.get(ev["regime"], 4)
        if ev["requires_road_closure"]:
            base += 2
        if ev["event_cause"] in ("accident", "water_logging", "tree_fall", "vip_movement"):
            base += 1
        overlap = ev.get("active_overlap", 0)
        if overlap >= 3:
            base += 1
        return int(min(base, MAX_OFFICERS_PER_EVENT))

    def _impact_weight(self, ev: Dict[str, Any]) -> float:
        w = ev["impact_score"] * REGIME_MULTIPLIER.get(ev["regime"], 1.0)
        w *= 1.0 + 0.12 * ev.get("active_overlap", 0)
        w *= 1.0 + 0.08 * ev.get("graph_features", {}).get("graph_node_betweenness", 0.0) * 100
        return max(0.5, w)

    def _eligible_stations(self, ev: Dict[str, Any]) -> List[str]:
        eligible = []
        for name, (slat, slon) in BTP_STATIONS.items():
            if response_mins(ev["lat"], ev["lon"], slat, slon) <= self.max_response_mins:
                eligible.append(name)
        if not eligible:
            # nearest station always eligible as fallback
            nearest = min(
                BTP_STATIONS.keys(),
                key=lambda s: haversine_km(ev["lat"], ev["lon"], *BTP_STATIONS[s]),
            )
            eligible = [nearest]
        return eligible

    def allocate(
        self,
        events: List[Dict[str, Any]],
        zone_capacity: Optional[Dict[str, int]] = None,
    ) -> ManpowerResult:
        if not events:
            return ManpowerResult({}, 0, False, "none")

        normalized = [normalize_v12_event(e) for e in events]
        zone_capacity = zone_capacity or {}

        if HAS_PULP:
            result = self._solve_pulp(normalized, zone_capacity)
            if result is not None:
                return result

        result = self._solve_scipy_milp(normalized, zone_capacity)
        if result is not None:
            return result

        return self._greedy_fallback(normalized, zone_capacity)

    def _solve_pulp(
        self,
        events: List[Dict[str, Any]],
        zone_capacity: Dict[str, int],
    ) -> Optional[ManpowerResult]:
        n = len(events)
        min_need = [self._min_officers(e) for e in events]
        weights = [self._impact_weight(e) for e in events]

        # Build eligible (event, station) pairs
        pairs: List[Tuple[int, str]] = []
        for i, ev in enumerate(events):
            for st in self._eligible_stations(ev):
                pairs.append((i, st))

        if not pairs:
            return None

        prob = LpProblem("BTP_Manpower_Allocation", LpMinimize)

        # z[i,s] = officers from station s to event i
        z = {
            (i, s): LpVariable(f"z_{i}_{s}", lowBound=0, upBound=self.station_capacity.get(s, DEFAULT_STATION_CAPACITY), cat=LpInteger)
            for i, s in pairs
        }
        # undercoverage slack per event
        u = {i: LpVariable(f"u_{i}", lowBound=0, cat="Continuous") for i in range(n)}
        # total officers per event
        x = {i: lpSum(z[i, s] for i2, s in pairs if i2 == i) for i in range(n)}

        # Objective: minimize weighted undercoverage
        prob += lpSum(weights[i] * u[i] for i in range(n))

        for i in range(n):
            prob += u[i] >= min_need[i] - x[i]

        # Station capacity
        for st in BTP_STATIONS:
            prob += lpSum(z[i, s] for i, s in pairs if s == st) <= self.station_capacity.get(st, DEFAULT_STATION_CAPACITY)

        # Zone capacity (soft upper bound on total in zone)
        for zone, cap in zone_capacity.items():
            idxs = [i for i, e in enumerate(events) if e.get("zone") == zone]
            if idxs:
                prob += lpSum(x[i] for i in idxs) <= cap

        # Reasonable per-event cap
        for i in range(n):
            prob += x[i] <= MAX_OFFICERS_PER_EVENT

        solved = False
        for solver in (
            HiGHS_CMD(msg=False),
            PULP_CBC_CMD(path=None, keepFiles=0, msg=0),
        ):
            try:
                prob.solve(solver)
                if LpStatus[prob.status] in ("Optimal", "Feasible"):
                    solved = True
                    break
            except (PulpSolverError, TypeError, OSError) as exc:
                logger.debug("Solver %s failed: %s", solver, exc)
        if not solved:
            logger.warning("PuLP solvers unavailable — using greedy fallback.")
            return None

        allocation: Dict[str, Dict[str, Any]] = {}
        station_assignments: Dict[str, Dict[str, int]] = {st: {} for st in BTP_STATIONS}

        for i, ev in enumerate(events):
            assigned = int(round(sum(value(z[i, s]) for i2, s in pairs if i2 == i)))
            needed = min_need[i]
            ratio = assigned / max(needed, 1)
            allocation[ev["event_id"]] = {
                "officers_assigned": assigned,
                "min_needed": needed,
                "impact_weight": round(weights[i], 3),
                "coverage_ratio": round(ratio, 3),
                "status": "FULLY COVERED" if assigned >= needed else "PARTIAL - TRIAGE ACTIVE",
                "eligible_stations": self._eligible_stations(ev),
                "nearest_station": self._nearest_station(ev),
            }
            for i2, st in pairs:
                if i2 == i:
                    cnt = int(round(value(z[i, st])))
                    if cnt > 0:
                        station_assignments[st][ev["event_id"]] = cnt

        triage = any(a["coverage_ratio"] < 1.0 for a in allocation.values())
        total = sum(a["officers_assigned"] for a in allocation.values())

        return ManpowerResult(
            allocation=allocation,
            total_deployed=total,
            triage_active=triage,
            solver="pulp",
            station_assignments={k: v for k, v in station_assignments.items() if v},
        )

    def _nearest_station(self, ev: Dict[str, Any]) -> str:
        return min(
            BTP_STATIONS.keys(),
            key=lambda s: haversine_km(ev["lat"], ev["lon"], *BTP_STATIONS[s]),
        )

    def _solve_scipy_milp(
        self,
        events: List[Dict[str, Any]],
        zone_capacity: Dict[str, int],
    ) -> Optional[ManpowerResult]:
        """
        PuLP-equivalent event-level MIP via scipy (Windows-safe when CBC path has spaces).
        Minimizes sum(impact_weight * undercoverage slack) subject to zone caps.
        """
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
        except ImportError:
            return None

        n = len(events)
        if n == 0:
            return None

        min_need = np.array([self._min_officers(e) for e in events], dtype=float)
        weights = np.array([self._impact_weight(e) for e in events], dtype=float)

        # Variables: x[0..n-1] integer officers, u[0..n-1] continuous undercoverage
        c = np.concatenate([np.zeros(n), weights])
        integrality = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)

        constraints = []
        # x_i + u_i >= min_i  →  -x_i - u_i <= -min_i
        a_lb = np.zeros((n, 2 * n))
        for i in range(n):
            a_lb[i, i] = -1.0
            a_lb[i, n + i] = -1.0
        constraints.append(LinearConstraint(a_lb, ub=-min_need))

        # Zone capacity: sum_{i in zone} x_i <= cap
        zones_seen = set()
        for zone, cap in zone_capacity.items():
            if zone in zones_seen:
                continue
            zones_seen.add(zone)
            idxs = [i for i, e in enumerate(events) if e.get("zone") == zone]
            if not idxs:
                continue
            row = np.zeros(2 * n)
            for i in idxs:
                row[i] = 1.0
            constraints.append(LinearConstraint(row.reshape(1, -1), ub=[cap]))

        bounds = Bounds(
            lb=np.concatenate([np.zeros(n), np.zeros(n)]),
            ub=np.concatenate([np.full(n, MAX_OFFICERS_PER_EVENT), np.full(n, np.inf)]),
        )

        try:
            res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        except Exception as exc:
            logger.debug("scipy MILP failed: %s", exc)
            return None

        if not res.success:
            return None

        x = res.x[:n]
        allocation: Dict[str, Dict[str, Any]] = {}
        for i, ev in enumerate(events):
            assigned = int(round(x[i]))
            needed = int(min_need[i])
            ratio = assigned / max(needed, 1)
            allocation[ev["event_id"]] = {
                "officers_assigned": assigned,
                "min_needed": needed,
                "impact_weight": round(float(weights[i]), 3),
                "coverage_ratio": round(ratio, 3),
                "status": "FULLY COVERED" if assigned >= needed else "PARTIAL - TRIAGE ACTIVE",
                "eligible_stations": self._eligible_stations(ev),
                "nearest_station": self._nearest_station(ev),
            }

        triage = any(a["coverage_ratio"] < 1.0 for a in allocation.values())
        return ManpowerResult(
            allocation=allocation,
            total_deployed=int(sum(round(x[i]) for i in range(n))),
            triage_active=triage,
            solver="scipy_milp",
            station_assignments={},
        )

    def _greedy_fallback(self, events: List[Dict[str, Any]], zone_capacity: Dict[str, int]) -> ManpowerResult:
        station_pool = {s: self.station_capacity.get(s, DEFAULT_STATION_CAPACITY) for s in BTP_STATIONS}
        zone_pool = dict(zone_capacity) if zone_capacity else {}

        sorted_ev = sorted(events, key=lambda e: self._impact_weight(e), reverse=True)
        allocation: Dict[str, Dict[str, Any]] = {}
        station_assignments: Dict[str, Dict[str, int]] = {}

        for ev in sorted_ev:
            needed = self._min_officers(ev)
            assigned = 0
            elig = self._eligible_stations(ev)
            zone = ev.get("zone", "Central")
            zone_left = zone_pool.get(zone, 999)

            for st in sorted(elig, key=lambda s: haversine_km(ev["lat"], ev["lon"], *BTP_STATIONS[s])):
                if assigned >= needed or zone_left <= 0:
                    break
                avail = station_pool.get(st, 0)
                take = min(avail, needed - assigned, zone_left, MAX_OFFICERS_PER_EVENT - assigned)
                if take > 0:
                    station_pool[st] -= take
                    zone_left -= take
                    assigned += take
                    station_assignments.setdefault(st, {})[ev["event_id"]] = take

            zone_pool[zone] = zone_left
            ratio = assigned / max(needed, 1)
            allocation[ev["event_id"]] = {
                "officers_assigned": assigned,
                "min_needed": needed,
                "impact_weight": round(self._impact_weight(ev), 3),
                "coverage_ratio": round(ratio, 3),
                "status": "FULLY COVERED" if assigned >= needed else "PARTIAL - TRIAGE ACTIVE",
                "eligible_stations": elig,
                "nearest_station": self._nearest_station(ev),
            }

        triage = any(a["coverage_ratio"] < 1.0 for a in allocation.values())
        return ManpowerResult(
            allocation=allocation,
            total_deployed=sum(a["officers_assigned"] for a in allocation.values()),
            triage_active=triage,
            solver="greedy_fallback",
            station_assignments=station_assignments,
        )


class BarricadeDiversionRecommender:
    """Graph-centrality upstream points + penalized shortest-path diversions."""

    def __init__(self, graph_engine=None):
        self._engine = graph_engine

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from src.corridor_graph import CorridorGraphEngine

            self._engine = CorridorGraphEngine.from_precomputed()
            return self._engine
        except Exception:
            return None

    def recommend(self, event: Dict[str, Any]) -> Dict[str, Any]:
        ev = normalize_v12_event(event)
        subgraph = ev.get("affected_subgraph") or {}
        engine = self._get_engine()

        barricade = self._barricade_plan(ev, subgraph)
        diversions = self._diversion_routes(ev, subgraph, engine)
        upstream = self._upstream_control_points(ev, subgraph)

        return {
            "barricade_level": barricade["level"],
            "barricade_points": barricade["points"],
            "upstream_control_points": upstream,
            "diversion_routes": diversions,
            "rationale": barricade["rationale"] + [d.get("rationale", "") for d in diversions],
        }

    def _barricade_plan(self, ev: Dict[str, Any], subgraph: Dict[str, Any]) -> Dict[str, Any]:
        closure = ev["requires_road_closure"]
        regime = ev["regime"]
        betweenness = _safe_float(ev.get("graph_features", {}).get("graph_node_betweenness", 0))

        if closure or regime == "long" or ev["impact_score"] >= 7:
            level = "FULL"
        elif betweenness > 0.01 or ev["impact_score"] >= 4:
            level = "PARTIAL"
        else:
            level = "MINIMAL"

        points = []
        for node in (subgraph.get("nodes") or [])[:6]:
            if node.get("betweenness", 0) >= 0.005 or node.get("is_origin"):
                points.append({
                    "label": node.get("label"),
                    "lat": node.get("lat"),
                    "lon": node.get("lon"),
                    "reason": "High betweenness upstream control point" if not node.get("is_origin") else "Event origin — primary barricade",
                })

        if not points:
            points.append({
                "label": ev["corridor"],
                "lat": ev["lat"],
                "lon": ev["lon"],
                "reason": "Event location — default barricade anchor",
            })

        rationale = [
            f"Barricade level {level} driven by regime={regime}, impact={ev['impact_score']:.1f}, closure={closure}.",
            f"Identified {len(points)} control point(s) using graph centrality on affected subgraph.",
        ]
        return {"level": level, "points": points[:4], "rationale": rationale}

    def _upstream_control_points(self, ev: Dict[str, Any], subgraph: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes = subgraph.get("nodes") or []
        origin = subgraph.get("origin_node")
        upstream = []
        for node in nodes:
            if node.get("is_origin"):
                continue
            if node.get("betweenness", 0) >= 0.003:
                upstream.append({
                    "node_id": node.get("node_id"),
                    "label": node.get("label"),
                    "lat": node.get("lat"),
                    "lon": node.get("lon"),
                    "betweenness": node.get("betweenness"),
                    "role": "upstream_detour_signage",
                })
        upstream.sort(key=lambda x: -x.get("betweenness", 0))
        return upstream[:3]

    def _diversion_routes(
        self,
        ev: Dict[str, Any],
        subgraph: Dict[str, Any],
        engine,
    ) -> List[Dict[str, Any]]:
        if engine is None or not engine.ready:
            return self._fallback_diversions(ev)

        art = engine.builder.artifacts
        if art is None:
            return self._fallback_diversions(ev)

        ug = art.undirected
        origin, _ = engine.builder.snap_to_node(ev["lat"], ev["lon"], ev["corridor"])
        if origin not in ug:
            return self._fallback_diversions(ev)

        affected_edges = set()
        for e in subgraph.get("edges") or []:
            affected_edges.add(tuple(sorted((e["source"], e["target"]))))

        def edge_weight(u, v, data):
            base = float(data.get("length_km", 1.0))
            penalty = 6.0 if tuple(sorted((u, v))) in affected_edges else 1.0
            impact = float(data.get("impact_score", 1.0))
            return base * penalty * (1.0 + 0.05 * impact)

        # Target: high-centrality corridor hub or major junction reachable without affected edges
        targets = [
            n for n, d in art.node_meta.items()
            if d.get("node_type") == "junction" and "major" in n
        ][:3]
        if not targets:
            targets = [
                n for n, d in art.node_meta.items()
                if d.get("node_type") == "corridor" and n != origin
            ][:3]

        routes = []
        for tgt in targets:
            if tgt == origin or tgt not in ug:
                continue
            try:
                path = nx.shortest_path(ug, origin, tgt, weight=edge_weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            baseline_path = []
            try:
                baseline_path = nx.shortest_path(
                    ug, origin, tgt,
                    weight=lambda u, v, d: float(d.get("length_km", 1.0)),
                )
            except Exception:
                baseline_path = path

            def path_km(p):
                total = 0.0
                for a, b in zip(p, p[1:]):
                    data = ug.get_edge_data(a, b) or {}
                    total += float(data.get("length_km", 0.5))
                return total

            base_km = path_km(baseline_path)
            alt_km = path_km(path)
            hour = ev.get("hour", 12)
            congestion = float(ISEC_CONGESTION.get(hour, 1.4))
            base_mins = base_km / 25 * 60 * congestion
            alt_mins = alt_km / 25 * 60 * congestion
            savings = max(0.0, base_mins - alt_mins)

            labels = [art.node_meta.get(n, {}).get("label", n) for n in path[:5]]
            routes.append({
                "route_id": f"alt_{len(routes)+1}",
                "target": art.node_meta.get(tgt, {}).get("label", tgt),
                "waypoints": labels,
                "path_nodes": path[:8],
                "distance_km": round(alt_km, 2),
                "est_travel_mins": round(alt_mins, 1),
                "est_time_savings_mins": round(savings, 1),
                "rationale": (
                    f"Avoids {len(affected_edges)} high-impact corridor segment(s); "
                    f"estimated {savings:.0f} min savings vs baseline during hour {hour}."
                ),
            })
            if len(routes) >= 2:
                break

        return routes or self._fallback_diversions(ev)

    def _fallback_diversions(self, ev: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{
            "route_id": "alt_parallel",
            "target": f"Parallel route to {ev['corridor']}",
            "waypoints": [ev["corridor"], "adjacent arterial"],
            "distance_km": None,
            "est_travel_mins": None,
            "est_time_savings_mins": round(max(3.0, ev["impact_score"] * 1.5), 1),
            "rationale": "Graph unavailable — heuristic parallel corridor diversion suggested.",
        }]


class WhatIfSimulator:
    """
    Lightweight queueing-style what-if:
    baseline delay units vs projected after recommendations.
    No SUMO — uses M/M/c-inspired service scaling from officer coverage.
    """

    def simulate(
        self,
        event: Dict[str, Any],
        manpower: Dict[str, Any],
        barricade_diversion: Dict[str, Any],
    ) -> Dict[str, Any]:
        ev = normalize_v12_event(event)
        hour = ev.get("hour", 12)
        congestion = float(ISEC_CONGESTION.get(hour, 1.4))
        overlap = ev.get("active_overlap", 0)

        p50 = ev["p50_hours"]
        impact = ev["impact_score"]
        regime_mult = REGIME_MULTIPLIER.get(ev["regime"], 1.0)

        # Baseline congestion-delay units (CIU proxy)
        baseline = impact * p50 * congestion * regime_mult * (1.0 + 0.18 * overlap)

        alloc = manpower.get("allocation", {})
        ev_alloc = alloc.get(ev["event_id"], {})
        coverage = _safe_float(ev_alloc.get("coverage_ratio"), 0.5)
        officers = int(ev_alloc.get("officers_assigned", 0))
        min_need = int(ev_alloc.get("min_needed", 4))

        # Service rate scales with sqrt(officers) — M/M/c approximation
        service_factor = min(1.0, math.sqrt(max(officers, 1) / max(min_need, 1)))
        manpower_reduction = 0.35 * coverage * service_factor

        barricade_level = barricade_diversion.get("barricade_level", "MINIMAL")
        barricade_reduction = {"FULL": 0.22, "PARTIAL": 0.12, "MINIMAL": 0.04}.get(barricade_level, 0.05)

        diversions = barricade_diversion.get("diversion_routes") or []
        div_savings_mins = sum(_safe_float(d.get("est_time_savings_mins"), 0) for d in diversions)
        diversion_reduction = min(0.25, div_savings_mins / max(p50 * 60, 30))

        projected = baseline * (1.0 - manpower_reduction) * (1.0 - barricade_reduction) * (1.0 - diversion_reduction)
        improvement_pct = (baseline - projected) / max(baseline, 0.01) * 100.0

        quant_spread = (ev["p65_hours"] - ev["p35_hours"]) / max(ev["p50_hours"], 0.1)
        confidence = self._confidence(coverage, quant_spread, barricade_diversion)

        return {
            "baseline_delay_units": round(baseline, 3),
            "projected_delay_units": round(projected, 3),
            "improvement_pct": round(improvement_pct, 1),
            "improvement_absolute": round(baseline - projected, 3),
            "decomposition": {
                "manpower_reduction_pct": round(manpower_reduction * 100, 1),
                "barricade_reduction_pct": round(barricade_reduction * 100, 1),
                "diversion_reduction_pct": round(diversion_reduction * 100, 1),
            },
            "confidence": confidence,
            "queueing_model": "lightweight_mmc_proxy",
        }

    def _confidence(self, coverage: float, quant_spread: float, bd: Dict[str, Any]) -> Dict[str, Any]:
        score = 0.55
        score += 0.15 * min(1.0, coverage)
        score -= 0.10 * min(1.0, quant_spread)
        if bd.get("diversion_routes") and bd["diversion_routes"][0].get("path_nodes"):
            score += 0.12
        if bd.get("upstream_control_points"):
            score += 0.08
        score = float(np.clip(score, 0.35, 0.92))
        label = "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.55 else "LOW"
        return {"score": round(score, 3), "label": label}


def _humanize_cause(cause: str) -> str:
    return str(cause).replace("_", " ").strip().title()


def _humanize_place(label: Optional[str], fallback: str = "the nearest junction") -> str:
    if not label:
        return fallback
    cleaned = str(label).strip()
    if cleaned.lower() in ("non-corridor", "unknown", "none", ""):
        return fallback
    if cleaned.islower() or cleaned.isupper():
        return cleaned.title()
    return cleaned


def _hour_context(hour: int) -> str:
    if 7 <= hour <= 10:
        return "morning peak (7–10 AM)"
    if 17 <= hour <= 20:
        return "evening rush (5–8 PM)"
    if 11 <= hour <= 16:
        return "midday traffic"
    if 21 <= hour <= 23 or 0 <= hour <= 6:
        return "off-peak hours"
    return f"hour {hour}:00"


def _improvement_phrase(pct: float) -> str:
    if pct >= 85:
        return "a major drop in gridlock — most through-traffic should flow once diversions are live"
    if pct >= 70:
        return "roughly two-thirds less gridlock"
    if pct >= 50:
        return "about half the congestion impact"
    if pct >= 30:
        return "a noticeable reduction in delays"
    if pct >= 15:
        return "a modest but worthwhile improvement"
    return "a small reduction in local delays"


def _confidence_explanation(label: str, pct: int) -> str:
    explanations = {
        "HIGH": "Model and corridor data align well — treat this as a reliable starting plan.",
        "MEDIUM": "Reasonable estimate, but verify officer availability and on-ground conditions.",
        "LOW": "Limited data for this corridor — use as guidance and adjust once units arrive.",
    }
    base = explanations.get(label, explanations["MEDIUM"])
    return f"{label.title()} confidence ({pct}%): {base}"


def _build_display_summary(
    ev: Dict[str, Any],
    ev_alloc: Dict[str, Any],
    bd: Dict[str, Any],
    wif: Dict[str, Any],
    mp_triage: bool,
) -> Dict[str, Any]:
    """Plain-English summary for the command-center UI."""
    cause = _humanize_cause(ev["event_cause"])
    corridor = ev["corridor"]
    zone = ev["zone"]
    p50 = ev["p50_hours"]
    p35 = ev.get("p35_hours", p50 * 0.65)
    p65 = ev.get("p65_hours", p50 * 1.45)
    overlap = ev.get("active_overlap", 0)
    hour_ctx = _hour_context(int(ev.get("hour", 12)))

    officers = int(ev_alloc.get("officers_assigned", 0))
    min_need = int(ev_alloc.get("min_needed", 0))
    station = _humanize_place(
        ev_alloc.get("nearest_station") or ev.get("police_station"),
        "the nearest BTP station",
    )
    status_raw = str(ev_alloc.get("status", "PENDING"))
    status = status_raw.replace("_", " ").replace(" - ", " — ").title()
    coverage = _safe_float(ev_alloc.get("coverage_ratio"), 0)

    barricade = bd.get("barricade_level", "MINIMAL")
    points = bd.get("barricade_points") or []
    upstream = bd.get("upstream_control_points") or []
    diversions = bd.get("diversion_routes") or []

    point_labels = [_humanize_place(p.get("label"), corridor) for p in points[:3]]
    imp_pct = _safe_float(wif.get("improvement_pct"), 0)
    decomp = wif.get("decomposition") or {}
    conf = wif.get("confidence") or {}
    conf_pct = round(_safe_float(conf.get("score"), 0.6) * 100)
    conf_label = conf.get("label", "MEDIUM")

    clearance_band = (
        f"about {p50:.1f} hours to clear (typical range {p35:.1f}–{p65:.1f} h)"
        if p50 >= 0.5
        else f"likely under {max(p50, 0.3):.1f} hour to clear"
    )

    if coverage >= 1.0:
        mp_msg = (
            f"Assign {officers} officers from {station} to manage this {cause.lower()} on {corridor} "
            f"in the {zone} zone. That meets the minimum of {min_need} officers needed for safe traffic control."
        )
    elif officers > 0:
        shortfall = max(0, min_need - officers)
        mp_msg = (
            f"You have only {officers} of {min_need} officers needed from {station} "
            f"({shortfall} short). Call for backup from a neighbouring zone before traffic backs up on {corridor}."
        )
    else:
        mp_msg = (
            f"No officers are assigned yet — dispatch at least {min_need} from {station} "
            f"to secure {corridor} before congestion spreads."
        )
    if mp_triage:
        mp_msg += (
            f" Triage mode is on: {overlap} other incident(s) within 2 km are sharing the same officer pool."
        )

    primary_point = point_labels[0] if point_labels else corridor
    barricade_text = {
        "FULL": (
            f"Shut the road completely at {primary_point}. "
            "No through-traffic until the scene is cleared and debris is removed."
        ),
        "PARTIAL": (
            f"Block one carriageway at {primary_point}, "
            "but keep one lane open for ambulances, buses, and emergency vehicles."
        ),
        "MINIMAL": (
            f"Use cones and warning boards at {corridor} only — "
            "let traffic pass slowly in single file past the incident."
        ),
    }
    bar_msg = barricade_text.get(barricade, barricade_text["PARTIAL"])

    if diversions:
        best = max(diversions, key=lambda d: _safe_float(d.get("est_time_savings_mins"), 0))
        waypoints = [_humanize_place(w, "") for w in (best.get("waypoints") or []) if w]
        route_str = " → ".join(waypoints[:4]) if waypoints else _humanize_place(best.get("target"), "a parallel arterial")
        savings = _safe_float(best.get("est_time_savings_mins"), 0)
        travel = best.get("est_travel_mins")
        if savings >= 5:
            div_msg = (
                f"Direct commuters away from {corridor} onto: {route_str}. "
                f"This alternate path saves about {savings:.0f} minutes compared with waiting in the jam."
            )
        elif travel and travel < 90:
            div_msg = (
                f"Post signs directing traffic via {route_str}. "
                f"Expect roughly {travel:.0f} minutes on this route during {hour_ctx}."
            )
        else:
            div_msg = (
                f"Use {route_str} as a bypass while {corridor} stays blocked. "
                f"It may be longer than usual, but it avoids the incident zone entirely."
            )
    else:
        div_msg = (
            f"No pre-mapped diversion for this spot — send units to guide drivers "
            f"onto parallel roads near {corridor}."
        )

    imp_phrase = _improvement_phrase(imp_pct)
    top_driver = max(
        [
            ("officer deployment", decomp.get("manpower_reduction_pct", 0)),
            ("barricades", decomp.get("barricade_reduction_pct", 0)),
            ("traffic diversions", decomp.get("diversion_reduction_pct", 0)),
        ],
        key=lambda x: x[1],
    )[0]
    wi_msg = (
        f"Following this plan should mean {imp_phrase} on {corridor} "
        f"(~{imp_pct:.0f}% less congestion than doing nothing). "
        f"The biggest benefit comes from {top_driver}."
    )

    barricade_word = {"FULL": "full road closure", "PARTIAL": "partial lane closure", "MINIMAL": "light cones only"}.get(
        barricade, barricade.lower()
    )
    headline = (
        f"{cause} on {corridor}: send {officers} officers, set up {barricade_word}, "
        f"{clearance_band}."
    )

    action_items = []
    if officers > 0:
        action_items.append(f"Dispatch {officers} officers from {station} — {status}")
    else:
        action_items.append(f"Request {min_need} officers from {station}")
    if points:
        primary = points[0]
        place = _humanize_place(primary.get("label"), corridor)
        reason = primary.get("reason", "")
        if "origin" in reason.lower():
            reason_plain = "this is where the incident started"
        elif "betweenness" in reason.lower():
            reason_plain = "this junction feeds most traffic into the corridor"
        elif reason:
            reason_plain = reason[0].lower() + reason[1:] if reason else ""
        else:
            reason_plain = ""
        action_items.append(
            f"Erect {barricade.lower()} barricade at {place}"
            + (f" ({reason_plain})" if reason_plain else "")
        )
    for pt in upstream[:2]:
        label = _humanize_place(pt.get("label"))
        if label != "the nearest junction":
            action_items.append(f"Put diversion signs at {label} (upstream of the block)")
    if diversions:
        wp = [_humanize_place(w, "") for w in (diversions[0].get("waypoints") or []) if w]
        if wp:
            action_items.append(f"Announce on radio/social: use {' → '.join(wp[:3])}")

    cascade_causes = [_humanize_cause(c) for c in (bd.get("cascade_watch") or [])]
    cascade = []
    if cascade_causes:
        if len(cascade_causes) == 1:
            joined = cascade_causes[0].lower()
        elif len(cascade_causes) == 2:
            joined = f"{cascade_causes[0].lower()} and {cascade_causes[1].lower()}"
        else:
            joined = ", ".join(c.lower() for c in cascade_causes[:-1]) + f" and {cascade_causes[-1].lower()}"
        cascade.append(
            f"After this {cause.lower()}, watch for secondary {joined} within 2 km — "
            "these often follow the first incident during peak traffic."
        )

    return {
        "headline": headline,
        "manpower_message": mp_msg,
        "barricade_message": bar_msg,
        "diversion_message": div_msg,
        "what_if_message": wi_msg,
        "action_items": action_items,
        "cascade_watch_messages": cascade,
        "confidence_label": conf_label,
        "confidence_pct": conf_pct,
        "confidence_message": _confidence_explanation(conf_label, conf_pct),
        "predicted_clearance_hrs": round(p50, 1),
        "officers_recommended": officers,
        "barricade_level": barricade,
        "improvement_pct": round(imp_pct, 1),
        "event_cause_label": cause,
        "time_context": hour_ctx,
    }


class PrescriptiveEngine:
    """
    Main entry point for Phase 3 recommendations.
    Accepts V12 outputs — never calls V12/inference directly.
    """

    def __init__(self):
        self.manpower = ManpowerAllocator()
        self.barricade = BarricadeDiversionRecommender()
        self.simulator = WhatIfSimulator()

    def _attach_graph_context(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Fill graph_features / affected_subgraph from Phase 2 if missing."""
        out = dict(event)
        if out.get("affected_subgraph") and out.get("graph_features"):
            return out
        try:
            from src.graph_feature_bridge import GraphFeatureEnricher

            enricher = GraphFeatureEnricher.from_precomputed()
            if enricher.ready:
                if not out.get("graph_features"):
                    out["graph_features"] = enricher.features_for_dict(out)
                if not out.get("affected_subgraph"):
                    out["affected_subgraph"] = enricher.affected_subgraph(out)
        except Exception as exc:
            logger.debug("Graph context unavailable: %s", exc)
        return out

    def recommend_one(
        self,
        event: Dict[str, Any],
        zone_capacity: Optional[Dict[str, int]] = None,
        concurrent_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        ev = self._attach_graph_context(event)
        peers = [self._attach_graph_context(e) for e in (concurrent_events or [ev])]
        if not any(p.get("event_id") == ev.get("event_id") for p in peers):
            peers.append(ev)

        mp = self.manpower.allocate(peers, zone_capacity=zone_capacity or {ev.get("zone", "Central"): 14})
        bd = self.barricade.recommend(ev)
        wif = self.simulator.simulate(ev, {"allocation": mp.allocation}, bd)

        ev_norm = normalize_v12_event(ev)
        ev_alloc = mp.allocation.get(ev_norm["event_id"], {})

        rationale = [
            f"MoE regime={ev_norm['regime']}: P50={ev_norm['p50_hours']:.1f}h "
            f"(band {ev_norm['p35_hours']:.1f}–{ev_norm['p65_hours']:.1f}h), impact={ev_norm['impact_score']:.1f}.",
            f"Active overlap within 2km: {ev_norm['active_overlap']} concurrent event(s).",
            f"Manpower ({mp.solver}): assign {ev_alloc.get('officers_assigned', 0)} officers "
            f"(need {ev_alloc.get('min_needed', 0)}, coverage {ev_alloc.get('coverage_ratio', 0):.0%}).",
            f"Barricade plan: {bd['barricade_level']} at {len(bd['barricade_points'])} point(s).",
        ]
        if bd.get("diversion_routes"):
            rationale.append(
                f"Diversion: {len(bd['diversion_routes'])} route(s); "
                f"best savings ~{bd['diversion_routes'][0].get('est_time_savings_mins', '?')} min."
            )
        rationale.append(
            f"What-if: {wif['improvement_pct']:.1f}% delay reduction expected (confidence {wif['confidence']['label']})."
        )

        bd_with_cascade = {**bd, "cascade_watch": self._cascade_watch(ev_norm)}
        display = _build_display_summary(ev_norm, ev_alloc, bd_with_cascade, wif, mp.triage_active)

        return {
            "event_id": ev_norm["event_id"],
            "display_summary": display,
            "recommendations": {
                "manpower": {
                    "officers_recommended": ev_alloc.get("officers_assigned", 0),
                    "min_needed": ev_alloc.get("min_needed", 0),
                    "nearest_station": ev_alloc.get("nearest_station"),
                    "coverage_ratio": ev_alloc.get("coverage_ratio", 0),
                    "status": ev_alloc.get("status", "PENDING"),
                },
                "barricading": {
                    "level": bd["barricade_level"],
                    "control_points": bd["barricade_points"],
                    "upstream_points": bd["upstream_control_points"],
                },
                "diversions": bd["diversion_routes"],
                "cascade_watch": bd_with_cascade.get("cascade_watch"),
            },
            "manpower_allocation": {
                "solver": mp.solver,
                "triage_active": mp.triage_active,
                "total_deployed": mp.total_deployed,
                "allocation": mp.allocation,
                "station_assignments": mp.station_assignments,
            },
            "what_if": wif,
            "rationale": rationale,
            "inputs_summary": {
                "regime": ev_norm["regime"],
                "p35_hours": ev_norm["p35_hours"],
                "p50_hours": ev_norm["p50_hours"],
                "p65_hours": ev_norm["p65_hours"],
                "impact_score": ev_norm["impact_score"],
                "active_overlap": ev_norm["active_overlap"],
                "corridor": ev_norm["corridor"],
                "zone": ev_norm["zone"],
            },
            "confidence": wif["confidence"],
        }

    def recommend_batch(
        self,
        events: List[Dict[str, Any]],
        zone_capacity: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        enriched = [self._attach_graph_context(e) for e in events]
        mp = self.manpower.allocate(enriched, zone_capacity=zone_capacity)
        per_event = []
        for ev in enriched:
            bd = self.barricade.recommend(ev)
            wif = self.simulator.simulate(ev, {"allocation": mp.allocation}, bd)
            norm = normalize_v12_event(ev)
            per_event.append({
                "event_id": norm["event_id"],
                "manpower": mp.allocation.get(norm["event_id"], {}),
                "barricading": bd,
                "what_if": wif,
            })

        return {
            "batch": True,
            "n_events": len(events),
            "manpower_allocation": {
                "solver": mp.solver,
                "triage_active": mp.triage_active,
                "total_deployed": mp.total_deployed,
                "allocation": mp.allocation,
                "station_assignments": mp.station_assignments,
            },
            "events": per_event,
            "confidence": {
                "label": "MEDIUM",
                "score": round(float(np.mean([e["what_if"]["confidence"]["score"] for e in per_event])) if per_event else 0.5, 3),
            },
        }

    @staticmethod
    def _cascade_watch(ev: Dict[str, Any]) -> List[str]:
        mapping = {
            "accident": ["vehicle_breakdown", "congestion"],
            "tree_fall": ["accident", "congestion"],
            "water_logging": ["accident", "vehicle_breakdown"],
            "construction": ["congestion", "vehicle_breakdown"],
            "public_event": ["congestion", "accident"],
            "procession": ["congestion", "public_event"],
        }
        return mapping.get(ev["event_cause"], ["congestion"])


def build_prescriptive_response(
    event: Dict[str, Any],
    zone_capacity: Optional[Dict[str, int]] = None,
    concurrent_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Convenience function for API / notebooks."""
    return PrescriptiveEngine().recommend_one(event, zone_capacity, concurrent_events)