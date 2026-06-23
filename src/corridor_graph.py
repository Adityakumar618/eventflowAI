"""
Phase 2 — Light Graph Features (NetworkX on Corridors/Junctions)
================================================================
Additive module only. Does NOT modify inference, V12, or production models.

Builds a corridor/junction graph from ASTraM event data, pre-computes
centrality and k-hop downstream features, and exposes subgraph queries
for Phase 3 recommendation work.

Artifacts (written by scripts/precompute_corridor_graph.py):
  data/precomputed/corridor_graph.pkl
  data/precomputed/corridor_graph_features.json
  data/precomputed/corridor_graph_snapper.pkl
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import joblib
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_PATH = BASE_DIR / "data" / "precomputed" / "corridor_graph.pkl"
DEFAULT_FEATURES_PATH = BASE_DIR / "data" / "precomputed" / "corridor_graph_features.json"
DEFAULT_SNAPPER_PATH = BASE_DIR / "data" / "precomputed" / "corridor_graph_snapper.pkl"

# Reuse established Bengaluru anchors (same coords as advanced_event_fe — read-only reference)
MAJOR_JUNCTIONS: Dict[str, Tuple[float, float]] = {
    "silk_board": (12.9177, 77.6228),
    "hebbal": (13.0354, 77.5910),
    "mekhri": (13.0090, 77.5770),
    "yeshwanthpur": (13.0267, 77.5361),
    "kr_puram": (13.0053, 77.6946),
    "madiwala": (12.9270, 77.6170),
}

CBD_ANCHOR = (12.9716, 77.5946)  # MG Road / city-centre proxy for downstream direction

GRAPH_FEATURE_NAMES = [
    "graph_node_betweenness",
    "graph_node_closeness",
    "graph_node_degree",
    "graph_corridor_edge_impact",
    "graph_corridor_edge_priority",
    "graph_snap_distance_km",
    "graph_downstream_k1_count",
    "graph_downstream_k2_count",
    "graph_upstream_k1_count",
    "graph_affected_edge_count_k2",
]

_JUNCTION_PATTERNS = re.compile(
    r"([\w\s]{3,60}?(?:junction|circle|signal|flyover|underpass|interchange))",
    re.IGNORECASE,
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return float(2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def _normalize_name(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text[:60] or "unknown"


def _extract_junction_label(address: str) -> Optional[str]:
    if not address or not isinstance(address, str):
        return None
    m = _JUNCTION_PATTERNS.search(address)
    if not m:
        return None
    return _normalize_name(m.group(1))


@dataclass
class GraphNode:
    node_id: str
    node_type: str  # junction | corridor | event_anchor
    lat: float
    lon: float
    label: str = ""
    corridor: Optional[str] = None
    event_count: int = 0


@dataclass
class CorridorGraphArtifacts:
    graph: nx.DiGraph
    undirected: nx.Graph
    node_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edge_meta: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    centralities: Dict[str, Dict[str, float]] = field(default_factory=dict)
    corridor_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    snap_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    snap_node_ids: List[str] = field(default_factory=list)


class CorridorGraphBuilder:
    """
    Builds a directed corridor/junction graph from historical ASTraM events.

    Nodes:
      - corridor:<name>  — corridor hub at median event location
      - junction:<name>  — named junction from address parsing + major anchors
    Edges:
      - corridor segment links (corridor hub ↔ junction, junction ↔ junction)
      - inter-corridor links when hubs are spatially close
    """

    def __init__(
        self,
        max_junctions: int = 80,
        inter_corridor_km: float = 7.0,
        junction_cluster_km: float = 0.35,
    ):
        self.max_junctions = max_junctions
        self.inter_corridor_km = inter_corridor_km
        self.junction_cluster_km = junction_cluster_km
        self.artifacts: Optional[CorridorGraphArtifacts] = None

    def fit(self, df: pd.DataFrame) -> CorridorGraphArtifacts:
        work = df.copy()
        work = work.dropna(subset=["latitude", "longitude"])
        work["corridor"] = work["corridor"].fillna("Non-corridor").astype(str)
        work["address"] = work.get("address", pd.Series([""] * len(work))).fillna("")

        if "duration_hrs" not in work.columns:
            work["duration_hrs"] = self._fallback_duration(work)

        corridor_stats = self._corridor_stats(work)
        nodes: Dict[str, GraphNode] = {}
        g = nx.DiGraph()
        ug = nx.Graph()

        # Corridor hub nodes
        for corridor, stats in corridor_stats.items():
            nid = f"corridor:{_normalize_name(corridor)}"
            nodes[nid] = GraphNode(
                node_id=nid,
                node_type="corridor",
                lat=stats["lat"],
                lon=stats["lon"],
                label=corridor,
                corridor=corridor,
                event_count=int(stats["event_count"]),
            )
            g.add_node(nid, **self._node_attrs(nodes[nid], stats))
            ug.add_node(nid, **self._node_attrs(nodes[nid], stats))

        # Major junction anchors
        for name, (lat, lon) in MAJOR_JUNCTIONS.items():
            nid = f"junction:major_{name}"
            nodes[nid] = GraphNode(
                node_id=nid,
                node_type="junction",
                lat=lat,
                lon=lon,
                label=name.replace("_", " ").title(),
                event_count=0,
            )
            g.add_node(nid, **self._node_attrs(nodes[nid]))
            ug.add_node(nid, **self._node_attrs(nodes[nid]))

        # Parsed junction nodes from addresses
        parsed = self._extract_junction_nodes(work)
        for nid, node in parsed.items():
            if nid not in nodes:
                nodes[nid] = node
                g.add_node(nid, **self._node_attrs(node))
                ug.add_node(nid, **self._node_attrs(node))

        edge_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Corridor hub → nearby junctions on same corridor
        for corridor, stats in corridor_stats.items():
            c_nid = f"corridor:{_normalize_name(corridor)}"
            c_lat, c_lon = stats["lat"], stats["lon"]
            local = work[work["corridor"] == corridor]
            for _, row in local.iterrows():
                jlabel = _extract_junction_label(str(row.get("address", "")))
                if not jlabel:
                    continue
                j_nid = f"junction:{jlabel}"
                if j_nid not in nodes:
                    nodes[j_nid] = GraphNode(
                        node_id=j_nid,
                        node_type="junction",
                        lat=float(row["latitude"]),
                        lon=float(row["longitude"]),
                        label=jlabel,
                        corridor=corridor,
                        event_count=1,
                    )
                    g.add_node(j_nid, **self._node_attrs(nodes[j_nid]))
                    ug.add_node(j_nid, **self._node_attrs(nodes[j_nid]))
                else:
                    nodes[j_nid].event_count += 1

                length_km = haversine_km(c_lat, c_lon, nodes[j_nid].lat, nodes[j_nid].lon)
                attrs = self._edge_attrs(
                    corridor=corridor,
                    length_km=length_km,
                    stats=stats,
                    priority=stats.get("priority", 1.0),
                )
                self._add_directed_edge(g, c_nid, j_nid, attrs, edge_meta, downstream=True)
                self._add_undirected_edge(ug, c_nid, j_nid, attrs)

        # Junction pairs on same corridor (segment mesh)
        for corridor in corridor_stats:
            j_nodes = [
                n for n in nodes.values()
                if n.node_type == "junction" and (n.corridor == corridor or n.corridor is None)
            ]
            j_on_corr = []
            for n in j_nodes:
                dist = haversine_km(
                    n.lat, n.lon,
                    corridor_stats[corridor]["lat"],
                    corridor_stats[corridor]["lon"],
                )
                if dist <= 4.0:
                    j_on_corr.append(n)
            j_on_corr = sorted(j_on_corr, key=lambda x: -x.event_count)[:12]
            stats = corridor_stats[corridor]
            for i in range(len(j_on_corr)):
                for j in range(i + 1, len(j_on_corr)):
                    a, b = j_on_corr[i], j_on_corr[j]
                    length_km = haversine_km(a.lat, a.lon, b.lat, b.lon)
                    if length_km > 5.0:
                        continue
                    attrs = self._edge_attrs(
                        corridor=corridor,
                        length_km=length_km,
                        stats=stats,
                        priority=stats.get("priority", 1.0) * 0.85,
                    )
                    # Direction: peripheral → toward CBD
                    if self._dist_to_cbd(a) > self._dist_to_cbd(b):
                        src, dst = a.node_id, b.node_id
                    else:
                        src, dst = b.node_id, a.node_id
                    self._add_directed_edge(g, src, dst, attrs, edge_meta, downstream=True)
                    self._add_undirected_edge(ug, a.node_id, b.node_id, attrs)

        # Inter-corridor edges between close hubs
        corr_ids = [f"corridor:{_normalize_name(c)}" for c in corridor_stats]
        for i, c1 in enumerate(corr_ids):
            for c2 in corr_ids[i + 1 :]:
                n1, n2 = nodes[c1], nodes[c2]
                length_km = haversine_km(n1.lat, n1.lon, n2.lat, n2.lon)
                if length_km > self.inter_corridor_km:
                    continue
                prio = (corridor_stats[n1.label]["priority"] + corridor_stats[n2.label]["priority"]) / 2
                attrs = self._edge_attrs(
                    corridor=f"{n1.label}|{n2.label}",
                    length_km=length_km,
                    stats={"med_duration_hrs": (corridor_stats[n1.label]["med_duration_hrs"] + corridor_stats[n2.label]["med_duration_hrs"]) / 2,
                           "impact_score": (corridor_stats[n1.label]["impact_score"] + corridor_stats[n2.label]["impact_score"]) / 2,
                           "closure_rate": (corridor_stats[n1.label]["closure_rate"] + corridor_stats[n2.label]["closure_rate"]) / 2},
                    priority=prio * 0.7,
                )
                if self._dist_to_cbd(n1) > self._dist_to_cbd(n2):
                    self._add_directed_edge(g, c1, c2, attrs, edge_meta, downstream=True)
                else:
                    self._add_directed_edge(g, c2, c1, attrs, edge_meta, downstream=True)
                self._add_undirected_edge(ug, c1, c2, attrs)

        centralities = self._compute_centralities(ug)
        snap_points, snap_ids = self._build_snap_index(nodes)

        node_meta = {
            nid: {
                "node_type": n.node_type,
                "lat": n.lat,
                "lon": n.lon,
                "label": n.label,
                "corridor": n.corridor,
                "event_count": n.event_count,
                **centralities.get(nid, {}),
            }
            for nid, n in nodes.items()
        }

        self.artifacts = CorridorGraphArtifacts(
            graph=g,
            undirected=ug,
            node_meta=node_meta,
            edge_meta=edge_meta,
            centralities=centralities,
            corridor_stats=corridor_stats,
            snap_points=snap_points,
            snap_node_ids=snap_ids,
        )
        logger.info(
            "Corridor graph built: %d nodes, %d directed edges, %d undirected edges",
            g.number_of_nodes(),
            g.number_of_edges(),
            ug.number_of_edges(),
        )
        return self.artifacts

    def save(self, graph_path=None, features_path=None, snapper_path=None) -> None:
        if self.artifacts is None:
            raise RuntimeError("Call fit() before save()")
        graph_path = Path(graph_path or DEFAULT_GRAPH_PATH)
        features_path = Path(features_path or DEFAULT_FEATURES_PATH)
        snapper_path = Path(snapper_path or DEFAULT_SNAPPER_PATH)
        graph_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.artifacts, graph_path)
        payload = {
            "node_meta": self.artifacts.node_meta,
            "edge_meta": {f"{a}|{b}": v for (a, b), v in self.artifacts.edge_meta.items()},
            "corridor_stats": self.artifacts.corridor_stats,
            "centralities": self.artifacts.centralities,
            "feature_names": GRAPH_FEATURE_NAMES,
            "n_nodes": self.artifacts.graph.number_of_nodes(),
            "n_edges": self.artifacts.graph.number_of_edges(),
        }
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        snapper = {
            "snap_points": self.artifacts.snap_points,
            "snap_node_ids": self.artifacts.snap_node_ids,
        }
        joblib.dump(snapper, snapper_path)
        logger.info("Saved graph artifacts to %s", graph_path.parent)

    @staticmethod
    def load(graph_path=None) -> CorridorGraphArtifacts:
        path = Path(graph_path or DEFAULT_GRAPH_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Graph not precomputed. Run scripts/precompute_corridor_graph.py — missing {path}")
        return joblib.load(path)

    # ── Query API (Phase 3 ready) ───────────────────────────────────────────

    def snap_to_node(self, lat: float, lon: float, corridor: Optional[str] = None) -> Tuple[str, float]:
        """Return (node_id, snap_distance_km)."""
        art = self.artifacts
        if art is None or len(art.snap_node_ids) == 0:
            return "unknown", 999.0

        dists = np.array([
            haversine_km(lat, lon, art.node_meta[nid]["lat"], art.node_meta[nid]["lon"])
            for nid in art.snap_node_ids
        ])
        idx = int(np.argmin(dists))

        # Prefer same-corridor hub when close enough
        if corridor:
            c_nid = f"corridor:{_normalize_name(corridor)}"
            if c_nid in art.node_meta:
                c_dist = haversine_km(lat, lon, art.node_meta[c_nid]["lat"], art.node_meta[c_nid]["lon"])
                if c_dist <= dists[idx] + 0.5:
                    return c_nid, c_dist

        return art.snap_node_ids[idx], float(dists[idx])

    def event_graph_features(
        self,
        lat: float,
        lon: float,
        corridor: str = "Non-corridor",
        k_hops: int = 2,
    ) -> Dict[str, float]:
        """Lookup precomputed graph features for a single event location."""
        art = self.artifacts
        if art is None:
            return {k: 0.0 for k in GRAPH_FEATURE_NAMES}

        node_id, snap_km = self.snap_to_node(lat, lon, corridor)
        meta = art.node_meta.get(node_id, {})
        c_stats = art.corridor_stats.get(corridor, art.corridor_stats.get("Non-corridor", {}))

        downstream_k1 = self._k_hop_neighbors(art.graph, node_id, 1, direction="successors")
        downstream_k2 = self._k_hop_neighbors(art.graph, node_id, k_hops, direction="successors")
        upstream_k1 = self._k_hop_neighbors(art.graph, node_id, 1, direction="predecessors")

        return {
            "graph_node_betweenness": float(meta.get("betweenness", 0.0)),
            "graph_node_closeness": float(meta.get("closeness", 0.0)),
            "graph_node_degree": float(meta.get("degree", 0.0)),
            "graph_corridor_edge_impact": float(c_stats.get("impact_score", 0.0)),
            "graph_corridor_edge_priority": float(c_stats.get("priority", 0.0)),
            "graph_snap_distance_km": float(snap_km),
            "graph_downstream_k1_count": float(len(downstream_k1)),
            "graph_downstream_k2_count": float(len(downstream_k2)),
            "graph_upstream_k1_count": float(len(upstream_k1)),
            "graph_affected_edge_count_k2": float(self._affected_edge_count(art.graph, node_id, k_hops)),
        }

    def get_affected_subgraph(
        self,
        lat: float,
        lon: float,
        corridor: str = "Non-corridor",
        k_hops: int = 2,
        max_nodes: int = 60,
    ) -> Dict[str, Any]:
        """
        Phase 3 query: return nodes/edges reachable within k hops downstream
        (directed) plus undirected neighborhood for context.
        """
        art = self.artifacts
        if art is None:
            return {"available": False, "message": "Graph not loaded"}

        origin, snap_km = self.snap_to_node(lat, lon, corridor)
        downstream = self._k_hop_neighbors(art.graph, origin, k_hops, direction="successors")
        undirected = self._k_hop_neighbors(art.undirected, origin, k_hops, direction="neighbors")
        affected_nodes: Set[str] = {origin} | downstream | undirected

        if len(affected_nodes) > max_nodes:
            # Keep closest nodes to origin
            ranked = sorted(
                affected_nodes,
                key=lambda nid: haversine_km(
                    lat, lon,
                    art.node_meta[nid]["lat"],
                    art.node_meta[nid]["lon"],
                ),
            )
            affected_nodes = set(ranked[:max_nodes])

        nodes_out = []
        for nid in affected_nodes:
            m = art.node_meta.get(nid, {})
            nodes_out.append({
                "node_id": nid,
                "node_type": m.get("node_type"),
                "label": m.get("label"),
                "lat": m.get("lat"),
                "lon": m.get("lon"),
                "corridor": m.get("corridor"),
                "betweenness": m.get("betweenness", 0.0),
                "closeness": m.get("closeness", 0.0),
                "is_origin": nid == origin,
            })

        edges_out = []
        for u, v, data in art.graph.edges(data=True):
            if u in affected_nodes and v in affected_nodes:
                edges_out.append({
                    "source": u,
                    "target": v,
                    "directed": True,
                    "length_km": data.get("length_km"),
                    "impact_score": data.get("impact_score"),
                    "priority": data.get("priority"),
                    "corridor": data.get("corridor"),
                })
        for u, v, data in art.undirected.edges(data=True):
            if u in affected_nodes and v in affected_nodes:
                key = tuple(sorted((u, v)))
                if any(e["source"] == u and e["target"] == v or e["source"] == v and e["target"] == u for e in edges_out):
                    continue
                edges_out.append({
                    "source": u,
                    "target": v,
                    "directed": False,
                    "length_km": data.get("length_km"),
                    "impact_score": data.get("impact_score"),
                    "priority": data.get("priority"),
                    "corridor": data.get("corridor"),
                })

        return {
            "available": True,
            "origin_node": origin,
            "snap_distance_km": snap_km,
            "corridor": corridor,
            "k_hops": k_hops,
            "graph_features": self.event_graph_features(lat, lon, corridor, k_hops),
            "nodes": nodes_out,
            "edges": edges_out,
            "n_nodes": len(nodes_out),
            "n_edges": len(edges_out),
        }

    # ── internals ───────────────────────────────────────────────────────────

    def _fallback_duration(self, df: pd.DataFrame) -> pd.Series:
        start = pd.to_datetime(df.get("start_datetime"), errors="coerce")
        closed = pd.to_datetime(df.get("closed_datetime"), errors="coerce")
        resolved = pd.to_datetime(df.get("resolved_datetime"), errors="coerce")
        max_d = start.max()
        dur = np.where(
            closed.notna(),
            (closed - start).dt.total_seconds() / 3600,
            np.where(
                resolved.notna(),
                (resolved - start).dt.total_seconds() / 3600,
                (max_d - start).dt.total_seconds() / 3600,
            ),
        )
        return pd.Series(np.maximum(dur, 0.05), index=df.index)

    def _corridor_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        total = len(df)
        max_count = max(df["corridor"].value_counts().max(), 1)

        impact_path = BASE_DIR / "data" / "precomputed" / "corridor_impact_stats.parquet"
        impact_df = None
        if impact_path.exists():
            impact_df = pd.read_parquet(impact_path).set_index("corridor", drop=False)

        for corridor, grp in df.groupby("corridor"):
            lat = float(grp["latitude"].median())
            lon = float(grp["longitude"].median())
            med_dur = float(grp["duration_hrs"].median())
            closure = float(grp.get("requires_road_closure", pd.Series(0, index=grp.index)).mean())
            count = len(grp)
            priority = count / max_count

            impact_score = med_dur * (1 + 1.5 * closure) * priority
            if impact_df is not None and corridor in impact_df.index:
                row = impact_df.loc[corridor]
                med_dur = float(row.get("med_dur_obs", med_dur))
                closure = float(row.get("closure_rate", closure))
                impact_score = med_dur * (1 + closure) * priority

            stats[corridor] = {
                "lat": lat,
                "lon": lon,
                "event_count": count,
                "med_duration_hrs": med_dur,
                "closure_rate": closure,
                "priority": priority,
                "impact_score": impact_score,
            }
        return stats

    def _extract_junction_nodes(self, df: pd.DataFrame) -> Dict[str, GraphNode]:
        records = []
        for _, row in df.iterrows():
            label = _extract_junction_label(str(row.get("address", "")))
            if not label:
                continue
            records.append((label, float(row["latitude"]), float(row["longitude"]), str(row.get("corridor", ""))))

        if not records:
            return {}

        jdf = pd.DataFrame(records, columns=["label", "lat", "lon", "corridor"])
        jdf = jdf.groupby("label").agg(
            lat=("lat", "median"),
            lon=("lon", "median"),
            corridor=("corridor", lambda x: x.mode().iloc[0] if len(x) else ""),
            count=("label", "count"),
        )
        jdf = jdf.sort_values("count", ascending=False).head(self.max_junctions)

        out: Dict[str, GraphNode] = {}
        for label, row in jdf.iterrows():
            nid = f"junction:{label}"
            out[nid] = GraphNode(
                node_id=nid,
                node_type="junction",
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                label=label,
                corridor=str(row["corridor"]) if row["corridor"] else None,
                event_count=int(row["count"]),
            )
        return out

    @staticmethod
    def _node_attrs(node: GraphNode, corridor_stats: Optional[Dict] = None) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {
            "node_type": node.node_type,
            "lat": node.lat,
            "lon": node.lon,
            "label": node.label,
            "corridor": node.corridor,
            "event_count": node.event_count,
        }
        if corridor_stats:
            attrs.update({f"corridor_{k}": v for k, v in corridor_stats.items() if k not in ("lat", "lon")})
        return attrs

    @staticmethod
    def _edge_attrs(corridor: str, length_km: float, stats: Dict, priority: float) -> Dict[str, Any]:
        return {
            "corridor": corridor,
            "length_km": round(length_km, 3),
            "med_duration_hrs": float(stats.get("med_duration_hrs", 1.0)),
            "impact_score": float(stats.get("impact_score", 1.0)),
            "closure_rate": float(stats.get("closure_rate", 0.0)),
            "priority": round(priority, 4),
        }

    @staticmethod
    def _add_directed_edge(g, src, dst, attrs, edge_meta, downstream=True):
        if src == dst:
            return
        g.add_edge(src, dst, **attrs)
        edge_meta[(src, dst)] = attrs

    @staticmethod
    def _add_undirected_edge(ug, a, b, attrs):
        if a == b:
            return
        if ug.has_edge(a, b):
            existing = ug[a][b]
            if attrs.get("priority", 0) > existing.get("priority", 0):
                ug[a][b].update(attrs)
        else:
            ug.add_edge(a, b, **attrs)

    @staticmethod
    def _dist_to_cbd(node: GraphNode) -> float:
        return haversine_km(node.lat, node.lon, CBD_ANCHOR[0], CBD_ANCHOR[1])

    def _compute_centralities(self, ug: nx.Graph) -> Dict[str, Dict[str, float]]:
        if ug.number_of_nodes() == 0:
            return {}
        bc = nx.betweenness_centrality(ug, weight="length_km", normalized=True)
        cc = nx.closeness_centrality(ug, distance="length_km")
        dc = dict(ug.degree())
        max_deg = max(dc.values()) if dc else 1
        return {
            nid: {
                "betweenness": round(float(bc.get(nid, 0.0)), 6),
                "closeness": round(float(cc.get(nid, 0.0)), 6),
                "degree": round(float(dc.get(nid, 0)) / max_deg, 6),
            }
            for nid in ug.nodes
        }

    def _build_snap_index(self, nodes: Dict[str, GraphNode]) -> Tuple[np.ndarray, List[str]]:
        ids = list(nodes.keys())
        pts = np.array([[nodes[i].lat, nodes[i].lon] for i in ids], dtype=np.float64)
        return pts, ids

    @staticmethod
    def _k_hop_neighbors(graph, origin: str, k: int, direction: str = "successors") -> Set[str]:
        if origin not in graph:
            return set()
        seen: Set[str] = set()
        frontier: Set[str] = {origin}
        for _ in range(k):
            nxt: Set[str] = set()
            for node in frontier:
                if direction == "successors":
                    nbrs = set(graph.successors(node)) if hasattr(graph, "successors") else set()
                elif direction == "predecessors":
                    nbrs = set(graph.predecessors(node)) if hasattr(graph, "predecessors") else set()
                else:
                    nbrs = set(graph.neighbors(node))
                nxt |= nbrs
            nxt -= seen
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        return seen

    @staticmethod
    def _affected_edge_count(graph: nx.DiGraph, origin: str, k: int) -> int:
        nodes = {origin} | CorridorGraphBuilder._k_hop_neighbors(graph, origin, k, "successors")
        return sum(1 for u, v in graph.edges() if u in nodes and v in nodes)


class CorridorGraphEngine:
    """Singleton-style loader for inference / API / experiment enrichment."""

    _instance: Optional["CorridorGraphEngine"] = None

    def __init__(self, artifacts: Optional[CorridorGraphArtifacts] = None):
        self.builder = CorridorGraphBuilder()
        self.builder.artifacts = artifacts
        self._ready = artifacts is not None

    @classmethod
    def from_precomputed(cls, graph_path=None) -> "CorridorGraphEngine":
        if cls._instance is not None and cls._instance._ready:
            return cls._instance
        art = CorridorGraphBuilder.load(graph_path)
        cls._instance = cls(art)
        return cls._instance

    @property
    def ready(self) -> bool:
        return self._ready and self.builder.artifacts is not None

    def features_for_event(self, event: Dict[str, Any], k_hops: int = 2) -> Dict[str, float]:
        if not self.ready:
            return {k: 0.0 for k in GRAPH_FEATURE_NAMES}
        return self.builder.event_graph_features(
            lat=float(event.get("lat", event.get("latitude", 13.0))),
            lon=float(event.get("lon", event.get("longitude", 77.6))),
            corridor=str(event.get("corridor", "Non-corridor")),
            k_hops=k_hops,
        )

    def affected_subgraph(self, event: Dict[str, Any], k_hops: int = 2) -> Dict[str, Any]:
        if not self.ready:
            return {"available": False, "message": "Graph not precomputed"}
        return self.builder.get_affected_subgraph(
            lat=float(event.get("lat", event.get("latitude", 13.0))),
            lon=float(event.get("lon", event.get("longitude", 77.6))),
            corridor=str(event.get("corridor", "Non-corridor")),
            k_hops=k_hops,
        )