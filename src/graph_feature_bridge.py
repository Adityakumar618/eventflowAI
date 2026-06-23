"""
Graph Feature Bridge — Phase 2 integration layer (additive only)
================================================================
Attaches precomputed NetworkX graph features to event rows or dicts
WITHOUT modifying inference.py, V12, or advanced_event_fe.py.

Usage in experiments / future V12 retrain:
    from src.graph_feature_bridge import GraphFeatureEnricher
    enricher = GraphFeatureEnricher.from_precomputed()
    df = enricher.transform(df)
    event_feats = enricher.features_for_dict(event_dict)

Safe to import even when graph is not precomputed (returns zeros).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.corridor_graph import (
    GRAPH_FEATURE_NAMES,
    CorridorGraphEngine,
    DEFAULT_GRAPH_PATH,
)

logger = logging.getLogger(__name__)


class GraphFeatureEnricher:
    """
    Optional enrichment step for any feature pipeline.
    Loads precomputed graph once; transform is row-wise lookup only.
    """

    def __init__(self, engine: Optional[CorridorGraphEngine] = None):
        self.engine = engine
        self.feature_names: List[str] = list(GRAPH_FEATURE_NAMES)
        self.ready = engine is not None and engine.ready

    @classmethod
    def from_precomputed(cls, graph_path=None) -> "GraphFeatureEnricher":
        path = graph_path or DEFAULT_GRAPH_PATH
        try:
            engine = CorridorGraphEngine.from_precomputed(path)
            return cls(engine)
        except FileNotFoundError:
            logger.warning("Corridor graph not found at %s — graph features will be zero.", path)
            return cls(None)

    def features_for_dict(self, event: Dict[str, Any], k_hops: int = 2) -> Dict[str, float]:
        if not self.ready or self.engine is None:
            return {k: 0.0 for k in self.feature_names}
        return self.engine.features_for_event(event, k_hops=k_hops)

    def transform(self, df: pd.DataFrame, k_hops: int = 2) -> pd.DataFrame:
        """Append graph feature columns to a DataFrame (vectorized via apply)."""
        out = df.copy()
        if not self.ready or self.engine is None:
            for col in self.feature_names:
                out[col] = 0.0
            return out

        def _row_feats(row) -> pd.Series:
            event = {
                "lat": row.get("latitude", row.get("lat", 13.0)),
                "lon": row.get("longitude", row.get("lon", 77.6)),
                "corridor": row.get("corridor", "Non-corridor"),
            }
            return pd.Series(self.engine.features_for_event(event, k_hops=k_hops))

        feats = out.apply(_row_feats, axis=1)
        return pd.concat([out, feats], axis=1)

    def affected_subgraph(self, event: Dict[str, Any], k_hops: int = 2) -> Dict[str, Any]:
        if not self.ready or self.engine is None:
            return {"available": False, "message": "Graph not precomputed"}
        return self.engine.affected_subgraph(event, k_hops=k_hops)


def enrich_event_dict(event: Dict[str, Any], k_hops: int = 2) -> Dict[str, Any]:
    """
    Convenience: merge graph features into an event dict in-place copy.
    Used by API layer and Phase 3 recommendation queries.
    """
    enricher = GraphFeatureEnricher.from_precomputed()
    feats = enricher.features_for_dict(event, k_hops=k_hops)
    return {**event, **feats}


def list_graph_feature_names() -> List[str]:
    return list(GRAPH_FEATURE_NAMES)