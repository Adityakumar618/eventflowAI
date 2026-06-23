#!/usr/bin/env python
"""
Standalone precompute for Phase 2 corridor graph features.
Does NOT modify src/precompute.py or any production ML paths.

Usage (from project root):
    python scripts/precompute_corridor_graph.py
    python scripts/precompute_corridor_graph.py --data data/processed/survival_ready.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.corridor_graph import CorridorGraphBuilder, DEFAULT_GRAPH_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Precompute corridor/junction NetworkX graph")
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "processed" / "survival_ready.parquet"),
        help="Path to processed events parquet",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    logger.info("Loading %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Events: %d | Corridors: %d", len(df), df["corridor"].nunique())

    builder = CorridorGraphBuilder()
    art = builder.fit(df)
    builder.save()

    # Smoke: sample event features + subgraph
    sample = df.dropna(subset=["latitude", "longitude"]).iloc[0]
    feats = builder.event_graph_features(
        float(sample["latitude"]),
        float(sample["longitude"]),
        str(sample.get("corridor", "Non-corridor")),
    )
    subgraph = builder.get_affected_subgraph(
        float(sample["latitude"]),
        float(sample["longitude"]),
        str(sample.get("corridor", "Non-corridor")),
        k_hops=2,
    )

    logger.info("Sample graph features: %s", {k: round(v, 4) for k, v in feats.items()})
    logger.info(
        "Sample affected subgraph: origin=%s nodes=%d edges=%d",
        subgraph.get("origin_node"),
        subgraph.get("n_nodes"),
        subgraph.get("n_edges"),
    )
    logger.info("Artifacts written under %s", DEFAULT_GRAPH_PATH.parent)
    logger.info("Done.")


if __name__ == "__main__":
    main()