#!/usr/bin/env python3
"""Collect Pruritus data from public databases and load into Neo4j."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pruritus.data_collector import PruritusDataCollector
from src.pruritus.graph_loader import PruritusGraphLoader


def main():
    print("\n🧪 STEP 1: Collecting Pruritus data from public databases...\n")
    collector = PruritusDataCollector()
    results = collector.collect_all()

    print("\n📊 STEP 2: Loading data into Neo4j knowledge graph...\n")
    loader = PruritusGraphLoader()
    loader.load_all()

    print("\n✅ Pruritus data pipeline complete!")


if __name__ == "__main__":
    main()
