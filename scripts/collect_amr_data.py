#!/usr/bin/env python3
"""Collect AMR data from public databases and load into Neo4j."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.amr.data_collector import AMRDataCollector
from src.amr.graph_loader import AMRGraphLoader


def main():
    # Step 1: Collect data
    print("\n🧬 STEP 1: Collecting AMR data from public databases...\n")
    collector = AMRDataCollector()
    results = collector.collect_all()

    # Step 2: Load into knowledge graph
    print("\n📊 STEP 2: Loading data into Neo4j knowledge graph...\n")
    loader = AMRGraphLoader()
    loader.load_all()

    print("\n✅ AMR data pipeline complete!")


if __name__ == "__main__":
    main()
