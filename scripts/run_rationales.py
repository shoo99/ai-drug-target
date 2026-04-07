#!/usr/bin/env python3
"""Generate LLM rationales for top targets + combinations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.target_rationale import TargetRationaleGenerator


def main():
    gen = TargetRationaleGenerator()

    # Target rationales
    gen.generate_all_rationales()

    # Combination rationales
    gen.generate_combination_rationales()

    print("\n✅ All rationales generated!")


if __name__ == "__main__":
    main()
