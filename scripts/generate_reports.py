#!/usr/bin/env python3
"""Generate all PDF reports for both tracks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.report_generator import ReportGenerator


def main():
    gen = ReportGenerator()
    reports = gen.generate_all_reports()
    print(f"\n✅ Generated {len(reports)} reports:")
    for r in reports:
        print(f"  📄 {r}")


if __name__ == "__main__":
    main()
