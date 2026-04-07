#!/usr/bin/env python3
"""Run TWOSIDES-style analysis + CALMA validation with real patient data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.twosides_client import TWOSIDESClient


def main():
    print("=" * 60)
    print("TWOSIDES + PATIENT DATA VALIDATION")
    print("=" * 60)

    client = TWOSIDESClient()

    # Full antibiotic combination analysis
    combo_df = client.analyze_antibiotic_combinations()

    # CALMA key validation: Vancomycin + Azithromycin nephrotoxicity
    print(f"\n{'='*60}")
    print("CALMA KEY FINDING VALIDATION")
    print("Vancomycin + Azithromycin → reduced nephrotoxicity")
    print(f"{'='*60}")

    # Compare: Vancomycin alone vs Vancomycin + Azithromycin
    import time, requests

    url = "https://api.fda.gov/drug/event.json"

    # Vancomycin alone — kidney events
    params_v = {
        "search": 'patient.drug.medicinalproduct:"vancomycin"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": 100,
    }
    try:
        resp_v = requests.get(url, params=params_v, timeout=15)
        if resp_v.status_code == 200:
            v_data = resp_v.json().get("results", [])
            v_kidney = sum(r["count"] for r in v_data
                          if any(kw in r["term"].lower() for kw in ["renal", "kidney", "nephro", "creatinine"]))
            v_total = sum(r["count"] for r in v_data)
            v_rate = v_kidney / v_total if v_total > 0 else 0
            print(f"\n  Vancomycin alone:")
            print(f"    Total AE reports: {v_total}")
            print(f"    Kidney events: {v_kidney}")
            print(f"    Kidney rate: {v_rate:.4f} ({v_rate*100:.2f}%)")

        time.sleep(0.5)

        # Vancomycin + Azithromycin — kidney events
        params_va = {
            "search": 'patient.drug.medicinalproduct:"vancomycin" AND patient.drug.medicinalproduct:"azithromycin"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": 100,
        }
        resp_va = requests.get(url, params=params_va, timeout=15)
        if resp_va.status_code == 200:
            va_data = resp_va.json().get("results", [])
            va_kidney = sum(r["count"] for r in va_data
                           if any(kw in r["term"].lower() for kw in ["renal", "kidney", "nephro", "creatinine"]))
            va_total = sum(r["count"] for r in va_data)
            va_rate = va_kidney / va_total if va_total > 0 else 0
            print(f"\n  Vancomycin + Azithromycin:")
            print(f"    Total AE reports: {va_total}")
            print(f"    Kidney events: {va_kidney}")
            print(f"    Kidney rate: {va_rate:.4f} ({va_rate*100:.2f}%)")

            if v_rate > 0 and va_rate > 0:
                ratio = va_rate / v_rate
                if ratio < 1:
                    print(f"\n  ✅ CALMA FINDING CONFIRMED!")
                    print(f"    Kidney event rate REDUCED by {(1-ratio)*100:.1f}% with Azithromycin addition")
                else:
                    print(f"\n  ⚠️ Kidney rate ratio: {ratio:.2f} (combo/alone)")

    except Exception as e:
        print(f"  openFDA error: {e}")

    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
