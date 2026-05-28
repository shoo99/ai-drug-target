#!/usr/bin/env python3
"""
Paper 3 — Hybrid Full-text Quantitative Extraction

Step 1: Parse PMC XML → extract Results/Methods sections
Step 2: Regex extract all numeric candidates (d, g, CI, p, N)
Step 3: Extract sentences containing numeric candidates + context
Step 4: (External) Send to 35B LLM for brain region × metric mapping

This script handles Steps 1-3 (CPU only, no LLM needed).
Output: JSON with sentences + numeric candidates ready for LLM mapping.
"""
import sys
import os
import re
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data" / "stress"
OUT_DIR = DATA_DIR / "fulltext_extract"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# STEP 1: PMC XML PARSING
# ============================================================

def parse_pmc_xml(xml_text):
    """Extract sections from PMC JATS XML."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    sections = {}

    # Find body sections
    for sec in root.findall(".//{http://jats.nlm.nih.gov}sec") + root.findall(".//sec"):
        title_el = sec.find("{http://jats.nlm.nih.gov}title") or sec.find("title")
        if title_el is None:
            continue
        title = (title_el.text or "").strip().lower()

        # Collect all text in this section
        text_parts = []
        for p in sec.findall(".//{http://jats.nlm.nih.gov}p") + sec.findall(".//p"):
            text = ET.tostring(p, encoding="unicode", method="text").strip()
            if text:
                text_parts.append(text)

        section_text = " ".join(text_parts)

        if any(k in title for k in ["result", "finding"]):
            sections["results"] = sections.get("results", "") + " " + section_text
        elif any(k in title for k in ["method", "material", "participant", "procedure"]):
            sections["methods"] = sections.get("methods", "") + " " + section_text
        elif any(k in title for k in ["discussion", "interpretation"]):
            sections["discussion"] = sections.get("discussion", "") + " " + section_text
        elif any(k in title for k in ["abstract", "summary"]):
            sections["abstract"] = sections.get("abstract", "") + " " + section_text

    # Fallback: get all body text if no sections found
    if not sections:
        body = root.find(".//{http://jats.nlm.nih.gov}body") or root.find(".//body")
        if body is not None:
            all_text = ET.tostring(body, encoding="unicode", method="text").strip()
            sections["full_body"] = all_text

    # Get PMID
    pmid = None
    for aid in root.findall(".//{http://jats.nlm.nih.gov}article-id") + root.findall(".//article-id"):
        if aid.get("pub-id-type") == "pmid":
            pmid = aid.text
    if not pmid:
        for aid in root.findall(".//{http://jats.nlm.nih.gov}article-id") + root.findall(".//article-id"):
            if aid.get("pub-id-type") == "pmc":
                pmid = "PMC" + aid.text

    return {"pmid": pmid, "sections": sections}


# ============================================================
# STEP 2: REGEX NUMERIC EXTRACTION
# ============================================================

# Effect size patterns
EFFECT_SIZE_PATTERNS = [
    # Cohen's d
    (r"[Cc]ohen'?s?\s*d\s*=\s*([-−–]?\d+\.?\d*)", "cohen_d"),
    (r"\bd\s*=\s*([-−–]?\d+\.?\d*)", "d"),
    # Hedges' g
    (r"[Hh]edges'?\s*g\s*=\s*([-−–]?\d+\.?\d*)", "hedges_g"),
    (r"\bg\s*=\s*([-−–]?\d+\.?\d*)", "g"),
    # Eta squared
    (r"η[²2p]\s*=\s*([-−–]?\d+\.?\d*)", "eta_sq"),
    (r"partial\s*η[²2]\s*=\s*([-−–]?\d+\.?\d*)", "partial_eta_sq"),
    # Beta
    (r"[βBb]eta\s*=\s*([-−–]?\d+\.?\d*)", "beta"),
    # F statistic
    (r"[Ff]\s*\(\s*\d+\s*,\s*\d+\s*\)\s*=\s*([-−–]?\d+\.?\d*)", "F"),
    # t statistic
    (r"[Tt]\s*\(\s*\d+\s*\)\s*=\s*([-−–]?\d+\.?\d*)", "t"),
    (r"\bt\s*=\s*([-−–]?\d+\.?\d*)", "t"),
    # r (correlation)
    (r"\br\s*=\s*([-−–]?\d+\.?\d*)", "r"),
]

CI_PATTERN = re.compile(
    r"95\s*%?\s*(?:CI|confidence\s+interval)\s*[=:,]?\s*"
    r"[\[\(]?\s*([-−–]?\d+\.?\d*)\s*[,;to–−-]+\s*([-−–]?\d+\.?\d*)\s*[\]\)]?",
    re.IGNORECASE
)

P_PATTERNS = [
    re.compile(r"[pP]\s*([=<>≤≥])\s*(\d+\.?\d*(?:[eE][-−]?\d+)?)"),
    re.compile(r"[pP]\s*-?\s*value\s*[=:<>]\s*(\d+\.?\d*(?:[eE][-−]?\d+)?)"),
]

N_PATTERN = re.compile(r"[nN]\s*=\s*(\d+)")

# Brain region patterns (for context matching)
BRAIN_REGIONS = [
    "hippocampus", "hippocampal", "amygdala", "amygdalar",
    "prefrontal cortex", "prefrontal", "anterior cingulate",
    "insula", "insular", "thalamus", "thalamic",
    "orbitofrontal", "dorsolateral", "ventromedial",
    "medial prefrontal", "dentate gyrus", "CA1", "CA3",
    "nucleus accumbens", "striatum", "putamen", "caudate",
    "cerebellum", "cerebellar", "posterior cingulate",
    "precuneus", "temporal", "parietal", "occipital",
    "basolateral amygdala", "central amygdala",
    "subgenual", "entorhinal", "parahippocampal",
    "white matter", "gray matter", "grey matter",
    "cortical thickness", "brain volume",
]
REGION_PATTERN = re.compile(
    "|".join(re.escape(r) for r in BRAIN_REGIONS),
    re.IGNORECASE
)


def extract_numerics_from_text(text):
    """Extract all numeric candidates from text."""
    if not text:
        return []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    results = []
    for sent in sentences:
        entry = {
            "sentence": sent.strip()[:500],  # cap at 500 chars
            "effect_sizes": [],
            "ci": None,
            "p_values": [],
            "sample_sizes": [],
            "brain_regions": [],
        }

        has_numeric = False

        # Effect sizes
        for pattern_str, es_type in EFFECT_SIZE_PATTERNS:
            for match in re.finditer(pattern_str, sent):
                try:
                    val = float(match.group(1).replace("−", "-").replace("–", "-"))
                    if abs(val) < 20:  # sanity check
                        entry["effect_sizes"].append({"type": es_type, "value": val})
                        has_numeric = True
                except ValueError:
                    pass

        # CI
        ci_match = CI_PATTERN.search(sent)
        if ci_match:
            try:
                lo = float(ci_match.group(1).replace("−", "-").replace("–", "-"))
                hi = float(ci_match.group(2).replace("−", "-").replace("–", "-"))
                entry["ci"] = [lo, hi]
                has_numeric = True
            except ValueError:
                pass

        # p-values
        for p_pat in P_PATTERNS:
            for match in p_pat.finditer(sent):
                try:
                    if match.lastindex == 2:
                        val = float(match.group(2))
                        op = match.group(1)
                        entry["p_values"].append({"op": op, "value": val})
                    else:
                        val = float(match.group(1))
                        entry["p_values"].append({"op": "=", "value": val})
                    has_numeric = True
                except ValueError:
                    pass

        # Sample sizes
        for match in N_PATTERN.finditer(sent):
            try:
                n = int(match.group(1))
                if 5 <= n <= 100000:  # sanity
                    entry["sample_sizes"].append(n)
                    has_numeric = True
            except ValueError:
                pass

        # Brain regions in sentence
        for match in REGION_PATTERN.finditer(sent):
            entry["brain_regions"].append(match.group(0).lower())

        # Only keep sentences with at least one numeric + one brain region
        # OR sentences with effect sizes (even without explicit region)
        if has_numeric and (entry["brain_regions"] or entry["effect_sizes"]):
            # Deduplicate regions
            entry["brain_regions"] = list(set(entry["brain_regions"]))
            results.append(entry)

    return results


# ============================================================
# STEP 3: PROCESS ALL PMC XMLs
# ============================================================

def process_tarball(tarball_path):
    """Process all XML files from the PMC tarball."""
    print(f"Processing tarball: {tarball_path}")
    all_results = []
    n_processed = 0
    n_with_numerics = 0

    with tarfile.open(tarball_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".xml")]
        print(f"  XML files in tarball: {len(members)}")

        for i, member in enumerate(members):
            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                xml_text = f.read().decode("utf-8", errors="ignore")
                parsed = parse_pmc_xml(xml_text)

                if parsed is None:
                    continue

                n_processed += 1

                # Prioritize Results section, fall back to full body
                text = parsed["sections"].get("results", "")
                if not text:
                    text = parsed["sections"].get("full_body", "")

                numerics = extract_numerics_from_text(text)

                if numerics:
                    n_with_numerics += 1
                    all_results.append({
                        "pmid": parsed["pmid"],
                        "filename": member.name,
                        "n_sentences": len(numerics),
                        "sentences": numerics,
                        "sections_available": list(parsed["sections"].keys()),
                    })

                if (i + 1) % 500 == 0:
                    print(f"    {i+1}/{len(members)}: processed={n_processed}, "
                          f"with_numerics={n_with_numerics}")

            except Exception as e:
                continue

    print(f"\n  Total processed: {n_processed}")
    print(f"  With numeric data: {n_with_numerics} ({n_with_numerics/n_processed*100:.1f}%)")
    return all_results


def generate_llm_prompts(results):
    """Generate minimal prompts for 35B LLM mapping (Step 4)."""
    prompts = []
    for article in results:
        for sent_data in article["sentences"]:
            if not sent_data["effect_sizes"]:
                continue

            prompt = {
                "pmid": article["pmid"],
                "sentence": sent_data["sentence"],
                "extracted_effect_sizes": sent_data["effect_sizes"],
                "extracted_ci": sent_data["ci"],
                "extracted_p_values": sent_data["p_values"],
                "extracted_sample_sizes": sent_data["sample_sizes"],
                "regex_brain_regions": sent_data["brain_regions"],
                "llm_prompt": (
                    f"Given this sentence from a neuroscience paper:\n"
                    f"\"{sent_data['sentence']}\"\n\n"
                    f"The following numeric values were found: "
                    f"{json.dumps(sent_data['effect_sizes'])}\n\n"
                    f"Map each effect size to:\n"
                    f"1. brain_region (specific name)\n"
                    f"2. metric (volume/connectivity/activation/neurogenesis/other)\n"
                    f"3. direction (decrease/increase)\n"
                    f"4. comparison (e.g., 'stress vs control', 'PTSD vs healthy')\n"
                    f"Return JSON array."
                ),
            }
            prompts.append(prompt)

    return prompts


def main():
    print("=" * 70)
    print("  HYBRID FULL-TEXT EXTRACTION (Steps 1-3: CPU only)")
    print("=" * 70)

    tarball = DATA_DIR / "pmc_fulltext.tar.gz"
    if not tarball.exists():
        print(f"  ERROR: {tarball} not found")
        print(f"  Download from HuggingFace first")
        return

    # Step 1-2: Parse XML + Regex extract
    results = process_tarball(tarball)

    # Save raw extraction
    raw_path = OUT_DIR / "regex_extractions.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved regex extractions: {raw_path} ({raw_path.stat().st_size//1024}KB)")

    # Step 3: Generate LLM prompts
    prompts = generate_llm_prompts(results)
    prompt_path = OUT_DIR / "llm_prompts.json"
    with open(prompt_path, "w") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"  Saved LLM prompts: {prompt_path} ({len(prompts)} prompts)")

    # Stats
    print(f"\n{'='*70}")
    print(f"  EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Articles with numeric data: {len(results)}")

    total_es = sum(len(s["effect_sizes"]) for r in results for s in r["sentences"])
    total_ci = sum(1 for r in results for s in r["sentences"] if s["ci"])
    total_p = sum(len(s["p_values"]) for r in results for s in r["sentences"])
    total_n = sum(len(s["sample_sizes"]) for r in results for s in r["sentences"])
    total_regions = sum(len(s["brain_regions"]) for r in results for s in r["sentences"])

    print(f"  Effect sizes found: {total_es}")
    print(f"  CIs found: {total_ci}")
    print(f"  p-values found: {total_p}")
    print(f"  Sample sizes found: {total_n}")
    print(f"  Brain region mentions: {total_regions}")
    print(f"  LLM prompts generated: {len(prompts)} (for 35B mapping)")

    # Effect size type distribution
    es_types = defaultdict(int)
    for r in results:
        for s in r["sentences"]:
            for es in s["effect_sizes"]:
                es_types[es["type"]] += 1
    print(f"\n  Effect size types:")
    for t, cnt in sorted(es_types.items(), key=lambda x: -x[1]):
        print(f"    {t:20s}: {cnt:>5d}")

    # Top brain regions in numeric sentences
    region_counts = defaultdict(int)
    for r in results:
        for s in r["sentences"]:
            for reg in s["brain_regions"]:
                region_counts[reg] += 1
    print(f"\n  Top brain regions (in sentences with numerics):")
    for reg, cnt in sorted(region_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {reg:25s}: {cnt:>5d}")

    print(f"\n{'='*70}")
    print(f"  NEXT STEP: Run llm_prompts.json through 35B on KBRI com02")
    print(f"  Command: python fulltext_llm_map.py --input llm_prompts.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
