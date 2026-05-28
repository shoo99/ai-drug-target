#!/usr/bin/env python3
"""
Paper 3 — Step 4: LLM-based brain region × metric mapping

Runs on KBRI com02 (Qwen3.6-35B-A3B MoE, port 11202).
Input: llm_prompts.json from Step 1-3 (regex extraction)
Output: mapped effect sizes with brain region + metric + direction

Each prompt is ~200 tokens → 35B on CPU handles in 1-5 seconds.
Total: N prompts × 3 seconds / 8 workers ≈ minutes to hours depending on N.
"""
import json
import time
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# KBRI server configuration
SERVERS = [
    "http://172.20.0.12:11202",  # com02: 35B-A3B
    # Add more servers if available
]

def query_llm(server_url, prompt, timeout=30):
    """Send prompt to llama.cpp server."""
    payload = {
        "prompt": prompt,
        "temperature": 0,
        "n_predict": 512,
        "stop": ["\n\n", "```"],
    }
    try:
        resp = requests.post(
            f"{server_url}/completion",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("content", "")
    except Exception as e:
        return f"ERROR: {e}"
    return None


def process_prompt(item, server_url):
    """Process a single prompt and parse result."""
    prompt = item["llm_prompt"]

    # Wrap in instruction format
    full_prompt = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    response = query_llm(server_url, full_prompt)

    result = {
        "pmid": item["pmid"],
        "sentence": item["sentence"],
        "regex_effect_sizes": item["extracted_effect_sizes"],
        "regex_ci": item["extracted_ci"],
        "regex_p_values": item["extracted_p_values"],
        "regex_sample_sizes": item["extracted_sample_sizes"],
        "regex_brain_regions": item["regex_brain_regions"],
        "llm_raw": response,
        "llm_parsed": None,
    }

    # Try to parse LLM response as JSON
    if response:
        try:
            # Find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                result["llm_parsed"] = json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/stress/fulltext_extract/llm_prompts.json")
    parser.add_argument("--output", default="data/stress/fulltext_extract/llm_mapped.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Limit prompts (0=all)")
    parser.add_argument("--server", default=SERVERS[0])
    args = parser.parse_args()

    with open(args.input) as f:
        prompts = json.load(f)

    if args.limit:
        prompts = prompts[:args.limit]

    print(f"Processing {len(prompts)} prompts on {args.server}")
    print(f"Workers: {args.workers}")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_prompt, item, args.server): i
            for i, item in enumerate(prompts)
        }

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results.append(result)

            if (len(results)) % 100 == 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed
                eta = (len(prompts) - len(results)) / rate if rate > 0 else 0
                parsed = sum(1 for r in results if r["llm_parsed"])
                print(f"  {len(results)}/{len(prompts)} done "
                      f"({parsed} parsed, {rate:.1f}/s, ETA {eta/60:.0f}min)")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    parsed = sum(1 for r in results if r["llm_parsed"])
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Parsed: {parsed}/{len(results)} ({parsed/len(results)*100:.1f}%)")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
