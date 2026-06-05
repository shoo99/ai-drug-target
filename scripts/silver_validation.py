#!/usr/bin/env python3
"""
Silver validation for Paper 3 v2 full-text extraction.

Reproduces the 'silver' check described in the manuscript: for each mapped
entry whose model direction is a structural/functional CHANGE (decrease/increase
-- NOT a correlation), derive an INDEPENDENT direction label from explicit
lexical cues in the sentence, blind to the model output. Keep only sentences
with an unambiguous single-polarity cue, then compare the model's mapped
direction to the lexical silver label and tabulate the decrease/increase
confusion matrix.

This is a fully automated check (lexical cues), NOT a human gold standard.
"""
import json, re, sys

ENTRIES = "/tmp/gold/mapped_entries.jsonl"

# Lexical cue lexicons (word-boundary matched, case-insensitive).
DEC = [r"reduc\w*", r"decreas\w*", r"smaller", r"lower\w*", r"diminish\w*",
       r"attenuat\w*", r"shrink\w*", r"atroph\w*", r"thinner", r"thinning",
       r"loss", r"lesser", r"deficit\w*", r"hypoactiv\w*", r"hypo-?activ\w*",
       r"weaker", r"declin\w*", r"less\b"]
INC = [r"increas\w*", r"greater", r"higher", r"larger", r"enhanc\w*",
       r"elevat\w*", r"thicker", r"bigger", r"hyperactiv\w*", r"hyper-?activ\w*",
       r"stronger", r"expand\w*", r"more\b", r"augment\w*"]

dec_re = re.compile("|".join(r"\b%s" % p for p in DEC), re.I)
inc_re = re.compile("|".join(r"\b%s" % p for p in INC), re.I)


def silver_label(sentence):
    has_dec = bool(dec_re.search(sentence))
    has_inc = bool(inc_re.search(sentence))
    if has_dec and not has_inc:
        return "decrease"
    if has_inc and not has_dec:
        return "increase"
    return None  # ambiguous (both) or no cue -> excluded


def main():
    rows = [json.loads(l) for l in open(ENTRIES) if l.strip()]
    # restrict to change-type model directions
    change = [r for r in rows if r.get("direction") in ("decrease", "increase")]
    entered, correct = 0, 0
    # confusion[model][silver]
    conf = {"decrease": {"decrease": 0, "increase": 0},
            "increase": {"decrease": 0, "increase": 0}}
    for r in change:
        sl = silver_label(r["sentence"])
        if sl is None:
            continue
        entered += 1
        md = r["direction"]
        conf[md][sl] += 1
        if md == sl:
            correct += 1
    n_change = len(change)
    print(f"total mapped entries:        {len(rows)}")
    print(f"change-type (dec/inc) model: {n_change}")
    print(f"entered silver (unambiguous cue): {entered}")
    print(f"correct:                     {correct}  ({100*correct/entered:.1f}%)")
    print()
    print("confusion matrix (rows=model direction, cols=silver/lexical label):")
    print(f"               silver=decrease  silver=increase")
    print(f"model=decrease     {conf['decrease']['decrease']:>6}          {conf['decrease']['increase']:>6}")
    print(f"model=increase     {conf['increase']['decrease']:>6}          {conf['increase']['increase']:>6}")
    print()
    # error breakdown
    false_inc = conf['increase']['decrease']  # model said increase, silver said decrease
    false_dec = conf['decrease']['increase']  # model said decrease, silver said increase
    print(f"model 'increase' but lexical 'decrease' (false increase): {false_inc}")
    print(f"model 'decrease' but lexical 'increase' (false decrease): {false_dec}")


if __name__ == "__main__":
    main()
