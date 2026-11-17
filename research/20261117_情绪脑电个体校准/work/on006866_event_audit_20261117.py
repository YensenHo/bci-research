"""Aggregate marker audit; pass a local events.tsv path, never emit ratings."""
import csv
import json
import sys
from collections import Counter

def audit(path):
    trials = []
    prompt = None
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            value = int(row["value"])
            if 18 <= value <= 23:
                trials.append({"condition": value, "arousal": 0, "valence": 0, "order": []})
                prompt = None
            elif trials and value in (48, 49):
                prompt = "arousal" if value == 49 else "valence"
                trials[-1]["order"].append(prompt)
            elif trials and 1 <= value <= 9 and prompt:
                trials[-1][prompt] += 1
                prompt = None
    conditions = Counter(t["condition"] for t in trials)
    complete = sum(t["arousal"] == t["valence"] == 1 and t["order"] == ["arousal", "valence"] for t in trials)
    return {"trials": len(trials), "conditions": dict(sorted(conditions.items())), "ordered_one_each": complete}

if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1]), ensure_ascii=False))
