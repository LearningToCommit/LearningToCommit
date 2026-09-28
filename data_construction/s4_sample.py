#!/usr/bin/env python3
"""S4: category-stratified sampling of learning and test commits.

Usage:
    python data_construction/s4_sample.py --input .../s3_tagged.jsonl --n-learn 10 --n-test 10
Output: benchmark_train<L>_test<T>.jsonl next to the input.
"""

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

from utils import load_jsonl, save_jsonl


def proportional_sample(groups: dict[str, list[dict]], total: int, seed: int = 42) -> list[dict]:
    """Sample `total` items with per-group quotas proportional to group size (at least 1 each)."""
    random.seed(seed)
    grand_total = sum(len(v) for v in groups.values())
    if grand_total == 0 or total <= 0:
        return []
    if total < len(groups):  # fewer slots than groups: one item from each of the largest groups
        largest = sorted(groups, key=lambda c: len(groups[c]), reverse=True)[:total]
        return [random.choice(groups[c]) for c in largest]

    quotas = {c: max(1, math.floor(len(items) / grand_total * total)) for c, items in groups.items()}
    by_size = sorted(quotas, key=lambda c: len(groups[c]), reverse=True)
    allocated, idx = sum(quotas.values()), 0
    while allocated != total and idx <= total * 2:
        cat = by_size[idx % len(by_size)]
        if allocated < total and quotas[cat] < len(groups[cat]):
            quotas[cat] += 1
            allocated += 1
        elif allocated > total and quotas[cat] > 1:
            quotas[cat] -= 1
            allocated -= 1
        idx += 1

    sampled = []
    for cat, n in quotas.items():
        sampled.extend(random.sample(groups[cat], min(n, len(groups[cat]))))
    return sampled


def main():
    parser = argparse.ArgumentParser(description="S4: stratified sampling")
    parser.add_argument("--input", required=True, help="s3_tagged.jsonl")
    parser.add_argument("--n-learn", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    tagged = [c for c in load_jsonl(input_path)
              if c.get("suitable") and c.get("category") and c["category"] != "uncategorized"]
    pools = {"learning": defaultdict(list), "test": defaultdict(list)}
    for c in tagged:
        pools[c["split"]][c["category"]].append(c)

    learn = proportional_sample(pools["learning"], args.n_learn, seed=args.seed)
    test = proportional_sample(pools["test"], args.n_test, seed=args.seed + 1)
    output = Path(args.output or input_path.parent / f"benchmark_train{len(learn)}_test{len(test)}.jsonl")
    save_jsonl(learn + test, output)
    print(f"sampled learning={len(learn)} test={len(test)} -> {output}")
    print(f"Next: python data_construction/s5_build.py --input {output} --repo-dir <repo>")


if __name__ == "__main__":
    main()
