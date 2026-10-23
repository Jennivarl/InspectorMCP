"""Compare op_bench runs: average of repeated runs per build, with run-to-run spread.

Usage: python scripts/compare_ops.py --base stock_a stock_b --new cool_a cool_b
"""

import argparse
import json


def load(labels):
    return [json.load(open(f"results/ops_{k}.json"))["us"] for k in labels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--new", nargs="+", required=True)
    args = ap.parse_args()
    base, new = load(args.base), load(args.new)
    print(f"{'op':30s} {'base us':>9s} {'new us':>9s} {'speedup':>8s}  spread base/new")
    for op in new[0]:
        b = [r[op] for r in base]
        n = [r[op] for r in new]
        bm, nm = sum(b) / len(b), sum(n) / len(n)
        print(f"{op:30s} {bm:9.1f} {nm:9.1f} {bm / nm:7.2f}x  {(max(b) - min(b)) / bm:.0%}/{(max(n) - min(n)) / nm:.0%}")


if __name__ == "__main__":
    main()
