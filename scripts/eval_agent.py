"""Run the agent on every held-out test cell (through MCP, in-process) and score it.

Compares: CellNet single pass vs agent (which may escalate unsure cells to a human).
Usage: python scripts/eval_agent.py --data ~/elpv-dataset/src/elpv_dataset/data [--brain policy]
Writes results/agent_<brain>.json
"""

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from inspector import agent
from inspector.server import mcp
from inspector.store import STORE


async def run(data: Path, brain, only=None):
    single = json.loads(Path("results/single_pass.json").read_text())
    test_ids = set(single["test_ids"])
    labels = {}
    for line in (data / "labels.csv").read_text().splitlines():
        path, prob, _ = line.split()
        if Path(path).name in test_ids:
            labels[Path(path).name] = int(float(prob) >= 0.5)

    rows, tool_counts, t0 = [], Counter(), time.perf_counter()
    names = sorted(labels.items())
    if only:
        names = [(n, y) for n, y in names if n in only]
    for name, y in names:
        loaded = agent._data(await _client.call_tool("load_dataset_cell", {"name": name}))
        out = await agent.inspect_one(mcp, loaded["image_id"], brain)
        tool_counts.update(s["tool"] for s in out["trace"] if s["kind"] == "tool")
        rows.append({"name": name, "y": y, "decision": out["decision"], "score": round(out["score"], 3),
                     "steps": len(out["trace"]), "by": out.get("by", "rules"),
                     "uncertain": any(s.get("tool") in ("second_look", "view_cell") for s in out["trace"])})
        if verbose:
            print(name, "truth", y, "->", out["decision"], out.get("by", "rules"), flush=True)
            for s in out["trace"]:
                if s["kind"] == "llm":
                    print("   LLM:", s["text"][:200], flush=True)
                elif s["kind"] == "tool":
                    print("   tool:", s["tool"], s["args"], flush=True)
    secs = time.perf_counter() - t0
    return rows, tool_counts, secs


def summarize(rows):
    auto = [r for r in rows if r["decision"] != "human"]
    human = [r for r in rows if r["decision"] == "human"]
    correct = sum((r["decision"] == "fail") == bool(r["y"]) for r in auto)
    caught = sum(r["decision"] == "fail" for r in auto if r["y"])
    false_alarm = sum(r["decision"] == "fail" for r in auto if not r["y"])
    missed = sum(r["decision"] == "pass" for r in auto if r["y"])
    return {
        "cells": len(rows),
        "auto_decided": len(auto),
        "sent_to_human": len(human),
        "human_rate": round(len(human) / len(rows), 4),
        "auto_accuracy": round(correct / max(len(auto), 1), 4),
        "auto_defects_caught": caught,
        "auto_false_alarms": false_alarm,
        "auto_missed_defects": missed,
        "defects_among_human_cases": sum(r["y"] for r in human),
        "avg_steps_per_cell": round(float(np.mean([r["steps"] for r in rows])), 2),
    }


_client = None
verbose = False


async def main_async(args):
    global _client, verbose
    from mcp import Client
    verbose = args.verbose
    only = None
    if args.uncertain_only:  # just the cells the rule run found unsure - where the LLM is used
        prev = json.loads(Path("results/agent_policy.json").read_text())["per_cell"]
        only = {r["name"] for r in prev if r["uncertain"]}
    if args.limit:
        only = set(sorted(only)[: args.limit]) if only else None
    async with Client(mcp) as c:
        _client = c
        if args.brain == "policy":
            brain = agent.PolicyBrain()
        else:
            from inspector.llm import GeminiBrain
            brain = GeminiBrain()
        rows, tool_counts, secs = await run(args.data.expanduser(), brain, only)
    result = {"brain": args.brain, "seconds": round(secs, 1), "summary": summarize(rows),
              "llm_requests": getattr(brain, "requests", 0), "tool_calls": dict(tool_counts), "per_cell": rows}
    suffix = "_uncertain" if args.uncertain_only else ""
    Path(f"results/agent_{args.brain}{suffix}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "per_cell"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--brain", default="policy", choices=["policy", "gemini"])
    ap.add_argument("--uncertain-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    asyncio.run(main_async(ap.parse_args()))
