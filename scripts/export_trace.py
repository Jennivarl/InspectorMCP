"""Save full agent traces as evidence: one demo panel with the rule brain, and optionally
one uncertain cell with the Gemini brain.

Usage: python -m scripts.export_trace [--gemini-cell cell0143.png]
Writes results/traces/*.json
"""

import argparse
import asyncio
import json
from pathlib import Path

from mcp import Client

from inspector import agent
from inspector.server import mcp


async def main(gemini_cell: str | None):
    out_dir = Path("results/traces")
    out_dir.mkdir(parents=True, exist_ok=True)
    async with Client(mcp) as c:
        panel = agent._data(await c.call_tool("make_demo_panel", {"rows": 4, "cols": 6, "seed": 7}))
    run = await agent.inspect_panel(mcp, panel["image_id"], 4, 6)
    (out_dir / "panel_policy.json").write_text(json.dumps(run, indent=1))
    print("panel:", {d: sum(r["decision"] == d for r in run["results"]) for d in ("pass", "fail", "human")})

    if gemini_cell:
        from inspector.llm import GeminiBrain
        async with Client(mcp) as c:
            cell = agent._data(await c.call_tool("load_dataset_cell", {"name": gemini_cell}))
        res = await agent.inspect_one(mcp, cell["image_id"], GeminiBrain())
        (out_dir / f"cell_gemini_{Path(gemini_cell).stem}.json").write_text(json.dumps(res, indent=1))
        print("gemini:", res["decision"], [s.get("tool") or s.get("text", "")[:60] for s in res["trace"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini-cell")
    asyncio.run(main(ap.parse_args().gemini_cell))
