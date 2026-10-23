"""Inspection agent: a perception -> decision -> action loop over the MCP tools.

Every step is an MCP tool call, and what the vision tools return decides the
next call. Two brains share the same loop and tools:

- PolicyBrain: fixed, auditable rules (no LLM needed)
- GeminiBrain: same rules for clear cells; a Gemini model picks the tools for
               uncertain cells (see llm.py)
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from mcp import Client

# quality gate: below these the cell gets enhanced before scoring
MIN_CONTRAST = 6.0
MIN_SHARPNESS = 15.0


def _data(result) -> Any:
    if result.is_error:
        raise RuntimeError(result.content[0].text)
    if result.structured_content is not None:
        return result.structured_content
    texts = [b.text for b in result.content if b.type == "text"]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]
    return texts


@dataclass
class Trace:
    """Everything the agent did, in order - shown live in the UI and saved as evidence."""
    steps: list[dict] = field(default_factory=list)
    on_step: Callable[[dict], None] | None = None

    def add(self, kind: str, **info) -> None:
        step = {"n": len(self.steps) + 1, "t": round(time.time(), 3), "kind": kind, **info}
        self.steps.append(step)
        if self.on_step:
            self.on_step(step)


class Tools:
    """Thin MCP client wrapper that records every call in the trace."""

    def __init__(self, client: Client, trace: Trace):
        self.client, self.trace = client, trace
        self.last_images: list[bytes] = []  # PNGs returned by the most recent call

    async def call(self, name: str, why: str = "", **args) -> Any:
        t0 = time.perf_counter()
        result = await self.client.call_tool(name, args)
        self.last_images = [base64.b64decode(b.data) for b in result.content if b.type == "image"]
        out = _data(result)
        self.trace.add("tool", tool=name, args=args, why=why, result=_brief(out),
                       images=len(self.last_images), ms=round(1000 * (time.perf_counter() - t0), 1))
        return out


def _brief(out: Any) -> Any:
    """Keep traces small: drop long lists, keep the numbers that drove decisions."""
    if isinstance(out, dict):
        keep = {}
        for k, val in out.items():
            if isinstance(val, list) and len(val) > 3 and k != "heat":
                keep[k] = f"[{len(val)} items]"
            else:
                keep[k] = val
        return keep
    return out


def _severity(score: float, cell: dict) -> str:
    if score > 0.9 or cell.get("dark_fraction", 0) > 0.05:
        return "high"
    if score > 0.75:
        return "medium"
    return "low"


class PolicyBrain:
    name = "policy"

    async def inspect_cell(self, tools: Tools, image_id: str) -> dict:
        q = await tools.call("check_quality", "is the image good enough to judge?", image_id=image_id)
        if q["contrast"] < MIN_CONTRAST or q["sharpness"] < MIN_SHARPNESS:
            e = await tools.call("enhance_image", "low contrast/sharpness - boost before judging",
                                 image_id=image_id, method="clahe")
            image_id = e["image_id"]

        scan = await tools.call("scan_cell", "score the cell and find marks", image_id=image_id)
        score, verdict = scan["score"], scan["verdict"]

        if verdict == "uncertain":
            return await self.resolve_uncertain(tools, image_id, scan)
        return await self.act(tools, image_id, scan, score, verdict)

    async def resolve_uncertain(self, tools: Tools, image_id: str, scan: dict) -> dict:
        look = await tools.call("second_look", f"score {scan['score']} is unsure - check other views", image_id=image_id)
        score, verdict = look["mean"], look["verdict"]
        if verdict == "uncertain" or look["spread"] > 0.4:
            where = await tools.call("locate_defect", "still unsure - where is the model looking?", image_id=image_id)
            x, y, w, h = where["hot_box"]
            region = await tools.call("inspect_region", "zoom into the suspicious patch",
                                      image_id=image_id, x=x, y=y, w=w, h=h)
            detail = region[0] if isinstance(region, list) else region
            detail = json.loads(detail) if isinstance(detail, str) else detail
            evidence = (f"score {score:.2f} (spread {look['spread']:.2f}); hottest patch {where['hot_box']} "
                        f"drop {where['hot_drop']:.2f}; line {detail.get('longest_line')} px, "
                        f"contrast {detail.get('contrast')}")
            await tools.call("request_approval", "evidence is mixed - a human should decide",
                             image_id=image_id, proposed_action="fail" if score >= 0.5 else "pass",
                             reason=evidence)
            return {"image_id": image_id, "decision": "human", "score": score}
        return await self.act(tools, image_id, scan, score, verdict)

    async def act(self, tools: Tools, image_id: str, scan: dict, score: float, verdict: str) -> dict:
        if verdict == "pass":
            await tools.call("mark_pass", "score is low", image_id=image_id, reason=f"defect score {score:.2f}")
            return {"image_id": image_id, "decision": "pass", "score": score}

        where = await tools.call("locate_defect", "defect found - locate it for the repair ticket", image_id=image_id)
        kind = "crack" if scan["cracks"] else "dark area" if scan["dark_fraction"] > 0.01 else "defect"
        await tools.call("open_ticket", "clear defect - send for repair", image_id=image_id, defect=kind,
                         severity=_severity(score, scan),
                         evidence=f"score {score:.2f}; location {where['hot_box']}; cracks {len(scan['cracks'])}; "
                                  f"dark {scan['dark_fraction']:.3f}")
        return {"image_id": image_id, "decision": "fail", "score": score}


async def inspect_panel(server: Any, panel_id: str, rows: int, cols: int, brain=None,
                        on_step: Callable[[dict], None] | None = None) -> dict:
    """Split a panel and inspect every cell. `server` is an MCP URL or an in-process MCPServer."""
    brain = brain or PolicyBrain()
    trace = Trace(on_step=on_step)
    async with Client(server) as client:
        tools = Tools(client, trace)
        split = await tools.call("split_panel", "cut the panel into cells", image_id=panel_id, rows=rows, cols=cols)
        results = []
        for cell in split["cells"]:
            trace.add("cell", row=cell["row"], col=cell["col"], image_id=cell["image_id"])
            res = await brain.inspect_cell(tools, cell["image_id"])
            results.append({**res, "row": cell["row"], "col": cell["col"]})
            trace.add("decision", row=cell["row"], col=cell["col"], decision=res["decision"], score=round(res["score"], 3))
    return {"panel": panel_id, "brain": brain.name, "results": results, "trace": trace.steps}


async def inspect_one(server: Any, image_id: str, brain=None) -> dict:
    brain = brain or PolicyBrain()
    trace = Trace()
    async with Client(server) as client:
        res = await brain.inspect_cell(Tools(client, trace), image_id)
    return {**res, "trace": trace.steps}
