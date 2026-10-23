"""MCP server + agent loop, in-process (no network, no dataset, no LLM)."""

import asyncio

from mcp import Client

from inspector import agent
from inspector.server import mcp
from inspector.store import STORE
from tests.test_vision import synthetic_cell


def test_mcp_exposes_all_tools():
    async def go():
        async with Client(mcp) as c:
            return {t.name for t in (await c.list_tools()).tools}
    names = asyncio.run(go())
    for tool in ("split_panel", "scan_cell", "second_look", "locate_defect", "inspect_region",
                 "mark_pass", "open_ticket", "request_approval"):
        assert tool in names
    assert len(names) == 15


def test_agent_ends_every_cell_with_one_action():
    image_id = STORE.put(synthetic_cell(crack=True), "cell")
    out = asyncio.run(agent.inspect_one(mcp, image_id))
    tools = [s["tool"] for s in out["trace"] if s["kind"] == "tool"]
    actions = [t for t in tools if t in ("mark_pass", "open_ticket", "request_approval")]
    assert tools[0] == "check_quality" and "scan_cell" in tools
    assert len(actions) == 1
    assert out["decision"] in ("pass", "fail", "human")
    assert STORE.verdicts[out["image_id"]]["verdict"] == out["decision"]


def test_vision_result_changes_next_step():
    """An uncertain score must trigger extra looks before any action (perception -> decision)."""
    image_id = STORE.put(synthetic_cell(), "cell")
    trace = agent.Trace()

    async def go():
        async with Client(mcp) as c:
            tools = agent.Tools(c, trace)
            scan = await tools.call("scan_cell", image_id=image_id)
            scan["verdict"], scan["score"] = "uncertain", 0.5
            return await agent.PolicyBrain().resolve_uncertain(tools, image_id, scan)

    asyncio.run(go())
    called = [s["tool"] for s in trace.steps if s["kind"] == "tool"]
    assert called[1] == "second_look"
