"""Smoke test: talk to the running MCP server over HTTP like an agent would.

Usage: python scripts/mcp_smoke.py [http://127.0.0.1:8001/mcp]
"""

import asyncio
import json
import sys

from mcp import Client


def data(result):
    if result.structured_content is not None:
        return result.structured_content
    if result.is_error:
        raise RuntimeError(result.content[0].text)
    if len(result.content) == 1 and result.content[0].type == "text":
        return json.loads(result.content[0].text)
    return [getattr(b, "text", None) or f"<{b.type} {getattr(b, 'mime_type', '')}>" for b in result.content]


async def main(url: str):
    async with Client(url) as c:
        tools = await c.list_tools()
        print("TOOLS:", [t.name for t in tools.tools])
        print("INFO:", data(await c.call_tool("opencv_info", {})))
        p = data(await c.call_tool("make_demo_panel", {"rows": 4, "cols": 6, "seed": 1}))
        print("PANEL:", p)
        s = data(await c.call_tool("split_panel", {"image_id": p["image_id"], "rows": 4, "cols": 6}))
        print("SPLIT ms:", s["ms"], "cells:", len(s["cells"]))
        for cell in s["cells"][:6]:
            r = data(await c.call_tool("scan_cell", {"image_id": cell["image_id"]}))
            print("SCAN", cell["row"], cell["col"], r["verdict"], r["score"], "cracks", len(r["cracks"]), "ms", r["ms"])
            if r["cracks"]:
                x, y, w, h = r["cracks"][0]["box"]
                region = await c.call_tool("inspect_region", {"image_id": cell["image_id"], "x": x, "y": y, "w": w, "h": h})
                print("  REGION:", data(region))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001/mcp"))
