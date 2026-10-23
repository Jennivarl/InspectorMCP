"""GeminiBrain: a Gemini model drives the MCP tools for cells the vision model is unsure about.

Clear cells keep the fast rule path (no LLM call). For an uncertain cell the model
sees the scan numbers and the cell picture, then chooses its own next tool calls
(second_look, locate_defect, inspect_region, ...) and must finish with exactly one
action: mark_pass, open_ticket or request_approval.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from google import genai
from google.genai import errors, types

from .agent import PolicyBrain, Tools

LOOK_TOOLS = ["second_look", "locate_defect", "inspect_region", "view_cell", "enhance_image", "scan_cell"]
ACTION_TOOLS = {"mark_pass": "pass", "open_ticket": "fail", "request_approval": "human"}
MAX_TURNS = 8

SYSTEM = """You inspect electroluminescence (EL) images of solar cells for a repair crew.
In EL images healthy silicon glows evenly grey. Two to three straight dark horizontal bars (busbars)
and dark cut corners are NORMAL. Grainy, cloudy texture is NORMAL for polycrystalline cells.
DEFECTS are: thin dark cracks (straight or branching, any direction), and dark inactive or
disconnected areas.

You get one cell the vision model is unsure about. Gather evidence with the tools - e.g.
second_look (score from several views), locate_defect (where the model sees the defect),
inspect_region (zoomed picture of a box), view_cell (marked-up picture). Use what each result
shows to choose the next call. Keep it short: usually 2-4 looks are enough.

Then finish with exactly ONE action:
- mark_pass        when the evidence says healthy
- open_ticket      when you see a real defect (defect: crack | dark area | other; severity: low|medium|high)
- request_approval when the evidence stays mixed - a human decides. Do not guess.
Before every tool call, say in one short sentence why you are calling it."""


def _load_env() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, val = line.split("=", 1)
                os.environ.setdefault(k.strip(), val.strip())


class GeminiBrain(PolicyBrain):
    name = "gemini"

    def __init__(self, model: str = "gemini-3.5-flash", fallback: str = "gemini-3.5-flash-lite"):
        _load_env()
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.models = [model, fallback]
        self.requests = 0
        self._decls = None

    async def _declarations(self, tools: Tools) -> list[types.FunctionDeclaration]:
        if self._decls is None:
            listed = await tools.client.list_tools()
            wanted = set(LOOK_TOOLS) | set(ACTION_TOOLS)
            self._decls = [
                types.FunctionDeclaration(name=t.name, description=t.description or "",
                                          parameters_json_schema=t.input_schema)
                for t in listed.tools if t.name in wanted
            ]
        return self._decls

    async def _generate(self, contents, config):
        """Call Gemini; on rate limits or overload, back off and fall back to the lighter model."""
        last = None
        for attempt in range(6):
            model = self.models[min(attempt // 2, 1)]
            try:
                self.requests += 1
                return await self.client.aio.models.generate_content(model=model, contents=contents, config=config), model
            except errors.APIError as e:
                last = e
                if e.code not in (429, 500, 503):
                    raise
                await asyncio.sleep(2 ** attempt)
        raise last

    async def resolve_uncertain(self, tools: Tools, image_id: str, scan: dict) -> dict:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(function_declarations=await self._declarations(tools))],
            temperature=0.2,
            max_output_tokens=600,
        )
        await tools.call("view_cell", "show the model the cell", image_id=image_id)
        picture = tools.last_images[0]
        brief = {k: scan[k] for k in ("score", "rule_score", "cracks", "spots", "dark_fraction", "grain")}
        contents = [types.Content(role="user", parts=[
            types.Part(text=f"Cell image_id={image_id}. The vision model is unsure. scan_cell result:\n"
                            f"{json.dumps(brief)}\nMarked-up picture attached (red=crack, yellow=spot, blue=dark)."),
            types.Part.from_bytes(data=picture, mime_type="image/png"),
        ])]

        score = scan["score"]
        for _ in range(MAX_TURNS):
            resp, model = await self._generate(contents, config)
            msg = resp.candidates[0].content
            contents.append(msg)
            said = " ".join(p.text for p in msg.parts or [] if p.text).strip()
            if said:
                tools.trace.add("llm", model=model, text=said[:500])
            calls = resp.function_calls or []
            if not calls:
                contents.append(types.Content(role="user", parts=[types.Part(
                    text="Finish now with exactly one action tool: mark_pass, open_ticket or request_approval.")]))
                continue

            replies, pictures = [], []
            for fc in calls:
                args = dict(fc.args or {})
                if fc.name not in set(LOOK_TOOLS) | set(ACTION_TOOLS):
                    replies.append(types.Part.from_function_response(name=fc.name, response={"error": "unknown tool"}))
                    continue
                args.setdefault("image_id", image_id)
                try:
                    out = await tools.call(fc.name, said or f"{model} chose this", **args)
                except Exception as e:  # bad arguments etc. - let the model correct itself
                    replies.append(types.Part.from_function_response(name=fc.name, response={"error": str(e)[:300]}))
                    continue
                if fc.name in ACTION_TOOLS:
                    return {"image_id": image_id, "decision": ACTION_TOOLS[fc.name], "score": score, "by": model}
                if fc.name == "second_look":
                    score = out["mean"]
                if isinstance(out, list) and out and isinstance(out[0], str):  # inspect_region: json text + picture
                    try:
                        out = json.loads(out[0])
                    except json.JSONDecodeError:
                        pass
                payload = out if isinstance(out, dict) else {"result": out}
                replies.append(types.Part.from_function_response(name=fc.name, response=payload))
                pictures += [types.Part.from_bytes(data=b, mime_type="image/png") for b in tools.last_images]
            contents.append(types.Content(role="user", parts=replies + pictures))

        await tools.call("request_approval", "the model ran out of turns - hand over to a human",
                         image_id=image_id, proposed_action="fail" if score >= 0.5 else "pass",
                         reason=f"no decision after {MAX_TURNS} turns; score {score:.2f}")
        return {"image_id": image_id, "decision": "human", "score": score, "by": "timeout"}
