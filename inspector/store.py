"""In-memory store shared by the MCP tools, the agent and the web app (one process)."""

from __future__ import annotations

import itertools
import threading
import time
from collections import OrderedDict

import numpy as np

MAX_IMAGES = 2000


class Store:
    def __init__(self):
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self.images: OrderedDict[str, np.ndarray] = OrderedDict()
        self.meta: dict[str, dict] = {}
        self.truth: dict[str, int] = {}  # hidden labels for demo images; never shown to the agent
        self.tickets: list[dict] = []
        self.approvals: dict[str, dict] = {}
        self.verdicts: dict[str, dict] = {}  # final call per cell: pass / fail / human

    def put(self, img: np.ndarray, prefix: str = "img", **meta) -> str:
        with self._lock:
            image_id = f"{prefix}-{next(self._ids)}"
            self.images[image_id] = img
            self.meta[image_id] = {"created": time.time(), "shape": list(img.shape), **meta}
            while len(self.images) > MAX_IMAGES:
                old, _ = self.images.popitem(last=False)
                self.meta.pop(old, None)
                self.truth.pop(old, None)
            return image_id

    def get(self, image_id: str) -> np.ndarray:
        try:
            return self.images[image_id]
        except KeyError:
            raise ValueError(f"unknown image_id: {image_id}") from None

    def set_verdict(self, image_id: str, verdict: str, reason: str, **extra) -> dict:
        with self._lock:
            v = {"image_id": image_id, "verdict": verdict, "reason": reason, "time": time.time(), **extra}
            self.verdicts[image_id] = v
            return v

    def add_ticket(self, **ticket) -> dict:
        with self._lock:
            ticket = {"ticket_id": f"T-{len(self.tickets) + 1:04d}", "created": time.time(), **ticket}
            self.tickets.append(ticket)
        self.set_verdict(ticket["image_id"], "fail", ticket.get("evidence", ""), ticket_id=ticket["ticket_id"])
        return ticket

    def request_approval(self, **req) -> dict:
        with self._lock:
            approval_id = f"A-{len(self.approvals) + 1:04d}"
            req = {"approval_id": approval_id, "status": "pending", "created": time.time(), **req}
            self.approvals[approval_id] = req
        self.set_verdict(req["image_id"], "human", req.get("reason", ""), approval_id=approval_id)
        return req

    def decide(self, approval_id: str, defect: bool, note: str = "") -> dict:
        with self._lock:
            req = self.approvals[approval_id]
            req.update(status="defect" if defect else "no_defect", note=note, decided=time.time())
            return req


STORE = Store()
