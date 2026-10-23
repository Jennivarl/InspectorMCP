"""Send run metrics and summaries to Amazon CloudWatch (best effort - never breaks a run)."""

from __future__ import annotations

import json
import logging
import threading
import time

import boto3

REGION = "us-east-1"
NAMESPACE = "InspectorMCP"
LOG_GROUP = "/inspector/runs"
log = logging.getLogger("inspector.telemetry")


def _send(summary: dict) -> None:
    try:
        cw = boto3.client("cloudwatch", region_name=REGION)
        dims = [{"Name": "Brain", "Value": summary.get("brain", "policy")}]
        metrics = {
            "CellsInspected": (summary["cells"], "Count"),
            "DefectsFound": (summary["fail"], "Count"),
            "HumanReviews": (summary["human"], "Count"),
            "MsPerCell": (summary["ms_per_cell"], "Milliseconds"),
            "LLMRequests": (summary.get("llm_requests", 0), "Count"),
        }
        cw.put_metric_data(Namespace=NAMESPACE, MetricData=[
            {"MetricName": k, "Value": float(val), "Unit": unit, "Dimensions": dims} for k, (val, unit) in metrics.items()
        ])
        logs = boto3.client("logs", region_name=REGION)
        stream = time.strftime("%Y-%m-%d")
        try:
            logs.create_log_group(logGroupName=LOG_GROUP)
        except logs.exceptions.ResourceAlreadyExistsException:
            pass
        try:
            logs.create_log_stream(logGroupName=LOG_GROUP, logStreamName=stream)
        except logs.exceptions.ResourceAlreadyExistsException:
            pass
        logs.put_log_events(logGroupName=LOG_GROUP, logStreamName=stream,
                            logEvents=[{"timestamp": int(time.time() * 1000), "message": json.dumps(summary)}])
    except Exception as e:  # metrics must never take the app down
        log.warning("telemetry failed: %s", e)


def report_run(summary: dict) -> None:
    threading.Thread(target=_send, args=(summary,), daemon=True).start()
