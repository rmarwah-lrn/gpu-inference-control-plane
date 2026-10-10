"""Request metadata for vLLM's real MooncakeConnector path.

The connector and the KV movement live inside vLLM/Mooncake. This module only
discovers the producer engine id and builds the documented per-request fields.
"""

from __future__ import annotations

import json
import os
import urllib.request
import uuid


def enabled() -> bool:
    return os.environ.get("MOONCAKE_REAL", "").strip().lower() in {"1", "true", "yes"}


def bootstrap_address() -> str:
    return os.environ.get("PREFILL_BOOTSTRAP_URL", "http://vllm-prefill:8998").rstrip("/")


def producer_engine_id(address: str | None = None) -> str:
    url = (address or bootstrap_address()).rstrip("/") + "/query"
    with urllib.request.urlopen(url, timeout=5) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise RuntimeError("Mooncake bootstrap returned a non-object response")
    entry = body.get("0") or body.get(0)
    if not isinstance(entry, dict) or not entry.get("engine_id"):
        raise RuntimeError("Mooncake bootstrap response has no producer engine_id")
    return str(entry["engine_id"])


def transfer_id(request_id: str | None = None) -> str:
    return f"xfer-{request_id or uuid.uuid4()}"


def producer_params(request_id: str, *, xfer_id: str | None = None) -> dict[str, object]:
    return {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": xfer_id or transfer_id(request_id),
    }


def consumer_params(
    request_id: str,
    engine_id: str,
    *,
    xfer_id: str | None = None,
) -> dict[str, object]:
    return {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_bootstrap_addr": bootstrap_address(),
        "remote_engine_id": engine_id,
        "transfer_id": xfer_id or transfer_id(request_id),
    }
