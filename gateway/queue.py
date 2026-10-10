from __future__ import annotations

import os
import threading
from contextlib import nullcontext

from router.kv.mooncake_connector import (
    consumer_params,
    enabled,
    producer_engine_id,
    producer_params,
    transfer_id,
)
from gateway.types import Handoff, Request

class KVTransferError(RuntimeError):
    """The real KV consumer did not accept the handoff."""


# vLLM's Mooncake callbacks are asynchronous, while the OpenAI adapter is
# synchronous.  On a single decode slice, overlapping producer/consumer
# submissions can complete out of order and expose a late callback to a
# request that vLLM has already removed.  Serialize the real handoff by
# default; operators can disable this only when the engine/connector version
# has a verified transfer-completion barrier.
_REAL_HANDOFF_LOCK = threading.Lock()


def _serialize_real_handoffs() -> bool:
    return os.environ.get("MOONCAKE_SERIALIZE_HANDOFFS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def enqueue(handoff: Handoff, req: Request) -> tuple[dict | None, dict | None]:

    if handoff.decode is handoff.prefill:
        out = handoff.prefill.enqueue(req, phase="both")
        return None, out if isinstance(out, dict) else None
    if enabled() and hasattr(handoff.prefill, "enqueue") and hasattr(handoff.decode, "enqueue"):
        request_id = req.id
        xfer_id = transfer_id(request_id)
        engine_id = producer_engine_id()
        try:
            # The consumer call is the authoritative transfer-acceptance
            # barrier.  On the single-slice deployment, keep the producer and
            # consumer lifecycle together so callbacks from overlapping
            # transfers cannot race vLLM request cleanup.
            lock = _REAL_HANDOFF_LOCK if _serialize_real_handoffs() else nullcontext()
            with lock:
                handoff.prefill.enqueue(
                    req,
                    phase="prefill",
                    kv_transfer_params=producer_params(request_id, xfer_id=xfer_id),
                    max_tokens=1,
                )
                out_d = handoff.decode.enqueue(
                    req,
                    phase="decode",
                    kv_transfer_params=consumer_params(request_id, engine_id, xfer_id=xfer_id),
                )
        except Exception as exc:
            for worker in (handoff.prefill, handoff.decode):
                report = getattr(worker, "report_transfer_failure", None)
                if report is not None:
                    report()
            raise KVTransferError(f"KV transfer was not accepted: {exc}") from exc
        for worker in (handoff.prefill, handoff.decode):
            report = getattr(worker, "report_transfer_success", None)
            if report is not None:
                report()
        bus = getattr(handoff.prefill, "bus", None) or getattr(handoff.decode, "bus", None)
        hop = bus.record_real_transfer(handoff.prefill.id, handoff.decode.id, req) if bus is not None else None
        return hop, out_d if isinstance(out_d, dict) else None

    out_p = handoff.prefill.enqueue(req, phase="prefill")
    hop = None
    bus = getattr(handoff.prefill, "bus", None) or getattr(handoff.decode, "bus", None)
    if bus is not None and hasattr(bus, "transfer"):
        hop = bus.transfer(handoff.prefill.id, handoff.decode.id, req)
    out_d = handoff.decode.enqueue(req, phase="decode")
    completion = out_d if isinstance(out_d, dict) else out_p if isinstance(out_p, dict) else None
    return hop, completion
