"""Normalized KV-cache lifecycle events consumed by the router.

The engine remains the owner of GPU KV blocks.  This module only defines the
small event contract a vLLM connector/event bridge must publish to the router.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KVEvent:
    worker_id: str
    kind: str
    block_hash: str
    tokens: int = 0
    parent_hash: str | None = None
    observed_at_s: float = 0.0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, default_worker: str = "") -> "KVEvent":
        kind = str(payload.get("kind") or payload.get("type") or payload.get("event") or "").lower()
        if kind in {"block_stored", "blockstored", "stored", "put", "added"}:
            kind = "stored"
        elif kind in {
            "block_removed", "blockremoved", "removed", "evicted", "delete", "deleted",
        }:
            kind = "removed"
        elif kind in {"all_blocks_cleared", "allblockscleared", "cleared"}:
            kind = "all_blocks_cleared"
        else:
            raise ValueError(f"unsupported KV event kind: {kind or '<empty>'}")

        timestamp = payload.get("observed_at_s", payload.get("timestamp", time.time()))
        block_hash = payload.get("block_hash") or payload.get("hash") or payload.get("key")
        if kind == "all_blocks_cleared":
            return cls(
                worker_id=str(payload.get("worker_id") or payload.get("worker") or default_worker),
                kind=kind,
                block_hash="",
                observed_at_s=float(timestamp),
            )
        if not block_hash:
            raise ValueError("KV event has no block_hash")
        return cls(
            worker_id=str(payload.get("worker_id") or payload.get("worker") or default_worker),
            kind=kind,
            block_hash=str(block_hash),
            tokens=max(0, int(payload.get("tokens", payload.get("block_size", 0)) or 0)),
            parent_hash=(str(payload["parent_hash"]) if payload.get("parent_hash") else None),
            observed_at_s=float(timestamp),
        )


class KVEventCollector:
    """Adapter for a vLLM KV event bridge.

    vLLM connector events are delivered to ``ingest`` by the bridge.  The
    optional HTTP method is intentionally generic: a sidecar or connector
    endpoint may expose either a JSON list, a single JSON object, or NDJSON.
    """

    def __init__(self, bus: Any, worker_id: str, *, stale_after_s: float = 5.0) -> None:
        self.bus = bus
        self.worker_id = worker_id
        self.stale_after_s = stale_after_s
        self.events_seen = 0
        self.events_rejected = 0

    def ingest(self, payload: KVEvent | dict[str, Any]) -> bool:
        try:
            event = payload if isinstance(payload, KVEvent) else KVEvent.from_mapping(
                payload, default_worker=self.worker_id
            )
            if not event.worker_id:
                event = KVEvent(
                    worker_id=self.worker_id,
                    kind=event.kind,
                    block_hash=event.block_hash,
                    tokens=event.tokens,
                    parent_hash=event.parent_hash,
                    observed_at_s=event.observed_at_s,
                )
            self.bus.apply_event(event)
        except (TypeError, ValueError, KeyError):
            self.events_rejected += 1
            return False
        self.events_seen += 1
        return True

    def ingest_many(self, payloads: list[KVEvent | dict[str, Any]]) -> int:
        return sum(1 for payload in payloads if self.ingest(payload))
