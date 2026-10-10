from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
import time

from gateway.metrics import METRICS
from gateway.types import Request
from router.kv.events import KVEvent


@dataclass
class _VerifiedResidency:
    tokens: int
    refcount: int
    observed_at_s: float

class KVBus:

    def __init__(self) -> None:
        self._cached: dict[tuple[str, str], int] = {}
        self._refcounts: dict[tuple[str, str], int] = {}
        self._verified: dict[tuple[str, str], _VerifiedResidency] = {}
        self.hops: list[dict] = []

    def apply_event(self, event: KVEvent) -> None:
        """Apply an engine-confirmed block lifecycle event.

        Stored events are reference-counted because multiple requests may
        share a prefix block. Removed events decrement the count and delete
        residency only when the engine reports the final reference gone.
        """
        if event.kind == "all_blocks_cleared":
            self._verified = {
                key: value for key, value in self._verified.items() if key[0] != event.worker_id
            }
            return
        key = (event.worker_id, event.block_hash)
        if event.kind == "stored":
            current = self._verified.get(key)
            self._verified[key] = _VerifiedResidency(
                tokens=max(event.tokens, current.tokens if current else 0),
                refcount=(current.refcount if current else 0) + 1,
                observed_at_s=event.observed_at_s or time.time(),
            )
            return
        current = self._verified.get(key)
        if current is None:
            return
        if current.refcount <= 1:
            self._verified.pop(key, None)
        else:
            current.refcount -= 1
            current.observed_at_s = event.observed_at_s or time.time()

    def verified_cached(self, worker_id: str, block_hash: str | None) -> int:
        if not block_hash:
            return 0
        item = self._verified.get((worker_id, block_hash))
        return 0 if item is None else item.tokens

    def verified_prefixes_for(self, worker_id: str, *, stale_after_s: float = 5.0) -> dict[str, int] | None:
        """Return exact event-backed residency, or ``None`` when stale/absent."""
        items = [
            (block_hash, item)
            for (wid, block_hash), item in self._verified.items()
            if wid == worker_id
        ]
        if not items:
            return None
        now = time.time()
        if any(now - item.observed_at_s > stale_after_s for _, item in items):
            return None
        return {block_hash: item.tokens for block_hash, item in items}

    def verified_event_age(self, worker_id: str) -> float:
        items = [item.observed_at_s for (wid, _), item in self._verified.items() if wid == worker_id]
        if not items:
            return float("inf")
        return max(0.0, time.time() - max(items))

    def cached(self, worker_id: str, prefix_hash: str | None) -> int:
        if not prefix_hash:
            return 0
        return self._cached.get((worker_id, prefix_hash), 0)

    def record(self, worker_id: str, prefix_hash: str | None, tokens: int) -> None:
        if not prefix_hash:
            return
        key = (worker_id, prefix_hash)
        self._cached[key] = max(self._cached.get(key, 0), tokens)
        self._refcounts[key] = self._refcounts.get(key, 0) + 1

    def refcount(self, worker_id: str, prefix_hash: str | None) -> int:
        if not prefix_hash:
            return 0
        return self._refcounts.get((worker_id, prefix_hash), 0)

    def forget_worker(self, worker_id: str) -> None:
        self._cached = {k: v for k, v in self._cached.items() if k[0] != worker_id}
        self._refcounts = {k: v for k, v in self._refcounts.items() if k[0] != worker_id}
        self._verified = {k: v for k, v in self._verified.items() if k[0] != worker_id}

    def prefixes_for(self, worker_id: str) -> dict[str, int]:
        return {h: n for (wid, h), n in self._cached.items() if wid == worker_id}

    def evict(self, worker_id: str, prefix_hash: str | None = None) -> int:

        if prefix_hash:
            key = (worker_id, prefix_hash)
            n = 1 if key in self._cached else 0
            self._cached.pop(key, None)
            self._refcounts.pop(key, None)
        else:
            keys = [k for k in self._cached if k[0] == worker_id]
            n = len(keys)
            for key in keys:
                del self._cached[key]
                self._refcounts.pop(key, None)
        if n:
            METRICS.inc_kv_evict(n)
        return n

    def evict_workers(self, worker_ids: list[str], prefix_hash: str | None = None) -> int:
        return sum(self.evict(wid, prefix_hash) for wid in worker_ids)

    def transfer(self, src: str, dst: str, req: Request) -> dict:

        tokens = self.cached(src, req.prefix_hash) or int(req.prompt_tokens)
        if req.prefix_hash:
            self.record(dst, req.prefix_hash, tokens)
        hop = {
            "src": src,
            "dst": dst,
            "req_id": req.id,
            "tokens": tokens,
            "backend": os.environ.get("KV_BACKEND", "mooncake"),
            "prefix": req.prefix_hash,
        }
        self.hops.append(hop)
        METRICS.inc_kv_transfer(tokens)
        _backend_put(hop)
        return hop

    def record_real_transfer(self, src: str, dst: str, req: Request) -> dict:
        """Record a transfer performed by vLLM's real MooncakeConnector."""
        tokens = self.cached(src, req.prefix_hash) or int(req.prompt_tokens)
        if req.prefix_hash:
            self.record(dst, req.prefix_hash, tokens)
        hop = {
            "src": src,
            "dst": dst,
            "req_id": req.id,
            "tokens": tokens,
            "backend": "mooncake-real",
            "prefix": req.prefix_hash,
        }
        self.hops.append(hop)
        METRICS.inc_kv_transfer(tokens)
        return hop

def _backend_put(hop: dict) -> None:
    backend = hop.get("backend") or "mooncake"
    if backend == "nccl":
        from router.nccl import put
        put(hop)
        return
    if backend == "nixl":
        from router.nixl import put
        put(hop)
        return
    _store_put(hop)

def _store_put(hop: dict) -> None:
    base = os.environ.get("MOONCAKE_URL", "").strip().rstrip("/")
    if not base:
        return
    data = json.dumps(hop).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/put",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=2)
    except (urllib.error.URLError, TimeoutError, OSError):
        return
