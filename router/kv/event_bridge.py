"""ZMQ bridge from vLLM KV event publishers into router workers."""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable

logger = logging.getLogger(__name__)

try:
    import msgspec

    class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
        ts: float
        events: list[Any]

    class KVCacheEvent(msgspec.Struct, array_like=True, omit_defaults=True, gc=False, tag=True):
        pass

    class BlockStored(KVCacheEvent):
        block_hashes: list[Any]
        parent_block_hash: Any = None
        token_ids: list[int] = msgspec.field(default_factory=list)
        block_size: int = 0
        lora_id: Any = None
        medium: Any = None
        lora_name: Any = None
        extra_keys: Any = None
        kv_cache_spec_group_id: Any = None
        kv_cache_spec_sliding_window: Any = None
        locality: Any = None
        ownership: Any = None

    class BlockRemoved(KVCacheEvent):
        block_hashes: list[Any]
        medium: Any = None
        group_idx: Any = None
        locality: Any = None
        ownership: Any = None

    class AllBlocksCleared(KVCacheEvent):
        pass

    class KVEventBatch(EventBatch):
        events: list[BlockStored | BlockRemoved | AllBlocksCleared]

except ImportError:  # pragma: no cover - installed in the gateway image
    msgspec = None
    KVEventBatch = None


def _event_mappings(batch: Any, worker_id: str) -> Iterable[dict[str, Any]]:
    """Translate vLLM's msgspec EventBatch into the router event contract."""
    timestamp = float(getattr(batch, "ts", 0.0))
    for event in getattr(batch, "events", ()):
        name = type(event).__name__
        hashes = getattr(event, "block_hashes", ())
        if name == "AllBlocksCleared":
            yield {
                "type": "all_blocks_cleared",
                "worker_id": worker_id,
                "observed_at_s": timestamp,
            }
            continue
        if name not in {"BlockStored", "BlockRemoved"}:
            continue
        kind = "stored" if name == "BlockStored" else "removed"
        for block_hash in hashes:
            yield {
                "type": kind,
                "worker_id": worker_id,
                "block_hash": str(block_hash),
                "parent_hash": (
                    str(getattr(event, "parent_block_hash"))
                    if getattr(event, "parent_block_hash", None) is not None
                    else None
                ),
                "tokens": int(getattr(event, "block_size", 0) or 0),
                "observed_at_s": timestamp,
            }


def _raw_event_mappings(payload: Any, worker_id: str) -> Iterable[dict[str, Any]]:
    """Fallback decoder for vLLM builds that emit map-encoded structs."""
    if isinstance(payload, dict):
        timestamp = float(payload.get("ts", 0.0))
        events = payload.get("events", ())
    elif isinstance(payload, (list, tuple)) and len(payload) >= 2:
        timestamp = float(payload[0])
        events = payload[1]
    else:
        return
    for event in events or ():
        if isinstance(event, dict):
            name = str(event.get("type") or event.get("event") or event.get("tag") or "")
            hashes = event.get("block_hashes") or event.get("hashes") or ()
            parent = event.get("parent_block_hash")
            tokens = event.get("block_size", 0)
        elif isinstance(event, (list, tuple)) and event:
            name = str(event[0])
            hashes = event[1] if len(event) > 1 else ()
            parent = event[2] if len(event) > 2 else None
            tokens = event[4] if len(event) > 4 else 0
        else:
            continue
        if name.lower() in {"allblockscleared", "all_blocks_cleared"}:
            yield {"type": "all_blocks_cleared", "worker_id": worker_id, "observed_at_s": timestamp}
            continue
        if name.lower() not in {"blockstored", "block_stored", "blockremoved", "block_removed"}:
            continue
        kind = "stored" if "stored" in name.lower() else "removed"
        for block_hash in hashes:
            yield {
                "type": kind,
                "worker_id": worker_id,
                "block_hash": str(block_hash),
                "parent_hash": str(parent) if parent is not None else None,
                "tokens": int(tokens or 0),
                "observed_at_s": timestamp,
            }


class KVEventBridge:
    """Subscribe to one vLLM ZMQ publisher per live worker.

    This bridge is deliberately best-effort. If ZMQ or the publisher is
    unavailable, workers remain healthy for serving but exact prefix affinity
    stays disabled and the router uses aggregate metrics.
    """

    def __init__(self, workers: Iterable[Any], endpoints: Iterable[str], *, topic: str = "kv-events") -> None:
        self.workers = {worker.id: worker for worker in workers}
        self.endpoints = list(endpoints)
        self.topic = topic
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if not self.endpoints or not self.workers:
            return
        for worker_id, endpoint in zip(self.workers, self.endpoints):
            thread = threading.Thread(
                target=self._consume,
                args=(worker_id, endpoint),
                daemon=True,
                name=f"kv-events-{worker_id}",
            )
            thread.start()
            self._threads.append(thread)

    def _consume(self, worker_id: str, endpoint: str) -> None:
        try:
            import zmq
            from msgspec.msgpack import Decoder
            decoder = Decoder(type=KVEventBatch)
            context = zmq.Context.instance()
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(endpoint)
            socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
            logger.warning("subscribed to vLLM KV events worker=%s endpoint=%s", worker_id, endpoint)
            while True:
                frames = socket.recv_multipart()
                if len(frames) != 3:
                    continue
                try:
                    batch = decoder.decode(frames[2])
                    events = _event_mappings(batch, worker_id)
                except Exception:
                    raw_batch = msgspec.msgpack.decode(frames[2])
                    events = _raw_event_mappings(raw_batch, worker_id)
                worker = self.workers.get(worker_id)
                if worker is None:
                    continue
                for event in events:
                    worker.ingest_kv_event(event)
        except Exception:
            logger.exception("vLLM KV event bridge stopped worker=%s endpoint=%s", worker_id, endpoint)
