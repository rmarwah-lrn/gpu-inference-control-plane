from __future__ import annotations

import time

from router.kv.events import KVEventCollector
from router.kv.event_bridge import _event_mappings
from router.kv.kvbus import KVBus
from router.pools import VLLMWorker


def test_event_collector_tracks_verified_residency_and_refcounts() -> None:
    bus = KVBus()
    collector = KVEventCollector(bus, "prefill-0")

    assert collector.ingest({"type": "BlockStored", "block_hash": "abc", "tokens": 64})
    assert collector.ingest({"type": "stored", "block_hash": "abc", "tokens": 64})
    assert bus.verified_cached("prefill-0", "abc") == 64
    assert bus.verified_prefixes_for("prefill-0") == {"abc": 64}

    assert collector.ingest({"type": "BlockRemoved", "block_hash": "abc"})
    assert bus.verified_cached("prefill-0", "abc") == 64
    assert collector.ingest({"type": "removed", "block_hash": "abc"})
    assert bus.verified_cached("prefill-0", "abc") == 0


def test_invalid_event_is_rejected() -> None:
    bus = KVBus()
    collector = KVEventCollector(bus, "decode-0")
    assert not collector.ingest({"type": "unknown", "block_hash": "x"})
    assert collector.events_rejected == 1


def test_vllm_event_batch_is_translated_to_worker_contract() -> None:
    class BlockStored:
        block_hashes = [123, 456]
        parent_block_hash = 99
        block_size = 16

    class Batch:
        ts = 42.0
        events = [BlockStored()]

    events = list(_event_mappings(Batch(), "prefill-0"))
    assert events == [
        {
            "type": "stored",
            "worker_id": "prefill-0",
            "block_hash": "123",
            "parent_hash": "99",
            "tokens": 16,
            "observed_at_s": 42.0,
        },
        {
            "type": "stored",
            "worker_id": "prefill-0",
            "block_hash": "456",
            "parent_hash": "99",
            "tokens": 16,
            "observed_at_s": 42.0,
        },
    ]


def test_vllm_worker_uses_exact_events_and_downgrades_when_stale(monkeypatch) -> None:
    bus = KVBus()
    worker = VLLMWorker("decode-0", "http://decode", bus=bus, model="text")
    assert worker.ingest_kv_event({"type": "stored", "block_hash": "prefix-1", "tokens": 32})
    monkeypatch.setattr(worker, "_get", lambda path, timeout_s=None: "vllm:kv_cache_usage_perc 0.1\n")

    fresh = worker.snapshot()
    assert fresh.prefix_affinity_available is True
    assert fresh.prefix_tokens == {"prefix-1": 32}
    assert fresh.kv_event_age_s is not None and fresh.kv_event_age_s < 5

    bus._verified[("decode-0", "prefix-1")].observed_at_s = time.time() - 10
    stale = worker.snapshot()
    assert stale.prefix_affinity_available is False
    assert stale.prefix_tokens is None
