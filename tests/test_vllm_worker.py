from __future__ import annotations

import os

import pytest

from router.pools import VLLMWorker, parse_vllm_metrics
from gateway.types import Request

pytestmark = getattr(pytest.mark, "lambda")

@pytest.mark.skipif(not os.environ.get("PREFILL_URLS"), reason="PREFILL_URLS not set")
def test_vllm_short_prompt_snapshot() -> None:
    url = os.environ["PREFILL_URLS"].split(",")[0].strip()
    worker = VLLMWorker("lambda-0", url)
    worker.enqueue(
        Request(
            id="lambda-ping",
            arrival_t=0.0,
            priority=1,
            prompt_tokens=8,
            max_new_tokens=4,
            prefix_hash=None,
            timeout_s=60.0,
            tenant="lab",
        )
    )
    snap = worker.snapshot()
    assert snap.kv_free_ratio is not None
    assert 0 < snap.kv_free_ratio <= 1
    assert snap.waiting is not None and snap.waiting < float("inf")
    assert snap.running is not None and snap.running < float("inf")


def test_vllm_worker_builds_mooncake_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = VLLMWorker("prefill-0", "http://prefill", model="text")
    sent: dict = {}

    def fake_post(path: str, payload: dict) -> str:
        sent["path"] = path
        sent["payload"] = payload
        return '{"id":"completion-1","choices":[]}'

    monkeypatch.setattr(worker, "_post", fake_post)
    req = Request(
        id="req-1",
        arrival_t=0.0,
        priority=1,
        prompt_tokens=32,
        max_new_tokens=64,
        prefix_hash="prefix-1",
        timeout_s=30.0,
        tenant="lab",
    )

    result = worker.enqueue(
        req,
        phase="prefill",
        max_tokens=1,
        kv_transfer_params={
            "do_remote_decode": True,
            "transfer_id": "xfer-req-1",
        },
    )

    assert result == {"id": "completion-1", "choices": []}
    assert sent == {
        "path": "/v1/chat/completions",
        "payload": {
            "model": "text",
            "messages": [{"role": "user", "content": "req-1"}],
            "max_tokens": 1,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "transfer_id": "xfer-req-1",
            },
        },
    }


def test_parse_live_vllm_cache_and_token_metrics() -> None:
    parsed = parse_vllm_metrics(
        """
vllm:prefix_cache_queries_total 10
vllm:prefix_cache_hits_total 4
vllm:prompt_tokens_total 120
vllm:generation_tokens_total 80
vllm:num_requests_waiting 2
vllm:num_requests_waiting_by_reason{reason="capacity"} 1
vllm:num_requests_waiting_by_reason{reason="deferred"} 1
vllm:num_requests_running 3
"""
    )
    assert parsed["prefix_cache_hit_ratio"] == 0.4
    assert parsed["prompt_tokens_total"] == 120
    assert parsed["generation_tokens_total"] == 80
    assert parsed["waiting_capacity"] == 1
    assert parsed["waiting_deferred"] == 1


def test_vllm_snapshot_exposes_live_observation_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = VLLMWorker("decode-0", "http://decode", model="text")
    metrics = """
vllm:kv_cache_usage_perc 0.25
vllm:num_requests_waiting 2
vllm:num_requests_waiting_by_reason{reason="capacity"} 1
vllm:num_requests_waiting_by_reason{reason="deferred"} 1
vllm:num_requests_running 3
vllm:prefix_cache_queries_total 10
vllm:prefix_cache_hits_total 4
vllm:prompt_tokens_total 120
vllm:generation_tokens_total 80
"""
    monkeypatch.setattr(worker, "_get", lambda path, timeout_s=None: metrics)

    snap = worker.snapshot()

    assert snap.observed_at_s is not None
    assert snap.age_s == 0.0
    assert snap.prefix_cache_hit_ratio == 0.4
    assert snap.prompt_tokens_total == 120
    assert snap.generation_tokens_total == 80
    assert snap.prompt_tokens_rate is None
    assert snap.generation_tokens_rate is None
    assert snap.waiting_capacity == 1
    assert snap.waiting_deferred == 1

    second = worker.snapshot()
    assert second.prompt_tokens_rate == 0.0
    assert second.generation_tokens_rate == 0.0
