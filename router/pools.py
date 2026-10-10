from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from router.kv.kvbus import KVBus
from router.kv.events import KVEventCollector
from gateway.types import KV_SATURATION, Request, SliceOOM, Snapshot, request_messages

PREFILL_CHUNK = 64
DECODE_PACK = 8

def parse_vllm_metrics(text: str) -> dict[str, float]:

    usage: float | None = None
    waiting: float | None = None
    running: float | None = None
    prefix_queries: float | None = None
    prefix_hits: float | None = None
    prompt_tokens: float | None = None
    generation_tokens: float | None = None
    waiting_capacity: float | None = None
    waiting_deferred: float | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(None, 1)[-1])
        except ValueError:
            continue

        if name in ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc") and usage is None:
            usage = value
        elif name == "vllm:num_requests_waiting" and waiting is None:
            waiting = value
        elif name == "vllm:num_requests_running" and running is None:
            running = value
        elif name == "vllm:num_requests_waiting_by_reason":
            reason = _label_value(line, "reason")
            if reason == "capacity" and waiting_capacity is None:
                waiting_capacity = value
            elif reason == "deferred" and waiting_deferred is None:
                waiting_deferred = value
        elif name == "vllm:prefix_cache_queries_total" and prefix_queries is None:
            prefix_queries = value
        elif name == "vllm:prefix_cache_hits_total" and prefix_hits is None:
            prefix_hits = value
        elif name == "vllm:prompt_tokens_total" and prompt_tokens is None:
            prompt_tokens = value
        elif name == "vllm:generation_tokens_total" and generation_tokens is None:
            generation_tokens = value
    if usage is not None and usage > 1.0:
        usage = usage / 100.0
    out: dict[str, float] = {}
    if usage is not None:
        out["gpu_cache_usage"] = usage
    if waiting is not None:
        out["waiting"] = waiting
    if running is not None:
        out["running"] = running
    if waiting_capacity is not None:
        out["waiting_capacity"] = waiting_capacity
    if waiting_deferred is not None:
        out["waiting_deferred"] = waiting_deferred
    if prefix_queries is not None:
        out["prefix_queries_total"] = prefix_queries
    if prefix_hits is not None:
        out["prefix_hits_total"] = prefix_hits
    if prompt_tokens is not None:
        out["prompt_tokens_total"] = prompt_tokens
    if generation_tokens is not None:
        out["generation_tokens_total"] = generation_tokens
    if prefix_queries and prefix_queries > 0 and prefix_hits is not None:
        out["prefix_cache_hit_ratio"] = max(0.0, min(1.0, prefix_hits / prefix_queries))
    return out

def _label_value(line: str, label: str) -> str | None:
    match = re.search(rf'{re.escape(label)}="([^"]*)"', line)
    return match.group(1) if match else None

class FakeWorker:

    def __init__(
        self,
        id: str,
        *,
        kv_capacity: int = 50_000,
        bus: KVBus | None = None,
    ) -> None:
        self.id = id
        self.kv_capacity = kv_capacity
        self.bus = bus or KVBus()
        self.healthy = True
        self.saturating = False
        self.age_override: float | None = None
        self.soak_weight = 1.0
        self.preempted = False
        self.last_error: str | None = None
        self.decode_tokens_emitted = 0
        self._kv_by_req: dict[str, int] = {}
        self._phase_by_req: dict[str, str] = {}
        self._prefill_left: dict[str, int] = {}
        self._decode_left: dict[str, int] = {}
        self._reqs: dict[str, Request] = {}

    @property
    def kv_used(self) -> int:
        return int(sum(self._kv_by_req.values()))

    def enqueue(
        self,
        req: Request,
        phase: str = "both",
        *,
        kv_transfer_params: dict | None = None,
        max_tokens: int | None = None,
    ) -> dict | None:
        if req.aborted:
            return
        cached = self.bus.cached(self.id, req.prefix_hash)
        need = 0
        if phase in ("prefill", "both"):
            need += max(0, int(req.prompt_tokens) - int(cached))
        if phase in ("decode", "both"):
            need += int(req.max_new_tokens)
        if self.kv_used + need > self.kv_capacity:
            self.last_error = "slice_oom"
            raise SliceOOM("slice_oom")
        self._reqs[req.id] = req
        self._phase_by_req[req.id] = phase
        if phase in ("prefill", "both"):
            uncached = max(0, int(req.prompt_tokens) - int(cached))
            self._prefill_left[req.id] = uncached

            self._kv_by_req[req.id] = 0
            self.bus.record(self.id, req.prefix_hash, req.prompt_tokens)
            self._drain_prefill(req.id)
        else:
            self._kv_by_req.setdefault(req.id, 0)
        if phase in ("decode", "both"):
            self._decode_left[req.id] = int(req.max_new_tokens)
        return None

    def step(self) -> None:
        ready = [
            rid
            for rid, left in self._decode_left.items()
            if left > 0 and rid not in self._prefill_left
        ]
        packed = 0
        for rid in ready:
            if packed >= DECODE_PACK:
                break
            take = min(DECODE_PACK - packed, self._decode_left[rid])
            self._decode_left[rid] -= take
            self._kv_by_req[rid] = self._kv_by_req.get(rid, 0) + take
            self.decode_tokens_emitted += take
            packed += take
            if self._decode_left[rid] == 0:
                del self._decode_left[rid]

    def run_until_idle(self) -> None:
        while self._prefill_left or self._decode_left:
            if self._prefill_left:
                for rid in list(self._prefill_left):
                    self._drain_prefill(rid)
            if self._decode_left:
                self.step()

    def abort(self, req_id: str) -> None:
        self._kv_by_req.pop(req_id, None)
        self._prefill_left.pop(req_id, None)
        self._decode_left.pop(req_id, None)
        self._phase_by_req.pop(req_id, None)
        req = self._reqs.pop(req_id, None)
        if req is not None:
            req.aborted = True

    def snapshot(self) -> Snapshot:
        used = self.kv_used
        free = 1.0 - (used / self.kv_capacity) if self.kv_capacity else 0.0
        free = max(0.0, min(1.0, free))
        n = len(self._reqs)
        in_flight = used
        uncached = int(sum(self._prefill_left.values()))
        saturating = self.saturating or (free < (1.0 - KV_SATURATION))
        return Snapshot(
            pod_id=self.id,
            age_s=0.0 if self.age_override is None else float(self.age_override),
            healthy=self.healthy,
            saturating=saturating,
            kv_free_ratio=free,
            tokens_in_flight=in_flight,
            uncached_prefill_tokens=uncached,
            active_requests=n,
            queue_depth=len(self._prefill_left) + len(self._decode_left),
            waiting=len(self._prefill_left),
            running=n,
            prefix_tokens=self.bus.prefixes_for(self.id),
            prefix_affinity_available=True,
            kv_event_age_s=0.0,
            soak_weight=self.soak_weight,
        )

    def _drain_prefill(self, req_id: str) -> None:
        left = self._prefill_left.get(req_id, 0)
        while left > 0:
            take = min(PREFILL_CHUNK, left)
            self._kv_by_req[req_id] = self._kv_by_req.get(req_id, 0) + take
            left -= take
        self._prefill_left.pop(req_id, None)

class VLLMWorker:

    def __init__(
        self,
        id: str,
        base_url: str,
        *,
        timeout_s: float = 8.0,
        bus: KVBus | None = None,
        model: str = "",
    ) -> None:
        self.id = id
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.bus = bus or KVBus()
        self.model = model or os.environ.get("LOCAL_MODEL", "lab")
        self._last: Snapshot | None = None
        self.healthy = True
        self.saturating = False
        self.kv_capacity = 0
        self._last_metrics_at: float | None = None
        self._last_prompt_tokens: float | None = None
        self._last_generation_tokens: float | None = None
        self.kv_event_stale_after_s = float(os.environ.get("KV_EVENT_STALE_S", "5"))
        self.transfer_cooldown_s = float(os.environ.get("KV_TRANSFER_COOLDOWN_S", "10"))
        self._transfer_failed_until = 0.0
        self._transfer_failures = 0
        self.kv_events = KVEventCollector(
            self.bus,
            self.id,
            stale_after_s=self.kv_event_stale_after_s,
        )

    def ingest_kv_event(self, event: dict[str, Any]) -> bool:
        """Accept one connector event from the vLLM KV-event bridge."""
        return self.kv_events.ingest(event)

    def report_transfer_failure(self) -> None:
        """Fail closed briefly after a connector/engine transfer failure.

        A failed KV load can leave vLLM restarting while its metrics endpoint
        still responds.  The cooldown keeps the router from piling more
        requests onto that engine during recovery.
        """
        self._transfer_failures += 1
        self._transfer_failed_until = time.monotonic() + self.transfer_cooldown_s
        self.healthy = False

    def report_transfer_success(self) -> None:
        self._transfer_failed_until = 0.0
        self._transfer_failures = 0
        self.healthy = True

    def enqueue(
        self,
        req: Request,
        phase: str = "both",
        *,
        kv_transfer_params: dict[str, object] | None = None,
        max_tokens: int | None = None,
    ) -> dict | None:
        """Submit one phase to vLLM.

        ``gateway.queue`` uses the optional transfer metadata for the real
        Mooncake producer/consumer path.  Keeping it here, at the worker
        adapter boundary, means the router still owns placement while vLLM
        owns execution and KV movement.
        """
        import json

        payload = {
            "model": self.model,
            "messages": request_messages(req),
            "max_tokens": max(
                1,
                int(max_tokens if max_tokens is not None else req.max_new_tokens),
            ),
        }
        if kv_transfer_params is not None:
            payload["kv_transfer_params"] = kv_transfer_params
        raw = self._post("/v1/chat/completions", payload)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"choices": [{"message": {"content": raw[:240]}}]}
        if not isinstance(data, dict):
            return {"choices": [{"message": {"content": str(data)[:240]}}]}
        return data

    def snapshot(self) -> Snapshot:
        now = time.time()
        monotonic_now = time.monotonic()
        try:
            raw = self._get("/metrics", timeout_s=3.0)
        except urllib.error.HTTPError:
            return self._unavailable_snapshot(monotonic_now)
        except (urllib.error.URLError, TimeoutError, OSError):
            return self._unavailable_snapshot(monotonic_now)
        parsed = parse_vllm_metrics(raw)
        usage = parsed.get("gpu_cache_usage")
        kv_free = None if usage is None else 1.0 - usage
        waiting = parsed.get("waiting")
        running = parsed.get("running")
        waiting_i = None if waiting is None else int(waiting)
        running_i = None if running is None else int(running)
        active = None
        if waiting_i is not None or running_i is not None:
            active = int((waiting_i or 0) + (running_i or 0))
        prompt_total = parsed.get("prompt_tokens_total")
        generation_total = parsed.get("generation_tokens_total")
        prompt_rate, generation_rate = self._rates(
            now,
            prompt_total,
            generation_total,
        )
        exact_prefixes = self.bus.verified_prefixes_for(
            self.id, stale_after_s=self.kv_event_stale_after_s
        )
        event_age = self.bus.verified_event_age(self.id)
        self._last_metrics_at = monotonic_now
        transfer_cooldown = monotonic_now < self._transfer_failed_until
        snap = Snapshot(
            pod_id=self.id,
            age_s=0.0,
            observed_at_s=now,
            healthy=not transfer_cooldown,
            saturating=self.saturating or bool(kv_free is not None and kv_free < (1.0 - KV_SATURATION)),
            kv_free_ratio=kv_free,
            tokens_in_flight=active,
            uncached_prefill_tokens=None,
            active_requests=active,
            queue_depth=waiting_i,
            waiting=waiting_i,
            running=running_i,
            waiting_capacity=(
                None if parsed.get("waiting_capacity") is None else int(parsed["waiting_capacity"])
            ),
            waiting_deferred=(
                None if parsed.get("waiting_deferred") is None else int(parsed["waiting_deferred"])
            ),
            prefix_tokens=exact_prefixes,
            prefix_cache_hit_ratio=parsed.get("prefix_cache_hit_ratio"),
            prompt_tokens_total=prompt_total,
            generation_tokens_total=generation_total,
            prompt_tokens_rate=prompt_rate,
            generation_tokens_rate=generation_rate,
            prefix_affinity_available=exact_prefixes is not None,
            kv_event_age_s=event_age,
            soak_weight=1.0,
        )
        self._last = snap
        return snap

    def _rates(
        self,
        now: float,
        prompt_total: float | None,
        generation_total: float | None,
    ) -> tuple[float | None, float | None]:
        previous_at = getattr(self, "_last_metrics_wall", None)
        previous_prompt = self._last_prompt_tokens
        previous_generation = self._last_generation_tokens
        self._last_metrics_wall = now
        self._last_prompt_tokens = prompt_total
        self._last_generation_tokens = generation_total
        if previous_at is None or now <= previous_at:
            return None, None
        interval = now - previous_at
        prompt_rate = None
        generation_rate = None
        if prompt_total is not None and previous_prompt is not None:
            prompt_rate = max(0.0, (prompt_total - previous_prompt) / interval)
        if generation_total is not None and previous_generation is not None:
            generation_rate = max(0.0, (generation_total - previous_generation) / interval)
        return prompt_rate, generation_rate

    def _unavailable_snapshot(self, monotonic_now: float) -> Snapshot:
        age = float("inf") if self._last_metrics_at is None else max(
            0.0, monotonic_now - self._last_metrics_at
        )
        return Snapshot(
            pod_id=self.id,
            age_s=age,
            healthy=False,
            saturating=self.saturating,
            observed_at_s=None,
            prefix_affinity_available=False,
            kv_event_age_s=self.bus.verified_event_age(self.id),
        )

    def abort(self, req_id: str) -> None:
        return None

    def _get(self, path: str, timeout_s: float | None = None) -> str:
        url = self.base_url + path
        with urllib.request.urlopen(url, timeout=timeout_s or self.timeout_s) as resp:
            return resp.read().decode("utf-8")

    def _post(self, path: str, payload: dict[str, Any]) -> str:
        import json

        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8")
            if exc.code >= 500:
                raise
            return raw

def _rewrite_url(url: str) -> str:
    return url.replace("YOUR_LAMBDA_IP", "127.0.0.1")

def _urls(env_name: str) -> list[str]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return []
    return [_rewrite_url(u.strip()) for u in raw.split(",") if u.strip()]

def build_pools(n_fake: int = 2, bus: KVBus | None = None) -> tuple[list[Any], list[Any]]:

    prefill_urls = _urls("PREFILL_URLS")
    if not prefill_urls:
        bus = bus or KVBus()
        prefill = [FakeWorker(f"prefill-{i}", bus=bus) for i in range(n_fake)]
        decode = [FakeWorker(f"decode-{i}", bus=bus) for i in range(n_fake)]
        return prefill, decode

    decode_urls = _urls("DECODE_URLS")
    topology = os.environ.get("LAB_TOPOLOGY", "disaggregated")
    same = (not decode_urls) or decode_urls == prefill_urls or topology == "aggregated"
    bus = bus or KVBus()
    text_model = os.environ.get("TEXT_MODEL", os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    split = os.environ.get("LAB_SPLIT", "capability")
    vision_model = (
        text_model
        if split == "phase"
        else os.environ.get("VISION_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    )
    if same:
        engines = [VLLMWorker(f"engine-{i}", url, bus=bus, model=text_model) for i, url in enumerate(prefill_urls)]
        return engines, engines

    text = [VLLMWorker(f"text-{i}", url, bus=bus, model=text_model) for i, url in enumerate(prefill_urls)]
    vision = [VLLMWorker(f"vision-{i}", url, bus=bus, model=vision_model) for i, url in enumerate(decode_urls)]
    return text, vision
