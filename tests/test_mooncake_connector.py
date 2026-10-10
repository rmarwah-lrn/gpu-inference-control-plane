from router.kv.mooncake_connector import consumer_params, producer_params
from gateway.queue import KVTransferError, enqueue
from gateway.types import Handoff, Request


def test_real_mooncake_request_params_share_transfer_id() -> None:
    producer = producer_params("req-1", xfer_id="xfer-unique")
    consumer = consumer_params("req-1", "engine-0", xfer_id="xfer-unique")
    assert producer["transfer_id"] == consumer["transfer_id"] == "xfer-unique"
    assert producer["do_remote_decode"] is True
    assert consumer["do_remote_prefill"] is True
    assert consumer["remote_engine_id"] == "engine-0"


def test_real_handoff_records_hop_only_after_consumer_accepts(monkeypatch) -> None:
    class Worker:
        def __init__(self, fail=False):
            self.id = "worker"
            self.fail = fail
            self.failed = 0
            self.succeeded = 0
        def enqueue(self, req, phase="both", **kwargs):
            if self.fail and phase == "decode":
                raise RuntimeError("consumer rejected KV")
            return {"id": req.id}
        def report_transfer_failure(self): self.failed += 1
        def report_transfer_success(self): self.succeeded += 1

    req = Request("r1", 0.0, 1, 32, 4, "p", 30.0, "lab")
    prefill = Worker()
    decode = Worker(fail=True)
    monkeypatch.setenv("MOONCAKE_REAL", "1")
    monkeypatch.setattr("gateway.queue.producer_engine_id", lambda: "engine-0")
    try:
        enqueue(Handoff(prefill, decode), req)
    except KVTransferError:
        pass
    else:
        raise AssertionError("expected KVTransferError")
    assert prefill.failed == 1 and decode.failed == 1
