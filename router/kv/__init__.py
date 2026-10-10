"""KV-cache control-plane bookkeeping and policy models.

This package deliberately does not own vLLM's GPU KV blocks. It contains
router-side metadata, transfer accounting, and the eviction policy model.
"""

from router.kv.kvbus import KVBus
from router.kv.events import KVEvent, KVEventCollector
from router.kv.event_bridge import KVEventBridge

__all__ = ["KVBus", "KVEvent", "KVEventCollector", "KVEventBridge"]
