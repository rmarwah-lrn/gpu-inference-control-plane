from __future__ import annotations

import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROLES = (
    ("researcher", "List two facts about GPU KV cache in one sentence."),
    ("writer", "Rewrite this for a slide: prefix cache should stick to a replica."),
    ("reviewer", "Approve or reject: shedding a 429 must stay local, never overflow."),
    ("planner", "One sentence: when should KEDA add a decode replica?"),
)

def _base() -> str:
    return (os.environ.get("ORCH_URL") or "http://127.0.0.1:8080/v1").rstrip("/")

def _pressure_prompt(role: str, prompt: str) -> str:
    """Build an optional large, varied context for KV-pressure experiments.

    The gateway estimates four characters per token and enforces an 8,000
    character prompt guard, so the default crew workload is unchanged and the
    pressure mode stays below that guard by construction.
    """
    target_tokens = int(os.environ.get("LOAD_PROMPT_TOKENS", "0"))
    if target_tokens <= 0:
        return f"[{role}] {prompt}"
    target_chars = min(max(256, target_tokens * 4), 7_600)
    parts = [f"[{role}] {prompt} pressure-context:"]
    size = len(parts[0])
    i = 0
    while size < target_chars:
        part = (
            f" segment-{i:05d} retrieval-observation-{i % 97:02d} "
            "replica-affinity decode-scheduling kv-residency"
        )
        parts.append(part)
        size += len(part)
        i += 1
    return "".join(parts)[:target_chars]

def _chat_once(role: str, prompt: str) -> str:
    payload = {
        "model": "text",
        "messages": [{"role": "user", "content": _pressure_prompt(role, prompt)}],
        "max_tokens": int(os.environ.get("LOAD_MAX_TOKENS", "32")),
        "tenant": "crew",
    }
    req = urllib.request.Request(
        f"{_base()}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode())
    choices = body.get("choices") or []
    if not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    msg = first.get("message") if isinstance(first.get("message"), dict) else {}
    return str(msg.get("content") or first.get("text") or "").strip()

def _run_threads(rounds: int, workers: int) -> list[str]:
    jobs = [ROLES[i % len(ROLES)] for i in range(rounds)]
    texts: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_chat_once, role, prompt) for role, prompt in jobs]
        for fut in as_completed(futs):
            texts.append(fut.result())
    return texts

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crew-style flood for class10 dashboards")
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    texts = _run_threads(args.rounds, args.workers)
    print(f"crew_flood rounds={len(texts)} workers={args.workers} model=text")
    for line in texts[:4]:
        if line:
            print(f"  TEXT {line[:160]}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
