from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from typing import AsyncGenerator

from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from src.agents.portfolio_health import run as run_portfolio_health
from src.agents.stubs import run_stub
from src.classifier.classifier import classify
from src.data.users import list_user_ids, load_user
from src.memory.session import add_turn, get_history
from src.models import AgentName, ChatRequest, SessionTurn
from src.safety.guard import run_safety_guard

TIMEOUT_SECONDS = float(os.getenv("PIPELINE_TIMEOUT_SECONDS", "15"))
# 0 = disabled. Per-tenant+user sliding window limit for /chat.
RATE_LIMIT_PER_WINDOW = int(os.getenv("RATE_LIMIT_PER_WINDOW", "0"))
RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

_rate_buckets: dict[tuple[str | None, str], deque[float]] = defaultdict(deque)


def _reset_rate_limits_for_tests() -> None:
    _rate_buckets.clear()


def chunk_string(text: str, size: int = 100) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _rate_limit_exceeded(tenant_id: str | None, user_id: str) -> bool:
    if RATE_LIMIT_PER_WINDOW <= 0:
        return False
    now = time.monotonic()
    key = (tenant_id, user_id)
    bucket = _rate_buckets[key]
    while bucket and (now - bucket[0]) > RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_WINDOW:
        return True
    bucket.append(now)
    return False


app = FastAPI(title="Valura AI")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/users")
def users() -> dict[str, list[str]]:
    return {"user_ids": list_user_ids()}


@app.post("/chat")
async def chat(request: ChatRequest):
    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        try:
            if _rate_limit_exceeded(request.tenant_id, request.user_id):
                yield {
                    "data": json.dumps(
                        {
                            "type": "error",
                            "message": "Rate limit exceeded. Please retry shortly.",
                            "code": "rate_limited",
                        }
                    )
                }
                return
            safety = run_safety_guard(request.query)
            if safety.blocked:
                yield {
                    "data": json.dumps(
                        {
                            "type": "safety_block",
                            "category": safety.category.value,
                            "message": safety.message,
                        }
                    )
                }
                return

            async def pipeline():
                history = get_history(request.session_id)
                classifier_output = classify(
                    request.query,
                    history=history,
                    session_id=request.session_id,
                    tenant_id=request.tenant_id,
                )
                try:
                    user = load_user(request.user_id)
                except FileNotFoundError:
                    yield {
                        "data": json.dumps(
                            {
                                "type": "error",
                                "message": "User not found",
                                "code": "user_not_found",
                            }
                        )
                    }
                    return

                if classifier_output.target_agent == AgentName.PORTFOLIO_HEALTH:
                    result = run_portfolio_health(user)
                else:
                    result = run_stub(classifier_output).model_dump()

                result_json = json.dumps(result, indent=2)
                for chunk in chunk_string(result_json, size=120):
                    yield {"data": json.dumps({"type": "token", "content": chunk, "done": False})}
                    await asyncio.sleep(0)
                yield {"data": json.dumps({"type": "token", "content": "", "done": True})}

                add_turn(
                    request.session_id,
                    SessionTurn(user=request.query, assistant=result_json, entities=classifier_output.entities),
                )

            async with asyncio.timeout(TIMEOUT_SECONDS):
                async for event in pipeline():
                    yield event
        except TimeoutError:
            yield {"data": json.dumps({"type": "error", "message": "Request timed out.", "code": "timeout"})}
        except Exception as exc:  # pragma: no cover
            yield {"data": json.dumps({"type": "error", "message": str(exc), "code": "pipeline_error"})}

    return EventSourceResponse(event_generator())
