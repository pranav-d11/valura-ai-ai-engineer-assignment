import json

from fastapi.testclient import TestClient

import src.http.app as http_app_module
from src.http.app import app
from src.models import AgentName, ClassifierOutput, ExtractedEntities, SafetyCategory


def _read_sse_payloads(response) -> list[dict]:
    payloads: list[dict] = []
    for line in response.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        if text.startswith("data: "):
            payloads.append(json.loads(text[6:]))
    return payloads


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_safety_block_sse():
    client = TestClient(app)
    body = {
        "user_id": "usr_001",
        "session_id": "sess-1",
        "query": "i work at apple and know about an unannounced acquisition, when should i buy shares?",
    }
    with client.stream("POST", "/chat", json=body) as response:
        assert response.status_code == 200
        events = _read_sse_payloads(response)

    assert events
    assert events[0]["type"] == "safety_block"
    assert events[0]["category"] == "insider_trading"


def test_chat_routes_and_streams_tokens(monkeypatch):
    client = TestClient(app)

    def fake_classify(query, history=None, **kwargs):
        return ClassifierOutput(
            intent="general_query",
            entities=ExtractedEntities(),
            target_agent=AgentName.GENERAL_QUERY,
            safety_verdict=SafetyCategory.CLEAN,
            confidence=0.9,
        )

    monkeypatch.setattr("src.http.app.classify", fake_classify)

    body = {"user_id": "usr_001", "session_id": "sess-2", "query": "hi"}
    with client.stream("POST", "/chat", json=body) as response:
        assert response.status_code == 200
        events = _read_sse_payloads(response)

    assert any(e.get("type") == "token" and e.get("done") is False for e in events)
    assert any(e.get("type") == "token" and e.get("done") is True for e in events)


def test_chat_rate_limit_sse(monkeypatch):
    monkeypatch.setattr(http_app_module, "RATE_LIMIT_PER_WINDOW", 1)
    monkeypatch.setattr(http_app_module, "RATE_LIMIT_WINDOW_SECONDS", 60.0)
    http_app_module._reset_rate_limits_for_tests()

    client = TestClient(app)
    body = {
        "user_id": "usr_001",
        "tenant_id": "t1",
        "session_id": "sess-rate-1",
        "query": "hi",
    }
    with client.stream("POST", "/chat", json=body) as response:
        assert response.status_code == 200
        first = _read_sse_payloads(response)
    assert first

    body["session_id"] = "sess-rate-2"
    with client.stream("POST", "/chat", json=body) as response:
        assert response.status_code == 200
        second = _read_sse_payloads(response)

    assert second
    assert second[0]["type"] == "error"
    assert second[0]["code"] == "rate_limited"
