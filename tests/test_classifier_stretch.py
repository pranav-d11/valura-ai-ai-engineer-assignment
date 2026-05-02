from unittest.mock import patch

from src.classifier import classifier as clf


def test_intra_session_identical_query_dedupe_cache():
    calls = {"count": 0}

    def fake_llm(query, history=None):
        calls["count"] += 1
        return {
            "intent": "market_research_query",
            "entities": {"tickers": ["AAPL"]},
            "target_agent": "market_research",
            "safety_verdict": "clean",
            "confidence": 0.9,
        }

    first = clf.classify("price of apple", llm=fake_llm, session_id="sess-cache-1")
    second = clf.classify("price of apple", llm=fake_llm, session_id="sess-cache-1")

    assert first.agent == "market_research"
    assert second.agent == "market_research"
    assert calls["count"] == 1


def test_per_tenant_model_override_resolution(monkeypatch):
    monkeypatch.setattr(clf, "DEFAULT_TENANT_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(clf, "TENANT_MODEL_OVERRIDES", {"premium": "gpt-4.1"}, raising=False)

    assert clf._resolve_model_for_tenant("premium") == "gpt-4.1"
    assert clf._resolve_model_for_tenant("free") == "gpt-4o-mini"


def test_preclassifier_skips_llm_when_confident(monkeypatch):
    monkeypatch.setattr(clf, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(clf, "PRE_CLASSIFIER_MIN_CONFIDENCE", 0.9)

    with patch.object(clf, "_classify_with_openai", side_effect=AssertionError("LLM should not be called")):
        result = clf.classify("hi", session_id="sess-pre-1")

    assert result.agent == "general_query"
    assert result.confidence >= 0.9
