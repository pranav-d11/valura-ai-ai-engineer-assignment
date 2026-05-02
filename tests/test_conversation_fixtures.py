import json
from pathlib import Path

from src.classifier import classify
from src.models import ExtractedEntities, SessionTurn


FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "conversations"
AGENT_ALIAS = {"portfolio_query": "portfolio_health"}


def _load_cases(name: str) -> list[dict]:
    with open(FIXTURES_ROOT / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)["test_cases"]


def test_conversation_fixture_routing_alignment():
    case_sets = ["follow_up_session", "multi_intent_session", "ambiguous_session"]
    for case_set in case_sets:
        for case in _load_cases(case_set):
            history = [
                SessionTurn(user=turn, assistant="", entities=ExtractedEntities())
                for turn in case.get("prior_user_turns", [])
            ]
            result = classify(case["current_user_turn"], history=history)
            expected = AGENT_ALIAS.get(case["expected"]["agent"], case["expected"]["agent"])
            assert result.agent == expected, f"{case_set}:{case['case_id']} expected {expected}, got {result.agent}"
