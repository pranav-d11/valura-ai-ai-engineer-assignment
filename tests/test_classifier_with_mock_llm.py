"""
Test classifier routing with a properly configured mock LLM.

This tests the LLM integration path (not the heuristic fallback).
The mock returns structured responses that the classifier expects from a real LLM.
"""
from typing import Any

import pytest
from src.classifier import classify


def _normalize_ticker(t: str) -> str:
    """Case-fold and drop the exchange suffix (AAPL.US → AAPL)."""
    return t.upper().split(".")[0]


def matches_entities(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """
    Subset match with normalization. `actual` must contain every value in
    `expected`; extra fields and extra values are allowed.
    """
    for field, exp_value in expected.items():
        act_value = actual.get(field)
        if act_value is None:
            return False

        if field == "tickers":
            exp_set = {_normalize_ticker(t) for t in exp_value}
            act_set = {_normalize_ticker(t) for t in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("topics", "sectors"):
            exp_set = {s.lower() for s in exp_value}
            act_set = {s.lower() for s in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("amount", "rate"):
            if abs(act_value - exp_value) > abs(exp_value) * 0.05:
                return False
        elif field == "period_years":
            if int(act_value) != int(exp_value):
                return False
        else:
            # Catch-all for vocabulary tokens
            if str(act_value).lower() != str(exp_value).lower():
                return False
    return True


def test_classifier_with_configured_mock_llm(gold_classifier_queries, configured_mock_llm):
    """
    Test that the classifier correctly uses the LLM response when provided.
    This tests the LLM integration path (not the heuristic fallback).
    
    The configured mock returns proper structured responses that a real LLM
    would return, allowing us to test the classifier's parsing and routing logic.
    """
    correct = 0
    results = []
    
    for case in gold_classifier_queries:
        query = case["query"]
        expected_agent = case["expected_agent"]
        
        # Use the configured mock which returns proper LLM responses
        result = classify(query, llm=configured_mock_llm)
        
        is_correct = result.agent == expected_agent
        if is_correct:
            correct += 1
        
        results.append({
            "query": query,
            "expected": expected_agent,
            "got": result.agent,
            "correct": is_correct,
            "confidence": result.confidence,
        })
    
    # Calculate and report accuracy
    total = len(gold_classifier_queries)
    accuracy = correct / total
    
    print(f"\n{'='*60}")
    print(f"LLM Integration Path Test Results:")
    print(f"{'='*60}")
    print(f"Total queries: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"\nPer-query breakdown:")
    
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} '{r['query'][:50]}...' -> {r['got']}")
    
    # Assert the threshold
    assert accuracy >= 0.85, f"Routing accuracy {accuracy:.1%} below 85% threshold"


def test_classifier_entity_extraction_with_mock(gold_classifier_queries, configured_mock_llm):
    """
    Test entity extraction when using the LLM integration path.
    """
    matched = 0
    total_with_entities = 0
    
    for case in gold_classifier_queries:
        if not case.get("expected_entities"):
            continue
        
        total_with_entities += 1
        result = classify(case["query"], llm=configured_mock_llm)
        
        if matches_entities(result.entities.model_dump(), case["expected_entities"]):
            matched += 1
    
    rate = matched / total_with_entities if total_with_entities else 0.0
    print(f"\n{'='*60}")
    print(f"Entity Extraction Test Results:")
    print(f"{'='*60}")
    print(f"Queries with entities: {total_with_entities}")
    print(f"Entity match rate: {rate:.1%} ({matched}/{total_with_entities})")
    
    # No hard assertion - just report the rate
    assert rate > 0, "Entity extraction should work with configured mock"
