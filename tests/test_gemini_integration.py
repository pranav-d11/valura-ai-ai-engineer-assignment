"""Tests for Gemini API integration."""

import json
import os
from unittest.mock import Mock, patch

import pytest

from src.classifier.classifier import _classify_with_gemini, classify
from src.models import AgentName, SafetyCategory, SessionTurn


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response."""
    return {
        "intent": "portfolio_health_query",
        "entities": {
            "tickers": ["AAPL", "GOOGL"],
            "amount": 1000.0,
            "currency": "USD"
        },
        "target_agent": "portfolio_health",
        "safety_verdict": "clean",
        "confidence": 0.95
    }


@patch("src.classifier.classifier.genai")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-2.5-flash"})
def test_classify_with_gemini_success(mock_genai, mock_gemini_response):
    """Test successful classification with Gemini API."""
    # Mock the Gemini response
    mock_model = Mock()
    mock_response = Mock()
    mock_response.text = json.dumps(mock_gemini_response)
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model
    
    result = _classify_with_gemini("How is my portfolio doing?", None)
    
    assert result.intent == "portfolio_health_query"
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH
    assert result.safety_verdict == SafetyCategory.CLEAN
    assert result.confidence == 0.95
    assert result.entities.tickers == ["AAPL", "GOOGL"]
    assert result.entities.amount == 1000.0
    assert result.entities.currency == "USD"


@patch("src.classifier.classifier.genai")
def test_classify_with_gemini_no_api_key(mock_genai):
    """Test fallback when no Gemini API key is provided."""
    # Ensure no API key
    with patch.dict(os.environ, {}, clear=True):
        result = _classify_with_gemini("How is my portfolio doing?", None)
    
    # Should fall back to heuristics
    assert result.intent == "portfolio_health_query"
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH
    assert result.confidence == 0.7  # Default heuristic confidence


@patch("src.classifier.classifier.genai")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
def test_classify_with_gemini_exception(mock_genai):
    """Test fallback when Gemini API throws an exception."""
    mock_genai.configure.side_effect = Exception("API Error")
    
    result = _classify_with_gemini("How is my portfolio doing?", None)
    
    # Should fall back to heuristics
    assert result.intent == "portfolio_health_query"
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH
    assert result.confidence == 0.7


@patch("src.classifier.classifier.genai")
@patch("src.classifier.classifier.OpenAI")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "OPENAI_API_KEY": ""})
def test_classify_prefers_gemini_over_openai(mock_openai, mock_genai, mock_gemini_response):
    """Test that classify() prefers Gemini when both are available."""
    # Mock Gemini response
    mock_model = Mock()
    mock_response = Mock()
    mock_response.text = json.dumps(mock_gemini_response)
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model
    
    result = classify("How is my portfolio doing?", session_id="test")
    
    # Should use Gemini, not OpenAI
    mock_openai.assert_not_called()
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH


@patch("src.classifier.classifier.genai")
@patch("src.classifier.classifier.OpenAI")
@patch.dict(os.environ, {"GEMINI_API_KEY": "", "OPENAI_API_KEY": "test-key"})
def test_classify_falls_back_to_openai(mock_openai, mock_genai):
    """Test that classify() falls back to OpenAI when Gemini is not available."""
    # Mock OpenAI response
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = json.dumps({
        "intent": "portfolio_health_query",
        "entities": {},
        "target_agent": "portfolio_health",
        "safety_verdict": "clean",
        "confidence": 0.95
    })
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client
    
    result = classify("How is my portfolio doing?", session_id="test")
    
    # Should use OpenAI
    mock_openai.assert_called_once()
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH


@patch("src.classifier.classifier.genai")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-2.5-flash-lite"})
def test_classify_with_gemini_lite_model(mock_genai, mock_gemini_response):
    """Test classification with Gemini Flash Lite model."""
    # Mock the Gemini response
    mock_model = Mock()
    mock_response = Mock()
    mock_response.text = json.dumps(mock_gemini_response)
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model
    
    result = _classify_with_gemini("How is my portfolio doing?", None)
    
    # Verify the correct model was used
    mock_genai.GenerativeModel.assert_called_once_with("gemini-2.5-flash-lite")
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH


@patch("src.classifier.classifier.genai")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
def test_classify_with_history(mock_genai, mock_gemini_response):
    """Test classification with conversation history."""
    # Mock the Gemini response
    mock_model = Mock()
    mock_response = Mock()
    mock_response.text = json.dumps(mock_gemini_response)
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model
    
    # Create mock history
    history = [
        SessionTurn(
            user="What's in my portfolio?",
            assistant="You have AAPL and GOOGL stocks.",
            entities={"tickers": ["AAPL", "GOOGL"]}
        )
    ]
    
    result = _classify_with_gemini("How are they doing?", history)
    
    # Verify history was included in the prompt
    call_args = mock_model.generate_content.call_args[0][0]
    assert "Previous query: What's in my portfolio?" in call_args
    assert "Previous entities:" in call_args
    assert "Current query: How are they doing?" in call_args
    
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH
