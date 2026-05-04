"""Tests for MiniMax API integration through OpenRouter."""

import json
import os
from unittest.mock import Mock, patch

import pytest

from src.classifier.classifier import _classify_with_openrouter, classify
from src.models import AgentName, SafetyCategory, SessionTurn


@pytest.fixture
def mock_minimax_response():
    """Mock MiniMax API response through OpenRouter."""
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


@patch("src.classifier.classifier.OpenAI")
@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MINIMAX_MODEL": "minimax/minimax-m2.5:free"})
def test_classify_with_openrouter_minimax_success(mock_openai, mock_minimax_response):
    """Test successful classification with MiniMax through OpenRouter."""
    # Mock the OpenRouter response
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = json.dumps(mock_minimax_response)
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client
    
    result = _classify_with_openrouter("How is my portfolio doing?", None, "minimax/minimax-m2.5:free")
    
    assert result.intent == "portfolio_health_query"
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH
    assert result.safety_verdict == SafetyCategory.CLEAN
    assert result.confidence == 0.95
    assert result.entities.tickers == ["AAPL", "GOOGL"]
    assert result.entities.amount == 1000.0
    assert result.entities.currency == "USD"
    
    # Verify OpenRouter was called with correct model
    mock_client.chat.completions.create.assert_called_once_with(
        model="minimax/minimax-m2.5:free",
        temperature=0,
        messages=[
            {"role": "system", "content": mock.ANY},
            {"role": "user", "content": mock.ANY}
        ]
    )


@patch("src.classifier.classifier.OpenAI")
def test_classify_with_openrouter_no_api_key(mock_openai):
    """Test fallback when no OpenRouter API key is provided."""
    # Ensure no API key
    with patch.dict(os.environ, {}, clear=True):
        result = _classify_with_openrouter("How is my portfolio doing?", None, "minimax/minimax-m2.5:free")
    
    # Should fall back to heuristics
    assert result.intent == "portfolio_health_query"
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH
    assert result.confidence == 0.7  # Default heuristic confidence


@patch("src.classifier.classifier.OpenAI")
@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
def test_classify_with_openrouter_exception(mock_openai):
    """Test fallback when OpenRouter API throws an exception."""
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    mock_openai.return_value = mock_client
    
    result = _classify_with_openrouter("How is my portfolio doing?", None, "minimax/minimax-m2.5:free")
    
    # Should fall back to heuristics
    assert result.intent == "portfolio_health_query"
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH
    assert result.confidence == 0.7


@patch("src.classifier.classifier.OpenAI")
@patch("src.classifier.classifier.genai")
@patch.dict(os.environ, {
    "OPENROUTER_API_KEY": "test-key", 
    "OPENROUTER_MINIMAX_MODEL": "minimax/minimax-m2.5:free",
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": ""
})
def test_classify_prefers_minimax_over_others(mock_genai, mock_openai, mock_minimax_response):
    """Test that classify() prefers MiniMax when OpenRouter is available and others are not."""
    # Mock MiniMax response
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = json.dumps(mock_minimax_response)
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client
    
    result = classify("How is my portfolio doing?", session_id="test")
    
    # Should use MiniMax through OpenRouter
    mock_openai.assert_called_once()
    mock_genai.assert_not_called()
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH


@patch("src.classifier.classifier.OpenAI")
@patch("src.classifier.classifier.genai")
@patch.dict(os.environ, {
    "OPENROUTER_API_KEY": "test-key",
    "OPENROUTER_MINIMAX_MODEL": "minimax/minimax-m2.5:free",
    "GEMINI_API_KEY": "test-key"
})
def test_classify_prefers_gemini_over_minimax(mock_genai, mock_openai, mock_minimax_response):
    """Test that classify() prefers Gemini over MiniMax when both are available."""
    # Mock Gemini response
    mock_model = Mock()
    mock_response = Mock()
    mock_response.text = json.dumps(mock_minimax_response)
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model
    
    result = classify("How is my portfolio doing?", session_id="test")
    
    # Should use Gemini, not MiniMax
    mock_genai.assert_called()
    mock_openai.assert_not_called()
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH


@patch("src.classifier.classifier.OpenAI")
@patch.dict(os.environ, {
    "OPENROUTER_API_KEY": "test-key", 
    "OPENROUTER_MINIMAX_MODEL": "minimax/minimax-m2.7",
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": ""
})
def test_classify_with_minimax_m27(mock_openai, mock_minimax_response):
    """Test classification with MiniMax M2.7 model."""
    # Mock the MiniMax response
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = json.dumps(mock_minimax_response)
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client
    
    result = _classify_with_openrouter("How is my portfolio doing?", None, "minimax/minimax-m2.7")
    
    # Verify the correct model was used
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]["model"] == "minimax/minimax-m2.7"
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH


@patch("src.classifier.classifier.OpenAI")
@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
def test_classify_with_history(mock_openai, mock_minimax_response):
    """Test classification with conversation history."""
    # Mock the MiniMax response
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = json.dumps(mock_minimax_response)
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client
    
    # Create mock history
    history = [
        SessionTurn(
            user="What's in my portfolio?",
            assistant="You have AAPL and GOOGL stocks.",
            entities={"tickers": ["AAPL", "GOOGL"]}
        )
    ]
    
    result = _classify_with_openrouter("How are they doing?", history, "minimax/minimax-m2.5:free")
    
    # Verify history was included in the prompt
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    
    # Check that user prompt contains history
    user_message = None
    for msg in messages:
        if msg["role"] == "user":
            user_message = msg["content"]
            break
    
    assert user_message is not None
    assert "Previous query: What's in my portfolio?" in user_message
    assert "Previous entities:" in user_message
    assert "Current query: How are they doing?" in user_message
    
    assert result.target_agent == AgentName.PORTFOLIO_HEALTH
