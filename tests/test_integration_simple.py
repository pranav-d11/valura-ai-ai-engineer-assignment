"""Simplified integration tests for Gemini and MiniMax APIs."""

import json
from unittest.mock import Mock, ANY, patch

import pytest

from src.classifier.classifier import _classify_with_gemini, _classify_with_openrouter
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


def test_classify_with_gemini_direct_mock(mock_gemini_response):
    """Test Gemini classification with direct mocking."""
    with patch("src.classifier.classifier.genai") as mock_genai:
        # Mock the Gemini response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(mock_gemini_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        
        # Temporarily set the API key for this test
        import src.classifier.classifier as classifier_module
        original_key = classifier_module.GEMINI_API_KEY
        classifier_module.GEMINI_API_KEY = "test-key"
        
        try:
            result = _classify_with_gemini("How is my portfolio doing?", None)
            
            assert result.intent == "portfolio_health_query"
            assert result.target_agent == AgentName.PORTFOLIO_HEALTH
            assert result.safety_verdict == SafetyCategory.CLEAN
            assert result.confidence == 0.95
            assert result.entities.tickers == ["AAPL", "GOOGL"]
            assert result.entities.amount == 1000.0
            assert result.entities.currency == "USD"
        finally:
            # Restore original key
            classifier_module.GEMINI_API_KEY = original_key


def test_classify_with_openrouter_direct_mock(mock_minimax_response):
    """Test OpenRouter classification with direct mocking."""
    with patch("src.classifier.classifier.OpenAI") as mock_openai:
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
        
        # Temporarily set the API key for this test
        import src.classifier.classifier as classifier_module
        original_key = classifier_module.OPENROUTER_API_KEY
        classifier_module.OPENROUTER_API_KEY = "test-key"
        
        try:
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
                    {"role": "system", "content": ANY},
                    {"role": "user", "content": ANY}
                ]
            )
        finally:
            # Restore original key
            classifier_module.OPENROUTER_API_KEY = original_key


def test_classify_with_gemini_no_api_key():
    """Test Gemini fallback when no API key is available."""
    with patch("src.classifier.classifier.genai") as mock_genai:
        # Temporarily clear the API key
        import src.classifier.classifier as classifier_module
        original_key = classifier_module.GEMINI_API_KEY
        classifier_module.GEMINI_API_KEY = None
        
        try:
            result = _classify_with_gemini("How is my portfolio doing?", None)
            
            # Should fall back to heuristics
            assert result.intent == "portfolio_health_query"
            assert result.target_agent == AgentName.PORTFOLIO_HEALTH
            assert result.confidence == 0.7  # Default heuristic confidence
        finally:
            # Restore original key
            classifier_module.GEMINI_API_KEY = original_key


def test_classify_with_openrouter_no_api_key():
    """Test OpenRouter fallback when no API key is available."""
    with patch("src.classifier.classifier.OpenAI") as mock_openai:
        # Temporarily clear the API key
        import src.classifier.classifier as classifier_module
        original_key = classifier_module.OPENROUTER_API_KEY
        classifier_module.OPENROUTER_API_KEY = None
        
        try:
            result = _classify_with_openrouter("How is my portfolio doing?", None, "minimax/minimax-m2.5:free")
            
            # Should fall back to heuristics
            assert result.intent == "portfolio_health_query"
            assert result.target_agent == AgentName.PORTFOLIO_HEALTH
            assert result.confidence == 0.7  # Default heuristic confidence
        finally:
            # Restore original key
            classifier_module.OPENROUTER_API_KEY = original_key


def test_classify_with_gemini_exception_handling(mock_gemini_response):
    """Test Gemini fallback when API throws an exception."""
    with patch("src.classifier.classifier.genai") as mock_genai:
        mock_genai.configure.side_effect = Exception("API Error")
        
        # Temporarily set the API key for this test
        import src.classifier.classifier as classifier_module
        original_key = classifier_module.GEMINI_API_KEY
        classifier_module.GEMINI_API_KEY = "test-key"
        
        try:
            result = _classify_with_gemini("How is my portfolio doing?", None)
            
            # Should fall back to heuristics
            assert result.intent == "portfolio_health_query"
            assert result.target_agent == AgentName.PORTFOLIO_HEALTH
            assert result.confidence == 0.7
        finally:
            # Restore original key
            classifier_module.GEMINI_API_KEY = original_key


def test_classify_with_openrouter_exception_handling(mock_minimax_response):
    """Test OpenRouter fallback when API throws an exception."""
    with patch("src.classifier.classifier.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        # Temporarily set the API key for this test
        import src.classifier.classifier as classifier_module
        original_key = classifier_module.OPENROUTER_API_KEY
        classifier_module.OPENROUTER_API_KEY = "test-key"
        
        try:
            result = _classify_with_openrouter("How is my portfolio doing?", None, "minimax/minimax-m2.5:free")
            
            # Should fall back to heuristics
            assert result.intent == "portfolio_health_query"
            assert result.target_agent == AgentName.PORTFOLIO_HEALTH
            assert result.confidence == 0.7
        finally:
            # Restore original key
            classifier_module.OPENROUTER_API_KEY = original_key


def test_classify_with_gemini_history(mock_gemini_response):
    """Test Gemini classification with conversation history."""
    with patch("src.classifier.classifier.genai") as mock_genai:
        # Mock the Gemini response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(mock_gemini_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        
        # Temporarily set the API key for this test
        import src.classifier.classifier as classifier_module
        original_key = classifier_module.GEMINI_API_KEY
        classifier_module.GEMINI_API_KEY = "test-key"
        
        try:
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
        finally:
            # Restore original key
            classifier_module.GEMINI_API_KEY = original_key


def test_classify_with_openrouter_history(mock_minimax_response):
    """Test OpenRouter classification with conversation history."""
    with patch("src.classifier.classifier.OpenAI") as mock_openai:
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
        
        # Temporarily set the API key for this test
        import src.classifier.classifier as classifier_module
        original_key = classifier_module.OPENROUTER_API_KEY
        classifier_module.OPENROUTER_API_KEY = "test-key"
        
        try:
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
        finally:
            # Restore original key
            classifier_module.OPENROUTER_API_KEY = original_key
