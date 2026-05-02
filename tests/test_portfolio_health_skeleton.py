"""
Skeleton test for the Portfolio Health agent.

Wire your agent import and remove the skip decorators.
"""
import pytest
import pandas as pd
from src.agents.portfolio_health import run


def test_portfolio_health_does_not_crash_on_empty_portfolio(load_user, mock_llm):
    """
    user_004 has no positions. Agent must not crash.
    """
    user = load_user("usr_004")
    response = run(user, llm=mock_llm)  # noqa: F821

    assert response is not None
    assert "disclaimer" in response


def test_portfolio_health_flags_concentration(load_user, mock_llm, monkeypatch):
    """
    user_003 has ~60% in NVDA. Agent must surface this.
    """
    class FakeTicker:
        def __init__(self, ticker: str):
            self.ticker = ticker

        def history(self, period="5d", auto_adjust=True):
            prices = {
                "NVDA": 1200.0,
                "VTI": 220.0,
                "VXUS": 58.0,
                "BND": 73.0,
                "AAPL": 180.0,
                "SPY": 500.0,
            }
            p = prices.get(self.ticker, 100.0)
            return pd.DataFrame({"Close": [p * 0.9, p]})

    monkeypatch.setattr("src.agents.portfolio_health.yf.Ticker", FakeTicker)

    user = load_user("usr_003")
    response = run(user, llm=mock_llm)  # noqa: F821

    assert response["concentration_risk"]["flag"] in {"high", "warning"}


def test_portfolio_health_includes_disclaimer(load_user, mock_llm, monkeypatch):
    class FakeTicker:
        def __init__(self, ticker: str):
            self.ticker = ticker

        def history(self, period="5d", auto_adjust=True):
            p = 200.0 if self.ticker != "SPY" else 450.0
            return pd.DataFrame({"Close": [p * 0.95, p]})

    monkeypatch.setattr("src.agents.portfolio_health.yf.Ticker", FakeTicker)

    user = load_user("usr_001")
    response = run(user, llm=mock_llm)  # noqa: F821
    assert response["disclaimer"]
    assert "not investment advice" in response["disclaimer"].lower()
