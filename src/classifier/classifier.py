from __future__ import annotations

import json
import os
import re
from typing import Callable

from src.models import AgentName, ClassifierOutput, ExtractedEntities, SafetyCategory, SessionTurn

MODEL_DEV = "gpt-4o-mini"
MODEL_EVAL = "gpt-4.1"
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("OPENAI_MODEL", MODEL_DEV))

TICKER_SYNONYMS = {
    "nvidia": "NVDA",
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "barclays": "BARC.L",
    "hsbc": "HSBA.L",
    "nikkei": "NIKKEI 225",
}


def _extract_entities(query: str) -> ExtractedEntities:
    text = query.lower()
    entities = ExtractedEntities()

    for token in re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b", query):
        entities.tickers.append(token.upper())

    for name, ticker in TICKER_SYNONYMS.items():
        if name in text:
            if " " in ticker:
                entities.index = ticker
            elif ticker not in entities.tickers:
                entities.tickers.append(ticker)

    amount_match = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(k|m)?", text)
    if amount_match:
        value = float(amount_match.group(1).replace(",", ""))
        suffix = amount_match.group(2)
        if suffix == "k":
            value *= 1000
        if suffix == "m":
            value *= 1_000_000
        entities.amount = value

    if "%" in text:
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if pct_match:
            entities.rate = float(pct_match.group(1)) / 100

    year_match = re.search(r"(\d+)\s*years?", text)
    if year_match:
        entities.period_years = int(year_match.group(1))

    if "monthly" in text:
        entities.frequency = "monthly"
    elif "weekly" in text:
        entities.frequency = "weekly"
    elif "yearly" in text or "annual" in text:
        entities.frequency = "yearly"

    if "6 months" in text:
        entities.horizon = "6_months"
    elif "1 year" in text:
        entities.horizon = "1_year"
    elif "5 years" in text:
        entities.horizon = "5_years"

    if "today" in text:
        entities.time_period = "today"
    elif "this week" in text:
        entities.time_period = "this_week"
    elif "this month" in text:
        entities.time_period = "this_month"

    for cur in ("USD", "EUR", "GBP", "JPY"):
        if cur.lower() in text:
            entities.currency = cur

    if "s&p 500" in text:
        entities.index = "S&P 500"
    elif "ftse" in text:
        entities.index = "FTSE 100"
    elif "msci world" in text:
        entities.index = "MSCI World"

    for action in ("buy", "sell", "hold", "hedge", "rebalance"):
        if action in text:
            entities.action = action
            break

    if "retire" in text or "retirement" in text:
        entities.goal = "retirement"
    elif "college" in text or "education" in text:
        entities.goal = "education"
    elif "house" in text:
        entities.goal = "house"
    elif "fire plan" in text or " fire " in f" {text} ":
        entities.goal = "FIRE"

    if "technology" in text or "tech" in text:
        entities.sectors.append("technology")

    for topic in (
        "mutual fund",
        "compound interest",
        "index fund",
        "p/e ratio",
        "fx",
        "beta",
        "max drawdown",
        "recession",
        "etf",
        "dividend",
        "login",
        "bank account",
        "transaction history",
        "recurring investment",
        "ltcg",
    ):
        if topic in text:
            entities.topics.append(topic.upper() if topic == "ltcg" else topic)

    return entities


def _heuristic_agent(query: str) -> AgentName:
    q = query.lower()
    if any(k in q for k in ["predict", "where will", "forecast"]):
        return AgentName.PREDICTIVE_ANALYSIS
    if any(k in q for k in ["risk", "beta", "drawdown", "stress test", "exposed"]):
        return AgentName.RISK_ASSESSMENT
    if any(k in q for k in ["retire", "retirement", "college fund", "save for a house", "fire plan"]):
        return AgentName.FINANCIAL_PLANNING
    if any(k in q for k in ["recommend", "which fund", "best low-cost"]):
        return AgentName.PRODUCT_RECOMMENDATION
    if any(k in q for k in ["calculate", "future value", "convert", "mortgage", "what will i have", "tax", "capital gains"]):
        return AgentName.FINANCIAL_CALCULATOR
    if any(k in q for k in ["price", "news", "markets today", "compare", "tell me about", "how is the ftse", "nikkei", "top gainers", "how is ", "eur/usd"]):
        return AgentName.MARKET_RESEARCH
    if any(k in q for k in ["should i", "rebalance", "good time to invest", "equity-bond split", "hedge"]):
        return AgentName.INVESTMENT_STRATEGY
    if any(k in q for k in ["login", "linked bank", "transaction history", "didn't go through"]):
        return AgentName.CUSTOMER_SUPPORT
    if any(k in q for k in ["portfolio", "diversified", "concentration risk", "am i beating the market", "health check", "holdings"]):
        return AgentName.PORTFOLIO_HEALTH
    stripped = query.strip()
    if re.fullmatch(r"[A-Za-z]{1,5}(?:\.[A-Za-z]{1,3})?", stripped) and any(ch.isupper() for ch in stripped):
        return AgentName.MARKET_RESEARCH
    return AgentName.GENERAL_QUERY


def _intent_for_agent(agent: AgentName) -> str:
    return f"{agent.value}_query"


def classify(query: str, history: list[SessionTurn] | None = None, llm: Callable | None = None) -> ClassifierOutput:
    try:
        if llm is not None:
            raw = llm(query=query, history=history or [])
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict) and "target_agent" in raw:
                return ClassifierOutput(
                    intent=raw.get("intent", "unknown"),
                    entities=ExtractedEntities(**raw.get("entities", {})),
                    target_agent=AgentName(raw["target_agent"]),
                    safety_verdict=SafetyCategory(raw.get("safety_verdict", "clean")),
                    confidence=float(raw.get("confidence", 0.0)),
                )
    except Exception:
        pass

    entities = _extract_entities(query)
    agent = _heuristic_agent(query)
    return ClassifierOutput(
        intent=_intent_for_agent(agent),
        entities=entities,
        target_agent=agent,
        safety_verdict=SafetyCategory.CLEAN,
        confidence=0.7,
    )
