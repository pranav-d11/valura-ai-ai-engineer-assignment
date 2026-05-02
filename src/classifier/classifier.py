from __future__ import annotations

import json
import os
import re
import time
from typing import Callable

from src.models import AgentName, ClassifierOutput, ExtractedEntities, SafetyCategory, SessionTurn

MODEL_DEV = "gpt-4o-mini"
MODEL_EVAL = "gpt-4.1"
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("OPENAI_MODEL", MODEL_DEV))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CACHE_TTL_SECONDS = int(os.getenv("CLASSIFIER_CACHE_TTL_SECONDS", "600"))
CACHE_MAX_ITEMS = int(os.getenv("CLASSIFIER_CACHE_MAX_ITEMS", "500"))
TENANT_MODEL_OVERRIDES_RAW = os.getenv("TENANT_MODEL_OVERRIDES", "{}")
DEFAULT_TENANT_MODEL = os.getenv("DEFAULT_TENANT_MODEL", MODEL_NAME)
# Heuristic pre-classifier: skip LLM when confidence is at/above this threshold (0 disables).
PRE_CLASSIFIER_MIN_CONFIDENCE = float(os.getenv("PRE_CLASSIFIER_MIN_CONFIDENCE", "0.92"))

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    TENANT_MODEL_OVERRIDES = json.loads(TENANT_MODEL_OVERRIDES_RAW)
except Exception:
    TENANT_MODEL_OVERRIDES = {}

_classifier_cache: dict[tuple[str, str], tuple[float, ClassifierOutput]] = {}


SYSTEM_PROMPT = """You are the intent classifier for Valura, an AI wealth management platform.

Analyze the user's query and return valid JSON with:
- intent: short snake_case string
- entities: object with keys [tickers, amount, currency, rate, period_years, frequency, horizon, time_period, topics, sectors, index, action, goal]
- target_agent: one of [portfolio_health, market_research, investment_strategy, financial_planning, financial_calculator, risk_assessment, product_recommendation, predictive_analysis, customer_support, general_query]
- safety_verdict: one of [clean, insider_trading, market_manipulation, guaranteed_returns, money_laundering, reckless_advice, sanctions_evasion, fraud]
- confidence: float in [0.0, 1.0]

Routing guidance:
- portfolio_health: portfolio review, diversification, concentration, benchmark checks
- market_research: stock/sector/index/market status and factual market updates
- investment_strategy: buy/sell/rebalance/hedge allocation guidance
- financial_planning: retirement/goal/savings plans
- financial_calculator: deterministic numeric calculations (FV, mortgage, conversion, tax math)
- risk_assessment: risk metrics, drawdown, stress/downside analysis
- product_recommendation: asking for specific product/fund recommendations
- predictive_analysis: forecasting / where will X be
- customer_support: platform/account/help flows
- general_query: greetings/education/ambiguous chat

Return only JSON. No prose. No markdown."""

TICKER_SYNONYMS = {
    "nvidia": "NVDA",
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "microsfot": "MSFT",
    "barclays": "BARC.L",
    "hsbc": "HSBA.L",
    "nikkei": "NIKKEI 225",
}


def _extract_tickers_from_text(text: str) -> list[str]:
    tickers: list[str] = []
    for token in re.findall(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,3})?\b", text):
        ticker = token.upper()
        if ticker in {"USD", "EUR", "GBP", "JPY"}:
            continue
        if ticker not in tickers:
            tickers.append(ticker)
    lowered = text.lower()
    for name, ticker in TICKER_SYNONYMS.items():
        if name in lowered and " " not in ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def _extract_entities(query: str, history: list[SessionTurn] | None = None) -> ExtractedEntities:
    text = query.lower()
    entities = ExtractedEntities()

    entities.tickers.extend(_extract_tickers_from_text(query))

    for name, ticker in TICKER_SYNONYMS.items():
        if name in text:
            if " " in ticker:
                entities.index = ticker
            elif ticker not in entities.tickers:
                entities.tickers.append(ticker)

    if history and not entities.tickers:
        if any(phrase in text for phrase in ["how much do i own", "compare them", "what about", "and ", "it", "them"]):
            recent_mentions: list[str] = []
            for turn in history[-3:]:
                for ticker in _extract_tickers_from_text(turn.user):
                    if ticker not in recent_mentions:
                        recent_mentions.append(ticker)
            if "compare them" in text and len(recent_mentions) >= 2:
                entities.tickers = recent_mentions[-2:]
            elif recent_mentions:
                entities.tickers = [recent_mentions[-1]]

    amount_match = re.search(r"(\d[\d,]*(?:\.\d+)?)(?:\s*(k|m)\b)?", text)
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

    if "monthly" in text or "a month" in text or "per month" in text:
        entities.frequency = "monthly"
    elif "weekly" in text:
        entities.frequency = "weekly"
    elif "yearly" in text or "annual" in text:
        entities.frequency = "yearly"

    if re.search(r"\b6\s+months?\b", text):
        entities.horizon = "6_months"
    elif re.search(r"\b1\s+year\b", text):
        entities.horizon = "1_year"
    elif re.search(r"\b5\s+years?\b", text):
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


def _heuristic_agent(query: str, history: list[SessionTurn] | None = None) -> AgentName:
    q = query.lower()
    if any(k in q for k in ["that thing you mentioned earlier", "thing you mentioned earlier"]):
        return AgentName.GENERAL_QUERY
    if any(k in q for k in ["how is my portfolio doing", "health check", "concentration risk", "portfolio summary"]):
        return AgentName.PORTFOLIO_HEALTH
    if any(k in q for k in ["how much do i own", "do i own"]):
        return AgentName.PORTFOLIO_HEALTH
    if any(k in q for k in ["predict", "where will", "forecast"]):
        return AgentName.PREDICTIVE_ANALYSIS
    if any(k in q for k in ["beta", "drawdown", "stress test", "exposed", "downside risk"]):
        return AgentName.RISK_ASSESSMENT
    if any(k in q for k in ["retire", "retirement", "college fund", "save for a house", "fire plan"]):
        return AgentName.FINANCIAL_PLANNING
    if any(k in q for k in ["recommend", "which fund", "best low-cost"]):
        return AgentName.PRODUCT_RECOMMENDATION
    if any(k in q for k in ["calculate", "future value", "convert", "mortgage", "what will i have", "tax", "capital gains"]):
        return AgentName.FINANCIAL_CALCULATOR
    if any(k in q for k in ["price", "news", "markets today", "compare", "tell me about", "how is the ftse", "nikkei", "top gainers", "eur/usd", "the markets", "what about"]):
        return AgentName.MARKET_RESEARCH
    if any(k in q for k in ["how is ", "hows "]) and any(k in q for k in ["tesla", "apple", "nvidia", "asml", "ftse", "nikkei", "microsoft", "microsfot"]):
        return AgentName.MARKET_RESEARCH
    if "?" in q and any(k in q for k in ["apple", "nvidia", "microsoft", "microsfot", "amd", "asml", "tesla"]):
        return AgentName.MARKET_RESEARCH
    if re.search(r"\b\d+(\.\d+)?\b", q) and ("monthly" in q or "weekly" in q or "year" in q):
        return AgentName.FINANCIAL_CALCULATOR
    if any(k in q for k in ["should i", "rebalance", "good time to invest", "equity-bond split", "hedge"]):
        return AgentName.INVESTMENT_STRATEGY
    if any(k in q for k in ["login", "linked bank", "transaction history", "didn't go through"]):
        return AgentName.CUSTOMER_SUPPORT
    if any(k in q for k in ["portfolio", "diversified", "concentration risk", "am i beating the market", "health check", "holdings"]):
        return AgentName.PORTFOLIO_HEALTH
    stripped = query.strip()
    if re.fullmatch(r"[A-Za-z]{1,5}(?:\.[A-Za-z]{1,3})?", stripped) and ("." in stripped or any(ch.isupper() for ch in stripped)):
        return AgentName.MARKET_RESEARCH
    if history and any(k in q for k in ["what about", "compare them"]):
        return AgentName.MARKET_RESEARCH
    return AgentName.GENERAL_QUERY


def _heuristic_confidence(agent: AgentName, query: str, history: list[SessionTurn] | None) -> float:
    """
    Lightweight confidence score for heuristic routing (not probabilistic).
    Used only to decide whether we can skip an LLM call safely.
    """
    q = query.lower().strip()
    if not q:
        return 0.2
    if agent == AgentName.GENERAL_QUERY and len(q) <= 12 and q.isalpha():
        return 0.95
    if agent == AgentName.PORTFOLIO_HEALTH and any(
        k in q for k in ["how is my portfolio doing", "health check", "concentration risk", "portfolio summary"]
    ):
        return 0.95
    if agent == AgentName.MARKET_RESEARCH and re.fullmatch(r"[a-z]{1,5}(?:\.[a-z]{1,3})?", q):
        return 0.93
    if agent == AgentName.FINANCIAL_CALCULATOR and re.search(r"\d", q) and (
        "monthly" in q or "weekly" in q or "year" in q or "mortgage" in q or "future value" in q
    ):
        return 0.93
    if history and any(k in q for k in ["what about", "compare them", "how much do i own"]):
        return 0.9
    if agent != AgentName.GENERAL_QUERY:
        return 0.85
    return 0.55


def _intent_for_agent(agent: AgentName) -> str:
    return f"{agent.value}_query"


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def _cache_get(session_id: str | None, query: str) -> ClassifierOutput | None:
    if not session_id:
        return None
    key = (session_id, _normalize_query(query))
    row = _classifier_cache.get(key)
    if not row:
        return None
    ts, result = row
    if (time.time() - ts) > CACHE_TTL_SECONDS:
        _classifier_cache.pop(key, None)
        return None
    return result


def _cache_set(session_id: str | None, query: str, result: ClassifierOutput) -> None:
    if not session_id:
        return
    if len(_classifier_cache) >= CACHE_MAX_ITEMS:
        oldest_key = min(_classifier_cache.items(), key=lambda item: item[1][0])[0]
        _classifier_cache.pop(oldest_key, None)
    _classifier_cache[(session_id, _normalize_query(query))] = (time.time(), result)


def _resolve_model_for_tenant(tenant_id: str | None) -> str:
    if tenant_id and isinstance(TENANT_MODEL_OVERRIDES, dict):
        candidate = TENANT_MODEL_OVERRIDES.get(tenant_id)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return DEFAULT_TENANT_MODEL


def _fallback_from_heuristics(query: str) -> ClassifierOutput:
    entities = _extract_entities(query, history=None)
    agent = _heuristic_agent(query, history=None)
    return ClassifierOutput(
        intent=_intent_for_agent(agent),
        entities=entities,
        target_agent=agent,
        safety_verdict=SafetyCategory.CLEAN,
        confidence=0.7,
    )


def _build_user_prompt(query: str, history: list[SessionTurn] | None) -> str:
    if history:
        last = history[-1]
        return (
            f"Previous query: {last.user}\n"
            f"Previous entities: {last.entities.model_dump()}\n\n"
            f"Current query: {query}"
        )
    return query


def _classify_with_openai(query: str, history: list[SessionTurn] | None, tenant_id: str | None = None) -> ClassifierOutput:
    if OpenAI is None or not OPENAI_API_KEY:
        return _fallback_from_heuristics(query)

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_prompt = _build_user_prompt(query, history)

    completion = client.chat.completions.create(
        model=_resolve_model_for_tenant(tenant_id),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content or "{}"
    payload = json.loads(content)
    return ClassifierOutput(
        intent=payload.get("intent", "unknown"),
        entities=ExtractedEntities(**payload.get("entities", {})),
        target_agent=AgentName(payload.get("target_agent", "general_query")),
        safety_verdict=SafetyCategory(payload.get("safety_verdict", "clean")),
        confidence=float(payload.get("confidence", 0.0)),
    )


def classify(
    query: str,
    history: list[SessionTurn] | None = None,
    llm: Callable | None = None,
    session_id: str | None = None,
    tenant_id: str | None = None,
) -> ClassifierOutput:
    try:
        cached = _cache_get(session_id, query)
        if cached is not None:
            return cached

        if llm is not None:
            raw = llm(query=query, history=history or [])
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict) and "target_agent" in raw:
                result = ClassifierOutput(
                    intent=raw.get("intent", "unknown"),
                    entities=ExtractedEntities(**raw.get("entities", {})),
                    target_agent=AgentName(raw["target_agent"]),
                    safety_verdict=SafetyCategory(raw.get("safety_verdict", "clean")),
                    confidence=float(raw.get("confidence", 0.0)),
                )
                _cache_set(session_id, query, result)
                return result
        if llm is None:
            entities = _extract_entities(query, history=history)
            agent = _heuristic_agent(query, history=history)
            h_conf = _heuristic_confidence(agent, query, history)
            if PRE_CLASSIFIER_MIN_CONFIDENCE > 0 and h_conf >= PRE_CLASSIFIER_MIN_CONFIDENCE and OPENAI_API_KEY:
                result = ClassifierOutput(
                    intent=_intent_for_agent(agent),
                    entities=entities,
                    target_agent=agent,
                    safety_verdict=SafetyCategory.CLEAN,
                    confidence=h_conf,
                )
                _cache_set(session_id, query, result)
                return result
            if OpenAI is None or not OPENAI_API_KEY:
                result = ClassifierOutput(
                    intent=_intent_for_agent(agent),
                    entities=entities,
                    target_agent=agent,
                    safety_verdict=SafetyCategory.CLEAN,
                    confidence=h_conf,
                )
                _cache_set(session_id, query, result)
                return result
        result = _classify_with_openai(query, history, tenant_id=tenant_id)
        _cache_set(session_id, query, result)
        return result
    except Exception:
        return ClassifierOutput(
            intent="unknown",
            entities=ExtractedEntities(),
            target_agent=AgentName.CUSTOMER_SUPPORT,
            safety_verdict=SafetyCategory.CLEAN,
            confidence=0.0,
        )
