# Valura AI Microservice — Product Requirements Document

**Version:** 1.0  
**Purpose:** Coding reference and build specification for AI-assisted implementation  
**Deadline:** 3 days from receipt  

---

## 1. What We Are Building

A FastAPI microservice that acts as an AI co-investor for Valura's wealth management platform. Users send financial queries. The system filters, classifies, routes, and streams back a response — all in real time via Server-Sent Events (SSE).

### End State Vision

A running server where:
1. `POST /chat` accepts a user query + user_id + session_id
2. The safety guard runs synchronously — harmful queries blocked immediately with a professional refusal
3. The classifier makes one LLM call and returns structured JSON: intent, entities, target agent, safety verdict
4. The router sends the query to the right agent
5. The Portfolio Health agent (only fully-implemented agent) fetches live prices, computes metrics, returns structured output
6. All other agents return a structured stub response — never an error
7. The entire response streams back as SSE tokens
8. Tests pass with `pytest tests/ -v` with no API key

---

## 2. Repository Structure

```
valura-ai/
├── src/
│   ├── models.py                    # All Pydantic schemas (shared)
│   ├── safety/
│   │   └── guard.py                 # Rule-based safety filter
│   ├── classifier/
│   │   └── classifier.py            # Single LLM call, structured output
│   ├── agents/
│   │   ├── portfolio_health.py      # Fully implemented agent
│   │   └── stubs.py                 # NotImplemented handler for all other agents
│   ├── memory/
│   │   └── session.py               # In-memory conversation store
│   └── http/
│       └── app.py                   # FastAPI app, SSE endpoint, pipeline wiring
├── tests/
│   ├── conftest.py                  # LLM mock fixtures
│   ├── test_safety.py
│   ├── test_classifier.py
│   ├── test_portfolio_health.py
│   └── test_routing.py
├── fixtures/
│   ├── README.md
│   ├── users/
│   │   ├── user_001_aggressive.json
│   │   ├── user_002_concentrated.json
│   │   ├── user_003_global.json
│   │   ├── user_004_empty.json
│   │   └── user_005_retiree.json
│   └── test_queries/
│       ├── intent_classification.json
│       └── safety_pairs.json
├── .env
├── .env.example
├── requirements.txt
└── README.md
```

---

## 3. Data Fixtures

### 3.1 User Profile Schema

Every user profile JSON must follow this shape:

```json
{
  "user_id": "user_001",
  "name": "Arjun Mehta",
  "risk_profile": "aggressive",
  "kyc_status": "verified",
  "base_currency": "USD",
  "account_inception_date": "2022-01-15",
  "holdings": [
    {
      "ticker": "NVDA",
      "exchange": "NASDAQ",
      "quantity": 100,
      "avg_cost_usd": 400.00,
      "asset_class": "equity"
    }
  ]
}
```

### 3.2 The 5 Users

| user_id | Profile | Key Edge Case |
|---|---|---|
| user_001 | Aggressive trader | High turnover, multiple tickers |
| user_002 | Concentrated single-stock | >60% in one position |
| user_003 | Multi-currency global investor | Non-USD holdings (ASML, Toyota) |
| user_004 | Empty portfolio | `holdings: []` — must not crash |
| user_005 | Dividend-focused retiree | Low risk, income-oriented holdings |

### 3.3 Fixture Files to Create

Create all 5 user JSONs manually under `fixtures/users/`. Use realistic tickers and quantities. `user_004_empty.json` must have `"holdings": []`.

---

## 4. Models (`src/models.py`)

Define all Pydantic v2 schemas here. Every module imports from here — no local schema definitions elsewhere.

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum

# --- Enums ---

class AgentName(str, Enum):
    PORTFOLIO_HEALTH = "portfolio_health"
    MARKET_RESEARCH = "market_research"
    INVESTMENT_STRATEGY = "investment_strategy"
    FINANCIAL_CALCULATOR = "financial_calculator"
    RISK_ANALYSIS = "risk_analysis"
    RECOMMENDATIONS = "recommendations"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    SUPPORT = "support"

class SafetyCategory(str, Enum):
    INSIDER_TRADING = "insider_trading"
    MARKET_MANIPULATION = "market_manipulation"
    GUARANTEED_RETURNS = "guaranteed_returns"
    MONEY_LAUNDERING = "money_laundering"
    RECKLESS_ADVICE = "reckless_advice"
    CLEAN = "clean"

class ObservationSeverity(str, Enum):
    WARNING = "warning"
    INFO = "info"
    CRITICAL = "critical"

# --- Safety ---

class SafetyResult(BaseModel):
    blocked: bool
    category: SafetyCategory
    refusal_message: Optional[str] = None

# --- Classifier ---

class ExtractedEntities(BaseModel):
    tickers: Optional[List[str]] = []
    amounts: Optional[List[float]] = []
    time_periods: Optional[List[str]] = []
    sectors: Optional[List[str]] = []
    topics: Optional[List[str]] = []

class ClassifierOutput(BaseModel):
    intent: str
    entities: ExtractedEntities
    target_agent: AgentName
    safety_verdict: SafetyCategory
    confidence: float = Field(ge=0.0, le=1.0)

# --- Portfolio Health Agent ---

class ConcentrationRisk(BaseModel):
    top_position_pct: float
    top_3_positions_pct: float
    flag: Literal["low", "medium", "high"]

class Performance(BaseModel):
    total_return_pct: float
    annualized_return_pct: float

class BenchmarkComparison(BaseModel):
    benchmark: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float

class Observation(BaseModel):
    severity: ObservationSeverity
    text: str

class HealthCheckOutput(BaseModel):
    mode: Literal["monitor", "build"]   # "build" for empty portfolio
    concentration_risk: Optional[ConcentrationRisk] = None
    performance: Optional[Performance] = None
    benchmark_comparison: Optional[BenchmarkComparison] = None
    observations: List[Observation]
    disclaimer: str

# --- Stub Response ---

class StubResponse(BaseModel):
    intent: str
    entities: ExtractedEntities
    target_agent: AgentName
    message: str = "This agent is not implemented in this build."

# --- HTTP Request/Response ---

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    query: str

class SessionTurn(BaseModel):
    user: str
    assistant: str
    entities: ExtractedEntities

# --- User Profile ---

class Holding(BaseModel):
    ticker: str
    exchange: str
    quantity: float
    avg_cost_usd: float
    asset_class: str

class UserProfile(BaseModel):
    user_id: str
    name: str
    risk_profile: str
    kyc_status: str
    base_currency: str
    account_inception_date: str
    holdings: List[Holding]
```

---

## 5. Safety Guard (`src/safety/guard.py`)

### Rules
- Pure Python. No imports beyond `re` and `src/models.py`
- Must complete in under 10ms for any input
- Each blocked category returns a distinct, professional refusal string
- Educational framing passes through (see edge cases below)

### Categories and Keyword Sets

```
INSIDER_TRADING:
  keywords: ["material non-public", "MNPI", "inside information", "tip from my", 
             "before the announcement", "not yet public", "my contact at"]

MARKET_MANIPULATION:
  keywords: ["pump and dump", "coordinate buying", "wash trade", "artificially inflate",
             "spread false", "fake volume", "move the price"]

GUARANTEED_RETURNS:
  keywords: ["guaranteed return", "100% return", "risk-free profit", "can't lose",
             "cannot lose", "sure profit", "no risk", "zero risk investment",
             "promise returns"]

MONEY_LAUNDERING:
  keywords: ["launder", "clean money", "hide proceeds", "shell company to hide",
             "conceal funds", "dirty money"]

RECKLESS_ADVICE:
  keywords: ["bet everything", "put all money in", "leverage everything",
             "mortgage my house to invest", "take a loan to buy crypto",
             "invest my emergency fund"]
```

### Distinct Refusal Messages per Category

```
INSIDER_TRADING:
  "This request involves acting on material non-public information, which constitutes 
   insider trading under securities law. Valura cannot assist with this."

MARKET_MANIPULATION:
  "This request describes market manipulation, which is illegal under securities regulations. 
   Valura cannot assist with this."

GUARANTEED_RETURNS:
  "No investment can guarantee returns. Valura does not endorse or facilitate 
   any claims of guaranteed or risk-free profits."

MONEY_LAUNDERING:
  "This request appears to involve concealing or laundering funds, which is a 
   serious financial crime. Valura cannot assist with this."

RECKLESS_ADVICE:
  "This request involves a level of financial risk that falls outside what Valura 
   can responsibly recommend. Please consult a certified financial advisor."
```

### Educational Pass-Through Logic

If the query contains `["how does", "what is", "explain", "define", "teach me", "example of", "history of"]` AND a blocked keyword → **pass through**. Do not block.

Example:
- BLOCKED: "I have inside information about AAPL earnings, should I buy?"
- PASSES: "How does insider trading work and why is it illegal?"

Document this tradeoff in README: educational queries on harmful topics pass through; this is intentional.

### Function Signature

```python
def run_safety_guard(query: str) -> SafetyResult:
    ...
```

Returns `SafetyResult(blocked=False, category=SafetyCategory.CLEAN)` if safe.

---

## 6. Session Memory (`src/memory/session.py`)

### Implementation

```python
from collections import defaultdict
from typing import Dict, List
from src.models import SessionTurn

# In-memory store: session_id → list of turns
_store: Dict[str, List[SessionTurn]] = defaultdict(list)

def get_history(session_id: str) -> List[SessionTurn]:
    return _store[session_id]

def add_turn(session_id: str, turn: SessionTurn) -> None:
    _store[session_id].append(turn)
    # Keep last 10 turns only — prevent unbounded growth
    _store[session_id] = _store[session_id][-10:]

def clear_session(session_id: str) -> None:
    _store[session_id] = []
```

### Justification for README
In-memory is sufficient for a demo/evaluation context. In production, replace with Redis (for horizontal scaling) or Postgres (for persistence across restarts). The interface is intentionally thin so swapping the backend is a 1-file change.

---

## 7. Intent Classifier (`src/classifier/classifier.py`)

### One LLM call. Returns `ClassifierOutput`. Never crashes.

### System Prompt Template

```
You are the intent classifier for Valura, an AI wealth management platform.

Your job: analyze the user's query and return a structured JSON with these fields:
- intent: a short snake_case string describing what the user wants
- entities: extracted tickers, amounts, time_periods, sectors, topics
- target_agent: one of [portfolio_health, market_research, investment_strategy, 
  financial_calculator, risk_analysis, recommendations, predictive_analysis, support]
- safety_verdict: one of [clean, insider_trading, market_manipulation, 
  guaranteed_returns, money_laundering, reckless_advice]
- confidence: float between 0.0 and 1.0

Agent routing rules:
- portfolio_health: "how is my portfolio", "health check", "am I diversified", 
  "portfolio performance", "concentration"
- market_research: questions about specific stocks, sectors, market news
- investment_strategy: "what should I buy", "how should I allocate", "asset allocation"
- financial_calculator: calculations — compound interest, returns, projections
- risk_analysis: "how risky is", "volatility", "drawdown", "VaR"
- recommendations: "suggest", "recommend", "what's a good stock"
- predictive_analysis: "will X go up", "forecast", "prediction"
- support: account questions, platform help, anything else

Follow-up handling: if the previous query is provided, resolve pronouns and references.
Example: previous="tell me about Microsoft", current="what about Apple?" → 
treat current as a market_research query about AAPL.

Respond ONLY with valid JSON. No explanation, no markdown.
```

### Follow-up Injection

```python
if history:
    last = history[-1]
    user_message = f"Previous query: {last.user}\nPrevious entities: {last.entities.model_dump()}\n\nCurrent query: {query}"
else:
    user_message = query
```

### Fallback on LLM Failure

```python
except Exception:
    return ClassifierOutput(
        intent="unknown",
        entities=ExtractedEntities(),
        target_agent=AgentName.SUPPORT,
        safety_verdict=SafetyCategory.CLEAN,
        confidence=0.0
    )
```

### Model Config

```python
MODEL_DEV = "gpt-4o-mini"
MODEL_EVAL = "gpt-4.1"
# Use env var MODEL_NAME, default to gpt-4o-mini
```

---

## 8. Portfolio Health Agent (`src/agents/portfolio_health.py`)

### This is the only fully-implemented agent.

### Inputs
- `user_profile: UserProfile` — passed in, not fetched
- `query: str` — user's original question

### Step-by-Step Logic

#### Step 1 — Empty Portfolio Check
```python
if not user_profile.holdings:
    return HealthCheckOutput(
        mode="build",
        observations=[
            Observation(severity=ObservationSeverity.INFO, 
                text="You have no holdings yet. Consider starting with a diversified ETF like VTI or a target-date fund aligned to your retirement horizon."),
            Observation(severity=ObservationSeverity.INFO, 
                text="Before investing, ensure you have 3–6 months of expenses in an emergency fund.")
        ],
        disclaimer=DISCLAIMER
    )
```

#### Step 2 — Fetch Current Prices via yfinance
```python
import yfinance as yf

tickers = [h.ticker for h in user_profile.holdings]
data = yf.download(tickers, period="1d", auto_adjust=True)
current_prices = {t: data["Close"][t].iloc[-1] for t in tickers}
```

Handle yfinance failure gracefully — if a ticker fetch fails, log a warning and skip that holding from calculations. Do not crash.

#### Step 3 — Compute Portfolio Value
```python
total_value = sum(h.quantity * current_prices[h.ticker] for h in holdings_with_price)
total_cost = sum(h.quantity * h.avg_cost_usd for h in holdings_with_price)
```

#### Step 4 — Concentration Risk
```python
position_values = sorted(
    [(h.ticker, h.quantity * current_prices[h.ticker]) for h in holdings_with_price],
    key=lambda x: x[1], reverse=True
)

top_1_pct = position_values[0][1] / total_value * 100
top_3_pct = sum(v for _, v in position_values[:3]) / total_value * 100

flag = "high" if top_1_pct > 40 else "medium" if top_1_pct > 20 else "low"
```

#### Step 5 — Performance
```python
total_return_pct = (total_value - total_cost) / total_cost * 100

# Annualize using account inception date
from datetime import date
days_held = (date.today() - date.fromisoformat(user_profile.account_inception_date)).days
years_held = days_held / 365.25
annualized_return_pct = ((1 + total_return_pct / 100) ** (1 / years_held) - 1) * 100 if years_held > 0 else 0.0
```

#### Step 6 — Benchmark Comparison
```python
# Fetch SPY for the same period
spy = yf.Ticker("SPY")
spy_hist = spy.history(start=user_profile.account_inception_date)
spy_return_pct = (spy_hist["Close"].iloc[-1] - spy_hist["Close"].iloc[0]) / spy_hist["Close"].iloc[0] * 100
alpha_pct = total_return_pct - spy_return_pct
```

#### Step 7 — Observations (Plain Language, Max 3)

Generate 2-3 observations using this priority logic:
1. If `flag == "high"` → WARNING: "X% of your portfolio is in [TICKER]. This is highly concentrated."
2. If `alpha_pct > 0` → INFO: "You are outperforming the S&P 500 by X% over the period."
3. If `alpha_pct < 0` → WARNING: "You are underperforming the S&P 500 by X%."
4. If single asset class dominates → INFO: "Your portfolio is entirely in equities. Consider diversifying."

Keep observations plain language. No jargon without explanation. Novice-first.

#### Step 8 — Disclaimer

```python
DISCLAIMER = (
    "This analysis is generated by an AI system and is for informational purposes only. "
    "It does not constitute financial advice, investment recommendations, or a solicitation "
    "to buy or sell any security. Past performance is not indicative of future results. "
    "Please consult a certified financial advisor before making investment decisions."
)
```

### Final Return Shape

```python
return HealthCheckOutput(
    mode="monitor",
    concentration_risk=ConcentrationRisk(
        top_position_pct=round(top_1_pct, 1),
        top_3_positions_pct=round(top_3_pct, 1),
        flag=flag
    ),
    performance=Performance(
        total_return_pct=round(total_return_pct, 2),
        annualized_return_pct=round(annualized_return_pct, 2)
    ),
    benchmark_comparison=BenchmarkComparison(
        benchmark="S&P 500",
        portfolio_return_pct=round(total_return_pct, 2),
        benchmark_return_pct=round(spy_return_pct, 2),
        alpha_pct=round(alpha_pct, 2)
    ),
    observations=observations,
    disclaimer=DISCLAIMER
)
```

---

## 9. Stub Agents (`src/agents/stubs.py`)

```python
from src.models import StubResponse, ClassifierOutput

def run_stub(classifier_output: ClassifierOutput) -> StubResponse:
    return StubResponse(
        intent=classifier_output.intent,
        entities=classifier_output.entities,
        target_agent=classifier_output.target_agent,
        message=f"The '{classifier_output.target_agent.value}' agent is not implemented in this build."
    )
```

---

## 10. HTTP Layer (`src/http/app.py`)

### Endpoint

```
POST /chat
Content-Type: application/json
Body: { "user_id": "user_001", "session_id": "abc123", "query": "how is my portfolio?" }
```

### Pipeline

```
1. Validate request body (Pydantic auto-validates)
2. Run safety guard (sync, <10ms)
   → If blocked: stream single SSE error event with refusal message, return
3. Load session history
4. Run classifier (async LLM call)
5. Load user profile from fixtures by user_id
6. Route to agent:
   → portfolio_health → run portfolio_health.py
   → everything else → run stubs.py
7. Stream result as SSE
8. Save turn to session memory
```

### SSE Event Format

```
data: {"type": "token", "content": "...", "done": false}\n\n
data: {"type": "token", "content": "", "done": true}\n\n
```

On error:
```
data: {"type": "error", "message": "...", "code": "pipeline_error"}\n\n
```

On safety block:
```
data: {"type": "safety_block", "category": "insider_trading", "message": "..."}\n\n
```

### Timeout

15 seconds total pipeline timeout. If exceeded → stream a structured timeout error event.

### Implementation Sketch

```python
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio, json

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    async def event_generator():
        try:
            # 1. Safety
            safety = run_safety_guard(request.query)
            if safety.blocked:
                yield {"data": json.dumps({
                    "type": "safety_block",
                    "category": safety.category.value,
                    "message": safety.refusal_message
                })}
                return

            # 2. Classify
            history = get_history(request.session_id)
            classifier_output = await classify(request.query, history)

            # 3. Load user
            user_profile = load_user(request.user_id)

            # 4. Route
            if classifier_output.target_agent == AgentName.PORTFOLIO_HEALTH:
                result = await run_portfolio_health(user_profile, request.query)
            else:
                result = run_stub(classifier_output)

            # 5. Stream result tokens
            result_json = result.model_dump_json(indent=2)
            for chunk in chunk_string(result_json, size=100):
                yield {"data": json.dumps({"type": "token", "content": chunk, "done": False})}
                await asyncio.sleep(0)

            yield {"data": json.dumps({"type": "token", "content": "", "done": True})}

            # 6. Save to memory
            add_turn(request.session_id, SessionTurn(
                user=request.query,
                assistant=result_json,
                entities=classifier_output.entities
            ))

        except asyncio.TimeoutError:
            yield {"data": json.dumps({"type": "error", "message": "Request timed out.", "code": "timeout"})}
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "message": str(e), "code": "pipeline_error"})}

    return EventSourceResponse(event_generator())
```

### Additional Endpoints

```
GET /health         → {"status": "ok"}
GET /users          → list of available user_ids from fixtures
```

---

## 11. Edge Cases Catalog

### Safety Guard
| Scenario | Expected Behavior |
|---|---|
| Query is purely educational about illegal topic | PASS THROUGH |
| Query contains blocked keyword mid-sentence innocuously | PASS THROUGH (use whole-phrase matching, not single-word) |
| Empty query string | PASS THROUGH — let classifier handle |
| Query in all caps | Normalize to lowercase before keyword matching |
| Multi-sentence query where one sentence is harmful | BLOCK |

### Classifier
| Scenario | Expected Behavior |
|---|---|
| LLM returns malformed JSON | Catch, return fallback ClassifierOutput with target_agent=support |
| LLM times out | Catch, return fallback |
| Follow-up query with pronoun ("what about it?") | Inject last turn into prompt context |
| Query about unrecognized agent type | Route to support |
| Confidence < 0.5 | Still route — just log low confidence. Don't block |

### Portfolio Health Agent
| Scenario | Expected Behavior |
|---|---|
| `user_004_empty` — no holdings | Return mode="build", observations with BUILD advice, no crash |
| Ticker delisted / not found in yfinance | Skip that holding, proceed with rest, add observation noting data unavailability |
| All tickers fail to fetch | Return graceful error observation, still return valid HealthCheckOutput |
| Single holding portfolio | top_3_positions_pct = top_1_pct, that's fine |
| account_inception_date is today | years_held = 0, annualized_return = 0.0, don't divide by zero |
| Non-USD holding (user_003) | Note in README: yfinance returns local currency prices — annex exchange rate note in observation |
| SPY fetch fails | Set benchmark fields to None, skip BenchmarkComparison block in output |

### HTTP Layer
| Scenario | Expected Behavior |
|---|---|
| Unknown user_id | Return SSE error event: "User not found", code: "user_not_found" |
| Missing session_id | Generate a new one server-side or reject with 422 |
| Pipeline exceeds 15s | Stream timeout SSE error, close connection |
| Client disconnects mid-stream | Generator exits naturally — no crash |

---

## 12. Testing Contract

### `tests/conftest.py` — Mock LLM

```python
import pytest
from unittest.mock import patch, MagicMock

MOCK_CLASSIFIER_RESPONSE = {
    "intent": "portfolio_health_check",
    "entities": {"tickers": [], "amounts": [], "time_periods": [], "sectors": [], "topics": []},
    "target_agent": "portfolio_health",
    "safety_verdict": "clean",
    "confidence": 0.95
}

@pytest.fixture
def mock_openai():
    with patch("openai.chat.completions.create") as mock:
        mock.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(MOCK_CLASSIFIER_RESPONSE)))]
        )
        yield mock
```

### `tests/test_safety.py`

- Load `fixtures/test_queries/safety_pairs.json`
- For each harmful query: assert `result.blocked == True`
- For each educational query: assert `result.blocked == False`
- Assert recall on harmful queries ≥ 95%
- Assert pass-through on educational queries ≥ 90%

### `tests/test_classifier.py`

- Load `fixtures/test_queries/intent_classification.json`
- Mock LLM to return per-query expected output
- Assert `classifier_output.target_agent == expected_agent` (string match, case-insensitive)
- Entity subset match with normalization:
  - Tickers: case-fold, strip exchange suffix (ASML.AS → ASML)
  - Numeric fields: ±5% tolerance
  - String lists: output must contain all expected values (extra allowed)
- Target routing accuracy ≥ 85%

### `tests/test_portfolio_health.py`

- Test `user_004_empty`: assert no exception, assert `mode == "build"`, assert `len(observations) > 0`
- Test `user_002_concentrated`: assert `concentration_risk.flag == "high"`
- Test output schema validity: assert all required fields present
- Mock yfinance in all tests

### `tests/test_routing.py`

- For each non-portfolio_health agent type: assert stub response has correct fields
- Assert `StubResponse.target_agent` matches classifier output
- Assert no exception thrown for any agent type

---

## 13. Environment Variables

`.env.example`:
```
OPENAI_API_KEY=your_key_here
MODEL_NAME=gpt-4o-mini
PIPELINE_TIMEOUT_SECONDS=15
SESSION_MAX_TURNS=10
```

---

## 14. Requirements

```
fastapi>=0.110.0
sse-starlette>=1.6.5
openai>=1.12.0
yfinance>=0.2.37
pydantic>=2.6.0
uvicorn>=0.27.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
python-dotenv>=1.0.0
```

---

## 15. README Must Cover (Minimum Sections)

1. **Setup** — clone, install, configure `.env`, run server
2. **Architecture** — ASCII diagram of request flow
3. **Library choices** — one-line justification per library
4. **Safety design** — what's blocked, what passes, why
5. **Session memory tradeoff** — in-memory vs Postgres, upgrade path
6. **Timeout rationale** — why 15s
7. **Latency measurement** — how you measured p95 (20 runs, `time.perf_counter()` around classifier call)
8. **What you'd do with more time** — must include: Postgres sessions, embedding pre-classifier, per-tenant model routing
9. **Video link** — placeholder until recorded

---

## 16. Performance Targets

| Target | Value | How to Measure |
|---|---|---|
| p95 first-token latency | < 2s | `time.perf_counter()` before/after first SSE chunk, 20 runs |
| p95 end-to-end | < 6s | time from request receipt to `done: true` event |
| Cost per query at gpt-4.1 pricing | < $0.05 | Log token counts, multiply by published rates |

Document measurements in README using `gpt-4o-mini` results and note that `gpt-4.1` results are estimated from token counts.

---

## 17. Git Commit Sequence (Minimum)

```
feat: initial repo structure, models, fixtures
feat: safety guard with all 5 categories
feat: session memory (in-memory)
feat: intent classifier with LLM + fallback
feat: portfolio health agent
feat: stub agents + router
feat: HTTP layer + SSE endpoint
test: full test suite passing
docs: README complete
```

No single-commit dumps. Each commit should represent a working, testable layer.

---

## 18. Defence Video Outline (10 min max)

```
0:00 - 2:00   Live demo: one query end-to-end, SSE tokens visible in terminal
2:00 - 5:00   Architecture walkthrough: guard → classifier → agent → SSE
5:00 - 7:30   One non-obvious decision (e.g. why guard is synchronous / how follow-up resolution works)
7:30 - 10:00  What you'd change with more time (Postgres, embedding pre-classifier, per-tenant routing)
```

---

*End of PRD. This document is the single source of truth for implementation. All decisions not specified here are left to developer judgment — document them in README.*
