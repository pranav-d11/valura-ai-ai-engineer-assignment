# Valura AI Microservice

FastAPI + SSE microservice for Valura's AI co-investor pipeline.

## Setup

- Python `3.11+`
- Windows (PowerShell):
  - `py -3.11 -m venv .venv`
  - `.\.venv\Scripts\activate`
  - `.\.venv\Scripts\python -m pip install -r requirements.txt`
- Copy `.env.example` to `.env`

## Environment Variables

- `OPENAI_API_KEY`: required only for live LLM-backed classifier mode
- `OPENAI_MODEL`/`MODEL_NAME`: model selection, default `gpt-4o-mini`
- `PIPELINE_TIMEOUT_SECONDS`: default `15`
- `SESSION_MAX_TURNS`: currently enforced as `10` in in-memory session store

## Run

- API: `.\.venv\Scripts\python -m uvicorn src.http.app:app --reload`
- Tests: `.\.venv\Scripts\python -m pytest tests -v`

## Architecture

```text
POST /chat
   |
   v
[Safety Guard - sync]
   | blocked? yes --> SSE safety_block and return
   |
   no
   v
[Session History (in-memory)]
   |
   v
[Classifier - one call / fallback]
   |
   v
[Router]
   |--> portfolio_health (implemented)
   |--> all other agents -> stub response
   |
   v
[SSE stream token chunks]
   |
   v
[Persist session turn]
```

## Library Choices

- `fastapi`: clean async API framework with strong typing
- `sse-starlette`: straightforward SSE integration with FastAPI
- `pydantic v2`: strict contracts for all pipeline payloads
- `openai`: production SDK for classifier model integration
- `yfinance`: quick live market data retrieval for portfolio checks
- `pytest`: fast test loop with fixture-friendly style

## Safety Design

- Guard runs before classifier (blocking authority)
- Category-based phrase matching for:
  - insider trading
  - market manipulation
  - money laundering
  - guaranteed returns
  - reckless advice
  - sanctions evasion / fraud patterns
- Educational framing is intentionally passed through (e.g., "what is insider trading?")
- Distinct refusal message per blocked category

## Session Memory Tradeoff

- Current implementation uses in-memory dictionary store keyed by `session_id`
- Why now: fastest implementation path, no infra needed for assignment runtime
- Production upgrade path:
  - Redis for horizontal scale and low latency
  - Postgres for durable audit/history requirements
- Store interface is intentionally thin, so backend swap is localized

## Timeout Rationale

- Pipeline timeout: `15s`
- Rationale: long enough for classifier + market calls under normal network jitter, short enough to avoid stalled client connections and runaway cost

## Latency Measurement Plan

- Method:
  - wrap `time.perf_counter()` around pipeline phases
  - record first-token latency and end-to-end completion
  - run 20 requests, report p95
- Status: placeholder section; final measured numbers will be filled before submission

## Cost Measurement Plan

- Log prompt/completion token counts from classifier calls
- Estimate per-query cost using published model rates
- Report observed `gpt-4o-mini` cost and estimated `gpt-4.1` cost

## Performance Targets

- p95 first token `< 2s` (target)
- p95 end-to-end `< 6s` (target)
- estimated per-query cost at `gpt-4.1` `< $0.05` (target)

## What I Would Do With More Time

- Replace in-memory sessions with Postgres-backed session memory
- Add embedding-based pre-classifier to reduce unnecessary LLM calls
- Per-tenant model routing (cost/performance tiering)
- Better observability (structured logs, trace ids, latency breakdown by stage)
- Stronger normalization and follow-up coreference for classifier entities

## Video Link

- TODO: add unlisted defence video URL
