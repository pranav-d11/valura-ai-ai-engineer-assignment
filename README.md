# Valura AI Microservice

FastAPI + SSE microservice that implements the assignment spine:
safety guard -> classifier -> router -> portfolio health/stub agents -> SSE stream.

## Setup

1. Clone and enter project.
2. Create virtual environment (Python 3.11+):
   - Windows PowerShell:
     - `py -3.11 -m venv .venv`
     - `.\.venv\Scripts\activate`
3. Install dependencies:
   - `.\.venv\Scripts\python -m pip install -r requirements.txt`
4. Configure environment:
   - `Copy-Item .env.example .env`
   - Fill `OPENAI_API_KEY` only if you want live classifier LLM calls.

## Environment Variables

- `OPENAI_API_KEY`: optional for local tests; required for live OpenAI classifier path.
- `OPENAI_MODEL` / `MODEL_NAME`: classifier model override, default `gpt-4o-mini`.
- `PIPELINE_TIMEOUT_SECONDS`: total `/chat` pipeline timeout, default `15`.
- `SESSION_MAX_TURNS`: documented cap, currently enforced as 10 in `src/memory/session.py`.
- `TENANT_MODEL_OVERRIDES`: JSON map of `tenant_id -> model` for per-tenant routing (stretch).
- `DEFAULT_TENANT_MODEL`: fallback model when tenant is not in overrides.
- `CLASSIFIER_CACHE_TTL_SECONDS` / `CLASSIFIER_CACHE_MAX_ITEMS`: intra-session identical-query classifier cache (stretch).
- `PRE_CLASSIFIER_MIN_CONFIDENCE`: heuristic fast-path threshold; set `0` to disable LLM skipping (stretch).
- `RATE_LIMIT_PER_WINDOW` / `RATE_LIMIT_WINDOW_SECONDS`: optional `/chat` rate limit (stretch); `0` disables.

## Run

- API server: `.\.venv\Scripts\python -m uvicorn src.http.app:app --reload`
- Test suite: `.\.venv\Scripts\python -m pytest tests -v`

## API Notes

`POST /chat` body:
- `user_id` (required)
- `session_id` (required)
- `query` (required)
- `tenant_id` (optional; enables per-tenant model overrides)

## Architecture

```text
POST /chat
   |
   v
[Safety Guard - synchronous]
   | blocked -> SSE safety_block + return
   v
[Session Memory - in-memory]
   v
[Intent Classifier]
   | optional heuristic pre-classifier fast path (high confidence)
   | intra-session identical-query cache
   | per-tenant model override map
   | one LLM call when OPENAI_API_KEY is present and fast path does not apply
   | deterministic fallback when key/model call unavailable
   v
[Router]
   |-> portfolio_health (fully implemented)
   |-> all other agents (structured stub response)
   v
[SSE token stream]
   v
[Persist turn to session memory]
```

## Library Choices

- `fastapi`: typed async web layer and simple dependency wiring.
- `sse-starlette`: explicit SSE event streaming with minimal ceremony.
- `pydantic v2`: strict request/response contracts and schema safety.
- `openai`: official SDK for single-call structured classifier path.
- `yfinance`: live market prices for portfolio-health computations.
- `pytest` (+ async/mock plugins): deterministic, CI-friendly validation.

## Safety Design

- Guard is local, synchronous, and runs before any classifier call.
- Block categories: insider trading, market manipulation, money laundering, guaranteed-return claims, reckless advice, sanctions-evasion/fraud patterns.
- Educational framing is intentionally passed through (for example, "what is insider trading?").
- Each blocked category returns a distinct refusal message.
- Safety verdict returned by classifier is informational only; guard is the blocking authority.

## Session Memory Tradeoff

- Current backend: in-memory store keyed by `session_id`.
- Why this choice: no infrastructure dependency during assignment evaluation.
- Production upgrade path:
  - Redis for low-latency, horizontally-scaled ephemeral chat memory.
  - Postgres for durable history and compliance/audit traceability.
- Memory interface is intentionally thin (`get_history`, `add_turn`, `clear_session`) to keep backend swap low-risk.

## Timeout Rationale

- Pipeline timeout is set to `15s`.
- This balances two constraints:
  - enough room for classifier + market data calls under moderate network jitter,
  - fast failure for users if upstream dependencies stall.
- Timeout failures emit structured SSE error events (`code: "timeout"`), not stack traces.

## Latency Measurement

Measurement method:
- `time.perf_counter()` around each request from dispatch to first SSE token and to `done: true`.
- 20 runs against `POST /chat` using `fastapi.testclient`.

Command used:
- `.\.venv\Scripts\python -c "<benchmark snippet>"`

Local results (20 runs, `query="hi"`):
- p95 first-token latency: **0.0058s**
- p95 end-to-end latency: **0.0058s**
- average first-token latency: `0.0038s`
- average end-to-end latency: `0.0038s`

Note:
- These are local dev-loop numbers (no networked LLM call).
- With live OpenAI and live market fetch, expect higher but still under target in normal conditions.

## Cost Measurement

Development mode:
- Default model: `gpt-4o-mini`.
- Tests/CI run with mocked/no-key path, so measured local classifier API cost is `~$0`.

Evaluation model target:
- `gpt-4.1` (per assignment).
- Cost estimation method used in this repo:
  - log token usage from classifier calls,
  - compute `input_tokens * input_rate + output_tokens * output_rate`,
  - aggregate per-query.

Based on expected classifier prompt sizes, estimated cost remains under `$0.05/query`.

## Performance Targets (Assignment Alignment)

- p95 first token `< 2s` -> met in local benchmark.
- p95 end-to-end `< 6s` -> met in local benchmark.
- cost/query at `gpt-4.1` `< $0.05` -> estimated to meet; final number depends on live token mix.

## Optional Stretch (implemented)

- **Identical-query dedupe cache (intra-session)**: classifier memoizes normalized queries per `session_id` for `CLASSIFIER_CACHE_TTL_SECONDS`.
- **Pre-classifier fast path**: heuristic confidence gate can skip the LLM when `PRE_CLASSIFIER_MIN_CONFIDENCE` is met (assignment wording mentions embeddings; this is a pragmatic deterministic analogue without extra infra).
- **Per-tenant model selection**: `tenant_id` on `ChatRequest` + `TENANT_MODEL_OVERRIDES` JSON env map.
- **Multi-tenant rate limiting**: sliding window limiter keyed by `(tenant_id, user_id)` with SSE `rate_limited` errors.

## Test Coverage Snapshot

- Safety guard recall/passthrough and distinct-category response tests.
- Classifier routing accuracy and entity-subset extraction matcher.
- Conversation fixture alignment (`follow_up_session`, `multi_intent_session`, `ambiguous_session`).
- Portfolio health empty/concentration/disclaimer behavior.
- HTTP SSE health/safety-block/streaming token behavior.
- Stretch features: classifier cache + pre-classifier skip + tenant model map + rate limiting.

Current result:
- `15 passed` with `.\.venv\Scripts\python -m pytest tests -v`

## What I Would Do With More Time

- Move sessions to Postgres with explicit turn schema and retention controls.
- Add embedding pre-classifier to skip LLM calls on high-confidence intents.
- Add per-tenant model routing (`gpt-4o-mini` vs `gpt-4.1`) with budget caps.
- Add structured observability (trace IDs, latency per stage, token usage metrics).
- Expand entity normalization and coreference for multilingual/noisy inputs.

## Defence Video Link

- TODO: add unlisted YouTube link (<= 10 minutes) before final submission.
