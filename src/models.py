from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentName(str, Enum):
    PORTFOLIO_HEALTH = "portfolio_health"
    MARKET_RESEARCH = "market_research"
    INVESTMENT_STRATEGY = "investment_strategy"
    FINANCIAL_PLANNING = "financial_planning"
    FINANCIAL_CALCULATOR = "financial_calculator"
    RISK_ASSESSMENT = "risk_assessment"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    CUSTOMER_SUPPORT = "customer_support"
    GENERAL_QUERY = "general_query"


class SafetyCategory(str, Enum):
    INSIDER_TRADING = "insider_trading"
    MARKET_MANIPULATION = "market_manipulation"
    GUARANTEED_RETURNS = "guaranteed_returns"
    MONEY_LAUNDERING = "money_laundering"
    RECKLESS_ADVICE = "reckless_advice"
    SANCTIONS_EVASION = "sanctions_evasion"
    FRAUD = "fraud"
    CLEAN = "clean"


class ObservationSeverity(str, Enum):
    WARNING = "warning"
    INFO = "info"
    CRITICAL = "critical"


class SafetyResult(BaseModel):
    blocked: bool
    category: SafetyCategory
    message: str | None = None

    @property
    def refusal_message(self) -> str | None:
        return self.message


class ExtractedEntities(BaseModel):
    tickers: list[str] | None = None
    amount: float | None = None
    currency: str | None = None
    rate: float | None = None
    period_years: int | None = None
    frequency: str | None = None
    horizon: str | None = None
    time_period: str | None = None
    topics: list[str] | None = None
    sectors: list[str] | None = None
    index: str | None = None
    action: str | None = None
    goal: str | None = None

    def model_post_init(self, __context: Any) -> None:
        """Ensure list fields are never None after validation."""
        if self.tickers is None:
            self.tickers = []
        if self.topics is None:
            self.topics = []
        if self.sectors is None:
            self.sectors = []


class ClassifierOutput(BaseModel):
    intent: str
    entities: ExtractedEntities
    target_agent: AgentName
    safety_verdict: SafetyCategory = SafetyCategory.CLEAN
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    @property
    def agent(self) -> str:
        return self.target_agent.value


class ConcentrationRisk(BaseModel):
    top_position_pct: float
    top_3_positions_pct: float
    flag: str


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
    mode: str
    concentration_risk: ConcentrationRisk | None = None
    performance: Performance | None = None
    benchmark_comparison: BenchmarkComparison | None = None
    observations: list[Observation]
    disclaimer: str


class StubResponse(BaseModel):
    intent: str
    entities: ExtractedEntities
    target_agent: AgentName
    message: str


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    query: str
    tenant_id: str | None = None


class SessionTurn(BaseModel):
    user: str
    assistant: str
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)


class Position(BaseModel):
    ticker: str
    exchange: str
    quantity: float
    avg_cost: float
    currency: str = "USD"
    purchased_at: str | None = None


class UserProfile(BaseModel):
    user_id: str
    name: str
    age: int | None = None
    country: str | None = None
    base_currency: str = "USD"
    kyc: dict[str, Any] = Field(default_factory=dict)
    risk_profile: str = "moderate"
    positions: list[Position] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
