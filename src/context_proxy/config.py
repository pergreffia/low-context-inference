from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    # M5: structured JSON logs for log shippers; False keeps human-readable.
    log_json: bool = False
    # Reject oversized request bodies before parsing (M5 resource limits).
    max_body_bytes: int = Field(default=8 * 1024 * 1024, gt=0)


class EndpointSettings(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str | None = None
    timeout_seconds: float = Field(default=600.0, gt=0)


class DatabaseSettings(BaseModel):
    url: str = "postgresql://context_proxy:context_proxy@localhost:5432/context_proxy"
    min_pool_size: int = Field(default=1, ge=1)
    max_pool_size: int = Field(default=10, ge=1)
    connect_timeout_seconds: float = Field(default=3.0, gt=0)


class ConversationSettings(BaseModel):
    """Conversation identity configuration (M2.1 §2).

    Precedence: body conversation_id > X-Conversation-ID > client_id_header
    > generated UUID. Do not hard-code a client-specific header.
    """

    client_id_header: str = "X-Session-ID"
    max_session_identity_chars: int = Field(default=128, gt=0)


class QdrantSettings(BaseModel):
    base_url: str = "http://localhost:6333"
    collection: str = "context_proxy"
    timeout_seconds: float = Field(default=5.0, gt=0)


class RetrievalSettings(BaseModel):
    """Hybrid retrieval scoring (master prompt §19). Weights configurable."""

    semantic_weight: float = 0.40
    lexical_weight: float = 0.25
    recency_weight: float = 0.15
    importance_weight: float = 0.15
    type_weight: float = 0.05
    limit_default: int = Field(default=8, ge=1)
    candidate_pool: int = Field(default=50, ge=1)


class MemorySettings(BaseModel):
    auto_index: bool = True  # chunk+index completed turns after each response
    max_embed_chars: int = Field(default=8000, gt=0)
    # Synchronous indexing latency budget (M3 review §4): the request waits at
    # most this long for chunking+embedding+vector upsert before giving up.
    index_timeout_seconds: float = Field(default=10.0, gt=0)


class AssemblySettings(BaseModel):
    """Context Assembly Engine knobs (master prompt §11).

    Scoring weights are shared with RetrievalSettings (single source of truth
    for hybrid relevance); this model adds MMR diversity and per-category
    budget caps. All selection is deterministic.
    """

    enabled: bool = True
    # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    max_retrieved_items: int = Field(default=8, ge=0)
    # Hard cap for the retrieved category (tier 5) within the usable budget.
    retrieved_budget_tokens: int = Field(default=4000, ge=0)


class ResilienceSettings(BaseModel):
    """Upstream resilience (M5): retries + circuit breaker.

    Retries apply ONLY to transport-level failures before any response byte
    is received (connect errors, connect timeouts). Upstream HTTP error
    responses are answers, never retried. Streaming: no retry once the stream
    opened.
    """

    max_retries: int = Field(default=2, ge=0)
    backoff_base_seconds: float = Field(default=0.2, ge=0)
    backoff_max_seconds: float = Field(default=2.0, ge=0)
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_reset_seconds: float = Field(default=30.0, gt=0)


class RateLimitSettings(BaseModel):
    """In-process token-bucket rate limiting (M5).

    Single-instance scope by design (master prompt §2.6: no unnecessary
    infrastructure — no Redis). Keyed per conversation id when available,
    else per client host.
    """

    enabled: bool = False
    requests_per_minute: int = Field(default=120, gt=0)
    burst: int = Field(default=30, gt=0)


class ContextSettings(BaseModel):
    """Context budget configuration (master prompt §14, §15).

    usable_budget = model_limit_tokens - safety_margin_tokens.
    Never use the theoretical model limit directly.
    """

    model_limit_tokens: int = Field(default=32768, gt=0)
    safety_margin_tokens: int = Field(default=2048, ge=0)
    pinned_budget_tokens: int = Field(default=2000, ge=0)  # reserved for M3+
    recent_target_tokens: int = Field(default=14000, gt=0)
    recent_min_tokens: int = Field(default=10000, gt=0)
    recent_max_tokens: int = Field(default=18000, gt=0)

    @property
    def usable_budget_tokens(self) -> int:
        return self.model_limit_tokens - self.safety_margin_tokens

    @model_validator(mode="after")
    def _margin_within_limit(self) -> ContextSettings:
        if self.safety_margin_tokens >= self.model_limit_tokens:
            raise ValueError("safety_margin_tokens must be smaller than model_limit_tokens")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    server: ServerSettings = ServerSettings()
    database: DatabaseSettings = DatabaseSettings()
    context: ContextSettings = ContextSettings()
    assembly: AssemblySettings = AssemblySettings()
    conversation: ConversationSettings = ConversationSettings()
    inference: EndpointSettings = EndpointSettings()
    qdrant: QdrantSettings = QdrantSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    memory: MemorySettings = MemorySettings()
    resilience: ResilienceSettings = ResilienceSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    compact: EndpointSettings = EndpointSettings(
        base_url="http://localhost:8001/v1",
        api_key="local",
        model="compact-model",
    )
    embeddings: EndpointSettings = EndpointSettings(
        base_url="http://localhost:8002/v1",
        api_key="local",
        model="embedding-model",
    )


@lru_cache
def load_settings() -> Settings:
    return Settings()
