from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"


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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    server: ServerSettings = ServerSettings()
    database: DatabaseSettings = DatabaseSettings()
    context: ContextSettings = ContextSettings()
    inference: EndpointSettings = EndpointSettings()
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
