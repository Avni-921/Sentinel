"""Centralized, Pydantic-validated application settings."""

import os
from urllib.parse import quote, urlunsplit

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load and validate environment variables from `.env`.

    - `extra="ignore"` lets `.env` keep legacy keys (`POSTGRES_URI`, `REDIS_URL`,
      old Groq keys, etc.) without failing startup.
    - Environment is loaded once, but service-specific requirements are enforced
      only when that service is actually used.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- JINA AI (embeddings + reranker) ---
    JINA_API_KEY: str | None = None

    # --- OPENAI LLM ---
    OPENAI_API_KEY: str | None = None
    JUDGE_OPENAI_API_KEY: str | None = None

    # --- PORTKEY LLM GATEWAY ---
    PORTKEY_API_KEY: str | None = None
    PORTKEY_PRIMARY_SLUG: str = "marathon-api"
    PORTKEY_FALLBACK_SLUG: str = "anthropic-fallback"
    # Portkey saved config is referenced by its system-generated `pc-...` ID.
    # Required when block_inline_config is enabled on the workspace.
    PORTKEY_PRIMARY_CONFIG_ID: str | None = None

    # --- QDRANT VECTOR DB ---
    QDRANT_URL: str | None = Field(default=None, validation_alias=AliasChoices("QDRANT_URL", "QDRANT_CLUSTER_ENDPOINT"))
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "enterprise_rag"

    # --- NEON SERVERLESS POSTGRES (LangGraph checkpointer) ---
    NEON_DB_URL: str | None = None

    # --- UPSTASH REDIS (rate limiting) ---
    UPSTASH_REDIS_REST_URL: str | None = None
    UPSTASH_REDIS_REST_TOKEN: str | None = None

    # --- API SAFETY ---
    API_KEY: str | None = Field(default=None, alias="RAG_API_KEY")
    RATE_LIMIT_PER_MINUTE: int = 20
    STRICT_STARTUP: bool = False

    # --- OBSERVABILITY ---
    LOGFIRE_TOKEN: str | None = None
    LOGFIRE_BASE_URL: str | None = None  # e.g. https://logfire-eu.pydantic.dev for EU tokens
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "rag_scale_test"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    @field_validator("QDRANT_API_KEY", mode="before")
    @classmethod
    def _empty_qdrant_key_as_none(cls, v):
        """Treat empty QDRANT_API_KEY as unset so local Qdrant doesn't receive a blank header."""
        if v == "" or v is None:
            return None
        return v

    @staticmethod
    def _required(value: str | None, env_name: str, purpose: str) -> str:
        """Raise a clear error only when a specific service needs a missing setting."""
        if value:
            return value
        raise ValueError(f"{env_name} is required for {purpose}. Set it in .env or the environment.")

    @property
    def judge_api_key(self) -> str:
        """Dedicated judge key, falling back to the main OpenAI key."""
        return self.JUDGE_OPENAI_API_KEY or self._required(
            self.OPENAI_API_KEY,
            "OPENAI_API_KEY",
            "judge LLM access",
        )

    @property
    def postgres_uri(self) -> str:
        """LangGraph Postgres checkpointer URI (Neon).

        Serverless Postgres closes idle connections, so append TCP keepalive
        options to keep the connection pool healthy between requests.
        """
        base = self._required(
            self.NEON_DB_URL,
            "NEON_DB_URL",
            "Postgres checkpointer",
        ).rstrip("/")
        keepalive = "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
        if "?" in base:
            return f"{base}&{keepalive}"
        return f"{base}?{keepalive}"

    @property
    def redis_url(self) -> str:
        """TLS Redis URL derived from Upstash REST credentials.

        Upstash exposes the same host for REST and TLS Redis. The REST token is
        used as the Redis password under the default username. The result is
        passed to `limits` for rate limiting and to the health checker.
        """
        host = self._required(
            self.UPSTASH_REDIS_REST_URL,
            "UPSTASH_REDIS_REST_URL",
            "Redis rate limiting",
        ).replace("https://", "").rstrip("/")
        token = quote(
            self._required(
                self.UPSTASH_REDIS_REST_TOKEN,
                "UPSTASH_REDIS_REST_TOKEN",
                "Redis rate limiting",
            ),
            safe="",
        )
        netloc = f"default:{token}@{host}"
        return urlunsplit(("rediss", netloc, "/0", "ssl_cert_reqs=required", ""))


# Singleton used across the app.
settings = Settings()


def apply_langchain_env():
    """Write LangSmith/LangChain settings to os.environ for automatic tracing.

    Tracing is only activated when both LANGSMITH_TRACING and LANGSMITH_API_KEY
    are set — enabling tracing without a key causes LangChain to emit 401 noise
    on every LangGraph step.
    """
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.LANGSMITH_TRACING)
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGSMITH_API_KEY)
    if settings.LANGSMITH_PROJECT:
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGSMITH_PROJECT)
    if settings.LANGSMITH_ENDPOINT:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.LANGSMITH_ENDPOINT)


apply_langchain_env()
