"""Provider abstractions.

The Context Proxy core depends only on these interfaces. Concrete
implementations are selected through configuration (master prompt §4).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConversationStore(Protocol):
    """Source-of-truth persistence for raw conversations."""

    async def ping(self) -> None:
        """Raise if the store is unreachable."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Persistence for derived memory records."""

    async def ping(self) -> None: ...


@runtime_checkable
class VectorStore(Protocol):
    """Derived vector index (e.g. Qdrant). Rebuildable from source of truth."""

    async def ping(self) -> None: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class LLMStream(Protocol):
    """Incremental passthrough handle over an upstream streaming response."""

    @property
    def status_code(self) -> int: ...

    @property
    def media_type(self) -> str: ...

    def passthrough_headers(self) -> dict[str, str]: ...

    def iter_bytes(self) -> AsyncIterator[bytes]: ...


@runtime_checkable
class LLMProvider(Protocol):
    async def list_models(self) -> tuple[int, dict[str, str], bytes]:
        """Return (status_code, headers, body) from the models endpoint."""
        ...

    async def complete(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        """Non-streaming chat completion passthrough."""
        ...

    async def open_stream(self, payload: dict[str, Any]) -> LLMStream:
        """Start a streaming chat completion and return its passthrough handle."""
        ...


@runtime_checkable
class CompactProvider(Protocol):
    async def compact(self, prompt: str) -> str: ...
