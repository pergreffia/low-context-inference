"""Compose configuration smoke checks (M0–M6 review P2).

Validates the default Compose configuration without booting the stack:
service-network env mappings must use service names, never localhost, for
dependencies that Compose itself provides.
"""

from __future__ import annotations

from pathlib import Path

COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


def _compose_text() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def test_qdrant_service_defined():
    text = _compose_text()
    assert "\n  qdrant:" in text


def _service_block(text: str, service: str, next_service: str) -> str:
    return text.split(f"{service}:", 1)[1].split(f"\n  {next_service}:", 1)[0]


def test_proxy_uses_qdrant_service_name_not_localhost():
    text = _compose_text()
    proxy_block = _service_block(text, "context-proxy", "postgres")
    assert "QDRANT__BASE_URL: http://qdrant:6333" in proxy_block
    qdrant_lines = [
        line for line in proxy_block.splitlines() if "QDRANT__BASE_URL" in line
    ]
    assert qdrant_lines and "localhost" not in qdrant_lines[0]


def test_proxy_depends_on_postgres_and_qdrant_health():
    text = _compose_text()
    block = _service_block(text, "context-proxy", "postgres")
    assert "postgres:" in block and "condition: service_healthy" in block
    assert "\n      qdrant:\n        condition: service_healthy" in block


def test_external_endpoints_documented_as_external():
    text = _compose_text()
    assert "EXTERNAL services" in text  # inference/embeddings policy comment
