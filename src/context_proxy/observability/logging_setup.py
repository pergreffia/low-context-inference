"""Logging setup for production (M5, master prompt §12).

- JSON line format when SERVER__LOG_JSON=true (log shippers), human-readable
  otherwise;
- a redaction filter that scrubs bearer tokens / api keys from any message or
  extra field, so credentials can never leak into log backends (§19);
- request-id correlation via the observability middleware context var.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from context_proxy.observability.middleware import current_request_id

_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
# Sensitive FIELD-NAME markers: matched against normalized keys only, never
# against free text — ordinary sentences containing the word "token" must not
# be masked (M6-final hardening P1.4).
_KEY_FIELDS = (
    "api_key",
    "apikey",
    "api-key",
    "token",
    "access_token",
    "authorization",
    "secret",
    "password",
    "credential",
    "client_secret",
)
_REDACTED = "[REDACTED]"


def _is_sensitive_key(key: Any) -> bool:
    normalized = re.sub(r"[-_\s]", "", str(key)).lower()
    return any(marker.replace("_", "") in normalized for marker in _KEY_FIELDS)


def redact_text(value: str) -> str:
    """Scrub inline Bearer credentials from free text."""
    return _BEARER_RE.sub(rf"\1{_REDACTED}", value)


def redact(value: Any) -> Any:
    """Centralized recursive redaction for structured log payloads.

    Strings get bearer scrubbing; sensitive-keyed fields are fully masked at
    any nesting depth; containers are traversed. Unknown types pass through.
    Free text is NOT key-masked: the word "token" alone never triggers.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            key: (_REDACTED if _is_sensitive_key(key) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list | tuple | set):
        converted = [redact(item) for item in value]
        return type(value)(converted) if isinstance(value, tuple) else converted
    return value


def _redact_mapping(items: dict[str, Any]) -> dict[str, Any]:
    return {key: redact(value) for key, value in items.items()}


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(str(record.getMessage()))
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": redact_text(record.getMessage()),
            "request_id": current_request_id(),
        }
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": redact_text(str(exc_value)) if exc_value else None,
                "traceback": redact_text(
                    self.formatException(record.exc_info)
                ),
            }
        reserved = set(vars(logging.LogRecord("%", 0, "", 0, "", (), None)).keys())
        for key, value in record.__dict__.items():
            if key in reserved or key in ("asctime", "message", "exc_info"):
                continue
            if not key.startswith("_"):
                payload[key] = redact(value)
        return json.dumps(payload, ensure_ascii=False, default=str)


_OWNED_HANDLER: logging.Handler | None = None


def configure_logging(log_level: str = "INFO", *, log_json: bool = False) -> None:
    """Idempotent setup bound to the application logger (never the root).

    Attaching to 'context_proxy' keeps host/test harnesses (pytest caplog,
    uvicorn) intact while still guaranteeing redaction + format for every
    record the proxy emits.
    """
    global _OWNED_HANDLER
    app_logger = logging.getLogger("context_proxy")
    app_logger.setLevel(log_level.upper())
    if _OWNED_HANDLER is not None:
        app_logger.removeHandler(_OWNED_HANDLER)
        _OWNED_HANDLER = None
    handler = logging.StreamHandler()
    handler.addFilter(RedactionFilter())
    if log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        )
    app_logger.addHandler(handler)
    _OWNED_HANDLER = handler
