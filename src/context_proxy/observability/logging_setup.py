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
_KEY_FIELDS = ("api_key", "authorization", "password", "secret", "token")


def redact_text(value: str) -> str:
    return _BEARER_RE.sub(r"\1[REDACTED]", value)


def _redact_mapping(items: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in items.items():
        if any(marker in key.lower() for marker in _KEY_FIELDS):
            safe[key] = "[REDACTED]"
        elif isinstance(value, str):
            safe[key] = redact_text(value)
        else:
            safe[key] = value
    return safe


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(str(record.getMessage()))
        record.args = ()
        if hasattr(record, "__dict__"):
            for field in ("error", "detail"):
                value = getattr(record, field, None)
                if isinstance(value, str) and "Bearer" in value:
                    setattr(record, field, redact_text(value))
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": current_request_id(),
        }
        reserved = set(vars(logging.LogRecord("%", 0, "", 0, "", (), None)).keys())
        for key, value in record.__dict__.items():
            if key in reserved or key in ("asctime", "message"):
                continue
            if not key.startswith("_"):
                payload[key] = value
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
