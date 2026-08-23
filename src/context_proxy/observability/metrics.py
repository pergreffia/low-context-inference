"""Minimal Prometheus text-format metrics registry (M5, master prompt §12).

Deliberately dependency-free (§2.6): counters and histograms with fixed
labels cover the operational surface — request totals/latency, upstream
latency, token accounting, degradation events, breaker state.
"""

from __future__ import annotations

import re
from threading import Lock

_LABEL_RE = re.compile(r"[^a-zA-Z0-9_]")
_NAME_RE = re.compile(r"[^a-zA-Z0-9_:]")


def _sanitize_label(value: str) -> str:
    """Escape per Prometheus text format; label values may contain slashes."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")[:128]
        or "unknown"
    )


def _sanitize_name(value: str) -> str:
    return _NAME_RE.sub("_", value)


class Counter:
    def __init__(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()):
        self.name = _sanitize_name(name)
        self.documentation = documentation
        self.labelnames = labelnames
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = Lock()

    def labels(self, **labels) -> Counter:
        key = tuple(_sanitize_label(str(labels.get(ln, ""))) for ln in self.labelnames)
        with self._lock:
            self._values.setdefault(key, 0.0)
        return _BoundCounter(self, key)

    def inc(self, key: tuple[str, ...] | None = None, amount: float = 1.0) -> None:
        if key is None:
            if self.labelnames:
                raise ValueError("labeled counter requires .labels()")
            key = ()
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            for key, value in sorted(self._values.items()):
                label_part = ""
                if self.labelnames:
                    pairs = ",".join(
                        f'{ln}="{kv}"' for ln, kv in zip(self.labelnames, key, strict=True)
                    )
                    label_part = f"{{{pairs}}}"
                rendered = int(value) if float(value).is_integer() else value
                lines.append(f"{self.name}{label_part} {rendered}")
        return lines


class Gauge:
    """Labeled gauge rendering only the current value per label set."""

    def __init__(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()):
        self.name = _sanitize_name(name)
        self.documentation = documentation
        self.labelnames = labelnames
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = Lock()

    def labels(self, **labels) -> _BoundGauge:
        key = tuple(_sanitize_label(str(labels.get(ln, ""))) for ln in self.labelnames)
        with self._lock:
            self._values.setdefault(key, 0.0)
        return _BoundGauge(self, key)

    def set(self, key: tuple[str, ...], value: float) -> None:
        with self._lock:
            self._values[key] = value

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} gauge"]
        with self._lock:
            for key, value in sorted(self._values.items()):
                label_part = ""
                if self.labelnames:
                    pairs = ",".join(
                        f'{ln}="{kv}"' for ln, kv in zip(self.labelnames, key, strict=True)
                    )
                    label_part = f"{{{pairs}}}"
                rendered = int(value) if float(value).is_integer() else value
                lines.append(f"{self.name}{label_part} {rendered}")
        return lines


class _BoundCounter:
    def __init__(self, parent: Counter, key: tuple[str, ...]):
        self._parent = parent
        self._key = key

    def inc(self, amount: float = 1.0) -> None:
        self._parent.inc(self._key, amount)


class Histogram:
    """Fixed-bucket cumulative histogram in seconds."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ):
        self.name = _sanitize_name(name)
        self.documentation = documentation
        self.labelnames = labelnames
        self.buckets = sorted(buckets)
        self._bucket_counts: dict[tuple[str, ...], list[int]] = {}
        self._sums: dict[tuple[str, ...], float] = {}
        self._counts: dict[tuple[str, ...], int] = {}
        self._lock = Lock()

    def labels(self, **labels) -> _BoundHistogram:
        key = tuple(_sanitize_label(str(labels.get(ln, ""))) for ln in self.labelnames)
        with self._lock:
            if key not in self._bucket_counts:
                self._bucket_counts[key] = [0] * len(self.buckets)
                self._sums[key] = 0.0
                self._counts[key] = 0
        return _BoundHistogram(self, key)

    def observe(self, key: tuple[str, ...], value: float) -> None:
        with self._lock:
            counts = self._bucket_counts[key]
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[i] += 1
            self._sums[key] = self._sums.get(key, 0.0) + value
            self._counts[key] = self._counts.get(key, 0) + 1

    def set_gauge_value(self, key: tuple[str, ...], value: float) -> None:
        """Breaker-style states report their current level as an observation."""
        with self._lock:
            if key not in self._bucket_counts:
                self._bucket_counts[key] = [0] * len(self.buckets)
                self._sums[key] = 0.0
                self._counts[key] = 0

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} histogram"]
        base = f"{self.name}"
        with self._lock:
            for key in sorted(self._bucket_counts):
                counts = self._bucket_counts[key]
                # observe() already marked EVERY bound >= value, so each
                # bucket renders its own cumulative count directly.
                label_pairs = [
                    (ln, kv) for ln, kv in zip(self.labelnames, key, strict=True)
                ]
                for i, bound in enumerate(self.buckets):
                    pairs = [*label_pairs, ("le", repr(bound))]
                    labels_str = ",".join(f'{n}="{v}"' for n, v in pairs)
                    lines.append(f"{base}_bucket{{{labels_str}}} {counts[i]}")
                pairs = [*label_pairs, ("le", "+Inf")]
                labels_str = ",".join(f'{n}="{v}"' for n, v in pairs)
                total = self._counts[key]
                lines.append(f"{base}_bucket{{{labels_str}}} {total}")
                sum_labels = (
                    "{" + ",".join(f'{n}="{v}"' for n, v in label_pairs) + "}"
                    if label_pairs
                    else ""
                )
                lines.append(f"{base}_sum{sum_labels} {self._sums[key]}")
                lines.append(f"{base}_count{sum_labels} {total}")
        return lines


class _BoundHistogram:
    def __init__(self, parent: Histogram, key: tuple[str, ...]):
        self._parent = parent
        self._key = key

    def observe(self, value: float) -> None:
        self._parent.observe(self._key, max(0.0, value))


class _BoundGauge:
    def __init__(self, parent: Gauge, key: tuple[str, ...]):
        self._parent = parent
        self._key = key

    def set(self, value: float) -> None:
        self._parent.set(self._key, value)


class Registry:
    def __init__(self):
        self._collectors: list[Counter | Histogram | Gauge] = []

    def register(self, collector: Counter | Histogram | Gauge) -> Counter | Histogram | Gauge:
        self._collectors.append(collector)
        return collector

    def reset(self) -> None:
        """Drop every time series (test isolation; never call at runtime)."""
        for collector in self._collectors:
            with collector._lock:
                if isinstance(collector, Histogram):
                    collector._bucket_counts.clear()
                    collector._sums.clear()
                    collector._counts.clear()
                else:
                    collector._values.clear()

    def render(self) -> str:
        lines: list[str] = []
        for collector in self._collectors:
            lines.extend(collector.render())
            lines.append("")
        return "\n".join(lines)


REGISTRY = Registry()

HTTP_REQUESTS_TOTAL: Counter = REGISTRY.register(
    Counter(
        "context_proxy_http_requests_total",
        "Total HTTP requests handled.",
        ("method", "route", "status"),
    )
)
HTTP_REQUEST_DURATION: Histogram = REGISTRY.register(
    Histogram(
        "context_proxy_http_request_duration_seconds",
        "End-to-end request latency.",
        ("route",),
    )
)
UPSTREAM_DURATION: Histogram = REGISTRY.register(
    Histogram(
        "context_proxy_upstream_duration_seconds",
        "Upstream inference call latency (headers received).",
        ("route",),
    )
)
LLM_TOKENS_TOTAL: Counter = REGISTRY.register(
    Counter(
        "context_proxy_llm_tokens_total",
        "Token accounting reported by the upstream provider.",
        ("direction",),
    )
)
DEGRADATIONS_TOTAL: Counter = REGISTRY.register(
    Counter(
        "context_proxy_degradations_total",
        "Degraded operations that continued without a derived service.",
        ("component",),
    )
)
RATE_LIMIT_REJECTS_TOTAL: Counter = REGISTRY.register(
    Counter(
        "context_proxy_rate_limit_rejects_total",
        "Requests rejected by the local rate limiter.",
    )
)
CIRCUIT_STATE: Gauge = REGISTRY.register(
    Gauge(
        "context_proxy_circuit_state",
        "Circuit-breaker state (1 = current state).",
        ("state",),
    )
)


def record_tokens(usage: dict | None, model: str | None = None) -> None:
    """Increment token counters from an OpenAI usage object; never raises."""
    if not isinstance(usage, dict):
        return
    try:
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, (int, float)):
            LLM_TOKENS_TOTAL.labels(direction="prompt").inc(prompt)
        if isinstance(completion, (int, float)):
            LLM_TOKENS_TOTAL.labels(direction="completion").inc(completion)
    except Exception:  # noqa: BLE001 - accounting must never break requests
        pass


def set_circuit_state(state: str) -> None:
    """Publish breaker state as labeled gauges (1 on active, 0 elsewhere)."""
    for known in ("closed", "open", "half_open"):
        CIRCUIT_STATE.labels(state=known).set(1.0 if known == state else 0.0)
