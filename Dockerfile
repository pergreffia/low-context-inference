FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && useradd --system --uid 10001 --no-create-home contextproxy \
    && chown -R contextproxy:contextproxy /app

# M5: run unprivileged; healthcheck drives compose dependency ordering.
USER contextproxy

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --retries=5 --start-period=5s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]

# SERVER__WEB_CONCURRENCY scales uvicorn workers in production.
CMD ["sh", "-c", "uvicorn context_proxy.main:app --host 0.0.0.0 --port ${SERVER__PORT:-8080} --workers ${SERVER__WEB_CONCURRENCY:-1}"]
