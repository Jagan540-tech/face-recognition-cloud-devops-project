"""Gunicorn settings for a small CPU-only host. Override with environment variables.

WORKERS MUST STAY 1: registration sessions and the embedding store live in process memory, so a
second worker would not see people registered through the first. Scale with threads instead
(requests overlap during image decode / JSON, while model inference is serialised by a lock).
"""
import os

bind = f"0.0.0.0:{os.environ.get('FR_PORT', '5000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
worker_class = "gthread"
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "60"))
graceful_timeout = 20
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
