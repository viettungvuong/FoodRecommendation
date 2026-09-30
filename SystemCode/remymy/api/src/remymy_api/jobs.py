from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Hashable

import redis

from .cache import DEFAULT_REDIS_URL


QUEUE_KEY = "ml:jobs"
RESULT_KEY_PREFIX = "ml:result:"
DEFAULT_RESULT_TIMEOUT_SECONDS = 10.0
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 0.5
RETRY_INTERVAL_SECONDS = 600

logger = logging.getLogger(__name__)


class JobFailed(Exception):
    pass


class JobRejected(JobFailed):
    pass


def result_timeout() -> float:
    return float(os.getenv("ML_QUEUE_TIMEOUT_SECONDS", DEFAULT_RESULT_TIMEOUT_SECONDS))


def connect_queue() -> redis.Redis:
    return redis.Redis.from_url(
        os.getenv("REDIS_URL", DEFAULT_REDIS_URL),
        socket_connect_timeout=2,
        socket_timeout=result_timeout() + 5,
    )


def submit(queue: redis.Redis, payload: dict) -> dict:
    user_id = payload.get("user_id", "")
    if user_id == "":
        raise JobFailed("unreadable reply")
    job_id = f"job:{user_id}:{uuid.uuid7().hex}"
    timeout = result_timeout()
    job = {"id": job_id, "deadline": time.time() + timeout, "payload": payload}
    try:
        queue.lpush(QUEUE_KEY, json.dumps(job))
        reply = queue.blpop([RESULT_KEY_PREFIX + job_id], timeout=timeout)
    except redis.RedisError as error:
        raise JobFailed(f"queue error: {error}") from error
    if reply is None:
        raise JobFailed(f"no reply within {timeout}s")
    try:
        body = json.loads(reply[1])
    except ValueError as error:
        raise JobFailed("unreadable reply") from error
    if "error" in body:
        if body.get("retryable", True):
            raise JobFailed(body["error"])
        raise JobRejected(body["error"])
    return body["result"]


def submit_with_retries(queue: redis.Redis, payload: dict) -> dict:
    attempt = 0
    while True:
        try:
            return submit(queue, payload)
        except JobRejected:
            raise
        except JobFailed as error:
            attempt += 1
            logger.warning("model job attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, error)
            if attempt >= MAX_ATTEMPTS:
                raise
            time.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))


class RetryScheduler:
    def __init__(self, interval: float = RETRY_INTERVAL_SECONDS) -> None:
        self.interval = interval
        self.stopped = threading.Event()
        self.pending: set[Hashable] = set()
        self.lock = threading.Lock()

    def schedule(self, key: Hashable, task: Callable[[], None]) -> bool:
        with self.lock:
            if self.stopped.is_set() or key in self.pending:
                return False
            self.pending.add(key)
        threading.Thread(target=self._run, args=(key, task), name=f"retry-{key}", daemon=True).start()
        return True

    def _run(self, key: Hashable, task: Callable[[], None]) -> None:
        try:
            while not self.stopped.wait(self.interval):
                try:
                    task()
                    return
                except Exception:
                    logger.warning("background retry for %s failed", key, exc_info=True)
        finally:
            with self.lock:
                self.pending.discard(key)

    def stop(self) -> None:
        self.stopped.set()
