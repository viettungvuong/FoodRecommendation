from __future__ import annotations

import json
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import redis

from remymy_api import jobs, main


class FakeCache:
    def __init__(self) -> None:
        self.down = False
        self.store: dict[str, bytes] = {}
        self.ttls: dict[str, int | None] = {}
        self.reads: list[str] = []

    def get(self, key: str) -> bytes | None:
        if self.down:
            raise redis.ConnectionError("down")
        self.reads.append(key)
        return self.store.get(key)

    def set(self, key: str, value: object, ex: int | None = None) -> None:
        if self.down:
            raise redis.ConnectionError("down")
        self.store[key] = str(value).encode()
        self.ttls[key] = ex

    def close(self) -> None:
        pass


@pytest.fixture
def cache(monkeypatch) -> FakeCache:
    fake = FakeCache()
    monkeypatch.setattr(main, "connect_cache", lambda: fake)
    return fake


class FakeQueue:
    def __init__(self) -> None:
        self.jobs: list[dict] = []
        self.handler: Callable[[dict], dict | None] = lambda payload: None
        self.down = False

    def lpush(self, key: str, value: str) -> None:
        if self.down:
            raise redis.ConnectionError("down")
        assert key == jobs.QUEUE_KEY
        self.jobs.append(json.loads(value))

    def blpop(self, keys: list[str], timeout: float) -> tuple[bytes, bytes] | None:
        job = self.jobs[-1]
        assert keys == [jobs.RESULT_KEY_PREFIX + job["id"]]
        reply = self.handler(job["payload"])
        return None if reply is None else (keys[0].encode(), json.dumps(reply).encode())

    @property
    def payloads(self) -> list[dict]:
        return [job["payload"] for job in self.jobs]

    def close(self) -> None:
        pass


@pytest.fixture
def queue(monkeypatch) -> FakeQueue:
    fake = FakeQueue()
    monkeypatch.setattr(main, "connect_queue", lambda: fake)
    return fake


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(jobs, "time", SimpleNamespace(time=time.time, sleep=recorded.append))
    return recorded
