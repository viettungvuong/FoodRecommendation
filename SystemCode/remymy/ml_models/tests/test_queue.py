from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest
import redis
from starlette.datastructures import State

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve


HISTORY = [{"rating": 5, "date": "2026-09-01", "ingredients": ["chicken breast"]}]


def job(payload: object, deadline: float | None = None, job_id: str = "abc") -> str:
    return json.dumps({"id": job_id, "deadline": time.time() + 10 if deadline is None else deadline,
                       "payload": payload})


@pytest.fixture
def state(monkeypatch) -> State:
    calls = []

    def fake_run(state, body):
        calls.append(body)
        return {"ingredients": [[]], "items": [{"fdc_id": 1, "top_k": body.top_k}]}

    monkeypatch.setattr(serve, "run_recommendation", fake_run)
    return State({"calls": calls})


def test_a_job_is_answered_on_its_reply_key(state):
    key, reply = serve.handle_job(state, job({"history": HISTORY, "top_k": 3}))

    assert key == "ml:result:abc"
    assert reply == {"result": {"ingredients": [[]], "items": [{"fdc_id": 1, "top_k": 3}]}}


def test_an_invalid_payload_is_rejected_as_not_retryable(state):
    key, reply = serve.handle_job(state, job({"history": [], "top_k": 3}))

    assert reply["retryable"] is False
    assert state.calls == []


def test_a_failing_model_is_reported_as_retryable(state, monkeypatch):
    def boom(state, body):
        raise RuntimeError("boom")

    monkeypatch.setattr(serve, "run_recommendation", boom)

    key, reply = serve.handle_job(state, job({"history": HISTORY}))

    assert reply == {"error": "RuntimeError('boom')", "retryable": True}


@pytest.mark.parametrize("raw", [b"not json", json.dumps({"payload": {}}), json.dumps([1])])
def test_malformed_jobs_are_dropped(state, raw):
    assert serve.handle_job(state, raw) is None


def test_expired_jobs_are_dropped_without_running_the_model(state):
    assert serve.handle_job(state, job({"history": HISTORY}, deadline=time.time() - 1)) is None
    assert state.calls == []


class FakeRedis:
    def __init__(self, items: list[object], stopped: threading.Event) -> None:
        self.items = items
        self.stopped = stopped
        self.replies: dict[str, list[str]] = {}
        self.expiries: dict[str, int] = {}

    def brpop(self, keys, timeout):
        if not self.items:
            self.stopped.set()
            return None
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return (keys[0].encode(), item)

    def pipeline(self):
        return self

    def lpush(self, key, value):
        self.replies.setdefault(key, []).append(value)
        return self

    def expire(self, key, seconds):
        self.expiries[key] = seconds
        return self

    def execute(self):
        return []


def test_the_worker_answers_every_job_and_survives_redis_errors(state, monkeypatch):
    monkeypatch.setattr(serve, "POLL_SECONDS", 0)
    stopped = threading.Event()
    client = FakeRedis(
        [job({"history": HISTORY}, job_id="one"), redis.ConnectionError("blip"), b"junk",
         job({"history": HISTORY, "top_k": 2}, job_id="two")],
        stopped,
    )

    serve.work(state, client, stopped)

    assert set(client.replies) == {"ml:result:one", "ml:result:two"}
    assert json.loads(client.replies["ml:result:two"][0])["result"]["items"][0]["top_k"] == 2
    assert client.expiries == {"ml:result:one": 60, "ml:result:two": 60}
