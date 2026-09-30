from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from remymy_api import jobs, recommend
from remymy_api.main import app
from remymy_api.users import add_history, create_user, init_user_database, open_user_database


CATALOG = {
    "tomatoes": {"fdc_id": 1, "name": "Tomatoes, red, ripe", "price": 2.0,
                 "nutrient_values": {"energy_kcal": 18.0, "protein_g": 0.9}},
    "salt": {"fdc_id": 2, "name": "Salt, table", "price": 1.0,
             "nutrient_values": {"energy_kcal": 0.0, "sodium_mg": 38758.0}},
    "apple": {"fdc_id": 3, "name": "Apples, raw", "price": 3.0,
              "nutrient_values": {"energy_kcal": 52.0}},
}
RECOMMENDED = [
    {"fdc_id": 167516, "name": "Waffles, buttermilk, frozen", "price": 6.85, "p_like": 0.8, "similarity": 0.9},
    {"fdc_id": 175237, "name": "Beans, black, cooked", "price": 2.72, "p_like": 0.7, "similarity": 0.9},
    {"fdc_id": 168516, "name": "Cabbage, savoy, cooked", "price": 4.2, "p_like": 0.6, "similarity": 0.9},
]


def answer(payload: dict) -> dict:
    matched = [
        [CATALOG[text] for text in dict.fromkeys(entry["ingredients"]) if text in CATALOG]
        for entry in payload["history"]
    ]
    return {"result": {"ingredients": matched, "items": RECOMMENDED if any(matched) else []}}


def failing(times: int) -> Callable[[dict], dict | None]:
    remaining = [times]

    def handler(payload: dict) -> dict | None:
        if remaining[0] > 0:
            remaining[0] -= 1
            return None
        return answer(payload)

    return handler


def wait_until(condition: Callable[[], bool], seconds: float = 5) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.fixture
def stores(tmp_path, monkeypatch):
    recipes = tmp_path / "recipes.db"
    with sqlite3.connect(recipes) as connection:
        connection.executescript(
            """
            CREATE TABLE recipe (recipe_id TEXT PRIMARY KEY, title TEXT, quality_status TEXT NOT NULL);
            CREATE TABLE recipe_ingredient (
              ingredient_id TEXT PRIMARY KEY, recipe_id TEXT NOT NULL, position INTEGER NOT NULL,
              raw_text TEXT NOT NULL, ingredient_text TEXT
            );
            CREATE TABLE recipe_instruction (recipe_id TEXT, position INTEGER, text TEXT);
            INSERT INTO recipe VALUES
              ('openrecipes:1', 'Tomato Soup', 'accepted'),
              ('openrecipes:2', 'Apple Salad', 'accepted'),
              ('openrecipes:3', 'Plain Water', 'accepted');
            INSERT INTO recipe_ingredient VALUES
              ('1:1', 'openrecipes:1', 1, '2 cups tomatoes', 'tomatoes'),
              ('1:2', 'openrecipes:1', 2, '1 tsp salt', 'salt'),
              ('1:3', 'openrecipes:1', 3, 'salt to taste', 'salt'),
              ('1:4', 'openrecipes:1', 4, '4 cups water', 'water'),
              ('2:1', 'openrecipes:2', 1, 'apple', NULL),
              ('3:1', 'openrecipes:3', 1, '1 cup water', 'water');
            """
        )
    connection.close()
    monkeypatch.setenv("DATABASE_PATH", str(recipes))
    monkeypatch.setenv("USER_DATABASE_PATH", str(tmp_path / "state" / "users.db"))
    init_user_database()
    connection = open_user_database()
    yield connection
    connection.close()


@pytest.fixture
def models(queue, sleeps):
    queue.handler = answer
    return queue


def seed_alice(stores, cache) -> int:
    alice = create_user(stores, cache, "alice")
    add_history(stores, alice.id, "openrecipes:1", rating=4)
    add_history(stores, alice.id, "openrecipes:2")
    return alice.id


def test_first_request_calls_the_model_once_and_caches_by_user_id(stores, models, cache):
    alice = seed_alice(stores, cache)

    with TestClient(app) as client:
        response = client.get("/api/recommend/alice", params={"top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == {
        "user_id": alice,
        "avg_price": pytest.approx(2.0),
        "avg_nutrient_values": {
            "energy_kcal": pytest.approx(70 / 3),
            "protein_g": pytest.approx(0.9),
            "sodium_mg": pytest.approx(38758.0),
        },
    }
    assert [item["id"] for item in body["items"]] == ["167516", "175237"]

    today = datetime.now(timezone.utc).date().isoformat()
    assert models.payloads == [
        {
            "history": [
                {"rating": None, "date": today, "ingredients": ["apple"]},
                {"rating": 4, "date": today, "ingredients": ["tomatoes", "salt", "salt", "water"]},
            ],
            "top_k": recommend.MAX_TOP_K,
        },
    ]

    cached = json.loads(cache.store[f"recommendations:{alice}"])
    assert len(cached["items"]) == 3
    assert cache.ttls[f"recommendations:{alice}"] == 86400


def test_later_requests_are_served_from_the_cache(stores, models, cache):
    seed_alice(stores, cache)

    with TestClient(app) as client:
        first = client.get("/api/recommend/alice", params={"top_k": 1}).json()
        second = client.get("/api/recommend/alice", params={"top_k": 3}).json()

    assert len(models.payloads) == 1
    assert [item["id"] for item in first["items"]] == ["167516"]
    assert [item["id"] for item in second["items"]] == ["167516", "175237", "168516"]
    assert second["profile"] == first["profile"]


def test_user_lookup_goes_to_redis_before_the_database(stores, models, cache):
    alice = seed_alice(stores, cache)
    cache.store.clear()

    with TestClient(app) as client:
        client.get("/api/recommend/alice")
        client.get("/api/recommend/alice")

    assert cache.reads[0] == "user:alice"
    assert cache.store["user:alice"] == str(alice).encode()
    assert cache.ttls["user:alice"] == 86400
    assert len(models.payloads) == 1


def test_unknown_user_neither_calls_the_model_nor_caches(stores, models, cache):
    with TestClient(app) as client:
        response = client.get("/api/recommend/nobody")

    assert response.json() == {"profile": None, "items": []}
    assert models.payloads == []
    assert cache.store == {}


def test_history_with_no_priced_ingredient_has_no_profile(stores, models, cache):
    bob = create_user(stores, cache, "bob")
    add_history(stores, bob.id, "openrecipes:3", rating=5)

    with TestClient(app) as client:
        response = client.get("/api/recommend/bob")

    assert response.json() == {"profile": None, "items": []}
    assert len(models.payloads) == 1
    assert f"recommendations:{bob.id}" in cache.store


def test_redis_down_still_answers_from_the_model(stores, models, cache):
    seed_alice(stores, cache)
    cache.down = True

    with TestClient(app) as client:
        first = client.get("/api/recommend/alice")
        second = client.get("/api/recommend/alice")

    assert first.status_code == second.status_code == 200
    assert len(first.json()["items"]) == 3
    assert len(models.payloads) == 2


def test_timeouts_are_retried_with_exponential_backoff(stores, models, sleeps, cache):
    alice = seed_alice(stores, cache)
    models.handler = failing(3)

    with TestClient(app) as client:
        response = client.get("/api/recommend/alice")

    assert response.status_code == 200
    assert len(models.jobs) == 4
    assert sleeps == [0.5, 1.0, 2.0]
    assert f"recommendations:{alice}" in cache.store


def test_every_job_has_a_deadline_and_its_own_reply_key(stores, models, cache):
    seed_alice(stores, cache)
    models.handler = failing(1)

    before = time.time()
    with TestClient(app) as client:
        client.get("/api/recommend/alice")

    first, second = models.jobs
    assert first["id"] != second["id"]
    assert before < first["deadline"] <= time.time() + jobs.DEFAULT_RESULT_TIMEOUT_SECONDS


def test_five_failures_answer_503_and_retry_in_the_background_until_cached(stores, models, sleeps, cache):
    alice = seed_alice(stores, cache)
    models.handler = failing(jobs.MAX_ATTEMPTS)

    with TestClient(app) as client:
        client.app.state.retries.interval = 0.01
        response = client.get("/api/recommend/alice")

        assert response.status_code == 503
        assert response.json() == {
            "detail": "Recommendations are unavailable right now; they will be retried in the background."
        }
        assert response.headers["Retry-After"] == "0"
        assert sleeps[:4] == [0.5, 1.0, 2.0, 4.0]

        wait_until(lambda: f"recommendations:{alice}" in cache.store)
        wait_until(lambda: not client.app.state.retries.pending)
        assert len(models.jobs) == jobs.MAX_ATTEMPTS + 1

        cached = client.get("/api/recommend/alice")

    assert cached.status_code == 200
    assert len(cached.json()["items"]) == 3
    assert len(models.jobs) == jobs.MAX_ATTEMPTS + 1


def test_background_retry_keeps_going_while_the_cache_is_down(stores, models, cache):
    alice = seed_alice(stores, cache)
    models.handler = failing(jobs.MAX_ATTEMPTS)

    with TestClient(app) as client:
        client.app.state.retries.interval = 0.01
        cache.down = True
        assert client.get("/api/recommend/alice").status_code == 503

        wait_until(lambda: len(models.jobs) >= jobs.MAX_ATTEMPTS + 3)
        assert client.app.state.retries.pending == {alice}
        cache.down = False

        wait_until(lambda: f"recommendations:{alice}" in cache.store)
        wait_until(lambda: not client.app.state.retries.pending)


def test_only_one_background_retry_runs_per_user(stores, models, cache):
    alice = seed_alice(stores, cache)
    models.handler = lambda payload: None

    with TestClient(app) as client:
        retries = client.app.state.retries
        retries.interval = 60
        assert client.get("/api/recommend/alice").status_code == 503
        assert client.get("/api/recommend/alice").status_code == 503

        assert retries.pending == {alice}
        assert len([thread for thread in threading.enumerate() if thread.name == f"retry-{alice}"]) == 1

    assert retries.stopped.is_set()


def test_unreachable_queue_is_503_and_not_cached(stores, models, sleeps, cache):
    alice = seed_alice(stores, cache)
    models.down = True

    with TestClient(app) as client:
        client.app.state.retries.interval = 60
        response = client.get("/api/recommend/alice")

    assert response.status_code == 503
    assert sleeps == [0.5, 1.0, 2.0, 4.0]
    assert f"recommendations:{alice}" not in cache.store


def test_rejected_job_is_502_without_retries(stores, models, sleeps, cache):
    alice = seed_alice(stores, cache)
    models.handler = lambda payload: {"error": "history: too long", "retryable": False}

    with TestClient(app) as client:
        response = client.get("/api/recommend/alice")
        pending = set(client.app.state.retries.pending)

    assert response.status_code == 502
    assert response.json() == {"detail": "The recommendation model rejected the request."}
    assert len(models.jobs) == 1
    assert sleeps == []
    assert pending == set()
    assert f"recommendations:{alice}" not in cache.store


def test_model_errors_are_retried(stores, models, sleeps, cache):
    seed_alice(stores, cache)
    calls = []

    def flaky(payload: dict) -> dict:
        calls.append(payload)
        return {"error": "RuntimeError('boom')", "retryable": True} if len(calls) == 1 else answer(payload)

    models.handler = flaky

    with TestClient(app) as client:
        response = client.get("/api/recommend/alice")

    assert response.status_code == 200
    assert sleeps == [0.5]
