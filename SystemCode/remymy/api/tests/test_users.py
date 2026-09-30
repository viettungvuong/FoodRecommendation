from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from remymy_api.main import app
from remymy_api.schemas import Preferences, UserHistory
from remymy_api.users import (
    add_history,
    create_user,
    fetch_user_history,
    init_user_database,
    open_user_database,
    username_exists,
)


@pytest.fixture
def users(tmp_path, monkeypatch):
    recipes = tmp_path / "recipes.db"
    with sqlite3.connect(recipes) as connection:
        connection.executescript(
            """
            CREATE TABLE recipe (recipe_id TEXT PRIMARY KEY, title TEXT, quality_status TEXT NOT NULL);
            CREATE TABLE recipe_ingredient (recipe_id TEXT);
            CREATE TABLE recipe_instruction (recipe_id TEXT);
            INSERT INTO recipe VALUES
              ('openrecipes:1', 'Tomato Soup', 'accepted'),
              ('openrecipes:2', 'Apple Salad', 'accepted');
            """
        )
    connection.close()
    monkeypatch.setenv("DATABASE_PATH", str(recipes))
    monkeypatch.setenv("USER_DATABASE_PATH", str(tmp_path / "state" / "users.db"))
    init_user_database()
    connection = open_user_database()
    yield connection
    connection.close()


def queries(connection: sqlite3.Connection) -> list[str]:
    seen: list[str] = []
    connection.set_trace_callback(seen.append)
    return seen


def store(connection, document: object) -> None:
    connection.execute(
        "INSERT INTO app_user (username, preferences) VALUES ('probe', jsonb(?))",
        (json.dumps(document),),
    )


VALID = {"special_diet": "halal", "cuisines": ["Thai"], "preferred_nutrient": "fibre"}


@pytest.mark.parametrize(
    "document",
    [
        {k: v for k, v in VALID.items() if k != "cuisines"},
        {**VALID, "allergies": []},
        {**VALID, "special_diet": "vegan"},
        {**VALID, "special_diet": 1},
        {**VALID, "cuisines": "Thai"},
        {**VALID, "cuisines": [f"c{i}" for i in range(41)]},
        {**VALID, "cuisines": ["Thai", 7]},
        {**VALID, "preferred_nutrient": "x" * 65},
        {**VALID, "preferred_nutrient": ["fibre"]},
        ["not", "an", "object"],
    ],
)
def test_schema_rejects_preferences_outside_the_model(users, document):
    with pytest.raises(sqlite3.IntegrityError):
        store(users, document)


def test_schema_accepts_the_model_and_stores_binary_jsonb(users):
    store(users, {**VALID, "cuisines": [f"c{i}" for i in range(40)]})
    kind = users.execute("SELECT typeof(preferences) FROM app_user").fetchone()[0]
    assert kind == "blob"


def test_schema_rejects_preferences_stored_as_text(users):
    with pytest.raises(sqlite3.IntegrityError):
        users.execute(
            "INSERT INTO app_user (username, preferences) VALUES ('probe', ?)",
            (json.dumps(VALID),),
        )


def test_schema_default_preferences_are_valid(users):
    users.execute("INSERT INTO app_user (username) VALUES ('probe')")
    (raw,) = users.execute("SELECT json(preferences) FROM app_user").fetchone()
    assert Preferences.model_validate_json(raw) == Preferences()


def test_model_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        Preferences.model_validate({**VALID, "allergies": []})


def test_create_user_writes_the_user_id_to_redis(users, cache):
    alice = create_user(users, cache, "alice")
    assert cache.store == {"user:alice": str(alice.id).encode()}


def test_username_exists_answers_from_redis_without_querying(users, cache):
    create_user(users, cache, "alice")
    seen = queries(users)
    assert username_exists(users, cache, "alice") is True
    assert seen == []


def test_username_exists_falls_back_to_the_database_and_caches(users, cache):
    alice = create_user(users, cache, "alice")
    cache.store.clear()

    seen = queries(users)
    assert username_exists(users, cache, "alice") is True
    assert len(seen) == 1
    assert cache.store == {"user:alice": str(alice.id).encode()}

    seen.clear()
    assert username_exists(users, cache, "alice") is True
    assert seen == []


def test_unknown_username_is_not_cached(users, cache):
    assert username_exists(users, cache, "bob") is False
    assert cache.reads == ["user:bob"]
    assert cache.store == {}


def test_redis_down_answers_from_the_database(users, cache):
    create_user(users, cache, "alice")
    cache.down = True
    assert username_exists(users, cache, "alice") is True
    assert username_exists(users, cache, "bob") is False


def test_duplicate_username_is_rejected(users, cache):
    create_user(users, cache, "alice")
    with pytest.raises(sqlite3.IntegrityError):
        create_user(users, cache, "alice")


def test_history_newest_first_and_limited(users, cache):
    alice = create_user(users, cache, "alice")
    first = add_history(users, alice.id, "openrecipes:1", rating=4)
    second = add_history(users, alice.id, "openrecipes:2")

    history = fetch_user_history(users, cache, "alice")
    assert history == [second, first]
    assert history[0].rating is None
    assert fetch_user_history(users, cache, "alice", limit=1) == [second]


def test_history_for_unknown_username_is_empty(users, cache):
    create_user(users, cache, "alice")
    assert fetch_user_history(users, cache, "bob") == []


def test_history_only_shows_that_users_rows(users, cache):
    alice = create_user(users, cache, "alice")
    create_user(users, cache, "bob")
    add_history(users, alice.id, "openrecipes:1", rating=5)
    assert fetch_user_history(users, cache, "bob") == []


def test_history_rejects_unknown_recipe(users, cache):
    alice = create_user(users, cache, "alice")
    with pytest.raises(LookupError):
        add_history(users, alice.id, "openrecipes:999")


@pytest.mark.parametrize("rating", [-1, 6])
def test_history_rejects_rating_outside_zero_to_five(users, cache, rating):
    alice = create_user(users, cache, "alice")
    with pytest.raises(sqlite3.IntegrityError):
        add_history(users, alice.id, "openrecipes:1", rating=rating)


def test_history_rejects_unknown_user(users, cache):
    with pytest.raises(sqlite3.IntegrityError):
        add_history(users, 999, "openrecipes:1")


def test_history_model_rejects_rating_outside_zero_to_five():
    with pytest.raises(ValidationError):
        UserHistory(user_id=1, timestamp="2026-09-28T00:00:00Z", rating=6, recipe_id="x")


def test_startup_creates_the_database_and_cache(users, cache):
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.app.state.cache is cache
