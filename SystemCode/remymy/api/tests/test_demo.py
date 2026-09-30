from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from remymy_api.demo import DEMO_USERNAME
from remymy_api.main import app
from remymy_api.users import open_user_database


@pytest.fixture
def recipes(tmp_path, monkeypatch):
    path = tmp_path / "recipes.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE recipe (recipe_id TEXT PRIMARY KEY, title TEXT, quality_status TEXT NOT NULL);
            CREATE TABLE recipe_ingredient (
              ingredient_id TEXT PRIMARY KEY, recipe_id TEXT NOT NULL, position INTEGER NOT NULL,
              raw_text TEXT NOT NULL, ingredient_text TEXT
            );
            CREATE TABLE recipe_instruction (recipe_id TEXT, position INTEGER, text TEXT);
            INSERT INTO recipe VALUES
              ('openrecipes:1', 'Roast Chicken', 'accepted'),
              ('openrecipes:2', 'Tomato Soup', 'accepted'),
              ('openrecipes:3', 'Greek Salad', 'accepted'),
              ('openrecipes:4', 'Chicken Salad Without Ingredients', 'accepted');
            INSERT INTO recipe_ingredient VALUES
              ('1:1', 'openrecipes:1', 1, '1 chicken', 'chicken'),
              ('2:1', 'openrecipes:2', 1, '3 tomatoes', 'tomatoes'),
              ('3:1', 'openrecipes:3', 1, '1 cucumber', 'cucumber');
            """
        )
    connection.close()
    monkeypatch.setenv("DATABASE_PATH", str(path))
    monkeypatch.setenv("USER_DATABASE_PATH", str(tmp_path / "state" / "users.db"))
    return path


def demo_history() -> list[tuple[str, int]]:
    connection = open_user_database()
    try:
        return [
            (row["recipe_id"], row["rating"])
            for row in connection.execute(
                "SELECT h.recipe_id, h.rating FROM user_history h "
                "JOIN app_user u ON u.user_id = h.user_id WHERE u.username = ? ORDER BY h.history_id",
                (DEMO_USERNAME,),
            )
        ]
    finally:
        connection.close()


def test_demo_user_is_seeded_once_from_matching_recipes(recipes, cache, monkeypatch):
    monkeypatch.setenv("SEED_DEMO_USER", "true")

    with TestClient(app):
        pass
    with TestClient(app):
        pass

    assert f"user:{DEMO_USERNAME}" in cache.store
    assert demo_history() == [
        ("openrecipes:1", 5),
        ("openrecipes:2", 4),
        ("openrecipes:3", 5),
    ]


def test_demo_user_is_not_seeded_by_default(recipes, cache, monkeypatch):
    monkeypatch.delenv("SEED_DEMO_USER", raising=False)

    with TestClient(app):
        pass

    assert demo_history() == []


def test_missing_recipe_database_skips_the_demo_user(recipes, cache, monkeypatch):
    monkeypatch.setenv("SEED_DEMO_USER", "true")
    recipes.unlink()

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert demo_history() == []


def test_swagger_is_served_with_the_demo_username_prefilled(recipes, cache):
    with TestClient(app) as client:
        assert client.get("/", follow_redirects=False).headers["location"] == "/docs"
        assert client.get("/docs").status_code == 200
        spec = client.get("/openapi.json").json()

    parameter = spec["paths"]["/api/recommend/{username}"]["get"]["parameters"][0]
    assert parameter["name"] == "username"
    assert parameter["examples"]["demo"]["value"] == DEMO_USERNAME
