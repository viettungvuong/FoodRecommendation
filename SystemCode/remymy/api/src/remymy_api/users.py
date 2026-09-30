from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from redis import Redis

from .cache import read_user_id, write_user_id
from .db import close_database, open_database
from .schemas import Preferences, User, UserHistory


DEFAULT_USER_DATABASE_PATH = "/state/remymy-users.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS app_user (
  user_id     INTEGER PRIMARY KEY,
  username    TEXT NOT NULL UNIQUE CHECK (length(username) > 0),
  preferences JSONB NOT NULL DEFAULT (
    jsonb('{"special_diet": null, "cuisines": [], "preferred_nutrient": null}')
  ),
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

  CHECK (json_valid(preferences, 8)),
  CHECK (
    json_type(preferences) = 'object'
    AND json_type(preferences, '$.special_diet') IS NOT NULL
    AND json_type(preferences, '$.cuisines') IS NOT NULL
    AND json_type(preferences, '$.preferred_nutrient') IS NOT NULL
    AND json_remove(preferences, '$.special_diet', '$.cuisines', '$.preferred_nutrient') = '{}'
  ),
  CHECK (
    json_type(preferences, '$.special_diet') = 'null'
    OR json_extract(preferences, '$.special_diet') IN ('vegetarian', 'halal', 'Hinduism')
  ),
  CHECK (
    json_type(preferences, '$.cuisines') = 'array'
    AND json_array_length(preferences, '$.cuisines') <= 40
  ),
  CHECK (
    json_type(preferences, '$.preferred_nutrient') = 'null'
    OR (
      json_type(preferences, '$.preferred_nutrient') = 'text'
      AND length(json_extract(preferences, '$.preferred_nutrient')) <= 64
    )
  )
);

CREATE TRIGGER IF NOT EXISTS app_user_cuisines_are_text_on_insert
BEFORE INSERT ON app_user
WHEN EXISTS (SELECT 1 FROM json_each(NEW.preferences, '$.cuisines') WHERE type <> 'text')
BEGIN
  SELECT RAISE(ABORT, 'preferences.cuisines must contain only strings');
END;

CREATE TRIGGER IF NOT EXISTS app_user_cuisines_are_text_on_update
BEFORE UPDATE OF preferences ON app_user
WHEN EXISTS (SELECT 1 FROM json_each(NEW.preferences, '$.cuisines') WHERE type <> 'text')
BEGIN
  SELECT RAISE(ABORT, 'preferences.cuisines must contain only strings');
END;

CREATE TABLE IF NOT EXISTS user_history (
  history_id INTEGER PRIMARY KEY,
  user_id    INTEGER NOT NULL REFERENCES app_user (user_id) ON DELETE CASCADE,
  timestamp  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  rating     INTEGER CHECK (rating IS NULL OR rating IN (0, 1, 2, 3, 4, 5)),
  recipe_id  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS user_history_by_user
  ON user_history (user_id, timestamp DESC);
"""


def user_database_path() -> Path:
    return Path(os.getenv("USER_DATABASE_PATH", DEFAULT_USER_DATABASE_PATH))


def open_user_database() -> sqlite3.Connection:
    connection = sqlite3.connect(user_database_path())
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_user_database() -> None:
    user_database_path().parent.mkdir(parents=True, exist_ok=True)
    connection = open_user_database()
    try:
        connection.executescript(SCHEMA)
    finally:
        connection.close()


def find_user_id(connection: sqlite3.Connection, cache: Redis, username: str) -> int | None:
    user_id = read_user_id(cache, username)
    if user_id is not None:
        return user_id
    row = connection.execute(
        "SELECT user_id FROM app_user WHERE username = ?", (username,)
    ).fetchone()
    if row is None:
        return None
    write_user_id(cache, username, row["user_id"])
    return row["user_id"]


def username_exists(connection: sqlite3.Connection, cache: Redis, username: str) -> bool:
    return find_user_id(connection, cache, username) is not None


def fetch_user_history(
    connection: sqlite3.Connection,
    cache: Redis,
    username: str,
    limit: int = 50,
) -> list[UserHistory]:
    user_id = find_user_id(connection, cache, username)
    if user_id is None:
        return []
    rows = connection.execute(
        """
        SELECT user_id, timestamp, rating, recipe_id
        FROM user_history
        WHERE user_id = ?
        ORDER BY timestamp DESC, history_id DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    return [UserHistory(**dict(row)) for row in rows]


def create_user(
    connection: sqlite3.Connection,
    cache: Redis,
    username: str,
    preferences: Preferences | None = None,
) -> User:
    preferences = preferences or Preferences()
    with connection:
        (row,) = connection.execute(
            "INSERT INTO app_user (username, preferences) VALUES (?, jsonb(?)) "
            "RETURNING user_id",
            (username, preferences.model_dump_json()),
        ).fetchall()
    write_user_id(cache, username, row["user_id"])
    return User(id=row["user_id"], username=username, preferences=preferences)


def add_history(
    connection: sqlite3.Connection,
    user_id: int,
    recipe_id: str,
    rating: int | None = None,
) -> UserHistory:
    recipes = open_database()
    try:
        known = recipes.execute(
            "SELECT 1 FROM recipe WHERE recipe_id = ?", (recipe_id,)
        ).fetchone()
    finally:
        close_database(recipes)
    if known is None:
        raise LookupError(f"Unknown recipe: {recipe_id}")

    with connection:
        (row,) = connection.execute(
            "INSERT INTO user_history (user_id, rating, recipe_id) VALUES (?, ?, ?) "
            "RETURNING timestamp",
            (user_id, rating, recipe_id),
        ).fetchall()
    return UserHistory(
        user_id=user_id, timestamp=row["timestamp"], rating=rating, recipe_id=recipe_id
    )
