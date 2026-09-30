from __future__ import annotations

import os
import sqlite3

from fastapi import HTTPException
from redis import Redis

from .db import close_database, open_database
from .schemas import Preferences
from .users import add_history, create_user, username_exists


DEMO_USERNAME = "demo"
DEMO_HISTORY = (
    ("chicken", 5),
    ("tomato", 4),
    ("salad", 5),
    ("soup", 3),
    ("chocolate", 2),
)


def demo_enabled() -> bool:
    return os.getenv("SEED_DEMO_USER", "false").strip().lower() in {"1", "true", "yes"}


def demo_recipes() -> list[tuple[str, int]]:
    try:
        recipes = open_database()
    except HTTPException:
        return []
    picked: dict[str, int] = {}
    try:
        for word, rating in DEMO_HISTORY:
            row = recipes.execute(
                "SELECT r.recipe_id FROM recipe r "
                "WHERE r.title LIKE ? COLLATE NOCASE "
                "AND EXISTS (SELECT 1 FROM recipe_ingredient i WHERE i.recipe_id = r.recipe_id) "
                "ORDER BY r.recipe_id LIMIT 1",
                (f"%{word}%",),
            ).fetchone()
            if row is not None:
                picked.setdefault(row["recipe_id"], rating)
    except sqlite3.Error:
        return []
    finally:
        close_database(recipes)
    return list(picked.items())


def seed_demo_user(connection: sqlite3.Connection, cache: Redis) -> None:
    if username_exists(connection, cache, DEMO_USERNAME):
        return
    history = demo_recipes()
    if not history:
        return
    user = create_user(
        connection,
        cache,
        DEMO_USERNAME,
        Preferences(cuisines=["Italian", "Mexican"], preferred_nutrient="protein"),
    )
    for recipe_id, rating in history:
        add_history(connection, user.id, recipe_id, rating)
