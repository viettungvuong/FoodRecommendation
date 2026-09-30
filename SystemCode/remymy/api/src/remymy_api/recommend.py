from __future__ import annotations

import sqlite3
from collections import defaultdict
from statistics import fmean

import redis
from fastapi import HTTPException

from .db import close_database, open_database
from .jobs import submit_with_retries
from .schemas import (
    Ingredient,
    RecommendationResponse,
    RecommendedRecipe,
    UserBuyingProfile,
    UserHistory,
)


MAX_TOP_K = 50


def recipe_ingredients(recipe_ids: set[str]) -> dict[str, list[str]]:
    connection = open_database()
    try:
        placeholders = ",".join("?" for _ in recipe_ids)
        rows = connection.execute(
            "SELECT recipe_id, COALESCE(ingredient_text, raw_text) AS text FROM recipe_ingredient "
            f"WHERE recipe_id IN ({placeholders}) ORDER BY recipe_id, position",
            tuple(recipe_ids),
        ).fetchall()
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="Recipe data is unavailable.")
    finally:
        close_database(connection)
    ingredients: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        ingredients[row["recipe_id"]].append(row["text"])
    return ingredients


def buying_profile(
    history: list[UserHistory], matched: list[list[Ingredient]]
) -> UserBuyingProfile | None:
    prices: list[float] = []
    nutrients: dict[str, list[float]] = defaultdict(list)
    for ingredients in matched:
        for ingredient in ingredients:
            prices.append(ingredient.price)
            for name, value in ingredient.nutrient_values.items():
                nutrients[name].append(value)
    if not prices:
        return None
    return UserBuyingProfile(
        user_id=history[0].user_id,
        avg_price=fmean(prices),
        avg_nutrient_values={name: fmean(values) for name, values in sorted(nutrients.items())},
    )


def recommend(queue: redis.Redis, history: list[UserHistory], user_id: str) -> RecommendationResponse:
    ingredients_by_recipe = recipe_ingredients({entry.recipe_id for entry in history})
    result = submit_with_retries(
        queue,
        {
            "user_id": user_id,
            "history": [
                {
                    "rating": entry.rating,
                    "date": entry.timestamp.date().isoformat(),
                    "ingredients": ingredients_by_recipe.get(entry.recipe_id, []),
                }
                for entry in history
            ],
            "top_k": MAX_TOP_K,
        },
    )
    matched = [[Ingredient(**ingredient) for ingredient in entry] for entry in result["ingredients"]]
    recommended_items = sorted(result["items"], key=lambda x: x["p_like"], reverse=True)
    return RecommendationResponse(
        profile=buying_profile(history, matched),
        items=[
            RecommendedRecipe(id=str(item["fdc_id"]), title=item["name"], score=item["p_like"])
            for item in recommended_items
        ],
    )
