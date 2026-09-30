from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecipeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    ingredient_count: int
    ingredient_preview: list[str]
    quality_status: str


class RecipeDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    ingredients: list[str]
    instructions: list[str]
    quality_status: str


class RecipeListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RecipeSummary]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str


Diet = Literal["vegetarian", "halal", "Hinduism"]


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    special_diet: Diet | None = None
    cuisines: list[str] = Field(default_factory=list, max_length=40)
    preferred_nutrient: str | None = Field(default=None, max_length=64)


class User(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    username: str = Field(min_length=1)
    preferences: Preferences = Field(default_factory=Preferences)

class UserHistory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int
    timestamp: datetime
    rating: int | None = Field(default=None, ge=0, le=5)
    recipe_id: str

class Ingredient(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fdc_id: int
    name: str
    price: float
    nutrient_values: dict[str, float]


class UserBuyingProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int
    avg_price: float
    avg_nutrient_values: dict[str, float]

class RecommendedRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    score: float


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: UserBuyingProfile | None
    items: list[RecommendedRecipe]