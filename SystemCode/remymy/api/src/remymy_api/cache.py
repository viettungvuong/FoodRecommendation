from __future__ import annotations

import os

import redis
from pydantic import ValidationError

from .schemas import RecommendationResponse


DEFAULT_REDIS_URL = "redis://redis:6379/0"
DEFAULT_TTL_SECONDS = 86400


def connect_cache() -> redis.Redis:
    return redis.Redis.from_url(
        os.getenv("REDIS_URL", DEFAULT_REDIS_URL),
        socket_connect_timeout=2,
        socket_timeout=2,
    )


def cache_ttl() -> int:
    return int(os.getenv("CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS))


def user_key(username: str) -> str:
    return f"user:{username}"


def read_user_id(cache: redis.Redis, username: str) -> int | None:
    try:
        raw = cache.get(user_key(username))
        return None if raw is None else int(raw)
    except (redis.RedisError, ValueError):
        return None


def write_user_id(cache: redis.Redis, username: str, user_id: int) -> None:
    try:
        cache.set(user_key(username), user_id, ex=cache_ttl())
    except redis.RedisError:
        pass


def recommendations_key(user_id: int) -> str:
    return f"recommendations:{user_id}"


def read_recommendations(cache: redis.Redis, user_id: int) -> RecommendationResponse | None:
    try:
        raw = cache.get(recommendations_key(user_id))
        return None if raw is None else RecommendationResponse.model_validate_json(raw)
    except (redis.RedisError, ValidationError):
        return None


def write_recommendations(
    cache: redis.Redis, user_id: int, recommendations: RecommendationResponse
) -> bool:
    try:
        cache.set(recommendations_key(user_id), recommendations.model_dump_json(), ex=cache_ttl())
    except redis.RedisError:
        return False
    return True
