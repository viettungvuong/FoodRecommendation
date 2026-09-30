from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException


DEFAULT_DATABASE_PATH = "/data/remymy-food.db"
REQUIRED_TABLES = {"recipe", "recipe_ingredient", "recipe_instruction"}


def database_path() -> Path:
    return Path(os.getenv("DATABASE_PATH", DEFAULT_DATABASE_PATH))


def open_database() -> sqlite3.Connection:
    """Open a valid recipe database or raise a generic, non-leaky 503."""

    path = database_path()
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Recipe data is unavailable.")
    try:
        # The API database is mounted read-only and is produced completely
        # before the API starts. Immutable read-only mode prevents SQLite from
        # attempting to create or update sidecar -shm/-wal files. Quote the
        # resolved path so spaces, query characters, and other path bytes
        # cannot alter the SQLite URI.
        uri_path = quote(str(path.resolve()), safe="/")
        connection = sqlite3.connect(
            f"file:{uri_path}?mode=ro&immutable=1",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in REQUIRED_TABLES)
        tables = {
            row[0]
            for row in connection.execute(
                f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({placeholders})",
                tuple(REQUIRED_TABLES),
            )
        }
        if tables != REQUIRED_TABLES:
            connection.close()
            raise HTTPException(status_code=503, detail="Recipe data is unavailable.")
        return connection
    except HTTPException:
        raise
    except (OSError, sqlite3.Error):
        raise HTTPException(status_code=503, detail="Recipe data is unavailable.")


def close_database(connection: sqlite3.Connection) -> None:
    try:
        connection.close()
    except sqlite3.Error:
        pass


def split_preview(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in value.split("\n") if part]
