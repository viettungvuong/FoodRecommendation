"""Validate the recipe database or build it atomically from explicit sources.

With no arguments this command is idempotent: an existing valid database is
accepted, while a missing database produces an actionable error. Ingestion is
only attempted when both recipe and FoodData Central source paths are supplied
explicitly through flags or environment variables.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

try:
    from ingest_food_data import main as ingest_main
except ModuleNotFoundError:  # Supports `python -m migrations.bootstrap`.
    from .ingest_food_data import main as ingest_main


DEFAULT_DATABASE_PATH = "/data/remymy-food.db"
REQUIRED_TABLES = {"recipe", "recipe_ingredient", "recipe_instruction"}


class DatabaseValidationError(RuntimeError):
    pass


def validate_database(path: Path) -> None:
    if not path.is_file():
        raise DatabaseValidationError(f"Database does not exist: {path}")
    try:
        with sqlite3.connect(path) as connection:
            result = connection.execute("PRAGMA quick_check").fetchone()[0]
            if str(result).lower() != "ok":
                raise DatabaseValidationError("SQLite quick_check did not return ok")
            placeholders = ",".join("?" for _ in REQUIRED_TABLES)
            tables = {
                row[0]
                for row in connection.execute(
                    f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({placeholders})",
                    tuple(REQUIRED_TABLES),
                )
            }
            if tables != REQUIRED_TABLES:
                missing = ", ".join(sorted(REQUIRED_TABLES - tables))
                raise DatabaseValidationError(f"required recipe tables are missing: {missing}")
    except sqlite3.Error as exc:
        raise DatabaseValidationError("SQLite validation failed") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-path",
        type=Path,
        default=Path(os.getenv("DATABASE_PATH", DEFAULT_DATABASE_PATH)),
        help="SQLite output path; defaults to DATABASE_PATH",
    )
    parser.add_argument(
        "--recipes",
        type=Path,
        default=Path(os.environ["RECIPES_SOURCE"]) if os.getenv("RECIPES_SOURCE") else None,
        help="Open Recipes .db/.csv; required for a missing database",
    )
    parser.add_argument(
        "--fdc-dir",
        type=Path,
        default=Path(os.environ["FDC_SOURCE_DIR"]) if os.getenv("FDC_SOURCE_DIR") else None,
        help="FoodData Central CSV directory; required for a missing database",
    )
    parser.add_argument(
        "--fdc-release",
        default=os.getenv("FDC_RELEASE", "unspecified"),
        help="FoodData Central release label",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an existing database; never ingest",
    )
    return parser.parse_args(argv)


def build_database(args: argparse.Namespace) -> None:
    if args.recipes is None or args.fdc_dir is None:
        raise DatabaseValidationError(
            "Database is missing. Provide both --recipes and --fdc-dir "
            "or set RECIPES_SOURCE and FDC_SOURCE_DIR to ingest it."
        )
    if not args.recipes.is_file():
        raise DatabaseValidationError(f"Recipe source does not exist: {args.recipes}")
    if not args.fdc_dir.is_dir():
        raise DatabaseValidationError(f"FDC source directory does not exist: {args.fdc_dir}")

    args.database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{args.database_path.name}.",
            suffix=".tmp",
            dir=args.database_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        ingest_main(
            [
                "--recipes",
                str(args.recipes),
                "--fdc-dir",
                str(args.fdc_dir),
                "--fdc-release",
                args.fdc_release,
                "--output",
                str(temporary_path),
                "--replace",
            ]
        )
        validate_database(temporary_path)
        temporary_path.replace(args.database_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        if args.database_path.is_file():
            validate_database(args.database_path)
            print(f"Database is valid: {args.database_path}")
            return 0
        if args.validate_only:
            raise DatabaseValidationError(
                f"Database does not exist: {args.database_path}"
            )
        build_database(args)
        print(f"Database created and validated: {args.database_path}")
        return 0
    except (DatabaseValidationError, OSError, SystemExit) as exc:
        print(f"Bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
