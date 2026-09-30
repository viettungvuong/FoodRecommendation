#!/usr/bin/env python3
"""Ingest Open Recipes 13k and FoodData Central Foundation CSVs into SQLite.

The loader keeps raw source records, normalizes recipe children, and loads FDC
food/nutrient/portion tables. It intentionally does not guess recipe nutrition
or ingredient-to-FDC matches; those are derived steps with their own versions.

Example:
  python migrations/ingest_food_data.py \
    --recipes /data/13k-recipes.db \
    --fdc-dir /data/FoodData_Central_foundation_food_csv_2026-04-30 \
    --output data/remymy-food.db
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE source_snapshot (
  snapshot_id TEXT PRIMARY KEY,
  source_name TEXT NOT NULL,
  source_version TEXT NOT NULL,
  source_uri TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  license_text TEXT,
  retrieved_at TEXT NOT NULL,
  row_count INTEGER NOT NULL DEFAULT 0,
  schema_hash TEXT NOT NULL
);

CREATE TABLE source_record (
  source_record_id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id),
  source_native_id TEXT NOT NULL,
  record_type TEXT NOT NULL,
  raw_payload TEXT NOT NULL,
  raw_hash TEXT NOT NULL,
  UNIQUE (snapshot_id, source_native_id, record_type)
);

CREATE TABLE quarantine_record (
  quarantine_id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id),
  source_native_id TEXT NOT NULL,
  record_type TEXT NOT NULL,
  reason TEXT NOT NULL,
  raw_payload TEXT NOT NULL
);

CREATE TABLE recipe (
  recipe_id TEXT PRIMARY KEY,
  source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id),
  source_record_id TEXT NOT NULL UNIQUE REFERENCES source_record(source_record_id),
  title TEXT,
  instructions_raw TEXT,
  description TEXT,
  cuisine TEXT,
  prep_minutes INTEGER,
  cook_minutes INTEGER,
  total_minutes INTEGER,
  yield_text TEXT,
  servings REAL,
  quality_status TEXT NOT NULL,
  raw_hash TEXT NOT NULL,
  normalization_version TEXT NOT NULL
);

CREATE TABLE recipe_ingredient (
  ingredient_id TEXT PRIMARY KEY,
  recipe_id TEXT NOT NULL REFERENCES recipe(recipe_id),
  position INTEGER NOT NULL,
  raw_text TEXT NOT NULL,
  quantity_value REAL,
  quantity_min REAL,
  quantity_max REAL,
  unit_raw TEXT,
  unit_normalized TEXT,
  ingredient_text TEXT,
  preparation TEXT,
  optional INTEGER,
  parse_status TEXT NOT NULL,
  parse_confidence REAL
);

CREATE TABLE recipe_instruction (
  recipe_id TEXT NOT NULL REFERENCES recipe(recipe_id),
  position INTEGER NOT NULL,
  text TEXT NOT NULL,
  PRIMARY KEY (recipe_id, position)
);

CREATE TABLE food_category (
  id TEXT PRIMARY KEY,
  code TEXT,
  description TEXT NOT NULL
);

CREATE TABLE measure_unit (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE nutrient (
  nutrient_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  unit_name TEXT NOT NULL,
  nutrient_number TEXT,
  rank INTEGER
);

CREATE TABLE food (
  fdc_id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id),
  data_type TEXT NOT NULL,
  description TEXT,
  food_category_id TEXT,
  publication_date TEXT,
  foundation_flag INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE foundation_food (
  fdc_id TEXT PRIMARY KEY REFERENCES food(fdc_id),
  ndb_number TEXT,
  footnote TEXT
);

CREATE TABLE food_nutrient (
  observation_id TEXT PRIMARY KEY,
  fdc_id TEXT NOT NULL REFERENCES food(fdc_id),
  nutrient_id TEXT NOT NULL REFERENCES nutrient(nutrient_id),
  snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id),
  amount REAL,
  data_points INTEGER,
  derivation_id TEXT,
  min_amount REAL,
  max_amount REAL,
  median_amount REAL,
  footnote TEXT,
  min_year_acquired INTEGER
);

CREATE TABLE food_portion (
  portion_id TEXT PRIMARY KEY,
  fdc_id TEXT REFERENCES food(fdc_id),
  seq_num INTEGER,
  amount REAL,
  measure_unit_id TEXT REFERENCES measure_unit(id),
  portion_description TEXT,
  modifier TEXT,
  gram_weight REAL,
  data_points INTEGER,
  footnote TEXT,
  min_year_acquired INTEGER
);

CREATE TABLE food_lineage (
  lineage_id TEXT PRIMARY KEY,
  fdc_id TEXT REFERENCES food(fdc_id),
  fdc_of_input_food TEXT,
  ingredient_code TEXT,
  ingredient_description TEXT,
  unit TEXT,
  portion_code TEXT,
  portion_description TEXT,
  gram_weight REAL,
  retention_code TEXT
);

CREATE TABLE ingredient_food_match (
  ingredient_id TEXT NOT NULL REFERENCES recipe_ingredient(ingredient_id),
  fdc_id TEXT NOT NULL REFERENCES food(fdc_id),
  candidate_rank INTEGER NOT NULL,
  match_method TEXT NOT NULL,
  match_score REAL NOT NULL,
  accepted INTEGER NOT NULL DEFAULT 0,
  review_status TEXT NOT NULL DEFAULT 'unreviewed',
  mapping_version TEXT NOT NULL,
  PRIMARY KEY (ingredient_id, fdc_id, mapping_version)
);

CREATE TABLE recipe_nutrition (
  recipe_id TEXT NOT NULL REFERENCES recipe(recipe_id),
  nutrient_id TEXT NOT NULL REFERENCES nutrient(nutrient_id),
  amount REAL,
  unit TEXT,
  basis TEXT NOT NULL,
  estimated INTEGER NOT NULL DEFAULT 1,
  ingredient_coverage_pct REAL,
  mass_coverage_pct REAL,
  serving_count REAL,
  confidence REAL,
  calculation_version TEXT,
  calculated_at TEXT,
  PRIMARY KEY (recipe_id, nutrient_id, basis, calculation_version)
);

CREATE INDEX recipe_title_idx ON recipe(title);
CREATE INDEX recipe_quality_idx ON recipe(quality_status);
CREATE INDEX recipe_ingredient_recipe_idx ON recipe_ingredient(recipe_id, position);
CREATE INDEX recipe_ingredient_text_idx ON recipe_ingredient(ingredient_text);
CREATE INDEX food_description_idx ON food(description);
CREATE INDEX food_foundation_idx ON food(foundation_flag, data_type);
CREATE INDEX food_nutrient_food_idx ON food_nutrient(fdc_id, nutrient_id);
CREATE INDEX food_portion_food_idx ON food_portion(fdc_id);
"""


NORMALIZATION_VERSION = "recipe-normalizer-v1"
FDC_SCHEMA_FILES = (
    "food.csv",
    "foundation_food.csv",
    "food_nutrient.csv",
    "nutrient.csv",
    "food_portion.csv",
    "measure_unit.csv",
    "food_category.csv",
    "input_food.csv",
)

UNICODE_FRACTIONS = {
    "¼": 0.25,
    "½": 0.5,
    "¾": 0.75,
    "⅐": 1 / 7,
    "⅑": 1 / 9,
    "⅒": 0.1,
    "⅓": 1 / 3,
    "⅔": 2 / 3,
    "⅕": 0.2,
    "⅖": 0.4,
    "⅗": 0.6,
    "⅘": 0.8,
    "⅙": 1 / 6,
    "⅚": 5 / 6,
    "⅛": 0.125,
    "⅜": 0.375,
    "⅝": 0.625,
    "⅞": 0.875,
}
UNIT_ALIASES = {
    "t": "teaspoon",
    "tsp": "teaspoon",
    "tsps": "teaspoon",
    "teaspoons": "teaspoon",
    "tbsp": "tablespoon",
    "tbsps": "tablespoon",
    "tablespoons": "tablespoon",
    "c": "cup",
    "cups": "cup",
    "oz": "ounce",
    "ounces": "ounce",
    "lb": "pound",
    "lbs": "pound",
    "pounds": "pound",
    "g": "gram",
    "grams": "gram",
    "kg": "kilogram",
    "ml": "milliliter",
    "l": "liter",
}
UNIT_WORDS = sorted(UNIT_ALIASES, key=len, reverse=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path, filenames: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(filenames):
        file_path = find_file(path, name)
        digest.update(name.encode("utf-8"))
        digest.update(sha256_file(file_path).encode("ascii"))
    return digest.hexdigest()


def find_file(base: Path, name: str) -> Path:
    matches = sorted(base.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name!r} below {base}")
    return matches[0]


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def integer(value: Any) -> int | None:
    try:
        return int(str(value).strip()) if clean(value) is not None else None
    except (TypeError, ValueError):
        return None


def number(value: Any) -> float | None:
    try:
        return float(str(value).strip()) if clean(value) is not None else None
    except (TypeError, ValueError):
        return None


def normalize_fraction_text(text: str) -> str:
    result = text
    for symbol, value in UNICODE_FRACTIONS.items():
        result = result.replace(symbol, f" {value}")
    return result


def parse_quantity_prefix(text: str) -> tuple[float | None, float | None, float | None, str]:
    """Parse common numeric prefixes, retaining the unparsed remainder."""
    source = normalize_fraction_text(text).replace("–", "-").replace("—", "-").strip()
    pattern = re.compile(
        r"^(?P<a>\d+(?:\.\d+)?)(?:\s+(?P<b>\d+(?:\.\d+)?))?"
        r"(?:\s*(?:-|to)\s*(?P<c>\d+(?:\.\d+)?))?\b\s*"
    )
    match = pattern.match(source)
    if not match:
        return None, None, None, text.strip()
    first = float(match.group("a"))
    value = first + (float(match.group("b")) if match.group("b") else 0.0)
    upper = float(match.group("c")) if match.group("c") else None
    if upper is not None:
        return None, value, upper, source[match.end() :].strip()
    return value, None, None, source[match.end() :].strip()


def parse_ingredient_line(raw_text: str) -> dict[str, Any]:
    raw = " ".join(raw_text.strip().split())
    value, minimum, maximum, remainder = parse_quantity_prefix(raw)
    unit_raw = None
    unit_normalized = None
    if remainder:
        unit_match = re.match(r"^(?P<unit>[A-Za-z]+)\b\s*(?P<rest>.*)$", remainder)
        if unit_match and unit_match.group("unit").lower() in UNIT_ALIASES:
            unit_raw = unit_match.group("unit")
            unit_normalized = UNIT_ALIASES[unit_raw.lower()]
            remainder = unit_match.group("rest").strip()
    ingredient_text = remainder.strip(" ,") or None
    preparation = None
    if ingredient_text and "," in ingredient_text:
        ingredient_text, preparation = (part.strip() or None for part in ingredient_text.split(",", 1))
    optional = int(bool(re.search(r"\b(optional|for serving|to serve)\b", raw, re.I)))
    status = "parsed" if ingredient_text and (value is not None or minimum is not None) else "partial"
    confidence = 1.0 if status == "parsed" else 0.5 if ingredient_text else 0.0
    return {
        "raw_text": raw,
        "quantity_value": value,
        "quantity_min": minimum,
        "quantity_max": maximum,
        "unit_raw": unit_raw,
        "unit_normalized": unit_normalized,
        "ingredient_text": ingredient_text,
        "preparation": preparation,
        "optional": optional,
        "parse_status": status,
        "parse_confidence": confidence,
    }


def open_csv(path: Path) -> tuple[csv.DictReader, Any]:
    handle = path.open("r", encoding="utf-8-sig", newline="")
    return csv.DictReader(handle), handle


def recipe_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        connection = sqlite3.connect(path)
        try:
            cursor = connection.execute("SELECT id, Title, Ingredients, Instructions FROM recipes")
            for row in cursor:
                yield {"id": row[0], "Title": row[1], "Ingredients": row[2], "Instructions": row[3]}
        finally:
            connection.close()
        return
    reader, handle = open_csv(path)
    try:
        headers = {str(field).strip().lower(): field for field in reader.fieldnames or []}
        required = {"id", "title", "ingredients", "instructions"}
        missing = required - headers.keys()
        if missing:
            raise ValueError(f"Recipe CSV missing columns: {sorted(missing)}")
        for row in reader:
            yield {
                "id": row[headers["id"]],
                "Title": row[headers["title"]],
                "Ingredients": row[headers["ingredients"]],
                "Instructions": row[headers["instructions"]],
            }
    finally:
        handle.close()


def fdc_rows(base: Path, filename: str) -> Iterator[dict[str, Any]]:
    reader, handle = open_csv(find_file(base, filename))
    try:
        yield from reader
    finally:
        handle.close()


def insert_source_record(
    connection: sqlite3.Connection,
    snapshot_id: str,
    native_id: str,
    record_type: str,
    payload: dict[str, Any],
) -> str:
    source_record_id = f"{snapshot_id}:{record_type}:{native_id}"
    raw = json_text(payload)
    raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    connection.execute(
        "INSERT INTO source_record VALUES (?, ?, ?, ?, ?, ?)",
        (source_record_id, snapshot_id, str(native_id), record_type, raw, raw_hash),
    )
    return source_record_id


def quarantine(
    connection: sqlite3.Connection,
    snapshot_id: str,
    native_id: str,
    record_type: str,
    reason: str,
    payload: dict[str, Any],
) -> None:
    connection.execute(
        "INSERT INTO quarantine_record(snapshot_id, source_native_id, record_type, reason, raw_payload) VALUES (?, ?, ?, ?, ?)",
        (snapshot_id, str(native_id), record_type, reason, json_text(payload)),
    )


def insert_recipes(connection: sqlite3.Connection, snapshot_id: str, path: Path) -> int:
    count = 0
    for row in recipe_rows(path):
        native_id = clean(row.get("id")) or f"row-{count + 1}"
        payload = {"id": row.get("id"), "Title": row.get("Title"), "Ingredients": row.get("Ingredients"), "Instructions": row.get("Instructions")}
        source_record_id = insert_source_record(connection, snapshot_id, native_id, "recipe", payload)
        title = clean(row.get("Title"))
        instructions = clean(row.get("Instructions"))
        raw_ingredients = row.get("Ingredients")
        reasons: list[str] = []
        if not title:
            reasons.append("blank_title")
        if not instructions:
            reasons.append("blank_instructions")
        try:
            parsed_ingredients = ast.literal_eval(str(raw_ingredients)) if raw_ingredients is not None else []
            if not isinstance(parsed_ingredients, list) or not all(isinstance(item, str) for item in parsed_ingredients):
                raise ValueError("expected list of strings")
        except (SyntaxError, ValueError, TypeError):
            parsed_ingredients = []
            reasons.append("malformed_ingredients")
        if not parsed_ingredients:
            reasons.append("empty_ingredients")
        quality = "quarantined" if not title else "partial" if reasons else "accepted"
        recipe_id = f"openrecipes:{native_id}"
        raw_hash = hashlib.sha256(json_text(payload).encode("utf-8")).hexdigest()
        connection.execute(
            "INSERT INTO recipe(recipe_id, source_snapshot_id, source_record_id, title, instructions_raw, quality_status, raw_hash, normalization_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (recipe_id, snapshot_id, source_record_id, title, instructions, quality, raw_hash, NORMALIZATION_VERSION),
        )
        for position, raw_line in enumerate(parsed_ingredients, start=1):
            parsed = parse_ingredient_line(raw_line)
            ingredient_id = f"{recipe_id}:ingredient:{position}"
            connection.execute(
                "INSERT INTO recipe_ingredient VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ingredient_id, recipe_id, position, parsed["raw_text"], parsed["quantity_value"], parsed["quantity_min"], parsed["quantity_max"], parsed["unit_raw"], parsed["unit_normalized"], parsed["ingredient_text"], parsed["preparation"], parsed["optional"], parsed["parse_status"], parsed["parse_confidence"]),
            )
        if instructions:
            steps = [part.strip() for part in re.split(r"\n\s*\n+", instructions) if part.strip()]
            if not steps:
                steps = [instructions]
            for position, step in enumerate(steps, start=1):
                connection.execute("INSERT INTO recipe_instruction VALUES (?, ?, ?)", (recipe_id, position, step))
        for reason in reasons:
            quarantine(connection, snapshot_id, native_id, "recipe", reason, payload)
        count += 1
    return count


def insert_fdc(connection: sqlite3.Connection, snapshot_id: str, base: Path) -> dict[str, int]:
    foundation_ids: set[str] = set()
    food_ids: set[str] = set()
    nutrient_ids: set[str] = set()
    measure_unit_ids: set[str] = set()
    counts: dict[str, int] = {}
    foundation_rows: list[dict[str, Any]] = []
    for row in fdc_rows(base, "foundation_food.csv"):
        fdc_id = clean(row.get("fdc_id"))
        if fdc_id:
            foundation_ids.add(fdc_id)
            foundation_rows.append(dict(row))

    for filename, table in (("food_category.csv", "food_category"), ("measure_unit.csv", "measure_unit"), ("nutrient.csv", "nutrient")):
        for row in fdc_rows(base, filename):
            key = clean(row.get("id"))
            if not key:
                continue
            insert_source_record(connection, snapshot_id, key, table, dict(row))
            if table == "food_category":
                connection.execute("INSERT INTO food_category VALUES (?, ?, ?)", (key, clean(row.get("code")), clean(row.get("description")) or ""))
            elif table == "measure_unit":
                connection.execute("INSERT INTO measure_unit VALUES (?, ?)", (key, clean(row.get("name")) or ""))
                measure_unit_ids.add(key)
            else:
                connection.execute("INSERT INTO nutrient VALUES (?, ?, ?, ?, ?)", (key, clean(row.get("name")) or "", clean(row.get("unit_name")) or "", clean(row.get("nutrient_nbr")), integer(row.get("rank"))))
                nutrient_ids.add(key)
            counts[table] = counts.get(table, 0) + 1

    for row in fdc_rows(base, "food.csv"):
        fdc_id = clean(row.get("fdc_id"))
        if not fdc_id:
            continue
        insert_source_record(connection, snapshot_id, fdc_id, "food", dict(row))
        connection.execute("INSERT INTO food VALUES (?, ?, ?, ?, ?, ?, ?)", (fdc_id, snapshot_id, clean(row.get("data_type")) or "", clean(row.get("description")), clean(row.get("food_category_id")), clean(row.get("publication_date")), int(fdc_id in foundation_ids)))
        food_ids.add(fdc_id)
        counts["food"] = counts.get("food", 0) + 1

    # The provenance table has a foreign key to food, so insert it after the
    # complete food dimension is present.
    for row in foundation_rows:
        fdc_id = clean(row.get("fdc_id"))
        assert fdc_id is not None
        insert_source_record(connection, snapshot_id, fdc_id, "foundation_food", row)
        if fdc_id not in food_ids:
            quarantine(connection, snapshot_id, fdc_id, "foundation_food", "orphan_food_fk", row)
            continue
        connection.execute("INSERT INTO foundation_food VALUES (?, ?, ?)", (fdc_id, clean(row.get("NDB_number")), clean(row.get("footnote"))))
        counts["foundation_food"] = counts.get("foundation_food", 0) + 1

    for filename, table in (("food_nutrient.csv", "food_nutrient"), ("food_portion.csv", "food_portion"), ("input_food.csv", "food_lineage")):
        for row in fdc_rows(base, filename):
            native_id = clean(row.get("id")) or clean(row.get("fdc_id"))
            if not native_id:
                continue
            payload = dict(row)
            insert_source_record(connection, snapshot_id, native_id, table, payload)
            if table == "food_nutrient":
                if clean(row.get("fdc_id")) not in food_ids or clean(row.get("nutrient_id")) not in nutrient_ids:
                    quarantine(connection, snapshot_id, native_id, table, "orphan_food_or_nutrient_fk", payload)
                    continue
                connection.execute("INSERT INTO food_nutrient VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (native_id, clean(row.get("fdc_id")), clean(row.get("nutrient_id")), snapshot_id, number(row.get("amount")), integer(row.get("data_points")), clean(row.get("derivation_id")), number(row.get("min")), number(row.get("max")), number(row.get("median")), clean(row.get("footnote")), integer(row.get("min_year_acquired"))))
            elif table == "food_portion":
                if clean(row.get("fdc_id")) not in food_ids or (clean(row.get("measure_unit_id")) and clean(row.get("measure_unit_id")) not in measure_unit_ids):
                    quarantine(connection, snapshot_id, native_id, table, "orphan_food_or_measure_unit_fk", payload)
                    continue
                connection.execute("INSERT INTO food_portion VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (native_id, clean(row.get("fdc_id")), integer(row.get("seq_num")), number(row.get("amount")), clean(row.get("measure_unit_id")), clean(row.get("portion_description")), clean(row.get("modifier")), number(row.get("gram_weight")), integer(row.get("data_points")), clean(row.get("footnote")), integer(row.get("min_year_acquired"))))
            else:
                if clean(row.get("fdc_id")) and clean(row.get("fdc_id")) not in food_ids:
                    quarantine(connection, snapshot_id, native_id, table, "orphan_food_fk", payload)
                    continue
                connection.execute("INSERT INTO food_lineage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (native_id, clean(row.get("fdc_id")), clean(row.get("fdc_of_input_food")), clean(row.get("ingredient_code")), clean(row.get("ingredient_description")), clean(row.get("unit")), clean(row.get("portion_code")), clean(row.get("portion_description")), number(row.get("gram_weight")), clean(row.get("retention_code"))))
            counts[table] = counts.get(table, 0) + 1
    return counts


def create_fts(connection: sqlite3.Connection) -> bool:
    try:
        connection.execute("CREATE VIRTUAL TABLE recipe_fts USING fts5(recipe_id UNINDEXED, title, ingredients, instructions)")
        connection.execute("INSERT INTO recipe_fts(recipe_id, title, ingredients, instructions) SELECT r.recipe_id, coalesce(r.title, ''), coalesce((SELECT group_concat(raw_text, ' ') FROM recipe_ingredient i WHERE i.recipe_id = r.recipe_id), ''), coalesce(r.instructions_raw, '') FROM recipe r")
        return True
    except sqlite3.OperationalError:
        return False


def schema_hash() -> str:
    return hashlib.sha256(SCHEMA.encode("utf-8")).hexdigest()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, required=True, help="Open Recipes .db or .csv")
    parser.add_argument("--fdc-dir", type=Path, required=True, help="Directory containing the FDC CSV archive")
    parser.add_argument("--output", type=Path, required=True, help="SQLite database to create")
    parser.add_argument("--fdc-release", default="unspecified", help="FDC release label, e.g. 2026-04")
    parser.add_argument("--replace", action="store_true", help="Replace an existing output database")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.recipes.is_file():
        raise SystemExit(f"Recipe source does not exist: {args.recipes}")
    if not args.fdc_dir.is_dir():
        raise SystemExit(f"FDC directory does not exist: {args.fdc_dir}")
    if args.output.exists():
        if not args.replace:
            raise SystemExit(f"Output exists; choose another path or pass --replace: {args.output}")
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    recipe_hash = sha256_file(args.recipes)
    fdc_hash = sha256_directory(args.fdc_dir, FDC_SCHEMA_FILES)
    retrieved_at = now_iso()
    connection = sqlite3.connect(args.output)
    connection.executescript(SCHEMA)
    connection.execute("PRAGMA journal_mode = WAL")
    recipe_snapshot = f"openrecipes-13k-{recipe_hash[:16]}"
    fdc_snapshot = f"fdc-foundation-{fdc_hash[:16]}"
    connection.execute("INSERT INTO source_snapshot VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (recipe_snapshot, "open-recipes-13k", recipe_hash, str(args.recipes), recipe_hash, "CC BY-SA 3.0 (as stated by source; verify scraped-content rights)", retrieved_at, 0, schema_hash()))
    connection.execute("INSERT INTO source_snapshot VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (fdc_snapshot, "fooddata-central-foundation", args.fdc_release, str(args.fdc_dir), fdc_hash, "USDA FoodData Central; see source release terms", retrieved_at, 0, schema_hash()))
    with connection:
        recipe_count = insert_recipes(connection, recipe_snapshot, args.recipes)
        fdc_counts = insert_fdc(connection, fdc_snapshot, args.fdc_dir)
        fts_created = create_fts(connection)
        connection.execute("UPDATE source_snapshot SET row_count = ? WHERE snapshot_id = ?", (recipe_count, recipe_snapshot))
        connection.execute("UPDATE source_snapshot SET row_count = ? WHERE snapshot_id = ?", (fdc_counts.get("food", 0), fdc_snapshot))
    connection.close()
    print(json.dumps({"output": str(args.output), "recipe_snapshot": recipe_snapshot, "fdc_snapshot": fdc_snapshot, "recipes": recipe_count, "fdc": fdc_counts, "fts5": fts_created}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

