# Remymy Recipe API

This is the narrow, read-only bridge between the normalized SQLite food
database and the Remymy frontend. It uses FastAPI, Pydantic response models,
and parameterized SQLite queries.

## Project layout

- `src/remymy_api/` contains the FastAPI app, routes, database access, and
  response schemas.
- `migrations/ingest_food_data.py` ingests Open Recipes 13k and FoodData
  Central Foundation CSVs without doing ingredient matching or nutrition
  inference.
- `migrations/bootstrap.py` validates an existing database or builds a
  missing database from explicitly supplied source paths.
- `tests/` contains the focused API tests.

## Public contract

- `GET /health` returns `{ "status": "ok" }` while the API process is running.
- `GET /ready` returns `{ "status": "ok" }` only when the normalized recipe
  database exists and has the required recipe tables.
- `GET /api/recipes?q=&limit=&offset=` returns `{ items, total, limit, offset }`.
  Each item contains only `id`, `title`, `ingredient_count`,
  `ingredient_preview`, and `quality_status`.
- `GET /api/recipes/{id}` returns only `id`, `title`, `ingredients`,
  `instructions`, and `quality_status`.

`limit` is between 1 and 100, and `offset` must be non-negative. Search uses
case-insensitive matching against recipe titles and raw ingredient text.
Recipe summaries exclude blank-title records and include at most three preview
ingredients; detail responses preserve all stored ingredient and instruction
strings in position order.

The API does not expose FoodData Central rows, raw source payloads, source
records, quarantine rows, nutrition, or arbitrary SQL.

## Database availability

Set `DATABASE_PATH` to the ingested SQLite file. The container listens on port
`8002`, with a default database path of `/data/remymy-food.db`. `/health`
reports process liveness, while `/ready` validates database availability for
container readiness. Recipe routes return a generic `503 Recipe data is
unavailable.` when the file is missing, invalid, or missing required recipe
tables. No local path or SQLite error is returned to callers.

## Users database

Users and their recipe history live in a second SQLite file,
`USER_DATABASE_PATH` (default `/state/remymy-users.db`, on the
`remymy-user-data` volume in compose). It is kept apart from the recipe
database because that one is read-only and replaced on every rebuild. The API
creates the schema on startup if it is missing, which needs SQLite 3.45 or
newer for JSONB. `user_history.recipe_id` points at `recipe.recipe_id` in the
recipe database; SQLite cannot enforce that across files, so `add_history`
checks it in code.

## Try it in Swagger

Open http://localhost:8002/docs (or just http://localhost:8002/). With
`SEED_DEMO_USER=true`, which compose sets by default, the API creates a `demo`
user on startup if it does not exist yet. Its history is the first recipe whose
title matches each of chicken, tomato, salad, soup and chocolate.
`GET /api/recommend/{username}` comes pre-filled with `demo`, so "Try it out",
then "Execute" returns its buying profile and recommendations. The model
service has its own Swagger at http://localhost:8003/docs, with an example
request body for `POST /v1/recommend`.

## Run locally

From this directory:

```bash
python -m pip install -r requirements.txt
python3 serve.py
```

`serve.py` fills in local defaults for anything you have not set:
`DATABASE_PATH=../data/remymy-food.db`, `USER_DATABASE_PATH=./remymy-users.db`,
`REDIS_URL=redis://localhost:6379/0` and `SEED_DEMO_USER=true`. Redis must be
running, since the API sends model jobs through it. Start the model service
with `python3 serve.py` in `../ml_models`; it takes jobs from the same Redis.

## Bootstrap and ingestion

The bootstrap command is idempotent for an existing valid database. It runs
SQLite `quick_check` and verifies the required recipe tables, then exits 0. If
the database is missing, it fails with an actionable message unless both
source paths are explicitly supplied:

```bash
DATABASE_PATH=/data/remymy-food.db \
RECIPES_SOURCE=/data/13k-recipes.db \
FDC_SOURCE_DIR=/data/FoodData_Central_foundation_food_csv_2026-04-30 \
python migrations/bootstrap.py
```

The same values can be passed with `--database-path`, `--recipes`, and
`--fdc-dir`. A missing database is written to a temporary file in the target
directory, validated, and atomically replaced into place only after ingestion
completes. Use `--validate-only` to prohibit ingestion.

The ingestion CLI can also be run directly:

```bash
python migrations/ingest_food_data.py \
  --recipes /data/13k-recipes.db \
  --fdc-dir /data/FoodData_Central_foundation_food_csv_2026-04-30 \
  --fdc-release 2026-04 \
  --output /data/remymy-food.db
```
