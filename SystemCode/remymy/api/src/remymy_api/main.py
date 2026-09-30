from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .cache import connect_cache
from .demo import demo_enabled, seed_demo_user
from .jobs import RetryScheduler, connect_queue
from .routes import router
from .users import init_user_database, open_user_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cache = connect_cache()
    app.state.queue = connect_queue()
    app.state.retries = RetryScheduler()
    init_user_database()
    if demo_enabled():
        connection = open_user_database()
        try:
            seed_demo_user(connection, app.state.cache)
        finally:
            connection.close()
    try:
        yield
    finally:
        app.state.retries.stop()
        app.state.queue.close()
        app.state.cache.close()


app = FastAPI(
    title="Remymy Recipe API",
    version="1.0.0",
    description="Recipes and personalised recommendations. Try `GET /api/recommend/demo` "
    "for the seeded demo user.",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/", include_in_schema=False)
def docs_redirect() -> RedirectResponse:
    return RedirectResponse(url="/docs")
