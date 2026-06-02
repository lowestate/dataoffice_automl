import os
from psycopg_pool import AsyncConnectionPool
from typing import cast

APP_DB_URI = os.getenv("APP_DB_URI")

# Async connection pool — mirror of ai_analyst pattern
pool = AsyncConnectionPool(
    conninfo=cast(str, APP_DB_URI),
    min_size=2,
    max_size=10,
    max_idle=300,
    timeout=10.0,
    kwargs={"autocommit": True, "prepare_threshold": 0},
    open=False
)
