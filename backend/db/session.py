import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from dotenv import load_dotenv
import typing
import contextlib

# Load .env file if it exists (for local development outside Docker)
load_dotenv()

# --- Asynchronous Setup ---
# The application now relies on a single DATABASE_URL environment variable.
# Ensure this URL is correctly set in your environment or .env file.
# For async, it should use the 'asyncpg' driver, e.g., "postgresql+asyncpg://..."
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set. Please configure it in your .env file.")

# The async URL is the primary one.
ASYNC_DATABASE_URL = DATABASE_URL

# --- Synchronous Setup ---
# The sync URL is derived by replacing the async driver with a sync one.
# This ensures both connections point to the same database.
SYNC_DATABASE_URL = ASYNC_DATABASE_URL.replace("postgresql+asyncpg", "postgresql+psycopg2")

if "psycopg2" not in SYNC_DATABASE_URL:
    # If the replacement didn't work (e.g., driver wasn't specified), raise an error.
    raise ValueError(
        "Could not derive a synchronous database URL from DATABASE_URL. "
        "Ensure DATABASE_URL is in the format 'postgresql+asyncpg://user:pass@host/db'"
    )


async_engine = create_async_engine(ASYNC_DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=async_engine, class_=AsyncSession, expire_on_commit=False)

sync_engine = create_engine(SYNC_DATABASE_URL)
SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


# --- Helper Functions ---

async def init_db_async():
    # This function is not used for table creation as Alembic handles migrations.
    # It can be used for initial checks or logging if needed.
    print("Database session module initialized.")

# Async session getter
@contextlib.asynccontextmanager
async def get_async_db_session() -> typing.AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

# Synchronous session getter for scripts
@contextlib.contextmanager
def get_sync_db_session() -> typing.Generator[SyncSession, None, None]:
    db = SyncSessionLocal()
    try:
        yield db
        db.commit() # Commit the transaction if everything was successful
    except Exception as e:
        db.rollback() # Rollback on any exception
        print(f"Database session rolled back due to an error: {e}")
        raise # Re-raise the exception after rollback
    finally:
        db.close()