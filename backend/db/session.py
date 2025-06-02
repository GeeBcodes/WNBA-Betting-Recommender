import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from dotenv import load_dotenv

# Load .env file if it exists (for local development outside Docker)
load_dotenv()

# Default to the hardcoded local URL if DATABASE_URL env var is not set
# Docker Compose will set DATABASE_URL for the backend service.
# Ensure this URL uses the asyncpg driver
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:Gr4t3fu143v3R@localhost:5432/wnba_db")

engine = create_async_engine(DATABASE_URL)

# expire_on_commit=False is often useful for async sessions
# to prevent attributes from being expired after commit, 
# which can cause issues if objects are accessed later.
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession, expire_on_commit=False)

# If you need a dependency to get a session, it would look something like:
# async def get_db() -> AsyncGenerator[AsyncSession, None]:
#     async with SessionLocal() as session:
#         yield session
