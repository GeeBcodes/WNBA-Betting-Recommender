from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from db.session import SessionLocal

# Dependency to get DB session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            # For async sessions, explicit close is often not needed as the context manager handles it.
            # However, if specific cleanup is required, it can be done here.
            # await session.close() # This is an option if needed, but usually handled by async with
            pass 