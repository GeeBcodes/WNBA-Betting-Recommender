from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.session import AsyncSessionLocal

# Dependency to get DB session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close() 