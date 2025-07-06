import logging
from typing import Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as SyncSession

from backend.db import models

logger = logging.getLogger(__name__)


def get_team_by_name(db: SyncSession, name: str) -> Optional[models.Team]:
    """Synchronous function to get a team by name."""
    if not name:
        return None
    stmt = select(models.Team).filter(models.Team.team_name == name)
    return db.execute(stmt).scalars().first()


async def get_team_by_api_id(
    db: AsyncSession, api_team_id: str
) -> Optional[models.Team]:
    """Asynchronously gets a team by its API-specific ID."""
    if not api_team_id:
        return None
    api_team_id_str = str(api_team_id)
    stmt = select(models.Team).filter(models.Team.api_team_id == api_team_id_str)
    result = await db.execute(stmt)
    return result.scalars().first()


async def create_team(
    db: AsyncSession, team_data: Dict[str, str]
) -> Optional[models.Team]:
    """
    Asynchronously creates a team if it doesn't exist, based on API data.
    """
    api_id = str(team_data.get("api_team_id"))
    name = team_data.get("name")

    if not api_id or not name:
        logger.warning(
            f"Cannot create team with missing api_id or name. Provided: {team_data}"
        )
        return None

    existing_team = await get_team_by_api_id(db, api_id)
    if existing_team:
        return existing_team

    logger.info(f"Creating new team: {name} ({api_id})")
    new_team = models.Team(api_team_id=api_id, team_name=name)

    try:
        db.add(new_team)
        await db.commit()
        await db.refresh(new_team)
        return new_team
    except IntegrityError:
        await db.rollback()
        logger.warning(f"Team with api_id {api_id} already exists. Fetching again.")
        return await get_team_by_api_id(db, api_id)
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating team {name}: {e}", exc_info=True)
        return None