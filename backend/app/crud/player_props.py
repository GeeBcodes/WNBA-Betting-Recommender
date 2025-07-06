from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
import uuid
from typing import List

from backend.db import models
from backend.app.core.config import DEFAULT_PROP_MARKET_TO_STAT_MAP

async def get_player_props_by_game_id(db: AsyncSession, game_id: uuid.UUID) -> List[models.PlayerProp]:
    """
    Fetches all player props for a specific game_id that are relevant for prediction.
    """
    stmt = (
        select(models.PlayerProp)
        .options(
            joinedload(models.PlayerProp.game).joinedload(models.Game.home_team_ref),
            joinedload(models.PlayerProp.game).joinedload(models.Game.away_team_ref),
            joinedload(models.PlayerProp.market),
            joinedload(models.PlayerProp.player),
        )
        .join(models.PlayerProp.market)
        .filter(models.PlayerProp.game_id == game_id)
        .filter(models.PlayerProp.player_id.isnot(None))
        .filter(models.Market.key.in_(DEFAULT_PROP_MARKET_TO_STAT_MAP.keys()))
    )
    result = await db.execute(stmt)
    props = result.scalars().all()
    return list(props) 