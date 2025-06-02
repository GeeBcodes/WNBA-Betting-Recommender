import sys
import os
import logging
import asyncio
from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from backend.db.session import SessionLocal
from backend.db import models as db_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def calculate_and_store_league_averages(db: AsyncSession):
    logger.info("Starting calculation of league season averages.")

    # Get distinct seasons from the Game table
    seasons_stmt = select(db_models.Game.season).distinct().order_by(db_models.Game.season)
    seasons_result = await db.execute(seasons_stmt)
    seasons = [s[0] for s in seasons_result.all() if s[0] is not None]

    if not seasons:
        logger.info("No seasons found in the games table. Exiting.")
        return

    logger.info(f"Found seasons to process: {seasons}")

    for season_year in seasons:
        logger.info(f"Calculating averages for season: {season_year}")

        # Query to get all game data for the current season
        games_stmt = select(db_models.Game).filter(db_models.Game.season == season_year)
        games_result = await db.execute(games_stmt)
        games_in_season = games_result.scalars().all()

        if not games_in_season:
            logger.warning(f"No games found for season {season_year}. Skipping.")
            continue

        num_games_in_season = len(games_in_season)
        num_team_games = num_games_in_season * 2 # Each game has two teams

        # Initialize totals
        total_pts = 0.0
        total_fgm = 0.0
        total_fga = 0.0
        total_3pm = 0.0
        total_3pa = 0.0
        total_ftm = 0.0
        total_fta = 0.0
        total_orb = 0.0
        total_drb = 0.0
        total_reb = 0.0
        total_ast = 0.0
        total_stl = 0.0
        total_blk = 0.0
        total_tov = 0.0
        total_pf = 0.0
        total_poss = 0.0
        
        valid_poss_games = 0 # Count games with valid possession data for averaging pace

        for game in games_in_season:
            # Sum stats from home and away teams
            # Points (using home_score and away_score which should be team points)
            total_pts += (game.home_score or 0) + (game.away_score or 0)
            
            total_fgm += (game.home_team_field_goals_made or 0) + (game.away_team_field_goals_made or 0)
            total_fga += (game.home_team_field_goals_attempted or 0) + (game.away_team_field_goals_attempted or 0)
            total_3pm += (game.home_team_three_pointers_made or 0) + (game.away_team_three_pointers_made or 0)
            total_3pa += (game.home_team_three_pointers_attempted or 0) + (game.away_team_three_pointers_attempted or 0)
            total_ftm += (game.home_team_free_throws_made or 0) + (game.away_team_free_throws_made or 0)
            total_fta += (game.home_team_free_throws_attempted or 0) + (game.away_team_free_throws_attempted or 0)
            total_orb += (game.home_team_offensive_rebounds or 0) + (game.away_team_offensive_rebounds or 0)
            total_drb += (game.home_team_defensive_rebounds or 0) + (game.away_team_defensive_rebounds or 0)
            total_reb += (game.home_team_total_rebounds or 0) + (game.away_team_total_rebounds or 0)
            total_ast += (game.home_team_assists or 0) + (game.away_team_assists or 0)
            total_stl += (game.home_team_steals or 0) + (game.away_team_steals or 0)
            total_blk += (game.home_team_blocks or 0) + (game.away_team_blocks or 0)
            total_tov += (game.home_team_turnovers or 0) + (game.away_team_turnovers or 0)
            total_pf += (game.home_team_fouls or 0) + (game.away_team_fouls or 0)
            
            if game.home_team_possessions is not None:
                total_poss += game.home_team_possessions
                valid_poss_games +=1
            if game.away_team_possessions is not None:
                total_poss += game.away_team_possessions
                valid_poss_games +=1

        if num_team_games == 0:
            logger.warning(f"Season {season_year} has num_team_games = 0. Skipping average calculation.")
            continue

        # Calculate per-team-game averages
        lg_averages = {
            "season_year": season_year,
            "games_played": num_games_in_season,
            "team_games_played": num_team_games,
            "lg_points": total_pts / num_team_games if num_team_games > 0 else 0,
            "lg_field_goals_made": total_fgm / num_team_games if num_team_games > 0 else 0,
            "lg_field_goals_attempted": total_fga / num_team_games if num_team_games > 0 else 0,
            "lg_three_pointers_made": total_3pm / num_team_games if num_team_games > 0 else 0,
            "lg_three_pointers_attempted": total_3pa / num_team_games if num_team_games > 0 else 0,
            "lg_free_throws_made": total_ftm / num_team_games if num_team_games > 0 else 0,
            "lg_free_throws_attempted": total_fta / num_team_games if num_team_games > 0 else 0,
            "lg_offensive_rebounds": total_orb / num_team_games if num_team_games > 0 else 0,
            "lg_defensive_rebounds": total_drb / num_team_games if num_team_games > 0 else 0,
            "lg_total_rebounds": total_reb / num_team_games if num_team_games > 0 else 0,
            "lg_assists": total_ast / num_team_games if num_team_games > 0 else 0,
            "lg_steals": total_stl / num_team_games if num_team_games > 0 else 0,
            "lg_blocks": total_blk / num_team_games if num_team_games > 0 else 0,
            "lg_turnovers": total_tov / num_team_games if num_team_games > 0 else 0,
            "lg_fouls": total_pf / num_team_games if num_team_games > 0 else 0,
            "lg_possessions": total_poss / valid_poss_games if valid_poss_games > 0 else 0, # Pace
        }

        # Calculate derived league metrics
        # VOP = Total Points / (Total FGA - Total ORB + Total TOV + 0.44 * Total FTA)
        denominator_vop = total_fga - total_orb + total_tov + (0.44 * total_fta)
        lg_averages["lg_value_of_possession"] = total_pts / denominator_vop if denominator_vop > 0 else 0
        
        # lgDRB% = Total DRB / (Total DRB + Total ORB that were opponent\'s)
        # Total ORB includes both home and away, so it represents all available offensive rebounds
        # Total DRB also. So, lgDRB% = total_drb / (total_drb + total_orb)
        if (total_drb + total_orb) > 0:
            lg_averages["lg_defensive_rebound_percentage"] = total_drb / (total_drb + total_orb)
        else:
            lg_averages["lg_defensive_rebound_percentage"] = 0

        # Upsert into the database
        stmt = pg_insert(db_models.LeagueSeasonAverages).values(lg_averages)
        on_conflict_stmt = stmt.on_conflict_do_update(
            index_elements=['season_year'],
            set_={key: getattr(stmt.excluded, key) for key in lg_averages if key != 'season_year'}
        )
        await db.execute(on_conflict_stmt)
        await db.commit()
        logger.info(f"Successfully calculated and stored/updated league averages for season {season_year}.")

    logger.info("Finished calculating all league season averages.")

async def main():
    logger.info("Connecting to database for league average calculation...")
    async with SessionLocal() as db_session:
        await calculate_and_store_league_averages(db_session)

if __name__ == "__main__":
    asyncio.run(main()) 