import sys
import os
import logging
import asyncio
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload # Used for eager loading relationships

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from backend.db.session import SessionLocal
from backend.db import models as db_models # Renamed to avoid conflict
from backend.db.models import PlayerStat, Game, Player, LeagueSeasonAverages # Specific imports

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper function for safe division ---
def safe_divide(numerator, denominator, default=0.0):
    if numerator is None or denominator is None or denominator == 0:
        return default
    return numerator / denominator

# --- Advanced Stat Calculation Functions ---

async def calculate_all_advanced_stats(db: AsyncSession):
    logger.info("Starting calculation of all advanced player statistics.")

    # Fetch all games, we might need to iterate or process season by season if too large
    games_stmt = select(Game).options(
        selectinload(Game.stats).joinedload(PlayerStat.player), # Eager load player stats and player info
        selectinload(Game.home_team_ref), # Eager load team details
        selectinload(Game.away_team_ref)
    )
    games_result = await db.execute(games_stmt)
    all_games = games_result.scalars().all()

    if not all_games:
        logger.info("No games found in the database. Exiting advanced stat calculation.")
        return

    logger.info(f"Found {len(all_games)} games to process for advanced stats.")
    updated_player_stats_count = 0

    for game_idx, game in enumerate(all_games):
        logger.info(f"Processing game {game_idx + 1}/{len(all_games)}: ID {game.id} ({game.game_datetime.date()})")
        
        player_stats_in_game = game.stats
        if not player_stats_in_game:
            logger.info(f"No player stats found for game {game.id}. Skipping.")
            continue

        # Ensure necessary team totals are present on the Game object
        team_minutes_home = game.home_team_minutes_played
        team_minutes_away = game.away_team_minutes_played
        team_possessions_home = game.home_team_possessions
        team_possessions_away = game.away_team_possessions
        team_fga_home = game.home_team_field_goals_attempted
        team_fga_away = game.away_team_field_goals_attempted
        team_fta_home = game.home_team_free_throws_attempted
        team_fta_away = game.away_team_free_throws_attempted
        team_tov_home = game.home_team_turnovers
        team_tov_away = game.away_team_turnovers
        team_orb_home = game.home_team_offensive_rebounds
        team_orb_away = game.away_team_offensive_rebounds
        team_drb_home = game.home_team_defensive_rebounds
        team_drb_away = game.away_team_defensive_rebounds
        team_fgm_home = game.home_team_field_goals_made
        team_fgm_away = game.away_team_field_goals_made
        team_ast_home = game.home_team_assists
        team_ast_away = game.away_team_assists
        team_stl_home = game.home_team_steals
        team_stl_away = game.away_team_steals
        team_blk_home = game.home_team_blocks
        team_blk_away = game.away_team_blocks

        game_total_minutes_for_calc = (team_minutes_home if team_minutes_home else 200)

        stats_to_update_for_game = []

        for p_stat in player_stats_in_game:
            if p_stat.minutes_played is None or p_stat.minutes_played <= 0:
                continue

            player_mp = p_stat.minutes_played
            updates_for_player_stat = {"id": p_stat.id}

            is_home = p_stat.is_home_team

            team_mp = team_minutes_home if is_home else team_minutes_away
            team_poss = team_possessions_home if is_home else team_possessions_away
            team_fga = team_fga_home if is_home else team_fga_away
            team_fta = team_fta_home if is_home else team_fta_away
            team_tov = team_tov_home if is_home else team_tov_away
            team_orb = team_orb_home if is_home else team_orb_away
            team_drb = team_drb_home if is_home else team_drb_away
            team_fgm = team_fgm_home if is_home else team_fgm_away
            team_ast = team_ast_home if is_home else team_ast_away

            opp_poss = team_possessions_away if is_home else team_possessions_home
            opp_orb = team_orb_away if is_home else team_orb_home
            opp_drb = team_drb_away if is_home else team_drb_home
            opp_fga = team_fga_away if is_home else team_fga_home

            if team_mp is None or team_mp <= 0: team_mp = game_total_minutes_for_calc / 2

            if p_stat.field_goals_attempted is not None and \
               p_stat.free_throws_attempted is not None and \
               p_stat.turnovers is not None and \
               team_fga is not None and team_fta is not None and team_tov is not None and team_mp is not None:
                
                player_usage_actions = p_stat.field_goals_attempted + \
                                     0.44 * p_stat.free_throws_attempted + \
                                     p_stat.turnovers
                team_usage_actions = team_fga + 0.44 * team_fta + team_tov
                
                if player_mp > 0 and team_usage_actions > 0 and team_mp > 0:
                    usg_rate = 100 * (player_usage_actions * (team_mp / 5.0)) / (player_mp * team_usage_actions)
                    updates_for_player_stat['usage_rate'] = usg_rate
            
            if p_stat.assists is not None and p_stat.field_goals_made is not None and \
               team_fgm is not None and team_mp is not None and player_mp > 0:
                team_minutes_per_position = team_mp / 5.0
                if team_minutes_per_position > 0: 
                    denominator_ast = ((player_mp / team_minutes_per_position) * team_fgm) - p_stat.field_goals_made
                    if denominator_ast > 0:
                        updates_for_player_stat['assist_percentage'] = 100 * p_stat.assists / denominator_ast

            player_orb_val = p_stat.offensive_rebounds
            player_drb_val = p_stat.defensive_rebounds
            player_trb_val = p_stat.rebounds

            team_total_reb = game.home_team_total_rebounds if is_home else game.away_team_total_rebounds
            opp_total_reb = game.away_team_total_rebounds if is_home else game.home_team_total_rebounds

            if team_mp is not None and player_mp > 0:
                team_minutes_per_position_reb = team_mp / 5.0
                if team_minutes_per_position_reb > 0:
                    if player_orb_val is not None and team_orb is not None and opp_drb is not None:
                        orb_chances = team_orb + opp_drb
                        if orb_chances > 0:
                            updates_for_player_stat['offensive_rebound_percentage'] = 100 * (player_orb_val * team_minutes_per_position_reb) / (player_mp * orb_chances)
                    
                    if player_drb_val is not None and team_drb is not None and opp_orb is not None:
                        drb_chances = team_drb + opp_orb
                        if drb_chances > 0:
                            updates_for_player_stat['defensive_rebound_percentage'] = 100 * (player_drb_val * team_minutes_per_position_reb) / (player_mp * drb_chances)

                    if player_trb_val is not None and team_total_reb is not None and opp_total_reb is not None:
                        total_reb_chances_game = team_total_reb + opp_total_reb
                        if total_reb_chances_game > 0:
                            updates_for_player_stat['total_rebound_percentage'] = 100 * (player_trb_val * team_minutes_per_position_reb) / (player_mp * total_reb_chances_game)

            if p_stat.steals is not None and opp_poss is not None and opp_poss > 0 and \
               team_mp is not None and player_mp > 0:
                team_minutes_per_position_stl = team_mp / 5.0
                if team_minutes_per_position_stl > 0:
                    updates_for_player_stat['steal_percentage'] = 100 * (p_stat.steals * team_minutes_per_position_stl) / (player_mp * opp_poss)

            if p_stat.blocks is not None and opp_fga is not None and opp_fga > 0 and \
               team_mp is not None and player_mp > 0:
                team_minutes_per_position_blk = team_mp / 5.0
                if team_minutes_per_position_blk > 0:
                    updates_for_player_stat['block_percentage'] = 100 * (p_stat.blocks * team_minutes_per_position_blk) / (player_mp * opp_fga)

            if player_mp > 0 and \
               p_stat.points is not None and p_stat.field_goals_made is not None and \
               p_stat.field_goals_attempted is not None and p_stat.free_throws_attempted is not None and \
               p_stat.free_throws_made is not None and \
               p_stat.offensive_rebounds is not None and p_stat.defensive_rebounds is not None and \
               p_stat.steals is not None and p_stat.assists is not None and \
               p_stat.blocks is not None and p_stat.fouls is not None and p_stat.turnovers is not None:
                
                uPER_value = (
                    p_stat.points +
                    (0.4 * p_stat.field_goals_made) -
                    (0.7 * p_stat.field_goals_attempted) -
                    (0.4 * (p_stat.free_throws_attempted - p_stat.free_throws_made)) +
                    (0.7 * p_stat.offensive_rebounds) + (0.3 * p_stat.defensive_rebounds) +
                    p_stat.steals +
                    (0.7 * p_stat.assists) +
                    (0.7 * p_stat.blocks) -
                    (0.4 * p_stat.fouls) -
                    p_stat.turnovers
                )
                updates_for_player_stat['player_efficiency_rating'] = uPER_value / player_mp

            if len(updates_for_player_stat) > 1:
                stats_to_update_for_game.append(updates_for_player_stat)
        
        if stats_to_update_for_game:
            try:
                await db.execute(update(PlayerStat), stats_to_update_for_game)
                await db.commit()
                updated_player_stats_count += len(stats_to_update_for_game)
                logger.info(f"Game {game.id}: Successfully updated {len(stats_to_update_for_game)} player stat records with advanced stats.")
            except Exception as e:
                await db.rollback()
                logger.error(f"Error bulk updating player stats for game {game.id}: {e}")

    logger.info(f"Finished advanced stat calculations. Total player stat records updated: {updated_player_stats_count}")

async def run_calculations(db: AsyncSession):
    logger.info("Starting advanced player stat calculation including standardized PER.")

    all_seasons_stmt = select(Game.season).distinct().order_by(Game.season)
    seasons_result = await db.execute(all_seasons_stmt)
    seasons = [s[0] for s in seasons_result.all() if s[0] is not None]

    if not seasons:
        logger.info("No seasons found in games table. Exiting.")
        return

    for season_year in seasons:
        logger.info(f"Processing season: {season_year}")

        # 1. Fetch league averages for this season
        lg_avg_stmt = select(LeagueSeasonAverages).filter_by(season_year=season_year)
        lg_avg_result = await db.execute(lg_avg_stmt)
        league_season_stats = lg_avg_result.scalar_one_or_none()

        if not league_season_stats:
            logger.warning(f"League averages not found for season {season_year}. Standardized PER calculation will be skipped for this season. Other stats will be calculated if possible.")
            # Fallback: uPER will be calculated and stored if other calculations proceed.
        
        lg_pace = league_season_stats.lg_possessions if league_season_stats and league_season_stats.lg_possessions is not None else 0.0
        if lg_pace == 0.0 and league_season_stats: # league_season_stats exists but pace is 0
             logger.warning(f"League pace (lg_possessions) is zero for season {season_year}. PER pace adjustment will effectively be skipped or result in zero if team pace is non-zero.")


        # Fetch all PlayerStat objects for the current season, including necessary relationships
        player_stats_stmt = (
            select(PlayerStat)
            .join(PlayerStat.game)
            .filter(Game.season == season_year)
            .options(
                selectinload(PlayerStat.player), # Player name for logging if needed
                selectinload(PlayerStat.game).options(
                    selectinload(Game.home_team_ref), # For team names if needed
                    selectinload(Game.away_team_ref),
                    # No need to selectinload Game.stats again here as we are querying PlayerStat
                )
            )
        )
        player_stats_result = await db.execute(player_stats_stmt)
        all_player_stats_for_season = player_stats_result.scalars().all()

        if not all_player_stats_for_season:
            logger.info(f"No player stats found for season {season_year}. Skipping.")
            continue

        # --- First pass: Calculate uPER and other stats for all players ---
        total_weighted_uper_for_season = 0.0
        total_minutes_for_season = 0.0
        temp_uper_map = {} # Stores {p_stat_id: uper_value}

        for p_stat in all_player_stats_for_season:
            game = p_stat.game
            player_mp = p_stat.minutes_played if p_stat.minutes_played is not None else 0.0

            if player_mp <= 0:
                p_stat.assist_percentage = 0.0
                p_stat.offensive_rebound_percentage = 0.0
                p_stat.defensive_rebound_percentage = 0.0
                p_stat.total_rebound_percentage = 0.0
                p_stat.steal_percentage = 0.0
                p_stat.block_percentage = 0.0
                p_stat.usage_rate = 0.0
                p_stat.player_efficiency_rating = 0.0 # This will hold standardized PER eventually
                temp_uper_map[p_stat.id] = 0.0
                # TOV% is calculated by scraper, leave as is or set to 0? For now, leave.
                continue

            # Determine team and opponent context
            is_home = p_stat.is_home_team
            
            team_mp = (game.home_team_minutes_played if is_home else game.away_team_minutes_played) or 200.0
            team_fgm = (game.home_team_field_goals_made if is_home else game.away_team_field_goals_made) or 0.0
            team_fga = (game.home_team_field_goals_attempted if is_home else game.away_team_field_goals_attempted) or 0.0
            team_fta = (game.home_team_free_throws_attempted if is_home else game.away_team_free_throws_attempted) or 0.0
            team_tov = (game.home_team_turnovers if is_home else game.away_team_turnovers) or 0.0
            team_orb = (game.home_team_offensive_rebounds if is_home else game.away_team_offensive_rebounds) or 0.0
            team_drb = (game.home_team_defensive_rebounds if is_home else game.away_team_defensive_rebounds) or 0.0
            team_total_reb = (game.home_team_total_rebounds if is_home else game.away_team_total_rebounds) or 0.0
            team_poss = (game.home_team_possessions if is_home else game.away_team_possessions) or 0.0

            opp_total_reb = (game.away_team_total_rebounds if is_home else game.home_team_total_rebounds) or 0.0
            opp_fga = (game.away_team_field_goals_attempted if is_home else game.home_team_field_goals_attempted) or 0.0
            opp_poss = (game.away_team_possessions if is_home else game.home_team_possessions) or 0.0
            opp_drb_from_game = (p_stat.game.away_team_defensive_rebounds if is_home else p_stat.game.home_team_defensive_rebounds) or 0.0
            opp_orb_from_game = (p_stat.game.away_team_offensive_rebounds if is_home else p_stat.game.home_team_offensive_rebounds) or 0.0
            
            # Ensure values are float for calculations
            p_ast = p_stat.assists or 0.0
            p_fgm = p_stat.field_goals_made or 0.0
            p_fga = p_stat.field_goals_attempted or 0.0
            p_fta = p_stat.free_throws_attempted or 0.0
            p_ftm = p_stat.free_throws_made or 0.0
            p_tov = p_stat.turnovers or 0.0
            p_orb = p_stat.offensive_rebounds or 0.0
            p_drb = p_stat.defensive_rebounds or 0.0
            p_trb = p_stat.rebounds or 0.0
            p_stl = p_stat.steals or 0.0
            p_blk = p_stat.blocks or 0.0
            p_pts = p_stat.points or 0.0
            p_pf = p_stat.fouls or 0.0

            # AST%
            if ((player_mp / (team_mp / 5.0)) * team_fgm) - p_fgm > 0:
                p_stat.assist_percentage = (100 * p_ast) / (((player_mp / (team_mp / 5.0)) * team_fgm) - p_fgm)
            else:
                p_stat.assist_percentage = 0.0
            
            # ORB%
            if player_mp > 0 and (team_orb + opp_total_reb - p_orb) > 0 : # Denominator: available ORBs for team
                 # More accurate: Team ORB / (Team ORB + Opponent DRB)
                 # Using simplified: Player ORB / (Player MP * (Team ORB + Opponent DRB)) * (Team MP / 5)
                 # Standard formula for ORB%: 100 * (Player ORB * (TeamMP / 5)) / (PlayerMP * (Team ORB + Opponent DRB))
                 # For now, use the prior implemented simpler version if team_orb and opp_total_reb are available
                 # Total Rebounding Chances for player: Player_MP * (Team_ORB + Opp_DRB) / (Team_MP / 5)
                 # This needs careful thought; let's use the existing one for now and flag for review.
                 # Previous: 100 * (p_orb * (team_mp / 5.0)) / (player_mp * (team_total_reb + opp_total_reb)) if (player_mp * (team_total_reb + opp_total_reb)) > 0 else 0.0
                 # Focusing on player's contribution to available rebounds while on court.
                 # Simpler: % of team's ORBs player got while on floor.
                 # More standard: 100 * (p_orb * (team_mp / 5)) / (player_mp * (team_orb + (opp_total_reb - (opp_total_reb* ( (league_season_stats.lg_offensive_rebounds / (league_season_stats.lg_offensive_rebounds + league_season_stats.lg_defensive_rebounds)) if league_season_stats and league_season_stats.lg_offensive_rebounds is not None and league_season_stats.lg_defensive_rebounds is not None and (league_season_stats.lg_offensive_rebounds + league_season_stats.lg_defensive_rebounds) > 0 else 0.4 )) ))) #This is getting too complex for now.
                 # Use existing:
                if (player_mp * (team_orb + (opp_total_reb - opp_orb_from_game))) > 0: # Denominator: Team ORB + Opponent DRB
                     p_stat.offensive_rebound_percentage = 100 * (p_orb * (team_mp / 5.0)) / (player_mp * (team_orb + (opp_total_reb - opp_orb_from_game))) # Opponent DRB = Opponent TRB - Opponent ORB
                else:
                     p_stat.offensive_rebound_percentage = 0.0
            else:
                p_stat.offensive_rebound_percentage = 0.0

            # DRB%
            # Standard: 100 * (Player DRB * (TeamMP / 5)) / (PlayerMP * (Team DRB + Opponent ORB))
            if player_mp > 0 and (team_drb + (opp_total_reb - opp_drb_from_game)) > 0: # Denom: Team DRB + Opp ORB
                p_stat.defensive_rebound_percentage = 100 * (p_drb * (team_mp / 5.0)) / (player_mp * (team_drb + (opp_total_reb - opp_drb_from_game))) # Opponent ORB = Opponent TRB - Opponent DRB
            else:
                p_stat.defensive_rebound_percentage = 0.0
            
            # TRB%
            if player_mp > 0 and (team_total_reb + opp_total_reb) > 0:
                p_stat.total_rebound_percentage = 100 * (p_trb * (team_mp / 5.0)) / (player_mp * (team_total_reb + opp_total_reb))
            else:
                p_stat.total_rebound_percentage = 0.0

            # STL%
            if player_mp > 0 and opp_poss > 0:
                p_stat.steal_percentage = 100 * (p_stl * (team_mp / 5.0)) / (player_mp * opp_poss)
            else:
                p_stat.steal_percentage = 0.0

            # BLK%
            if player_mp > 0 and opp_fga > 0: # Opponent FGA (2pt and 3pt)
                p_stat.block_percentage = 100 * (p_blk * (team_mp / 5.0)) / (player_mp * opp_fga)
            else:
                p_stat.block_percentage = 0.0

            # USG%
            denominator_usg = player_mp * (team_fga + 0.44 * team_fta + team_tov)
            if denominator_usg > 0:
                p_stat.usage_rate = 100 * ((p_fga + 0.44 * p_fta + p_tov) * (team_mp / 5.0)) / denominator_usg
            else:
                p_stat.usage_rate = 0.0
            
            # uPER Calculation
            uPER_value = (1 / player_mp) * (
                p_pts + (0.4 * p_fgm) - (0.7 * p_fga) - (0.4 * (p_fta - p_ftm)) +
                (0.7 * p_orb) + (0.3 * p_drb) + p_stl + (0.7 * p_ast) +
                (0.7 * p_blk) - (0.4 * p_pf) - p_tov
            )
            temp_uper_map[p_stat.id] = uPER_value
            total_weighted_uper_for_season += uPER_value * player_mp
            total_minutes_for_season += player_mp
        
        # After processing all players in the season, calculate lguPER for this season
        lguPER_for_season = 0.0
        if total_minutes_for_season > 0:
            lguPER_for_season = total_weighted_uper_for_season / total_minutes_for_season
        else:
            logger.warning(f"No player minutes recorded for season {season_year}. lguPER defaults to 0.")

        # --- Second pass: Standardize PER using lguPER and league_pace ---
        for p_stat in all_player_stats_for_season:
            if p_stat.minutes_played is None or p_stat.minutes_played <= 0:
                p_stat.player_efficiency_rating = 0.0 # Already set, but for clarity
                continue

            uper = temp_uper_map.get(p_stat.id, 0.0)
            standardized_per = 0.0 # Default

            if league_season_stats and lg_pace > 0: # Check if league stats for pace adjustment are available
                current_game = p_stat.game
                team_poss_for_player_game = (current_game.home_team_possessions if p_stat.is_home_team else current_game.away_team_possessions) or 0.0

                if team_poss_for_player_game > 0:
                    pace_adjustment_factor = lg_pace / team_poss_for_player_game
                    adjusted_uper = uper * pace_adjustment_factor

                    if lguPER_for_season != 0:
                        standardized_per = adjusted_uper * (15.0 / lguPER_for_season)
                    # If lguPER_for_season is 0, standardized_per remains 0.0
                else: # Team possessions missing/zero, cannot pace adjust accurately
                    logger.warning(f"Team possessions missing or zero for player {p_stat.player_id} in game {p_stat.game_id} (season {season_year}). Using uPER for PER normalization (no pace adjustment).")
                    if lguPER_for_season != 0:
                         standardized_per = uper * (15.0 / lguPER_for_season) # Normalize uPER directly
            else: # League pace missing or other league_season_stats issue, cannot pace adjust
                logger.warning(f"League pace missing or zero for season {season_year}. Using uPER for PER normalization (no pace adjustment).")
                if lguPER_for_season != 0:
                    standardized_per = uper * (15.0 / lguPER_for_season) # Normalize uPER directly
            
            p_stat.player_efficiency_rating = standardized_per
        
        await db.commit()
        logger.info(f"Successfully calculated and updated advanced stats (including standardized PER) for season {season_year}")

    logger.info("Finished processing all seasons for advanced stats.")

async def main():
    logger.info("Connecting to database for advanced stat calculation...")
    async with SessionLocal() as db_session:
        await run_calculations(db_session)

if __name__ == "__main__":
    # Ensure the script can be run directly
    # This setup allows `python utils/calculate_advanced_stats.py`
    if PROJECT_ROOT not in sys.path: # Redundant if global sys.path.insert(0,PROJECT_ROOT) worked
        sys.path.insert(0, PROJECT_ROOT)
        
    # Need to re-import if path was just added and modules were not found initially
    # from backend.db.session import SessionLocal # Already imported
    # from backend.db import models # Already imported
    
    asyncio.run(main())

# TODO for future refinement:
# 1. Add Offensive and Defensive Rebounds to PlayerStat model and scraper for more accurate rebound % and PER.
# 2. For PER: Implement league average calculations (lgPace, league factors) for a more standardized PER.
#    This would involve another script or part of this one to calculate league averages per season.
# 3. For BLK%: Refine opponent FGA to be 2-point FGA if possible (might need PBP analysis).
# 4. Add error handling and logging for missing critical data points for specific calculations.
# 5. Consider processing season by season or in smaller batches if the dataset becomes very large.
# 6. Team Minutes (TeamMP / 5): Ensure TeamMP is consistently total minutes played by all 5 positions (e.g., 200 for regulation).
#    The current scraper sums player minutes, which should be correct. 