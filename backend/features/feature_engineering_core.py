import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# --- Core Feature Generation Orchestrator ---

def generate_full_feature_set(
    base_df: pd.DataFrame, 
    target_stat: str, 
    team_context_df: pd.DataFrame,
    rolling_windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Orchestrates the generation of all features for the model.
    """
    if base_df.empty:
        logger.warning("Base DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df = base_df.copy()

    # Ensure UTC timezone consistency
    df['game_datetime'] = pd.to_datetime(df['game_datetime'], utc=True)
    team_context_df['game_datetime'] = pd.to_datetime(team_context_df['game_datetime'], utc=True)
    
    if 'prop_line' in df.columns:
        df['prop_line'] = pd.to_numeric(df['prop_line'], errors='coerce')

    # Generate foundational features
    df = generate_foundational_features(df)

    # --- Generate Team & Opponent Features ---
    team_performance_features = get_team_performance_rolling_averages(team_context_df, rolling_windows)
    opponent_defense_features = get_team_defensive_rolling_averages(team_context_df, rolling_windows)
    
    # Merge team performance features
    df = pd.merge(
        df,
        team_performance_features,
        on=['team_id', 'game_datetime'],
        how='left'
    )
    
    # Merge opponent defensive features
    df = pd.merge(
        df,
        opponent_defense_features,
        left_on=['opponent_team_id', 'game_datetime'],
        right_on=['team_id', 'game_datetime'],
        how='left',
        suffixes=('', '_opponent_conceded_stats') # Suffix to avoid column name conflicts
    )
    # Drop the redundant team_id from the opponent merge
    if 'team_id_opponent_conceded_stats' in df.columns:
        df.drop(columns=['team_id_opponent_conceded_stats'], inplace=True)

    return df


# --- Foundational & Player-Centric Features ---

def generate_foundational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates basic time-based and rolling average features for the player.
    """
    df = df.sort_values(by=['player_id', 'game_datetime'])
    
    # Days since last game
    df['days_since_last_game'] = df.groupby('player_id')['game_datetime'].diff().dt.days
    
    # Rolling averages for player's own stats
        # Rolling averages for player's own stats
    stats_to_roll = [
        'minutes', 'points', 'rebounds', 'assists', 'steals', 'blocks', 
        'turnovers', 'pra', 'pr', 'pa', 'ra', 'blocks_plus_steals'
    ]
    windows = [3, 5, 10]

    for stat in stats_to_roll:
        if stat in df.columns:
            for window in windows:
                df[f'{stat}_rolling_{window}'] = df.groupby('player_id')[stat].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
    
    return df


# --- Team-Level Feature Calculation ---

STATS_TO_AGGREGATE = [
    'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 
    'field_goals_made', 'field_goals_attempted', 
    'three_pointers_made', 'three_pointers_attempted',
    'free_throws_made', 'free_throws_attempted', 'possessions'
]

def _calculate_team_game_stats(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal helper to aggregate player stats into team totals for each game.
    """
    for col in STATS_TO_AGGREGATE:
        if col not in player_stats_df.columns:
            player_stats_df[col] = 0
            
    team_game_stats = player_stats_df.groupby(
        ['game_id', 'team_id', 'game_datetime', 'opponent_team_id']
    )[STATS_TO_AGGREGATE].sum().reset_index()

    # Safely calculate advanced stats
    efg_numerator = team_game_stats['field_goals_made'] + 0.5 * team_game_stats['three_pointers_made']
    efg_denominator = team_game_stats['field_goals_attempted'].replace(0, np.nan)
    team_game_stats['effective_field_goal_percentage'] = (efg_numerator / efg_denominator)

    ts_numerator = team_game_stats['points']
    ts_denominator = (2 * (team_game_stats['field_goals_attempted'] + 0.44 * team_game_stats['free_throws_attempted'])).replace(0, np.nan)
    team_game_stats['true_shooting_percentage'] = (ts_numerator / ts_denominator)
    
    team_game_stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    team_game_stats.fillna(0, inplace=True)

    return team_game_stats


def get_team_performance_rolling_averages(all_player_stats_df: pd.DataFrame, rolling_windows: list) -> pd.DataFrame:
    """
    Calculates rolling averages of a team's own offensive performance.
    """
    team_game_stats = _calculate_team_game_stats(all_player_stats_df)
    
    if team_game_stats.empty:
        logger.warning("Team game stats are empty, cannot calculate performance rolling averages.")
        return pd.DataFrame()
        
    team_game_stats.sort_values('game_datetime', inplace=True)
    
    rolling_cols = [
        'points', 'possessions', 'effective_field_goal_percentage', 'true_shooting_percentage',
        'assists', 'rebounds', 'turnovers'
    ]
    
    # Set game_datetime as index to perform time-based rolling window operations
    indexed_team_stats = team_game_stats.set_index('game_datetime')
    
    all_rolling_dfs = []
    for w in rolling_windows:
        rolling_result = indexed_team_stats.groupby('team_id')[rolling_cols].rolling(window=f'{w}D', closed='left').mean()
        rolling_result.rename(columns={c: f'team_{c}_rolling_{w}d' for c in rolling_cols}, inplace=True)
        all_rolling_dfs.append(rolling_result)
        
    if not all_rolling_dfs:
        return pd.DataFrame()

    final_rolling_df = pd.concat(all_rolling_dfs, axis=1).reset_index()

    return final_rolling_df


def get_team_defensive_rolling_averages(all_player_stats_df: pd.DataFrame, rolling_windows: list) -> pd.DataFrame:
    """
    Calculates rolling averages of stats CONCEDED by a team to their opponents.
    """
    team_game_stats = _calculate_team_game_stats(all_player_stats_df)
    
    if team_game_stats.empty:
        logger.warning("Team game stats are empty, cannot calculate defensive rolling averages.")
        return pd.DataFrame()

    conceded_df = pd.merge(
        team_game_stats,
        team_game_stats,
        on='game_id',
        how='left',
        suffixes=('_team', '_opponent')
    )
    
    conceded_df = conceded_df[conceded_df['team_id_team'] != conceded_df['team_id_opponent']]
    
    conceded_cols_map = {f'{col}_opponent': f'{col}_conceded' for col in STATS_TO_AGGREGATE + ['effective_field_goal_percentage', 'true_shooting_percentage']}
    conceded_df.rename(columns=conceded_cols_map, inplace=True)
    
    final_cols = ['team_id_team', 'game_datetime_team'] + list(conceded_cols_map.values())
    conceded_df = conceded_df[final_cols]
    conceded_df.rename(columns={'team_id_team': 'team_id', 'game_datetime_team': 'game_datetime'}, inplace=True)

    conceded_df.sort_values('game_datetime', inplace=True)

    rolling_conceded_cols = [col for col in conceded_df.columns if '_conceded' in col]
    
    indexed_conceded_stats = conceded_df.set_index('game_datetime')

    all_rolling_dfs = []
    for w in rolling_windows:
        rolling_result = indexed_conceded_stats.groupby('team_id')[rolling_conceded_cols].rolling(window=f'{w}D', closed='left').mean()
        rolling_result.rename(columns={c: f'opponent_{c}_rolling_{w}d' for c in rolling_conceded_cols}, inplace=True)
        all_rolling_dfs.append(rolling_result)

    if not all_rolling_dfs:
        return pd.DataFrame()

    final_rolling_df = pd.concat(all_rolling_dfs, axis=1).reset_index()
    
    return final_rolling_df