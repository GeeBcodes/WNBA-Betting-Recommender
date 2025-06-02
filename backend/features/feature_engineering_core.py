import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def generate_full_feature_set(
    base_df: pd.DataFrame, 
    target_stat: str, 
    opponent_defense_df: Optional[pd.DataFrame] = None,
    team_performance_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Orchestrates the generation of all features for the model.
    This function will call other specific feature generation functions.

    Args:
        base_df: DataFrame containing raw loaded data. Expected columns include 
                 player_id, game_datetime, target_stat, minutes_played,
                 is_home_team, home_team_name, away_team_name, season, game_id.
        target_stat: The name of the target statistic column.
        opponent_defense_df: Optional DataFrame with pre-calculated opponent 
                             defensive rolling averages. Expected columns:
                             game_id, defending_team_name (as opponent_team_name), 
                             opponent_{target_stat}_conceded_roll_avg.
        team_performance_df: Optional DataFrame with pre-calculated team 
                             performance rolling averages. Expected columns: 
                             game_id, player_actual_team_name, 
                             team_{target_stat}_roll_avg, team_total_points_roll_avg.

    Returns:
        A DataFrame with all engineered features.
    """
    if base_df.empty:
        logger.warning("Base DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df = base_df.copy()

    # Sort by player_id and game_datetime (crucial for time-based features)
    df = df.sort_values(by=['player_id', 'game_datetime'])

    # 1. Generate time-based player features (lags, rolls, rest days)
    df = _generate_player_time_series_features(df, target_stat)

    # 2. Generate game context features (is_home, opponent_name)
    df = _generate_game_context_features(df)
    
    # 3. Merge opponent defensive stats
    df = _merge_opponent_defensive_stats(df, target_stat, opponent_defense_df)

    # 4. Merge player's team performance stats
    df = _merge_team_performance_stats(df, target_stat, team_performance_df)

    # 5. Generate interaction features
    df = _generate_interaction_features(df, target_stat)

    logger.info(f"Feature engineering complete. Final DataFrame shape: {df.shape}")
    return df

def _generate_player_time_series_features(df: pd.DataFrame, target_stat: str) -> pd.DataFrame:
    logger.info(f"Generating player time-series features for target: {target_stat}")
    
    # Lagged features for the target stat
    for lag in [1, 2, 3]:
        df[f'{target_stat}_lag_{lag}'] = df.groupby('player_id')[target_stat].shift(lag)
    
    # Rolling average for target_stat
    df[f'{target_stat}_roll_avg_3'] = df.groupby('player_id')[target_stat].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1) # shift(1) to avoid data leakage
    )
    # Rolling average for minutes_played
    if 'minutes_played' in df.columns:
        df[f'minutes_played_roll_avg_3'] = df.groupby('player_id')['minutes_played'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
    else:
        logger.warning("'minutes_played' column not found for rolling average. Skipping minutes_played_roll_avg_3.")

    # Days since last game
    df['days_since_last_game'] = df.groupby('player_id')['game_datetime'].diff().dt.days.fillna(7)
    return df

def _generate_game_context_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating game context features (is_home, opponent_team_name).")
    # Create is_home feature
    if 'is_home_team' in df.columns:
        df['is_home'] = df['is_home_team'].astype(int)
    else:
        logger.warning("'is_home_team' column not found. 'is_home' feature will default to 0 or be missing.")
        df['is_home'] = 0 

    # Create opponent_team_name feature
    if 'is_home_team' in df.columns and 'home_team_name' in df.columns and 'away_team_name' in df.columns:
        df['opponent_team_name'] = df.apply(
            lambda row: row['away_team_name'] if row['is_home_team'] else row['home_team_name'], 
            axis=1
        )
    else:
        logger.warning("Required columns for opponent_team_name not found. Feature will be default or missing.")
        df['opponent_team_name'] = "UnknownOpponent"
    return df

def _merge_opponent_defensive_stats(df: pd.DataFrame, target_stat: str, opponent_defense_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if opponent_defense_df is None or opponent_defense_df.empty:
        logger.info("No opponent defensive stats provided or it's empty. Skipping merge.")
        return df

    logger.info("Merging opponent defensive stats.")
    if 'opponent_team_name' not in df.columns:
        logger.warning("'opponent_team_name' not in main DataFrame. Cannot merge opponent defensive strength.")
        return df

    opponent_feature_col = f'opponent_{target_stat}_conceded_roll_avg'
    if opponent_feature_col not in opponent_defense_df.columns:
        logger.warning(f"Column '{opponent_feature_col}' not found in opponent_defense_df. Skipping merge.")
        return df

    temp_opponent_defense_df = opponent_defense_df[['game_id', 'defending_team_name', opponent_feature_col]].copy()
    temp_opponent_defense_df.rename(columns={'defending_team_name': 'opponent_team_name'}, inplace=True)
    
    df = pd.merge(df, temp_opponent_defense_df, on=['game_id', 'opponent_team_name'], how='left')
    logger.info(f"Successfully merged opponent defensive strength for {target_stat}.")
    return df

def _merge_team_performance_stats(df: pd.DataFrame, target_stat: str, team_performance_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if team_performance_df is None or team_performance_df.empty:
        logger.info("No team performance stats provided or it's empty. Skipping merge.")
        return df
    
    logger.info("Merging team performance stats.")
    # Determine player's actual team for merging
    if 'is_home_team' in df.columns and 'home_team_name' in df.columns and 'away_team_name' in df.columns:
        df['player_actual_team_name_for_merge'] = df.apply(
            lambda row: row['home_team_name'] if row['is_home_team'] else row['away_team_name'], 
            axis=1
        )
    else:
        logger.warning("Required columns for player_actual_team_name_for_merge not found. Team performance merge might fail.")
        df['player_actual_team_name_for_merge'] = "UnknownPlayerTeam"
        return df # Cannot merge without player's team

    team_target_stat_col = f'team_{target_stat}_roll_avg'
    team_total_points_col = 'team_total_points_roll_avg'
    
    cols_to_merge_from_source = ['game_id', 'player_actual_team_name']
    features_added_log = []

    if team_target_stat_col in team_performance_df.columns:
        cols_to_merge_from_source.append(team_target_stat_col)
        features_added_log.append(team_target_stat_col)
    if team_total_points_col in team_performance_df.columns:
        cols_to_merge_from_source.append(team_total_points_col)
        features_added_log.append(team_total_points_col)

    if not features_added_log: # No relevant feature columns found in team_performance_df
        logger.warning(f"Relevant feature columns not found in team_performance_df for target {target_stat}. Skipping merge.")
        return df

    temp_team_performance_df = team_performance_df[cols_to_merge_from_source].copy()
    temp_team_performance_df.rename(columns={'player_actual_team_name': 'player_actual_team_name_for_merge'}, inplace=True)

    df = pd.merge(df, temp_team_performance_df, on=['game_id', 'player_actual_team_name_for_merge'], how='left')
    logger.info(f"Successfully merged team performance stats. Features added: {features_added_log}")
    
    # Clean up the temporary merge key if it's not needed downstream
    # df.drop(columns=['player_actual_team_name_for_merge'], inplace=True, errors='ignore')
    return df

def _generate_interaction_features(df: pd.DataFrame, target_stat: str) -> pd.DataFrame:
    logger.info("Creating interaction features...")
    
    interaction_feature_1_name = f'player_{target_stat}_roll_avg_vs_opp_conceded'
    player_stat_roll_avg_col = f'{target_stat}_roll_avg_3'
    opp_conceded_roll_avg_col = f'opponent_{target_stat}_conceded_roll_avg'

    if player_stat_roll_avg_col in df.columns and opp_conceded_roll_avg_col in df.columns:
        df[interaction_feature_1_name] = df[player_stat_roll_avg_col] * df[opp_conceded_roll_avg_col]
        logger.info(f"Created interaction feature: {interaction_feature_1_name}")
    else:
        missing_cols = []
        if player_stat_roll_avg_col not in df.columns: missing_cols.append(player_stat_roll_avg_col)
        if opp_conceded_roll_avg_col not in df.columns: missing_cols.append(opp_conceded_roll_avg_col)
        logger.warning(f"Could not create '{interaction_feature_1_name}'. Missing columns: {missing_cols}")

    interaction_feature_2_name = f'player_minutes_roll_avg_vs_opp_conceded'
    player_minutes_roll_avg_col = 'minutes_played_roll_avg_3'
    
    if player_minutes_roll_avg_col in df.columns and opp_conceded_roll_avg_col in df.columns:
        df[interaction_feature_2_name] = df[player_minutes_roll_avg_col] * df[opp_conceded_roll_avg_col]
        logger.info(f"Created interaction feature: {interaction_feature_2_name}")
    else:
        missing_cols_2 = []
        if player_minutes_roll_avg_col not in df.columns: missing_cols_2.append(player_minutes_roll_avg_col)
        if opp_conceded_roll_avg_col not in df.columns: missing_cols_2.append(opp_conceded_roll_avg_col)
        logger.warning(f"Could not create '{interaction_feature_2_name}'. Missing columns: {missing_cols_2}")
    return df 