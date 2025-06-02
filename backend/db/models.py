import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, Text, Boolean, UniqueConstraint, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from .base import Base

class Team(Base):
    __tablename__ = "teams"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_name = Column(String, unique=True, index=True, nullable=False)
    api_team_id = Column(String, unique=True, nullable=True)

    # Relationship to players is implicitly handled by Player.team backref
    # Relationship to games as home/away team is handled by Game.home_team_ref/away_team_ref
    pbp_events = relationship("PlayByPlayEvent", back_populates="team")

class Player(Base):
    __tablename__ = "players"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_name = Column(String, unique=False, index=True, nullable=False)
    api_player_id = Column(String, unique=True, nullable=True, index=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)

    # Relationships
    team = relationship("Team", backref="players")
    stats = relationship("PlayerStat", back_populates="player")
    prop_odds = relationship("PlayerProp", back_populates="player")
    pbp_events_as_player1 = relationship("PlayByPlayEvent", foreign_keys="PlayByPlayEvent.player1_id", back_populates="player1")
    pbp_events_as_player2 = relationship("PlayByPlayEvent", foreign_keys="PlayByPlayEvent.player2_id", back_populates="player2")
    pbp_events_as_player3 = relationship("PlayByPlayEvent", foreign_keys="PlayByPlayEvent.player3_id", back_populates="player3")

    @property
    def team_name(self) -> str | None:
        if self.team:
            return self.team.team_name
        return None

class PlayerStat(Base):
    __tablename__ = "player_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=False)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)
    season = Column(Integer, nullable=True, index=True)
    true_shooting_percentage = Column(Float, nullable=True)
    effective_field_goal_percentage = Column(Float, nullable=True)
    assist_percentage = Column(Float, nullable=True)
    offensive_rebound_percentage = Column(Float, nullable=True)
    defensive_rebound_percentage = Column(Float, nullable=True)
    total_rebound_percentage = Column(Float, nullable=True)
    steal_percentage = Column(Float, nullable=True)
    block_percentage = Column(Float, nullable=True)
    turnover_percentage = Column(Float, nullable=True)
    usage_rate = Column(Float, nullable=True)
    player_efficiency_rating = Column(Float, nullable=True)
    is_home_team = Column(Boolean, nullable=False, default=False)
    points = Column(Float, nullable=True)
    rebounds = Column(Float, nullable=True)
    offensive_rebounds = Column(Integer, nullable=True)
    defensive_rebounds = Column(Integer, nullable=True)
    assists = Column(Float, nullable=True)
    steals = Column(Float, nullable=True)
    blocks = Column(Float, nullable=True)
    turnovers = Column(Float, nullable=True)
    fouls = Column(Integer, nullable=True)
    minutes_played = Column(Float, nullable=True)
    game_date = Column(Date, nullable=True)
    field_goals_made = Column(Integer, nullable=True)
    field_goals_attempted = Column(Integer, nullable=True)
    three_pointers_made = Column(Integer, nullable=True)
    three_pointers_attempted = Column(Integer, nullable=True)
    free_throws_made = Column(Integer, nullable=True)
    free_throws_attempted = Column(Integer, nullable=True)
    plus_minus = Column(Integer, nullable=True)

    # Relationship
    player = relationship("Player", back_populates="stats")
    game = relationship("Game", back_populates="stats")
    team = relationship("Team")

class Game(Base):
    __tablename__ = "games"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String, unique=True, index=True, nullable=True)
    home_team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)
    away_team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)
    game_datetime = Column(DateTime(timezone=True), nullable=False)
    season = Column(Integer, nullable=True, index=True)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)

    # Team Aggregates (NEW)
    home_team_minutes_played = Column(Float, nullable=True)
    away_team_minutes_played = Column(Float, nullable=True)
    home_team_field_goals_made = Column(Integer, nullable=True)
    away_team_field_goals_made = Column(Integer, nullable=True)
    home_team_field_goals_attempted = Column(Integer, nullable=True)
    away_team_field_goals_attempted = Column(Integer, nullable=True)
    home_team_three_pointers_made = Column(Integer, nullable=True)
    away_team_three_pointers_made = Column(Integer, nullable=True)
    home_team_three_pointers_attempted = Column(Integer, nullable=True)
    away_team_three_pointers_attempted = Column(Integer, nullable=True)
    home_team_free_throws_made = Column(Integer, nullable=True)
    away_team_free_throws_made = Column(Integer, nullable=True)
    home_team_free_throws_attempted = Column(Integer, nullable=True)
    away_team_free_throws_attempted = Column(Integer, nullable=True)
    home_team_offensive_rebounds = Column(Integer, nullable=True)
    away_team_offensive_rebounds = Column(Integer, nullable=True)
    home_team_defensive_rebounds = Column(Integer, nullable=True)
    away_team_defensive_rebounds = Column(Integer, nullable=True)
    home_team_total_rebounds = Column(Integer, nullable=True)
    away_team_total_rebounds = Column(Integer, nullable=True)
    home_team_assists = Column(Integer, nullable=True)
    away_team_assists = Column(Integer, nullable=True)
    home_team_steals = Column(Integer, nullable=True)
    away_team_steals = Column(Integer, nullable=True)
    home_team_blocks = Column(Integer, nullable=True)
    away_team_blocks = Column(Integer, nullable=True)
    home_team_turnovers = Column(Integer, nullable=True)
    away_team_turnovers = Column(Integer, nullable=True)
    home_team_fouls = Column(Integer, nullable=True)
    away_team_fouls = Column(Integer, nullable=True)
    home_team_possessions = Column(Float, nullable=True)
    away_team_possessions = Column(Float, nullable=True)

    # Relationships for Game to Team
    home_team_ref = relationship("Team", foreign_keys=[home_team_id])
    away_team_ref = relationship("Team", foreign_keys=[away_team_id])

    @property
    def home_team(self) -> str | None:
        if self.home_team_ref:
            return self.home_team_ref.team_name
        return None

    @property
    def away_team(self) -> str | None:
        if self.away_team_ref:
            return self.away_team_ref.team_name
        return None

    # Relationship
    odds = relationship("GameOdd", back_populates="game")
    stats = relationship("PlayerStat", back_populates="game")
    play_by_play_events = relationship("PlayByPlayEvent", back_populates="game", cascade="all, delete-orphan")

class GameOdd(Base):
    __tablename__ = "game_odds"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False, index=True)
    bookmaker_id = Column(UUID(as_uuid=True), ForeignKey("bookmakers.id"), nullable=False, index=True)
    market_id = Column(UUID(as_uuid=True), ForeignKey("markets.id"), nullable=False, index=True)
        
    last_update_api = Column(DateTime(timezone=True))
    outcomes = Column(JSON) 

    game = relationship("Game", back_populates="odds")
    bookmaker = relationship("Bookmaker", back_populates="game_odds")
    market = relationship("Market", back_populates="game_odds")

    __table_args__ = (UniqueConstraint('game_id', 'bookmaker_id', 'market_id', name='_game_bookmaker_market_uc'),)

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_name = Column(String, nullable=False, unique=True)
    trained_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    description = Column(String, nullable=True)
    model_path = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)

    predictions = relationship("Prediction", back_populates="model_version")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_prop_id = Column(UUID(as_uuid=True), ForeignKey("player_props.id"), nullable=False)
    model_version_id = Column(UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=False)
    
    predicted_value = Column(Float, nullable=True)
    predicted_over_probability = Column(Float, nullable=True)
    predicted_under_probability = Column(Float, nullable=True)
    prediction_datetime = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # New fields for tracking actual outcomes
    actual_value = Column(Float, nullable=True)  # The actual stat value achieved by the player
    outcome = Column(String, nullable=True)  # e.g., 'OVER', 'UNDER', 'PUSH'
    outcome_processed_at = Column(DateTime, nullable=True) # When the outcome was recorded

    player_prop = relationship("PlayerProp", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")

class Parlay(Base):
    __tablename__ = "parlays"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    selections = Column(JSONB, nullable=False)
    combined_probability = Column(Float, nullable=True)
    total_odds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class Sport(Base):
    __tablename__ = "sports"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String, unique=True, index=True, nullable=False)
    group_name = Column(String)
    title = Column(String)
    description = Column(String, nullable=True)
    active = Column(Boolean, default=True)
    has_outrights = Column(Boolean, default=False)

class Bookmaker(Base):
    __tablename__ = "bookmakers"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    
    game_odds = relationship("GameOdd", back_populates="bookmaker")
    player_props = relationship("PlayerProp", back_populates="bookmaker")

class Market(Base):
    __tablename__ = "markets"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String, unique=True, index=True, nullable=False)
    description = Column(String, nullable=True)
    
    game_odds = relationship("GameOdd", back_populates="market")
    player_props = relationship("PlayerProp", back_populates="market")

class PlayerProp(Base):
    __tablename__ = "player_props"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False, index=True)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=True, index=True)
    bookmaker_id = Column(UUID(as_uuid=True), ForeignKey("bookmakers.id"), nullable=False, index=True)
    market_id = Column(UUID(as_uuid=True), ForeignKey("markets.id"), nullable=False, index=True)
    
    player_name_api = Column(String, index=True)
    last_update_api = Column(DateTime(timezone=True))
    outcomes = Column(JSON)

    game = relationship("Game")
    player = relationship("Player", back_populates="prop_odds")
    bookmaker = relationship("Bookmaker", back_populates="player_props")
    market = relationship("Market", back_populates="player_props")
    
    # Add relationship to Prediction
    predictions = relationship("Prediction", back_populates="player_prop", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint('game_id', 'player_name_api', 'bookmaker_id', 'market_id', name='_game_player_bookmaker_market_uc'),)

# New Model for Play-By-Play Events
class PlayByPlayEvent(Base):
    __tablename__ = "play_by_play_events"

    id = Column(Integer, primary_key=True, index=True)
    external_event_id = Column(String, index=True, comment="Source's unique ID for the event, e.g., from ESPN API's play 'id' field")
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False, index=True)
    
    sequence_number = Column(String, nullable=True, comment="Event sequence number within the game")
    event_text = Column(String, nullable=True, comment="Textual description of the play")
    
    away_score_after_event = Column(Integer, nullable=True)
    home_score_after_event = Column(Integer, nullable=True)
    scoring_play = Column(Boolean, nullable=True)
    score_value = Column(Integer, nullable=True, comment="Point value of the score if a scoring play")

    wallclock_time = Column(String, nullable=True, comment="Wallclock time of the event as a string")
    game_clock_display = Column(String, nullable=True, comment="Game clock time, e.g., '10:34'")
    period = Column(Integer, nullable=True, comment="Game period (quarter/overtime)")
    period_display_value = Column(String, nullable=True, comment="e.g., '1st', 'OT1'")

    event_type_id = Column(Integer, nullable=True, comment="Source's numeric ID for event type")
    event_type_text = Column(String, nullable=True, index=True, comment="Source's textual description of event type, e.g., 'Made Shot', 'Rebound'")

    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True, index=True, comment="ID of the team that initiated or is primarily associated with the event")
    
    # Player IDs involved in the event
    # Using ESPN's athlete_id_1, athlete_id_2, athlete_id_3 convention
    player1_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=True, index=True, comment="Primary player in the event (e.g., shooter, rebounder, fouler)")
    player2_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=True, index=True, comment="Secondary player (e.g., assister, fouled by)")
    player3_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=True, index=True, comment="Tertiary player, if applicable")

    # Shot details (if applicable)
    shooting_play = Column(Boolean, nullable=True)
    shot_made = Column(Boolean, nullable=True, comment="True if a shot or free throw was made (relevant if shooting_play is True)")
    # Example: type_text might be "Field Goal Attempt". shot_made determines if it was successful.
    # For free throws, type_text might be "Free Throw Attempt"
    # Further detail on shot type (2pt, 3pt) might need to be inferred from event_text or coordinates
    # or if sportsdataverse provides a more specific field not shown in the immediate R docs.
    shot_type_detail = Column(String, nullable=True, comment="e.g., 'Jump Shot', 'Layup', 'Free Throw', '3-pointer' - often in event_text")

    coordinate_x_raw = Column(Float, nullable=True, comment="Raw X coordinate from source")
    coordinate_y_raw = Column(Float, nullable=True, comment="Raw Y coordinate from source")
    coordinate_x = Column(Float, nullable=True, comment="Standardized X coordinate (e.g., 0-94 feet)")
    coordinate_y = Column(Float, nullable=True, comment="Standardized Y coordinate (e.g., 0-50 feet)")

    # Relationship back to Game, Player, Team
    game = relationship("Game", back_populates="play_by_play_events")
    team = relationship("Team", back_populates="pbp_events")
    player1 = relationship("Player", foreign_keys=[player1_id], back_populates="pbp_events_as_player1")
    player2 = relationship("Player", foreign_keys=[player2_id], back_populates="pbp_events_as_player2")
    player3 = relationship("Player", foreign_keys=[player3_id], back_populates="pbp_events_as_player3")

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<PlayByPlayEvent(id={self.id}, game_id={self.game_id}, period={self.period}, clock='{self.game_clock_display}', type='{self.event_type_text}', text='{self.event_text[:50]}...')>"

# Update existing models to have relationships to PlayByPlayEvent if desired
# For example, in Game model:
# play_by_play_events = relationship("PlayByPlayEvent", back_populates="game", cascade="all, delete-orphan")
# In Player model (though less direct, might be complex due to player1/2/3_id):
# pbp_events_as_player1 = relationship("PlayByPlayEvent", foreign_keys="[PlayByPlayEvent.player1_id]", back_populates="player1")

# In Team model:
# pbp_events = relationship("PlayByPlayEvent", back_populates="team")

class LeagueSeasonAverages(Base):
    __tablename__ = "league_season_averages"

    season_year = Column(Integer, primary_key=True, index=True)
    games_played = Column(Integer, nullable=True) # Total number of games in the season for reference
    team_games_played = Column(Integer, nullable=True) # Total team-game instances (games_played * 2)

    lg_points = Column(Float, nullable=True)
    lg_field_goals_made = Column(Float, nullable=True)
    lg_field_goals_attempted = Column(Float, nullable=True)
    lg_three_pointers_made = Column(Float, nullable=True)
    lg_three_pointers_attempted = Column(Float, nullable=True)
    lg_free_throws_made = Column(Float, nullable=True)
    lg_free_throws_attempted = Column(Float, nullable=True)
    lg_offensive_rebounds = Column(Float, nullable=True)
    lg_defensive_rebounds = Column(Float, nullable=True)
    lg_total_rebounds = Column(Float, nullable=True)
    lg_assists = Column(Float, nullable=True)
    lg_steals = Column(Float, nullable=True)
    lg_blocks = Column(Float, nullable=True)
    lg_turnovers = Column(Float, nullable=True)
    lg_fouls = Column(Float, nullable=True)
    lg_possessions = Column(Float, nullable=True)  # This is lgPace (average possessions per team per game)
    
    # Derived league metrics
    lg_value_of_possession = Column(Float, nullable=True) # VOP
    lg_defensive_rebound_percentage = Column(Float, nullable=True) # lgDRB%
    # Could also add lgFactor if needed for more complex PER versions later

    def __repr__(self):
        return f"<LeagueSeasonAverages(season={self.season_year}, lg_pts={self.lg_points:.1f}, lg_pace={self.lg_possessions:.1f})>"