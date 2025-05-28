import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, Text, Boolean, UniqueConstraint, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from .base import Base

class Team(Base):
    __tablename__ = "teams"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_name = Column(String, unique=True, index=True, nullable=False)
    api_team_id = Column(String, unique=True, nullable=True)

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

class PlayerStat(Base):
    __tablename__ = "player_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=False)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False)
    is_home_team = Column(Boolean, nullable=False, default=False)
    points = Column(Float, nullable=True)
    rebounds = Column(Float, nullable=True)
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

class Game(Base):
    __tablename__ = "games"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String, unique=True, index=True, nullable=True)
    home_team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)
    away_team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)
    game_datetime = Column(DateTime, nullable=True)
    season = Column(Integer, nullable=True, index=True)

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
    prediction_datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

    player_prop = relationship("PlayerProp")
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

    __table_args__ = (UniqueConstraint('game_id', 'player_name_api', 'bookmaker_id', 'market_id', name='_game_player_bookmaker_market_uc'),)