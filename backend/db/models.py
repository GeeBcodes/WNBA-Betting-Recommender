import uuid
from datetime import datetime # Added datetime import
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from .base import Base # Assuming base.py is in the same directory

class Player(Base):
    __tablename__ = "players"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_name = Column(String, unique=True, index=True, nullable=False)
    team_name = Column(String, nullable=True)

    # Relationships
    stats = relationship("PlayerStat", back_populates="player")
    prop_odds = relationship("PlayerPropOdd", back_populates="player")

class PlayerStat(Base):
    __tablename__ = "player_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=False)
    # player_name = Column(String, index=True) # Redundant if we link to Player table
    # team_name = Column(String) # Redundant if we link to Player table or a Team table
    points = Column(Float, nullable=True)
    rebounds = Column(Float, nullable=True)
    assists = Column(Float, nullable=True)
    steals = Column(Float, nullable=True)
    blocks = Column(Float, nullable=True)
    turnovers = Column(Float, nullable=True)
    minutes_played = Column(Float, nullable=True)
    game_date = Column(Date, nullable=True)

    # Relationship
    player = relationship("Player", back_populates="stats")

class Game(Base):
    __tablename__ = "games"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4) # Game identifier
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    game_datetime = Column(DateTime, nullable=True) # To store specific game time

    # Relationship
    odds = relationship("GameOdd", back_populates="game")

class GameOdd(Base):
    __tablename__ = "game_odds"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False)
    home_team_odds = Column(Float, nullable=True)
    away_team_odds = Column(Float, nullable=True)
    spread = Column(Float, nullable=True)
    over_under = Column(Float, nullable=True)
    source = Column(String, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    game = relationship("Game", back_populates="odds")

class PlayerPropOdd(Base):
    __tablename__ = "player_prop_odds"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=False)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=True) # Prop might be tied to a game
    stat_type = Column(String, nullable=False)  # e.g., "points", "rebounds"
    line = Column(Float, nullable=False)
    over_odds = Column(Integer, nullable=False)
    under_odds = Column(Integer, nullable=False)
    source = Column(String, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    player = relationship("Player", back_populates="prop_odds")
    # game = relationship("Game") # If you need to link props to games directly 

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_name = Column(String, nullable=False, unique=True) # e.g., "20240516_v1"
    trained_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    description = Column(String, nullable=True)
    # parameters = Column(JSONB, nullable=True) # Store model parameters if needed
    # accuracy = Column(Float, nullable=True) # Store model accuracy if needed

    predictions = relationship("Prediction", back_populates="model_version")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_prop_odd_id = Column(UUID(as_uuid=True), ForeignKey("player_prop_odds.id"), nullable=False)
    model_version_id = Column(UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=False)
    # game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False) # Could be derived from player_prop_odd_id
    # player_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=False) # Could be derived from player_prop_odd_id
    
    predicted_over_probability = Column(Float, nullable=True)
    predicted_under_probability = Column(Float, nullable=True)
    # Could also store the actual line used for this prediction if it can vary from the PlayerPropOdd's line
    # predicted_line = Column(Float, nullable=True)
    prediction_datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

    player_prop_odd = relationship("PlayerPropOdd") # Add back_populates if needed in PlayerPropOdd
    model_version = relationship("ModelVersion", back_populates="predictions")

class Parlay(Base):
    __tablename__ = "parlays"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True) # If you add users later
    selections = Column(JSONB, nullable=False) # Store array of prediction IDs or prop details
    combined_probability = Column(Float, nullable=True)
    total_odds = Column(Float, nullable=True) # If you calculate and store combined odds
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # If selections are prediction_ids, you might not need direct relationships here,
    # or you could have a many-to-many through an association table if parlays can share prediction instances.
    # For simplicity, storing IDs in JSONB is often easiest for V1. 