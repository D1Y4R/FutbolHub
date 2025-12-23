"""
Database models for Football Prediction Hub
Using SQLAlchemy with PostgreSQL
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
from datetime import datetime
import os

Base = declarative_base()

class Team(Base):
    """Team model for storing team information"""
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.id'))
    logo_url = Column(String(500))
    stadium = Column(String(255))
    city = Column(String(255))
    country = Column(String(100))
    founded = Column(Integer)
    
    # Performance metrics
    elo_rating = Column(Float, default=1500.0)
    attack_strength = Column(Float, default=1.0)
    defense_strength = Column(Float, default=1.0)
    form_rating = Column(Float, default=0.0)
    
    # Relationships
    league = relationship("League", back_populates="teams")
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    statistics = relationship("TeamStatistics", back_populates="team")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_team_name', 'name'),
        Index('idx_team_league', 'league_id'),
        UniqueConstraint('name', 'league_id', name='uq_team_league')
    )

class League(Base):
    """League model for storing league/competition information"""
    __tablename__ = 'leagues'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    country = Column(String(100))
    logo_url = Column(String(500))
    season = Column(String(50))
    api_id = Column(Integer, unique=True)  # External API ID
    
    # League characteristics
    avg_goals_per_match = Column(Float, default=2.5)
    league_type = Column(String(50), default='normal')  # high_scoring, low_scoring, normal
    
    # Relationships
    teams = relationship("Team", back_populates="league")
    matches = relationship("Match", back_populates="league")
    
    # Indexes
    __table_args__ = (
        Index('idx_league_name', 'name'),
        Index('idx_league_api_id', 'api_id'),
    )

class Match(Base):
    """Match model for storing match information and results"""
    __tablename__ = 'matches'
    
    id = Column(Integer, primary_key=True)
    home_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    away_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.id'), nullable=False)
    
    # Match information
    match_date = Column(DateTime, nullable=False)
    round = Column(String(50))
    venue = Column(String(255))
    referee = Column(String(255))
    api_fixture_id = Column(Integer, unique=True)  # External API fixture ID
    
    # Match status
    status = Column(String(50), default='SCHEDULED')  # SCHEDULED, LIVE, FINISHED, POSTPONED
    elapsed_time = Column(Integer, default=0)
    
    # Results
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_score_halftime = Column(Integer)
    away_score_halftime = Column(Integer)
    
    # Advanced statistics
    home_xg = Column(Float)
    away_xg = Column(Float)
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_shots_on_target = Column(Integer)
    away_shots_on_target = Column(Integer)
    home_possession = Column(Float)
    away_possession = Column(Float)
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    league = relationship("League", back_populates="matches")
    predictions = relationship("Prediction", back_populates="match")
    
    # Indexes
    __table_args__ = (
        Index('idx_match_date', 'match_date'),
        Index('idx_match_teams', 'home_team_id', 'away_team_id'),
        Index('idx_match_status', 'status'),
        Index('idx_match_api_id', 'api_fixture_id'),
    )

class Prediction(Base):
    """Prediction model for storing match predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey('matches.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Basic predictions
    predicted_winner = Column(String(50))  # HOME, DRAW, AWAY
    home_win_probability = Column(Float)
    draw_probability = Column(Float)
    away_win_probability = Column(Float)
    
    # Score predictions
    predicted_home_score = Column(Float)
    predicted_away_score = Column(Float)
    most_likely_score = Column(String(20))  # e.g., "2-1"
    
    # Betting predictions
    over_2_5_probability = Column(Float)
    under_2_5_probability = Column(Float)
    btts_yes_probability = Column(Float)
    btts_no_probability = Column(Float)
    
    # Half-time/Full-time predictions
    ht_ft_prediction = Column(String(50))
    ht_ft_probabilities = Column(JSON)  # Store all HT/FT combinations
    
    # Advanced predictions
    asian_handicap = Column(JSON)  # Store handicap lines and probabilities
    correct_score_probabilities = Column(JSON)  # Store all score probabilities
    
    # Confidence and metadata
    confidence_score = Column(Float)
    algorithm_weights = Column(JSON)  # Store which algorithms were used and their weights
    prediction_factors = Column(JSON)  # Store key factors that influenced the prediction
    
    # Relationships
    match = relationship("Match", back_populates="predictions")
    
    # Indexes
    __table_args__ = (
        Index('idx_prediction_match', 'match_id'),
        Index('idx_prediction_created', 'created_at'),
        UniqueConstraint('match_id', 'created_at', name='uq_match_prediction')
    )

class TeamStatistics(Base):
    """Team statistics model for storing performance metrics"""
    __tablename__ = 'team_statistics'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    season = Column(String(50), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Overall statistics
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    
    # Goal statistics
    goals_scored = Column(Integer, default=0)
    goals_conceded = Column(Integer, default=0)
    clean_sheets = Column(Integer, default=0)
    btts_count = Column(Integer, default=0)
    
    # Home/Away splits
    home_wins = Column(Integer, default=0)
    home_draws = Column(Integer, default=0)
    home_losses = Column(Integer, default=0)
    home_goals_scored = Column(Integer, default=0)
    home_goals_conceded = Column(Integer, default=0)
    
    away_wins = Column(Integer, default=0)
    away_draws = Column(Integer, default=0)
    away_losses = Column(Integer, default=0)
    away_goals_scored = Column(Integer, default=0)
    away_goals_conceded = Column(Integer, default=0)
    
    # Form and streaks
    current_form = Column(String(10))  # Last 5 matches: WWDLW
    current_streak = Column(JSON)  # {"type": "win", "count": 3}
    
    # Advanced metrics
    xg_for = Column(Float, default=0.0)
    xg_against = Column(Float, default=0.0)
    avg_possession = Column(Float, default=50.0)
    shots_per_game = Column(Float, default=0.0)
    shots_on_target_per_game = Column(Float, default=0.0)
    
    # Relationships
    team = relationship("Team", back_populates="statistics")
    
    # Indexes
    __table_args__ = (
        Index('idx_teamstats_team_season', 'team_id', 'season'),
        UniqueConstraint('team_id', 'season', name='uq_team_season_stats')
    )

class ModelPerformance(Base):
    """Track performance of prediction models"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.id'))
    evaluated_at = Column(DateTime, default=datetime.utcnow)
    
    # Performance metrics
    accuracy_1x2 = Column(Float)
    accuracy_btts = Column(Float)
    accuracy_over_under = Column(Float)
    roi_percentage = Column(Float)
    
    # Sample size
    predictions_count = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    
    # Detailed metrics
    performance_by_market = Column(JSON)  # Store performance for each betting market
    performance_by_odds_range = Column(JSON)  # Performance in different odds ranges
    
    # Indexes
    __table_args__ = (
        Index('idx_model_performance', 'model_name', 'league_id'),
        Index('idx_model_evaluated', 'evaluated_at'),
    )

class APICache(Base):
    """Cache for API responses to reduce API calls"""
    __tablename__ = 'api_cache'
    
    id = Column(Integer, primary_key=True)
    endpoint = Column(String(500), nullable=False)
    params_hash = Column(String(64), nullable=False)  # MD5 hash of parameters
    response_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_cache_endpoint_hash', 'endpoint', 'params_hash'),
        Index('idx_cache_expires', 'expires_at'),
        UniqueConstraint('endpoint', 'params_hash', name='uq_endpoint_params')
    )

# Database configuration
def get_engine(pool_size=10, max_overflow=20):
    """Create database engine with connection pooling"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    # Use connection pooling for better performance
    engine = create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,  # Verify connections before using
        echo=False  # Set to True for SQL debugging
    )
    return engine

def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)

def get_session(engine):
    """Get a database session"""
    Session = sessionmaker(bind=engine)
    return Session()

# Initialize database
if __name__ == "__main__":
    engine = get_engine()
    create_tables(engine)
    print("Database tables created successfully!")