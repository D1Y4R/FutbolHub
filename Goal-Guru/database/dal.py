"""
Data Access Layer (DAL) for Football Prediction Hub
Provides clean interface for database operations with error handling
"""

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import and_, or_, func
import hashlib
import json

from database.models import (
    Team, League, Match, Prediction, TeamStatistics, 
    ModelPerformance, APICache, get_engine, get_session
)

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom database error"""
    pass

class DAL:
    """Data Access Layer for database operations"""
    
    def __init__(self, engine=None):
        """Initialize DAL with database engine"""
        self.engine = engine or get_engine()
        self._session = None
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope for database operations"""
        session = get_session(self.engine)
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            session.close()
    
    # Team operations
    def get_team_by_name(self, name: str, league_id: Optional[int] = None) -> Optional[Team]:
        """Get team by name and optionally league"""
        try:
            with self.session_scope() as session:
                query = session.query(Team).filter(Team.name == name)
                if league_id:
                    query = query.filter(Team.league_id == league_id)
                return query.first()
        except Exception as e:
            logger.error(f"Error getting team by name: {str(e)}")
            return None
    
    def create_or_update_team(self, team_data: Dict[str, Any]) -> Team:
        """Create or update team"""
        try:
            with self.session_scope() as session:
                team = session.query(Team).filter(
                    Team.name == team_data.get('name'),
                    Team.league_id == team_data.get('league_id')
                ).first()
                
                if team:
                    # Update existing team
                    for key, value in team_data.items():
                        setattr(team, key, value)
                else:
                    # Create new team
                    team = Team(**team_data)
                    session.add(team)
                
                session.commit()
                return team
        except IntegrityError as e:
            logger.error(f"Integrity error creating/updating team: {str(e)}")
            raise DatabaseError("Team already exists or constraint violation")
        except Exception as e:
            logger.error(f"Error creating/updating team: {str(e)}")
            raise DatabaseError(f"Failed to create/update team: {str(e)}")
    
    def get_team_statistics(self, team_id: int, season: str) -> Optional[TeamStatistics]:
        """Get team statistics for a specific season"""
        try:
            with self.session_scope() as session:
                return session.query(TeamStatistics).filter(
                    TeamStatistics.team_id == team_id,
                    TeamStatistics.season == season
                ).first()
        except Exception as e:
            logger.error(f"Error getting team statistics: {str(e)}")
            return None
    
    def update_team_statistics(self, team_id: int, season: str, stats_data: Dict[str, Any]) -> TeamStatistics:
        """Update team statistics"""
        try:
            with self.session_scope() as session:
                stats = session.query(TeamStatistics).filter(
                    TeamStatistics.team_id == team_id,
                    TeamStatistics.season == season
                ).first()
                
                if stats:
                    for key, value in stats_data.items():
                        setattr(stats, key, value)
                    setattr(stats, 'updated_at', datetime.utcnow())
                else:
                    stats = TeamStatistics(
                        team_id=team_id,
                        season=season,
                        updated_at=datetime.utcnow(),
                        **stats_data
                    )
                    session.add(stats)
                
                session.commit()
                return stats
        except Exception as e:
            logger.error(f"Error updating team statistics: {str(e)}")
            raise DatabaseError(f"Failed to update team statistics: {str(e)}")
    
    # League operations
    def get_league_by_api_id(self, api_id: int) -> Optional[League]:
        """Get league by external API ID"""
        try:
            with self.session_scope() as session:
                return session.query(League).filter(League.api_id == api_id).first()
        except Exception as e:
            logger.error(f"Error getting league by API ID: {str(e)}")
            return None
    
    def create_or_update_league(self, league_data: Dict[str, Any]) -> League:
        """Create or update league"""
        try:
            with self.session_scope() as session:
                league = session.query(League).filter(
                    League.api_id == league_data.get('api_id')
                ).first()
                
                if league:
                    for key, value in league_data.items():
                        setattr(league, key, value)
                else:
                    league = League(**league_data)
                    session.add(league)
                
                session.commit()
                return league
        except Exception as e:
            logger.error(f"Error creating/updating league: {str(e)}")
            raise DatabaseError(f"Failed to create/update league: {str(e)}")
    
    # Match operations
    def get_match_by_fixture_id(self, fixture_id: int) -> Optional[Match]:
        """Get match by external fixture ID"""
        try:
            with self.session_scope() as session:
                return session.query(Match).filter(
                    Match.api_fixture_id == fixture_id
                ).first()
        except Exception as e:
            logger.error(f"Error getting match by fixture ID: {str(e)}")
            return None
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> List[Match]:
        """Get upcoming matches within specified days"""
        try:
            with self.session_scope() as session:
                cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
                return session.query(Match).filter(
                    Match.match_date >= datetime.utcnow(),
                    Match.match_date <= cutoff_date,
                    Match.status == 'SCHEDULED'
                ).order_by(Match.match_date).all()
        except Exception as e:
            logger.error(f"Error getting upcoming matches: {str(e)}")
            return []
    
    def get_team_recent_matches(self, team_id: int, limit: int = 10) -> List[Match]:
        """Get team's recent matches"""
        try:
            with self.session_scope() as session:
                return session.query(Match).filter(
                    or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
                    Match.status == 'FINISHED'
                ).order_by(Match.match_date.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting team recent matches: {str(e)}")
            return []
    
    def get_h2h_matches(self, team1_id: int, team2_id: int, limit: int = 10) -> List[Match]:
        """Get head-to-head matches between two teams"""
        try:
            with self.session_scope() as session:
                return session.query(Match).filter(
                    or_(
                        and_(Match.home_team_id == team1_id, Match.away_team_id == team2_id),
                        and_(Match.home_team_id == team2_id, Match.away_team_id == team1_id)
                    ),
                    Match.status == 'FINISHED'
                ).order_by(Match.match_date.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting H2H matches: {str(e)}")
            return []
    
    def create_or_update_match(self, match_data: Dict[str, Any]) -> Match:
        """Create or update match"""
        try:
            with self.session_scope() as session:
                match = session.query(Match).filter(
                    Match.api_fixture_id == match_data.get('api_fixture_id')
                ).first()
                
                if match:
                    for key, value in match_data.items():
                        setattr(match, key, value)
                else:
                    match = Match(**match_data)
                    session.add(match)
                
                session.commit()
                return match
        except Exception as e:
            logger.error(f"Error creating/updating match: {str(e)}")
            raise DatabaseError(f"Failed to create/update match: {str(e)}")
    
    # Prediction operations
    def get_match_prediction(self, match_id: int) -> Optional[Prediction]:
        """Get most recent prediction for a match"""
        try:
            with self.session_scope() as session:
                return session.query(Prediction).filter(
                    Prediction.match_id == match_id
                ).order_by(Prediction.created_at.desc()).first()
        except Exception as e:
            logger.error(f"Error getting match prediction: {str(e)}")
            return None
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Save new prediction"""
        try:
            with self.session_scope() as session:
                prediction = Prediction(**prediction_data)
                session.add(prediction)
                session.commit()
                return prediction
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            raise DatabaseError(f"Failed to save prediction: {str(e)}")
    
    def get_recent_predictions(self, limit: int = 100) -> List[Prediction]:
        """Get recent predictions for model evaluation"""
        try:
            with self.session_scope() as session:
                return session.query(Prediction).join(Match).filter(
                    Match.status == 'FINISHED'
                ).order_by(Prediction.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting recent predictions: {str(e)}")
            return []
    
    # Model performance operations
    def save_model_performance(self, performance_data: Dict[str, Any]) -> ModelPerformance:
        """Save model performance metrics"""
        try:
            with self.session_scope() as session:
                performance = ModelPerformance(**performance_data)
                session.add(performance)
                session.commit()
                return performance
        except Exception as e:
            logger.error(f"Error saving model performance: {str(e)}")
            raise DatabaseError(f"Failed to save model performance: {str(e)}")
    
    def get_model_performance(self, model_name: str, league_id: Optional[int] = None) -> List[ModelPerformance]:
        """Get model performance history"""
        try:
            with self.session_scope() as session:
                query = session.query(ModelPerformance).filter(
                    ModelPerformance.model_name == model_name
                )
                if league_id:
                    query = query.filter(ModelPerformance.league_id == league_id)
                return query.order_by(ModelPerformance.evaluated_at.desc()).limit(10).all()
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return []
    
    # Cache operations
    def get_cached_response(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Get cached API response"""
        try:
            params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
            
            with self.session_scope() as session:
                cache_entry = session.query(APICache).filter(
                    APICache.endpoint == endpoint,
                    APICache.params_hash == params_hash,
                    APICache.expires_at > datetime.utcnow()
                ).first()
                
                if cache_entry:
                    # Return the JSON data - cast for type checker
                    return cache_entry.response_data  # type: ignore
                return None
        except Exception as e:
            logger.error(f"Error getting cached response: {str(e)}")
            return None
    
    def save_cached_response(self, endpoint: str, params: Dict[str, Any], 
                           response_data: Dict, cache_duration_hours: int = 24) -> None:
        """Save API response to cache"""
        try:
            params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
            expires_at = datetime.utcnow() + timedelta(hours=cache_duration_hours)
            
            with self.session_scope() as session:
                # Delete existing cache entry if exists
                session.query(APICache).filter(
                    APICache.endpoint == endpoint,
                    APICache.params_hash == params_hash
                ).delete()
                
                # Create new cache entry
                cache_entry = APICache(
                    endpoint=endpoint,
                    params_hash=params_hash,
                    response_data=response_data,
                    expires_at=expires_at
                )
                session.add(cache_entry)
                session.commit()
        except Exception as e:
            logger.error(f"Error saving cached response: {str(e)}")
    
    def clean_expired_cache(self) -> int:
        """Clean expired cache entries"""
        try:
            with self.session_scope() as session:
                deleted = session.query(APICache).filter(
                    APICache.expires_at < datetime.utcnow()
                ).delete()
                session.commit()
                return deleted
        except Exception as e:
            logger.error(f"Error cleaning expired cache: {str(e)}")
            return 0
    
    # Aggregate operations
    def get_league_statistics(self, league_id: int) -> Dict[str, Any]:
        """Get aggregate statistics for a league"""
        try:
            with self.session_scope() as session:
                matches = session.query(Match).filter(
                    Match.league_id == league_id,
                    Match.status == 'FINISHED'
                ).all()
                
                if not matches:
                    return {}
                
                # Access attributes as Python values from loaded objects
                total_goals = sum((getattr(m, 'home_score', 0) or 0) + (getattr(m, 'away_score', 0) or 0) for m in matches)
                total_matches = len(matches)
                
                home_wins = 0
                draws = 0
                away_wins = 0
                btts_count = 0
                over_2_5_count = 0
                
                for match in matches:
                    home_score = getattr(match, 'home_score', 0) or 0
                    away_score = getattr(match, 'away_score', 0) or 0
                    
                    if home_score > away_score:
                        home_wins += 1
                    elif home_score == away_score:
                        draws += 1
                    else:
                        away_wins += 1
                    
                    if home_score > 0 and away_score > 0:
                        btts_count += 1
                    
                    if home_score + away_score > 2.5:
                        over_2_5_count += 1
                
                return {
                    'total_matches': total_matches,
                    'avg_goals_per_match': total_goals / total_matches if total_matches > 0 else 0,
                    'home_wins': home_wins,
                    'draws': draws,
                    'away_wins': away_wins,
                    'btts_percentage': (btts_count / total_matches * 100) if total_matches > 0 else 0,
                    'over_2_5_percentage': (over_2_5_count / total_matches * 100) if total_matches > 0 else 0
                }
        except Exception as e:
            logger.error(f"Error getting league statistics: {str(e)}")
            return {}

# Singleton instance
_dal_instance = None

def get_dal() -> DAL:
    """Get DAL singleton instance"""
    global _dal_instance
    if _dal_instance is None:
        _dal_instance = DAL()
    return _dal_instance