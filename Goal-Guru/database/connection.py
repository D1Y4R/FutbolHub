"""
Database connection manager with error handling
"""

import logging
from functools import wraps
from flask import jsonify
from database.dal import get_dal, DatabaseError

logger = logging.getLogger(__name__)

def db_error_handler(func):
    """Decorator to handle database errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DatabaseError as e:
            logger.error(f"Database error in {func.__name__}: {str(e)}")
            return jsonify({
                'error': 'Database operation failed',
                'message': str(e)
            }), 500
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            }), 500
    return wrapper

class DatabaseManager:
    """Database connection and operation manager"""
    
    def __init__(self):
        self.dal = get_dal()
    
    def save_match_data(self, match_data):
        """Save match data to database"""
        try:
            # Save home team
            home_team = self.dal.create_or_update_team({
                'name': match_data['teams']['home']['name'],
                'league_id': match_data.get('league_id'),
                'logo_url': match_data['teams']['home'].get('logo')
            })
            
            # Save away team
            away_team = self.dal.create_or_update_team({
                'name': match_data['teams']['away']['name'],
                'league_id': match_data.get('league_id'),
                'logo_url': match_data['teams']['away'].get('logo')
            })
            
            # Save match
            match = self.dal.create_or_update_match({
                'home_team_id': home_team.id,
                'away_team_id': away_team.id,
                'league_id': match_data.get('league_id'),
                'match_date': match_data['date'],
                'status': match_data.get('status', 'SCHEDULED'),
                'api_fixture_id': match_data.get('fixture_id'),
                'venue': match_data.get('venue', {}).get('name'),
                'home_score': match_data.get('goals', {}).get('home'),
                'away_score': match_data.get('goals', {}).get('away')
            })
            
            return match
            
        except Exception as e:
            logger.error(f"Error saving match data: {str(e)}")
            raise DatabaseError(f"Failed to save match data: {str(e)}")
    
    def save_prediction_data(self, match_id, prediction_data):
        """Save prediction to database"""
        try:
            # Prepare prediction data
            db_prediction = {
                'match_id': match_id,
                'predicted_winner': prediction_data.get('predicted_winner'),
                'home_win_probability': prediction_data.get('probabilities', {}).get('home'),
                'draw_probability': prediction_data.get('probabilities', {}).get('draw'),
                'away_win_probability': prediction_data.get('probabilities', {}).get('away'),
                'predicted_home_score': prediction_data.get('expected_goals', {}).get('home'),
                'predicted_away_score': prediction_data.get('expected_goals', {}).get('away'),
                'most_likely_score': prediction_data.get('most_likely_score'),
                'over_2_5_probability': prediction_data.get('betting_predictions', {}).get('over_2_5_goals', {}).get('probability'),
                'btts_yes_probability': prediction_data.get('betting_predictions', {}).get('both_teams_to_score', {}).get('probability'),
                'ht_ft_prediction': prediction_data.get('half_time_full_time', {}).get('prediction'),
                'ht_ft_probabilities': prediction_data.get('half_time_full_time', {}).get('probabilities'),
                'confidence_score': prediction_data.get('confidence', 0),
                'algorithm_weights': prediction_data.get('algorithm_weights'),
                'prediction_factors': prediction_data.get('factors')
            }
            
            # Remove None values
            db_prediction = {k: v for k, v in db_prediction.items() if v is not None}
            
            return self.dal.save_prediction(db_prediction)
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            raise DatabaseError(f"Failed to save prediction: {str(e)}")
    
    def get_cached_api_response(self, endpoint, params):
        """Get cached API response"""
        return self.dal.get_cached_response(endpoint, params)
    
    def save_api_response_cache(self, endpoint, params, response_data, cache_hours=24):
        """Save API response to cache"""
        return self.dal.save_cached_response(endpoint, params, response_data, cache_hours)
    
    def get_team_recent_matches(self, team_name, limit=10):
        """Get team's recent matches"""
        team = self.dal.get_team_by_name(team_name)
        if team:
            return self.dal.get_team_recent_matches(team.id, limit)
        return []
    
    def get_h2h_matches(self, team1_name, team2_name, limit=10):
        """Get head-to-head matches between two teams"""
        team1 = self.dal.get_team_by_name(team1_name)
        team2 = self.dal.get_team_by_name(team2_name)
        
        if team1 and team2:
            return self.dal.get_h2h_matches(team1.id, team2.id, limit)
        return []
    
    def update_team_statistics(self, team_name, season, stats_data):
        """Update team statistics"""
        team = self.dal.get_team_by_name(team_name)
        if team:
            return self.dal.update_team_statistics(team.id, season, stats_data)
        return None
    
    def save_model_performance(self, model_name, league_name, performance_data):
        """Save model performance metrics"""
        # Get league ID if provided
        league_id = None
        if league_name:
            league = self.dal.get_league_by_api_id(league_name)  # Assuming league_name is API ID
            if league:
                league_id = league.id
        
        performance_data.update({
            'model_name': model_name,
            'league_id': league_id
        })
        
        return self.dal.save_model_performance(performance_data)

# Singleton instance
_db_manager = None

def get_db_manager():
    """Get database manager singleton instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager