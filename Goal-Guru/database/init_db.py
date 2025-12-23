"""
Database initialization and migration script
"""

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import Base, get_engine, create_tables
from database.dal import get_dal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the database with tables and initial data"""
    try:
        logger.info("Initializing PostgreSQL database...")
        
        # Create engine
        engine = get_engine()
        
        # Create all tables
        logger.info("Creating database tables...")
        create_tables(engine)
        
        # Get DAL instance
        dal = get_dal()
        
        # Create default leagues if they don't exist
        default_leagues = [
            {
                'name': 'Premier League',
                'country': 'England',
                'season': '2024/2025',
                'avg_goals_per_match': 2.8,
                'league_type': 'normal',
                'api_id': 39
            },
            {
                'name': 'La Liga',
                'country': 'Spain',
                'season': '2024/2025',
                'avg_goals_per_match': 2.5,
                'league_type': 'low_scoring',
                'api_id': 140
            },
            {
                'name': 'Bundesliga',
                'country': 'Germany',
                'season': '2024/2025',
                'avg_goals_per_match': 3.1,
                'league_type': 'high_scoring',
                'api_id': 78
            },
            {
                'name': 'Serie A',
                'country': 'Italy',
                'season': '2024/2025',
                'avg_goals_per_match': 2.4,
                'league_type': 'low_scoring',
                'api_id': 135
            },
            {
                'name': 'Ligue 1',
                'country': 'France',
                'season': '2024/2025',
                'avg_goals_per_match': 2.5,
                'league_type': 'low_scoring',
                'api_id': 61
            },
            {
                'name': 'Süper Lig',
                'country': 'Turkey',
                'season': '2024/2025',
                'avg_goals_per_match': 2.7,
                'league_type': 'normal',
                'api_id': 203
            }
        ]
        
        for league_data in default_leagues:
            try:
                existing = dal.get_league_by_api_id(league_data['api_id'])
                if not existing:
                    dal.create_or_update_league(league_data)
                    logger.info(f"Created league: {league_data['name']}")
                else:
                    logger.info(f"League already exists: {league_data['name']}")
            except Exception as e:
                logger.error(f"Error creating league {league_data['name']}: {str(e)}")
        
        logger.info("Database initialization completed successfully!")
        
        # Clean expired cache entries
        deleted = dal.clean_expired_cache()
        logger.info(f"Cleaned {deleted} expired cache entries")
        
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

def verify_database():
    """Verify database connection and tables"""
    try:
        logger.info("Verifying database connection...")
        
        engine = get_engine()
        
        # Test connection
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
        
        # Check tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        expected_tables = ['teams', 'leagues', 'matches', 'predictions', 
                          'team_statistics', 'model_performance', 'api_cache']
        
        for table in expected_tables:
            if table in tables:
                logger.info(f"✓ Table '{table}' exists")
            else:
                logger.warning(f"✗ Table '{table}' missing")
        
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    if verify_database():
        logger.info("Database already configured")
    else:
        init_database()
        verify_database()