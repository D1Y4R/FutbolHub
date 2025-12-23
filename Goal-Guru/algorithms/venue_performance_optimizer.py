"""
Venue Performance Optimizer Module for Football Prediction System
Analyzes home/away performance, venue-specific effects, and optimizes predictions.

This module provides:
1. Home Advantage Analysis - Ev sahibi avantaj analizi
2. Venue-Specific Performance Modeling - Venue-specific performans modelleme
3. Dynamic Venue Adjustment System - Dinamik venue ayarlama sistemi
4. Cross-League Venue Intelligence - Çapraz lig venue zekası

Author: Football Prediction System
Date: September 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from scipy import stats
import json
import math
import requests
import geopy.distance
from geopy.geocoders import Nominatim

# Import existing modules for integration
from .league_normalization_engine import LeagueNormalizationEngine
from .fixture_congestion_analyzer import FixtureCongestionAnalyzer

logger = logging.getLogger(__name__)

class VenuePerformanceOptimizer:
    """
    Gelişmiş venue performans optimizasyon sistemi
    Ev ve deplasman performansını optimize eden ve venue-specific etkilerini analiz eden sistem
    
    Ana bileşenler:
    - Home Advantage Analysis
    - Venue-Specific Performance Modeling  
    - Dynamic Venue Adjustment System
    - Cross-League Venue Intelligence
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Venue Performance Optimizer
        
        Args:
            config: Configuration dictionary for customization
        """
        self.config = config or self._get_default_config()
        
        # Initialize existing analyzers for integration
        self.league_normalizer = LeagueNormalizationEngine()
        self.congestion_analyzer = FixtureCongestionAnalyzer()
        
        # Venue performance databases
        self.venue_profiles = {}
        self.stadium_database = {}
        self.weather_cache = {}
        self.travel_cache = {}
        self.home_advantage_coefficients = {}
        
        # Cross-league intelligence
        self.venue_difficulty_rankings = {}
        self.international_venue_data = {}
        self.cultural_adaptation_factors = {}
        
        # Real-time adjustment factors
        self.dynamic_adjustments = {}
        self.seasonal_patterns = {}
        self.crowd_impact_models = {}
        
        # Geographic and weather services
        self.geolocator = Nominatim(user_agent="venue_optimizer")
        
        # Initialize stadium database with major European stadiums
        self._initialize_stadium_database()
        
        logger.info("Venue Performance Optimizer initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for the venue optimizer"""
        return {
            'home_advantage': {
                'base_coefficient': 1.05,
                'min_coefficient': 0.95,
                'max_coefficient': 1.15,
                'min_matches_for_analysis': 10
            },
            'venue_factors': {
                'altitude_threshold': 1000,  # meters
                'crowd_capacity_levels': {
                    'small': 20000,
                    'medium': 40000,
                    'large': 60000,
                    'mega': 80000
                },
                'surface_types': ['grass', 'artificial_turf', 'hybrid'],
                'weather_impact_threshold': 0.15
            },
            'travel_factors': {
                'local_threshold': 100,      # km
                'domestic_threshold': 800,   # km
                'european_threshold': 3000,  # km
                'intercontinental_threshold': 8000,  # km
                'time_zone_impact': True,
                'cultural_adaptation': True
            },
            'dynamic_adjustments': {
                'weather_api_enabled': True,
                'real_time_conditions': True,
                'crowd_presence_tracking': True,
                'seasonal_calibration': True
            },
            'performance_thresholds': {
                'excellent': 80,
                'good': 65,
                'average': 50,
                'poor': 35,
                'terrible': 20
            }
        }
    
    def _initialize_stadium_database(self):
        """Initialize comprehensive stadium database with major venues"""
        self.stadium_database = {
            # Premier League
            'Old Trafford': {
                'id': 'old_trafford',
                'name': 'Old Trafford',
                'city': 'Manchester',
                'country': 'England',
                'league': 'Premier League',
                'capacity': 74140,
                'coordinates': (53.4631, -2.2914),
                'altitude': 38,
                'surface': 'grass',
                'roof_type': 'partial',
                'atmosphere_rating': 9.2,
                'home_advantage_factor': 1.08
            },
            'Anfield': {
                'id': 'anfield',
                'name': 'Anfield',
                'city': 'Liverpool',
                'country': 'England',
                'league': 'Premier League',
                'capacity': 53394,
                'coordinates': (53.4308, -2.9609),
                'altitude': 35,
                'surface': 'grass',
                'roof_type': 'partial',
                'atmosphere_rating': 9.5,
                'home_advantage_factor': 1.30
            },
            'Emirates Stadium': {
                'id': 'emirates',
                'name': 'Emirates Stadium',
                'city': 'London',
                'country': 'England',
                'league': 'Premier League',
                'capacity': 60704,
                'coordinates': (51.5549, -0.1084),
                'altitude': 41,
                'surface': 'grass',
                'roof_type': 'partial',
                'atmosphere_rating': 7.8,
                'home_advantage_factor': 1.12
            },
            
            # La Liga
            'Camp Nou': {
                'id': 'camp_nou',
                'name': 'Camp Nou',
                'city': 'Barcelona',
                'country': 'Spain',
                'league': 'La Liga',
                'capacity': 99354,
                'coordinates': (41.3809, 2.1228),
                'altitude': 12,
                'surface': 'grass',
                'roof_type': 'open',
                'atmosphere_rating': 9.0,
                'home_advantage_factor': 1.08
            },
            'Santiago Bernabéu': {
                'id': 'bernabeu',
                'name': 'Santiago Bernabéu',
                'city': 'Madrid',
                'country': 'Spain',
                'league': 'La Liga',
                'capacity': 81044,
                'coordinates': (40.4530, -3.6883),
                'altitude': 650,
                'surface': 'grass',
                'roof_type': 'retractable',
                'atmosphere_rating': 8.8,
                'home_advantage_factor': 1.07
            },
            
            # Serie A
            'San Siro': {
                'id': 'san_siro',
                'name': 'San Siro',
                'city': 'Milan',
                'country': 'Italy',
                'league': 'Serie A',
                'capacity': 75923,
                'coordinates': (45.4781, 9.1240),
                'altitude': 122,
                'surface': 'grass',
                'roof_type': 'partial',
                'atmosphere_rating': 8.9,
                'home_advantage_factor': 1.06
            },
            'Stadio Olimpico': {
                'id': 'olimpico_roma',
                'name': 'Stadio Olimpico',
                'city': 'Rome',
                'country': 'Italy',
                'league': 'Serie A',
                'capacity': 70634,
                'coordinates': (41.9342, 12.4549),
                'altitude': 20,
                'surface': 'grass',
                'roof_type': 'partial',
                'atmosphere_rating': 8.5,
                'home_advantage_factor': 1.06
            },
            
            # Bundesliga
            'Allianz Arena': {
                'id': 'allianz_arena',
                'name': 'Allianz Arena',
                'city': 'Munich',
                'country': 'Germany',
                'league': 'Bundesliga',
                'capacity': 75024,
                'coordinates': (48.2188, 11.6242),
                'altitude': 515,
                'surface': 'grass',
                'roof_type': 'partial',
                'atmosphere_rating': 8.7,
                'home_advantage_factor': 1.08
            },
            'Signal Iduna Park': {
                'id': 'signal_iduna',
                'name': 'Signal Iduna Park',
                'city': 'Dortmund',
                'country': 'Germany',
                'league': 'Bundesliga',
                'capacity': 81365,
                'coordinates': (51.4925, 7.4517),
                'altitude': 85,
                'surface': 'grass',
                'roof_type': 'partial',
                'atmosphere_rating': 9.3,
                'home_advantage_factor': 1.09
            },
            
            # Süper Lig
            'Türk Telekom Stadium': {
                'id': 'turk_telekom',
                'name': 'Türk Telekom Stadium',
                'city': 'Istanbul',
                'country': 'Turkey',
                'league': 'Süper Lig',
                'capacity': 52652,
                'coordinates': (41.1039, 28.9994),
                'altitude': 150,
                'surface': 'grass',
                'roof_type': 'closed',
                'atmosphere_rating': 8.8,
                'home_advantage_factor': 1.07
            },
            'Fenerbahçe Şükrü Saracoğlu Stadium': {
                'id': 'sukru_saracoglu',
                'name': 'Fenerbahçe Şükrü Saracoğlu Stadium',
                'city': 'Istanbul',
                'country': 'Turkey',
                'league': 'Süper Lig',
                'capacity': 50530,
                'coordinates': (40.9897, 29.0364),
                'altitude': 30,
                'surface': 'grass',
                'roof_type': 'partial',
                'atmosphere_rating': 8.9,
                'home_advantage_factor': 1.08
            }
        }
    
    def analyze_comprehensive_venue_performance(self, home_team_id: int, away_team_id: int,
                                              venue_info: Dict, match_context: Dict,
                                              historical_matches: List[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive venue performance analysis for a specific match
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier  
            venue_info: Venue information (stadium, location, etc.)
            match_context: Match context (date, time, conditions)
            historical_matches: Historical match data for analysis
            
        Returns:
            Comprehensive venue performance analysis
        """
        try:
            logger.info(f"Analyzing venue performance for match: {home_team_id} vs {away_team_id}")
            
            # 1. Home Advantage Analysis
            home_advantage = self._analyze_home_advantage(
                home_team_id, venue_info, historical_matches, match_context
            )
            
            # 2. Venue-Specific Performance Modeling
            venue_modeling = self._model_venue_specific_performance(
                venue_info, home_team_id, away_team_id, historical_matches
            )
            
            # 3. Travel Impact Assessment
            travel_impact = self._assess_travel_impact(
                home_team_id, away_team_id, venue_info, match_context
            )
            
            # 4. Weather and Climate Analysis
            weather_analysis = self._analyze_weather_climate_impact(
                venue_info, match_context
            )
            
            # 5. Dynamic Venue Adjustments
            dynamic_adjustments = self._calculate_dynamic_venue_adjustments(
                venue_info, match_context, home_advantage, weather_analysis
            )
            
            # 6. Cross-League Intelligence
            cross_league_intelligence = self._analyze_cross_league_venue_intelligence(
                venue_info, home_team_id, away_team_id, match_context
            )
            
            # 7. Calculate venue difficulty score
            venue_difficulty = self._calculate_venue_difficulty_score(
                venue_info, home_advantage, travel_impact, weather_analysis
            )
            
            # 8. Generate performance predictions
            performance_predictions = self._generate_venue_adjusted_predictions(
                home_advantage, venue_modeling, travel_impact, 
                weather_analysis, dynamic_adjustments
            )
            
            # 9. Optimal performance conditions
            optimal_conditions = self._identify_optimal_performance_conditions(
                venue_info, weather_analysis, dynamic_adjustments
            )
            
            return {
                'match_info': {
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'venue': venue_info.get('name', 'Unknown'),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'home_advantage_analysis': home_advantage,
                'venue_specific_modeling': venue_modeling,
                'travel_impact_assessment': travel_impact,
                'weather_climate_analysis': weather_analysis,
                'dynamic_adjustments': dynamic_adjustments,
                'cross_league_intelligence': cross_league_intelligence,
                'venue_difficulty_score': venue_difficulty,
                'performance_predictions': performance_predictions,
                'optimal_conditions': optimal_conditions,
                'recommendations': self._generate_venue_recommendations(
                    home_advantage, venue_difficulty, travel_impact, weather_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive venue analysis: {str(e)}")
            return self._get_default_venue_analysis()
    
    def _analyze_home_advantage(self, home_team_id: int, venue_info: Dict,
                               historical_matches: List[Dict] = None,
                               match_context: Dict = None) -> Dict[str, Any]:
        """
        Gelişmiş ev sahibi avantaj analizi
        League-specific home advantage coefficients ve stadium effects
        """
        try:
            # Base home advantage from config
            base_coefficient = self.config['home_advantage']['base_coefficient']
            
            # Stadium-specific factors
            stadium_factor = self._calculate_stadium_advantage_factor(venue_info)
            
            # Historical performance analysis
            historical_factor = self._analyze_historical_home_performance(
                home_team_id, venue_info, historical_matches
            )
            
            # League-specific calibration
            league_factor = self._get_league_home_advantage_factor(
                venue_info.get('league_id'), venue_info.get('league_name')
            )
            
            # Atmosphere and crowd impact
            atmosphere_factor = self._calculate_atmosphere_impact(venue_info)
            
            # Season-specific adjustments
            seasonal_factor = self._calculate_seasonal_home_advantage(
                match_context, historical_matches
            )
            
            # Calculate final home advantage coefficient
            final_coefficient = (
                base_coefficient * 
                stadium_factor * 
                league_factor * 
                atmosphere_factor * 
                seasonal_factor * 
                historical_factor
            )
            
            # CROSS-LEAGUE ADJUSTMENT: Reduce home advantage when away team is from stronger league
            if match_context:
                league_strength_context = match_context.get('league_strength_context')
                cross_league = match_context.get('cross_league', False)
                
                if cross_league and league_strength_context:
                    away_info = league_strength_context.get('away', {})
                    home_info = league_strength_context.get('home', {})
                    away_strength = away_info.get('strength_score', 50)
                    home_strength = home_info.get('strength_score', 50)
                    
                    if away_strength > home_strength:
                        # Away team from stronger league - reduce home advantage significantly
                        strength_diff = away_strength - home_strength
                        
                        if strength_diff > 40:  # Elite away team (e.g., Liverpool)
                            reduction_factor = 0.3  # 70% reduction - ultra aggressive
                        elif strength_diff > 25:
                            reduction_factor = 0.5  # 50% reduction
                        elif strength_diff > 15:
                            reduction_factor = 0.65
                        else:
                            reduction_factor = 0.75
                        
                        logger.info(f"🌍 CROSS-LEAGUE HOME ADVANTAGE REDUCTION: {reduction_factor:.2f}x (Away team from stronger league)")
                        logger.info(f"   {home_info.get('league_name')} (strength: {home_strength}) vs {away_info.get('league_name')} (strength: {away_strength})")
                        logger.info(f"   Original home advantage: {final_coefficient:.3f} → Adjusted: {final_coefficient * reduction_factor:.3f}")
                        
                        final_coefficient *= reduction_factor
            
            # Ensure coefficient is within reasonable bounds
            final_coefficient = max(
                self.config['home_advantage']['min_coefficient'],
                min(self.config['home_advantage']['max_coefficient'], final_coefficient)
            )
            
            return {
                'base_coefficient': base_coefficient,
                'stadium_factor': stadium_factor,
                'historical_factor': historical_factor,
                'league_factor': league_factor,
                'atmosphere_factor': atmosphere_factor,
                'seasonal_factor': seasonal_factor,
                'final_coefficient': final_coefficient,
                'advantage_strength': self._classify_home_advantage_strength(final_coefficient),
                'confidence_level': self._calculate_confidence_level(historical_matches),
                'contributing_factors': self._identify_contributing_factors(
                    stadium_factor, atmosphere_factor, historical_factor
                )
            }
            
        except Exception as e:
            logger.error(f"Error in home advantage analysis: {str(e)}")
            return self._get_default_home_advantage()
    
    def _calculate_stadium_advantage_factor(self, venue_info: Dict) -> float:
        """Calculate stadium-specific advantage factor"""
        try:
            venue_name = venue_info.get('name', '').lower()
            venue_id = venue_info.get('id', venue_name)
            
            # Check if stadium is in our database
            for stadium_id, stadium_data in self.stadium_database.items():
                if (stadium_data['name'].lower() in venue_name or 
                    venue_name in stadium_data['name'].lower() or
                    venue_id == stadium_data['id']):
                    return stadium_data.get('home_advantage_factor', 1.1)
            
            # Default calculation based on available info
            capacity = venue_info.get('capacity', 30000)
            
            # Capacity-based factor
            if capacity >= 80000:
                capacity_factor = 1.15
            elif capacity >= 60000:
                capacity_factor = 1.12
            elif capacity >= 40000:
                capacity_factor = 1.08
            elif capacity >= 20000:
                capacity_factor = 1.05
            else:
                capacity_factor = 1.02
            
            return capacity_factor
            
        except Exception as e:
            logger.error(f"Error calculating stadium factor: {str(e)}")
            return 1.1
    
    def _analyze_historical_home_performance(self, home_team_id: int, venue_info: Dict,
                                           historical_matches: List[Dict] = None) -> float:
        """Analyze historical home performance for the team at this venue"""
        if not historical_matches:
            return 1.0
        
        try:
            home_matches = [
                match for match in historical_matches
                if match.get('home_team_id') == home_team_id
            ]
            
            if len(home_matches) < self.config['home_advantage']['min_matches_for_analysis']:
                return 1.0
            
            # Calculate home performance metrics
            home_wins = sum(1 for match in home_matches 
                          if (match.get('home_score', 0) or 0) > (match.get('away_score', 0) or 0))
            home_draws = sum(1 for match in home_matches 
                           if (match.get('home_score', 0) or 0) == (match.get('away_score', 0) or 0))
            
            total_matches = len(home_matches)
            home_points = (home_wins * 3 + home_draws) / total_matches
            
            # Expected home points (league average ~1.7)
            expected_home_points = 1.7
            
            # Calculate factor based on performance vs expectation
            if home_points > expected_home_points:
                factor = 1.0 + min(0.2, (home_points - expected_home_points) / 3.0)
            else:
                factor = 1.0 - min(0.15, (expected_home_points - home_points) / 3.0)
            
            return max(0.85, min(1.25, factor))
            
        except Exception as e:
            logger.error(f"Error in historical home performance analysis: {str(e)}")
            return 1.0
    
    def _get_league_home_advantage_factor(self, league_id: Optional[str] = None,
                                         league_name: Optional[str] = None) -> float:
        """Get league-specific home advantage factor"""
        try:
            # Use existing league normalization data if available
            if hasattr(self.league_normalizer, 'league_profiles'):
                for profile_id, profile in self.league_normalizer.league_profiles.items():
                    if (str(profile_id) == str(league_id) or 
                        profile.get('league_name', '').lower() == str(league_name).lower()):
                        home_adv = profile.get('characteristics', {}).get('home_advantage', {})
                        return home_adv.get('coefficient', 1.1)
            
            # League-specific factors based on known characteristics
            league_factors = {
                # Turkish leagues - strong home advantage
                'süper lig': 1.18,
                'tff 1. lig': 1.15,
                
                # Top European leagues
                'premier league': 1.12,
                'la liga': 1.14,
                'serie a': 1.13,
                'bundesliga': 1.11,
                'ligue 1': 1.10,
                
                # Other European leagues
                'eredivisie': 1.13,
                'primeira liga': 1.16,
                'pro league': 1.12,
                
                # South American leagues - very strong home advantage
                'brasileirão': 1.20,
                'liga profesional': 1.19,
                
                # Default
                'default': 1.11
            }
            
            if league_name:
                league_key = league_name.lower()
                for key, factor in league_factors.items():
                    if key in league_key:
                        return factor
            
            return league_factors['default']
            
        except Exception as e:
            logger.error(f"Error getting league home advantage factor: {str(e)}")
            return 1.11
    
    def _calculate_atmosphere_impact(self, venue_info: Dict) -> float:
        """Calculate atmosphere and crowd impact factor"""
        try:
            capacity = venue_info.get('capacity', 30000)
            atmosphere_rating = venue_info.get('atmosphere_rating', 7.0)
            
            # Capacity-based atmosphere
            if capacity >= 80000:
                capacity_impact = 1.08
            elif capacity >= 60000:
                capacity_impact = 1.06
            elif capacity >= 40000:
                capacity_impact = 1.04
            elif capacity >= 20000:
                capacity_impact = 1.02
            else:
                capacity_impact = 1.0
            
            # Atmosphere rating impact (scale 1-10) - AZALTILDI
            if atmosphere_rating >= 9.0:
                atmosphere_impact = 1.03  # Azaltıldı (1.06 -> 1.03)
            elif atmosphere_rating >= 8.0:
                atmosphere_impact = 1.02  # Azaltıldı (1.04 -> 1.02)
            elif atmosphere_rating >= 7.0:
                atmosphere_impact = 1.01  # Azaltıldı (1.02 -> 1.01)
            else:
                atmosphere_impact = 1.0
            
            # Stadium design factors
            roof_type = venue_info.get('roof_type', 'open')
            if roof_type == 'closed':
                roof_impact = 1.03  # Enclosed stadiums amplify noise
            elif roof_type == 'partial':
                roof_impact = 1.02
            else:
                roof_impact = 1.0
            
            return capacity_impact * atmosphere_impact * roof_impact
            
        except Exception as e:
            logger.error(f"Error calculating atmosphere impact: {str(e)}")
            return 1.02
    
    def _calculate_seasonal_home_advantage(self, match_context: Dict = None,
                                         historical_matches: List[Dict] = None) -> float:
        """Calculate seasonal adjustments to home advantage"""
        try:
            if not match_context:
                return 1.0
            
            match_date = match_context.get('date')
            if isinstance(match_date, str):
                match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
            elif not isinstance(match_date, datetime):
                return 1.0
            
            # Month-based seasonal effects
            month = match_date.month
            
            # Winter months typically have stronger home advantage
            if month in [12, 1, 2]:  # Winter
                seasonal_factor = 1.04
            elif month in [3, 4, 5]:  # Spring
                seasonal_factor = 1.02
            elif month in [6, 7, 8]:  # Summer
                seasonal_factor = 0.98
            else:  # Autumn
                seasonal_factor = 1.01
            
            # Holiday periods might affect atmosphere
            day_of_year = match_date.timetuple().tm_yday
            if 355 <= day_of_year <= 365 or 1 <= day_of_year <= 15:  # Christmas/New Year
                seasonal_factor *= 1.02
            
            return seasonal_factor
            
        except Exception as e:
            logger.error(f"Error calculating seasonal home advantage: {str(e)}")
            return 1.0
    
    def _classify_home_advantage_strength(self, coefficient: float) -> str:
        """Classify home advantage strength based on coefficient"""
        if coefficient >= 1.25:
            return "Çok Güçlü"  # Very Strong
        elif coefficient >= 1.15:
            return "Güçlü"  # Strong
        elif coefficient >= 1.05:
            return "Orta"  # Medium
        elif coefficient >= 0.98:
            return "Zayıf"  # Weak
        else:
            return "Çok Zayıf"  # Very Weak
    
    def _calculate_confidence_level(self, historical_matches: List[Dict] = None) -> float:
        """Calculate confidence level of home advantage analysis"""
        if not historical_matches:
            return 0.3
        
        match_count = len(historical_matches)
        if match_count >= 50:
            return 0.95
        elif match_count >= 30:
            return 0.85
        elif match_count >= 20:
            return 0.75
        elif match_count >= 10:
            return 0.65
        else:
            return 0.45
    
    def _identify_contributing_factors(self, stadium_factor: float, 
                                     atmosphere_factor: float, 
                                     historical_factor: float) -> List[str]:
        """Identify main contributing factors to home advantage"""
        factors = []
        
        if stadium_factor >= 1.1:
            factors.append("Stadium Capacity")
        if atmosphere_factor >= 1.05:
            factors.append("Atmosphere Quality")
        if historical_factor >= 1.1:
            factors.append("Historical Performance")
        if stadium_factor >= 1.15:
            factors.append("Iconic Venue")
        
        return factors or ["Standard Home Advantage"]
    
    def _get_default_home_advantage(self) -> Dict[str, Any]:
        """Get default home advantage analysis"""
        return {
            'base_coefficient': 1.1,
            'stadium_factor': 1.05,
            'historical_factor': 1.0,
            'league_factor': 1.1,
            'atmosphere_factor': 1.02,
            'seasonal_factor': 1.0,
            'final_coefficient': 1.1,
            'advantage_strength': "Orta",
            'confidence_level': 0.5,
            'contributing_factors': ["Standard Home Advantage"]
        }
    
    def _model_venue_specific_performance(self, venue_info: Dict, home_team_id: int,
                                         away_team_id: int, historical_matches: List[Dict] = None) -> Dict[str, Any]:
        """
        Venue-specific performans modelleme
        Individual stadium performance profiles ve surface/altitude effects
        """
        try:
            # Stadium profile analysis
            stadium_profile = self._analyze_stadium_profile(venue_info)
            
            # Surface type effects
            surface_effects = self._analyze_surface_effects(venue_info, historical_matches)
            
            # Altitude and geographic effects
            geographic_effects = self._analyze_geographic_effects(venue_info)
            
            # Stadium size and crowd noise impacts
            crowd_noise_impact = self._analyze_crowd_noise_impact(venue_info)
            
            # Historical venue head-to-head records
            h2h_venue_records = self._analyze_venue_h2h_records(
                home_team_id, away_team_id, venue_info, historical_matches
            )
            
            # Team-specific venue performance
            team_venue_performance = self._analyze_team_venue_performance(
                home_team_id, away_team_id, venue_info, historical_matches
            )
            
            return {
                'stadium_profile': stadium_profile,
                'surface_effects': surface_effects,
                'geographic_effects': geographic_effects,
                'crowd_noise_impact': crowd_noise_impact,
                'h2h_venue_records': h2h_venue_records,
                'team_venue_performance': team_venue_performance,
                'overall_venue_factor': self._calculate_overall_venue_factor(
                    stadium_profile, surface_effects, geographic_effects, crowd_noise_impact
                )
            }
            
        except Exception as e:
            logger.error(f"Error in venue-specific performance modeling: {str(e)}")
            return self._get_default_venue_modeling()
    
    def _analyze_stadium_profile(self, venue_info: Dict) -> Dict[str, Any]:
        """Analyze individual stadium profile and characteristics"""
        venue_name = venue_info.get('name', '').lower()
        venue_id = venue_info.get('id', venue_name)
        
        # Check if stadium is in our database
        stadium_data = None
        for stadium_id, data in self.stadium_database.items():
            if (data['name'].lower() in venue_name or 
                venue_name in data['name'].lower() or
                venue_id == data['id']):
                stadium_data = data
                break
        
        if stadium_data:
            return {
                'stadium_type': self._classify_stadium_type(stadium_data),
                'capacity_category': self._classify_capacity(stadium_data['capacity']),
                'atmosphere_rating': stadium_data.get('atmosphere_rating', 7.0),
                'architectural_advantages': self._identify_architectural_advantages(stadium_data),
                'historical_significance': self._assess_historical_significance(stadium_data),
                'modern_facilities': self._assess_modern_facilities(stadium_data),
                'intimidation_factor': self._calculate_intimidation_factor(stadium_data)
            }
        else:
            # Generic profile based on available info
            capacity = venue_info.get('capacity', 30000)
            return {
                'stadium_type': 'Standard',
                'capacity_category': self._classify_capacity(capacity),
                'atmosphere_rating': 7.0,
                'architectural_advantages': [],
                'historical_significance': 'Medium',
                'modern_facilities': 'Standard',
                'intimidation_factor': self._calculate_generic_intimidation_factor(capacity)
            }
    
    def _analyze_surface_effects(self, venue_info: Dict, historical_matches: List[Dict] = None) -> Dict[str, Any]:
        """Analyze surface type effects on team performance"""
        surface_type = venue_info.get('surface', 'grass').lower()
        
        # Surface characteristics
        surface_characteristics = {
            'grass': {
                'speed': 'medium',
                'bounce': 'natural',
                'technical_advantage': 1.0,
                'physical_advantage': 1.0,
                'weather_sensitivity': 'high'
            },
            'artificial_turf': {
                'speed': 'fast',
                'bounce': 'consistent',
                'technical_advantage': 0.95,
                'physical_advantage': 1.05,
                'weather_sensitivity': 'low'
            },
            'hybrid': {
                'speed': 'medium-fast',
                'bounce': 'semi-natural',
                'technical_advantage': 1.02,
                'physical_advantage': 1.01,
                'weather_sensitivity': 'medium'
            }
        }
        
        surface_data = surface_characteristics.get(surface_type, surface_characteristics['grass'])
        
        # Performance impact based on team style (if available)
        team_style_impact = self._calculate_surface_team_style_impact(surface_type, historical_matches)
        
        return {
            'surface_type': surface_type,
            'characteristics': surface_data,
            'team_style_impact': team_style_impact,
            'maintenance_quality': venue_info.get('maintenance_quality', 'standard'),
            'seasonal_condition_variation': self._assess_seasonal_surface_variation(surface_type)
        }
    
    def _analyze_geographic_effects(self, venue_info: Dict) -> Dict[str, Any]:
        """Analyze altitude and geographic effects"""
        coordinates = venue_info.get('coordinates', (0, 0))
        altitude = venue_info.get('altitude', 0)
        
        # Altitude effects
        altitude_effects = self._calculate_altitude_effects(altitude)
        
        # Climate zone effects
        climate_effects = self._assess_climate_zone_effects(coordinates)
        
        # Geographic isolation effects
        isolation_effects = self._assess_geographic_isolation(coordinates)
        
        return {
            'altitude': altitude,
            'altitude_effects': altitude_effects,
            'coordinates': coordinates,
            'climate_effects': climate_effects,
            'isolation_effects': isolation_effects,
            'geographic_advantage': self._calculate_geographic_advantage(
                altitude_effects, climate_effects, isolation_effects
            )
        }
    
    def _analyze_crowd_noise_impact(self, venue_info: Dict) -> Dict[str, Any]:
        """Analyze stadium size and crowd noise impacts"""
        capacity = venue_info.get('capacity', 30000)
        roof_type = venue_info.get('roof_type', 'open')
        atmosphere_rating = venue_info.get('atmosphere_rating', 7.0)
        
        # Noise amplification based on stadium design
        noise_amplification = self._calculate_noise_amplification(capacity, roof_type)
        
        # Crowd density impact
        crowd_density_impact = self._calculate_crowd_density_impact(capacity)
        
        # Psychological pressure on away team
        psychological_pressure = self._calculate_psychological_pressure(
            atmosphere_rating, noise_amplification, capacity
        )
        
        return {
            'capacity': capacity,
            'noise_amplification_factor': noise_amplification,
            'crowd_density_impact': crowd_density_impact,
            'psychological_pressure': psychological_pressure,
            'referee_influence': self._assess_crowd_referee_influence(
                capacity, atmosphere_rating
            ),
            'player_concentration_impact': self._assess_concentration_impact(
                noise_amplification, psychological_pressure
            )
        }
    
    def _assess_travel_impact(self, home_team_id: int, away_team_id: int,
                             venue_info: Dict, match_context: Dict) -> Dict[str, Any]:
        """
        Travel impact assessment with distance, time zones, and cultural factors
        """
        try:
            # Get team locations (simplified - would need team database)
            home_location = self._get_team_location(home_team_id, venue_info)
            away_location = self._get_team_location(away_team_id)
            
            # Calculate travel distance
            travel_distance = self._calculate_travel_distance(home_location, away_location)
            
            # Time zone impact
            timezone_impact = self._calculate_timezone_impact(home_location, away_location)
            
            # Travel fatigue assessment
            travel_fatigue = self._assess_travel_fatigue(
                travel_distance, timezone_impact, match_context
            )
            
            # Cultural adaptation factors
            cultural_adaptation = self._assess_cultural_adaptation(
                home_location, away_location, away_team_id
            )
            
            # Integration with existing congestion analyzer
            congestion_impact = self._integrate_congestion_analysis(
                away_team_id, match_context, travel_distance
            )
            
            return {
                'travel_distance_km': travel_distance,
                'travel_category': self._classify_travel_distance(travel_distance),
                'timezone_difference': timezone_impact,
                'travel_fatigue_score': travel_fatigue,
                'cultural_adaptation': cultural_adaptation,
                'congestion_impact': congestion_impact,
                'overall_travel_penalty': self._calculate_overall_travel_penalty(
                    travel_fatigue, timezone_impact, cultural_adaptation, congestion_impact
                ),
                'recovery_time_needed': self._estimate_recovery_time(travel_distance, timezone_impact)
            }
            
        except Exception as e:
            logger.error(f"Error in travel impact assessment: {str(e)}")
            return self._get_default_travel_impact()
    
    def _analyze_weather_climate_impact(self, venue_info: Dict, match_context: Dict) -> Dict[str, Any]:
        """
        Weather and climate impact analysis with API integration
        """
        try:
            coordinates = venue_info.get('coordinates', (0, 0))
            match_date = match_context.get('date')
            
            # Current weather conditions (mock - would integrate with weather API)
            current_weather = self._get_weather_conditions(coordinates, match_date)
            
            # Climate analysis
            climate_profile = self._analyze_climate_profile(coordinates)
            
            # Weather impact on playing style
            playing_style_impact = self._assess_weather_playing_style_impact(current_weather)
            
            # Seasonal weather patterns
            seasonal_patterns = self._analyze_seasonal_weather_patterns(coordinates, match_date)
            
            # Extreme weather adjustments
            extreme_weather_adjustments = self._calculate_extreme_weather_adjustments(current_weather)
            
            return {
                'current_conditions': current_weather,
                'climate_profile': climate_profile,
                'playing_style_impact': playing_style_impact,
                'seasonal_patterns': seasonal_patterns,
                'extreme_weather_adjustments': extreme_weather_adjustments,
                'weather_advantage': self._determine_weather_advantage(
                    current_weather, playing_style_impact
                ),
                'surface_condition_impact': self._assess_weather_surface_impact(
                    current_weather, venue_info.get('surface', 'grass')
                )
            }
            
        except Exception as e:
            logger.error(f"Error in weather climate impact analysis: {str(e)}")
            return self._get_default_weather_analysis()
    
    def _calculate_dynamic_venue_adjustments(self, venue_info: Dict, match_context: Dict,
                                           home_advantage: Dict, weather_analysis: Dict) -> Dict[str, Any]:
        """
        Dynamic venue adjustment system with real-time conditions
        """
        try:
            # Real-time venue condition assessment
            real_time_conditions = self._assess_real_time_venue_conditions(
                venue_info, match_context, weather_analysis
            )
            
            # Season-specific venue advantages
            seasonal_advantages = self._calculate_seasonal_venue_advantages(
                venue_info, match_context
            )
            
            # Crowd presence optimization
            crowd_presence = self._optimize_crowd_presence(venue_info, match_context)
            
            # Time-of-day venue effects
            time_effects = self._calculate_time_of_day_effects(match_context)
            
            # Weather condition adaptations
            weather_adaptations = self._calculate_weather_adaptations(
                weather_analysis, venue_info
            )
            
            # Overall dynamic adjustment factor
            dynamic_factor = self._calculate_overall_dynamic_factor(
                real_time_conditions, seasonal_advantages, crowd_presence,
                time_effects, weather_adaptations
            )
            
            return {
                'real_time_conditions': real_time_conditions,
                'seasonal_advantages': seasonal_advantages,
                'crowd_presence_optimization': crowd_presence,
                'time_of_day_effects': time_effects,
                'weather_adaptations': weather_adaptations,
                'dynamic_adjustment_factor': dynamic_factor,
                'adjustment_confidence': self._calculate_adjustment_confidence(
                    real_time_conditions, weather_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error in dynamic venue adjustments: {str(e)}")
            return self._get_default_dynamic_adjustments()
    
    def _analyze_cross_league_venue_intelligence(self, venue_info: Dict, home_team_id: int,
                                                away_team_id: int, match_context: Dict) -> Dict[str, Any]:
        """
        Cross-league venue intelligence for international comparisons
        """
        try:
            # International venue performance comparison
            international_comparison = self._compare_international_venue_performance(venue_info)
            
            # European away performance analysis
            european_analysis = self._analyze_european_away_performance(away_team_id, venue_info)
            
            # Venue difficulty ranking
            difficulty_ranking = self._calculate_venue_difficulty_ranking(venue_info)
            
            # Cultural adaptation factors
            cultural_factors = self._analyze_cultural_adaptation_factors(
                venue_info, away_team_id, match_context
            )
            
            return {
                'international_comparison': international_comparison,
                'european_away_analysis': european_analysis,
                'difficulty_ranking': difficulty_ranking,
                'cultural_adaptation_factors': cultural_factors,
                'cross_league_insights': self._generate_cross_league_insights(
                    international_comparison, difficulty_ranking, cultural_factors
                )
            }
            
        except Exception as e:
            logger.error(f"Error in cross-league venue intelligence: {str(e)}")
            return self._get_default_cross_league_intelligence()
    
    def _calculate_venue_difficulty_score(self, venue_info: Dict, home_advantage: Dict,
                                         travel_impact: Dict, weather_analysis: Dict) -> int:
        """
        Calculate venue difficulty score (0-100)
        """
        try:
            # Base difficulty from home advantage
            home_advantage_score = min(40, (home_advantage.get('final_coefficient', 1.1) - 1.0) * 400)
            
            # Travel difficulty
            travel_score = min(25, travel_impact.get('overall_travel_penalty', 0) * 250)
            
            # Weather/climate difficulty
            weather_score = min(15, abs(weather_analysis.get('weather_advantage', 0)) * 150)
            
            # Stadium atmosphere and intimidation
            atmosphere_score = min(20, venue_info.get('atmosphere_rating', 7.0) * 2)
            
            # Combine scores
            total_score = home_advantage_score + travel_score + weather_score + atmosphere_score
            
            return max(0, min(100, int(total_score)))
            
        except Exception as e:
            logger.error(f"Error calculating venue difficulty score: {str(e)}")
            return 50
    
    def _generate_venue_adjusted_predictions(self, home_advantage: Dict, venue_modeling: Dict,
                                           travel_impact: Dict, weather_analysis: Dict,
                                           dynamic_adjustments: Dict) -> Dict[str, Any]:
        """
        Generate venue-adjusted performance predictions
        """
        try:
            # Home team boost calculation
            home_boost = (
                home_advantage.get('final_coefficient', 1.1) *
                venue_modeling.get('overall_venue_factor', 1.0) *
                dynamic_adjustments.get('dynamic_adjustment_factor', 1.0)
            )
            
            # Away team penalty calculation
            away_penalty = (
                1.0 - travel_impact.get('overall_travel_penalty', 0.05) -
                weather_analysis.get('weather_advantage', 0) * 0.1
            )
            
            # Goal expectation adjustments
            goal_adjustments = self._calculate_goal_expectation_adjustments(
                home_advantage, venue_modeling, travel_impact
            )
            
            # Win probability adjustments
            win_prob_adjustments = self._calculate_win_probability_adjustments(
                home_boost, away_penalty
            )
            
            return {
                'home_team_boost': max(0.95, min(1.15, home_boost)),
                'away_team_penalty': max(0.85, min(1.05, away_penalty)),
                'goal_expectation_adjustments': goal_adjustments,
                'win_probability_adjustments': win_prob_adjustments,
                'confidence_intervals': self._calculate_prediction_confidence_intervals(
                    home_advantage, travel_impact
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating venue-adjusted predictions: {str(e)}")
            return self._get_default_predictions()
    
    def _identify_optimal_performance_conditions(self, venue_info: Dict, weather_analysis: Dict,
                                               dynamic_adjustments: Dict) -> Dict[str, Any]:
        """
        Identify optimal performance venue conditions
        """
        try:
            # Optimal weather conditions
            optimal_weather = self._identify_optimal_weather_conditions(weather_analysis)
            
            # Optimal time conditions
            optimal_timing = self._identify_optimal_timing_conditions(dynamic_adjustments)
            
            # Optimal crowd conditions
            optimal_crowd = self._identify_optimal_crowd_conditions(venue_info)
            
            # Surface conditions
            optimal_surface = self._identify_optimal_surface_conditions(venue_info)
            
            return {
                'optimal_weather': optimal_weather,
                'optimal_timing': optimal_timing,
                'optimal_crowd': optimal_crowd,
                'optimal_surface': optimal_surface,
                'performance_score': self._calculate_optimal_performance_score(
                    optimal_weather, optimal_timing, optimal_crowd, optimal_surface
                ),
                'recommendations': self._generate_optimal_condition_recommendations(
                    optimal_weather, optimal_timing, optimal_crowd
                )
            }
            
        except Exception as e:
            logger.error(f"Error identifying optimal performance conditions: {str(e)}")
            return self._get_default_optimal_conditions()
    
    def _generate_venue_recommendations(self, home_advantage: Dict, venue_difficulty: int,
                                       travel_impact: Dict, weather_analysis: Dict) -> List[str]:
        """Generate venue-specific recommendations"""
        recommendations = []
        
        # Home advantage recommendations
        if home_advantage.get('final_coefficient', 1.1) >= 1.2:
            recommendations.append("Güçlü ev sahibi avantajı - Home team favored")
        elif home_advantage.get('final_coefficient', 1.1) <= 1.05:
            recommendations.append("Zayıf ev sahibi avantajı - Neutral venue characteristics")
        
        # Venue difficulty recommendations
        if venue_difficulty >= 75:
            recommendations.append("Çok zor deplasman - Very difficult away venue")
        elif venue_difficulty >= 60:
            recommendations.append("Zor deplasman - Challenging away venue")
        elif venue_difficulty <= 30:
            recommendations.append("Kolay deplasman - Favorable away venue")
        
        # Travel impact recommendations
        if travel_impact.get('overall_travel_penalty', 0) >= 0.1:
            recommendations.append("Yüksek seyahat etkisi - Significant travel fatigue expected")
        
        # Weather recommendations
        weather_advantage = weather_analysis.get('weather_advantage', 0)
        if abs(weather_advantage) >= 0.15:
            if weather_advantage > 0:
                recommendations.append("Hava koşulları ev sahibi lehine - Weather favors home team")
            else:
                recommendations.append("Hava koşulları deplasman lehine - Weather favors away team")
        
        return recommendations or ["Standart venue analizi uygulandı"]
    
    # Helper methods (simplified implementations)
    def _classify_stadium_type(self, stadium_data: Dict) -> str:
        capacity = stadium_data.get('capacity', 0)
        if capacity >= 80000:
            return "Mega Stadium"
        elif capacity >= 60000:
            return "Large Stadium"
        elif capacity >= 40000:
            return "Medium Stadium"
        else:
            return "Small Stadium"
    
    def _classify_capacity(self, capacity: int) -> str:
        if capacity >= 80000:
            return "Mega"
        elif capacity >= 60000:
            return "Large"
        elif capacity >= 40000:
            return "Medium"
        elif capacity >= 20000:
            return "Small"
        else:
            return "Mini"
    
    def _calculate_intimidation_factor(self, stadium_data: Dict) -> float:
        atmosphere = stadium_data.get('atmosphere_rating', 7.0)
        capacity = stadium_data.get('capacity', 30000)
        
        base_factor = atmosphere / 10.0
        capacity_factor = min(1.2, capacity / 80000)
        
        return base_factor * (1 + capacity_factor)
    
    def _calculate_generic_intimidation_factor(self, capacity: int) -> float:
        return min(0.8, capacity / 100000) + 0.7
    
    def _get_default_venue_modeling(self) -> Dict[str, Any]:
        """Default venue modeling when errors occur"""
        return {
            'stadium_profile': {'stadium_type': 'Standard', 'intimidation_factor': 0.75},
            'overall_venue_factor': 1.0
        }
    
    def _get_default_travel_impact(self) -> Dict[str, Any]:
        """Default travel impact when errors occur"""
        return {
            'travel_distance_km': 200,
            'travel_category': 'Domestic',
            'overall_travel_penalty': 0.05
        }
    
    def _get_default_weather_analysis(self) -> Dict[str, Any]:
        """Default weather analysis when errors occur"""
        return {
            'current_conditions': {'temperature': 15, 'weather': 'clear'},
            'weather_advantage': 0
        }
    
    def _get_default_dynamic_adjustments(self) -> Dict[str, Any]:
        """Default dynamic adjustments when errors occur"""
        return {
            'dynamic_adjustment_factor': 1.0,
            'adjustment_confidence': 0.5
        }
    
    def _get_default_cross_league_intelligence(self) -> Dict[str, Any]:
        """Default cross-league intelligence when errors occur"""
        return {
            'difficulty_ranking': 50,
            'cross_league_insights': []
        }
    
    def _get_default_predictions(self) -> Dict[str, Any]:
        """Default predictions when errors occur"""
        return {
            'home_team_boost': 1.1,
            'away_team_penalty': 0.95,
            'confidence_intervals': {'low': 0.4, 'high': 0.6}
        }
    
    def _get_default_optimal_conditions(self) -> Dict[str, Any]:
        """Default optimal conditions when errors occur"""
        return {
            'performance_score': 70,
            'recommendations': ["Standard conditions assumed"]
        }
    
    # Placeholder methods for complex calculations (would be implemented with real data)
    def _calculate_surface_team_style_impact(self, surface_type: str, historical_matches: List[Dict] = None) -> Dict:
        return {'technical_teams': 1.0, 'physical_teams': 1.0}
    
    def _assess_seasonal_surface_variation(self, surface_type: str) -> Dict:
        return {'winter': 'standard', 'summer': 'good'}
    
    def _calculate_altitude_effects(self, altitude: int) -> Dict:
        if altitude >= 1000:
            return {'effect': 'significant', 'factor': 1.1, 'adaptation_time': '3-5 days'}
        else:
            return {'effect': 'minimal', 'factor': 1.0, 'adaptation_time': '0 days'}
    
    def _assess_climate_zone_effects(self, coordinates: Tuple) -> Dict:
        return {'zone': 'temperate', 'adaptation_difficulty': 'low'}
    
    def _assess_geographic_isolation(self, coordinates: Tuple) -> Dict:
        return {'isolation_level': 'low', 'factor': 1.0}
    
    def _calculate_geographic_advantage(self, altitude_effects: Dict, climate_effects: Dict, isolation_effects: Dict) -> float:
        return 1.0
    
    def _calculate_noise_amplification(self, capacity: int, roof_type: str) -> float:
        base = capacity / 50000
        if roof_type == 'closed':
            return base * 1.3
        elif roof_type == 'partial':
            return base * 1.15
        else:
            return base
    
    def _calculate_crowd_density_impact(self, capacity: int) -> float:
        return min(1.2, capacity / 80000)
    
    def _calculate_psychological_pressure(self, atmosphere_rating: float, noise_amplification: float, capacity: int) -> float:
        return (atmosphere_rating / 10.0) * noise_amplification * (capacity / 50000)
    
    def _assess_crowd_referee_influence(self, capacity: int, atmosphere_rating: float) -> Dict:
        influence = (capacity / 100000) * (atmosphere_rating / 10.0)
        return {'influence_level': 'medium' if influence > 0.5 else 'low', 'factor': influence}
    
    def _assess_concentration_impact(self, noise_amplification: float, psychological_pressure: float) -> Dict:
        impact = (noise_amplification + psychological_pressure) / 2
        return {'impact_level': 'high' if impact > 1.0 else 'medium', 'factor': impact}
    
    def _get_team_location(self, team_id: int, venue_info: Dict = None) -> Tuple:
        if venue_info:
            return venue_info.get('coordinates', (41.0, 29.0))  # Default to Istanbul
        return (41.0, 29.0)  # Default location
    
    def _calculate_travel_distance(self, location1: Tuple, location2: Tuple) -> float:
        try:
            return geopy.distance.geodesic(location1, location2).kilometers
        except:
            return 200  # Default distance
    
    def _classify_travel_distance(self, distance: float) -> str:
        if distance <= 100:
            return "Local"
        elif distance <= 800:
            return "Domestic"
        elif distance <= 3000:
            return "European"
        else:
            return "Intercontinental"
    
    def _calculate_timezone_impact(self, location1: Tuple, location2: Tuple) -> Dict:
        # Simplified timezone calculation
        time_diff = abs(location1[1] - location2[1]) / 15  # Rough timezone difference
        return {'difference_hours': time_diff, 'impact_factor': min(1.0, time_diff / 6)}
    
    def _assess_travel_fatigue(self, distance: float, timezone_impact: Dict, match_context: Dict) -> float:
        base_fatigue = min(1.0, distance / 5000)
        timezone_fatigue = timezone_impact.get('impact_factor', 0) * 0.5
        return base_fatigue + timezone_fatigue
    
    def _assess_cultural_adaptation(self, home_location: Tuple, away_location: Tuple, away_team_id: int) -> Dict:
        # Simplified cultural adaptation assessment
        distance = self._calculate_travel_distance(home_location, away_location)
        if distance > 3000:
            return {'adaptation_difficulty': 'high', 'factor': 0.9}
        elif distance > 800:
            return {'adaptation_difficulty': 'medium', 'factor': 0.95}
        else:
            return {'adaptation_difficulty': 'low', 'factor': 1.0}
    
    def _integrate_congestion_analysis(self, team_id: int, match_context: Dict, travel_distance: float) -> Dict:
        # Integration with existing congestion analyzer would go here
        return {'congestion_factor': 1.0, 'integration': 'placeholder'}
    
    def _calculate_overall_travel_penalty(self, travel_fatigue: float, timezone_impact: Dict, 
                                        cultural_adaptation: Dict, congestion_impact: Dict) -> float:
        penalty = (
            travel_fatigue * 0.4 +
            timezone_impact.get('impact_factor', 0) * 0.3 +
            (1 - cultural_adaptation.get('factor', 1.0)) * 0.2 +
            congestion_impact.get('congestion_factor', 0) * 0.1
        )
        return min(0.3, penalty)  # Cap at 30% penalty
    
    def _estimate_recovery_time(self, distance: float, timezone_impact: Dict) -> Dict:
        base_recovery = distance / 1000  # 1 day per 1000km
        timezone_recovery = timezone_impact.get('difference_hours', 0) * 0.5
        total_days = base_recovery + timezone_recovery
        return {'days': min(7, total_days), 'quality': 'full' if total_days <= 2 else 'partial'}
    
    def _get_weather_conditions(self, coordinates: Tuple, match_date) -> Dict:
        # Placeholder for weather API integration
        return {
            'temperature': 15,
            'humidity': 60,
            'wind_speed': 10,
            'precipitation': 0,
            'weather': 'clear',
            'visibility': 'good'
        }
    
    def _analyze_climate_profile(self, coordinates: Tuple) -> Dict:
        return {'climate_type': 'temperate', 'seasonal_variation': 'moderate'}
    
    def _assess_weather_playing_style_impact(self, weather: Dict) -> Dict:
        temp = weather.get('temperature', 15)
        if temp < 5:
            return {'impact': 'significant', 'favors': 'physical_play'}
        elif temp > 30:
            return {'impact': 'moderate', 'favors': 'technical_play'}
        else:
            return {'impact': 'minimal', 'favors': 'balanced'}
    
    def _analyze_seasonal_weather_patterns(self, coordinates: Tuple, match_date) -> Dict:
        return {'pattern': 'stable', 'predictability': 'high'}
    
    def _calculate_extreme_weather_adjustments(self, weather: Dict) -> Dict:
        temp = weather.get('temperature', 15)
        wind = weather.get('wind_speed', 10)
        
        adjustments = {}
        if temp < 0 or temp > 35:
            adjustments['temperature'] = 'extreme'
        if wind > 20:
            adjustments['wind'] = 'high'
        
        return adjustments
    
    def _determine_weather_advantage(self, weather: Dict, playing_style_impact: Dict) -> float:
        # Return value between -0.2 and 0.2 indicating weather advantage
        if playing_style_impact.get('impact') == 'significant':
            return 0.1 if 'physical' in playing_style_impact.get('favors', '') else -0.1
        return 0.0
    
    def _assess_weather_surface_impact(self, weather: Dict, surface: str) -> Dict:
        temp = weather.get('temperature', 15)
        precipitation = weather.get('precipitation', 0)
        
        if surface == 'grass' and precipitation > 0:
            return {'condition': 'wet', 'impact': 'moderate'}
        elif surface == 'artificial_turf' and temp > 30:
            return {'condition': 'hot', 'impact': 'moderate'}
        else:
            return {'condition': 'normal', 'impact': 'minimal'}
    
    # Additional placeholder methods for remaining functionality
    def _assess_real_time_venue_conditions(self, venue_info: Dict, match_context: Dict, weather_analysis: Dict) -> Dict:
        return {'condition': 'good', 'factor': 1.0}
    
    def _calculate_seasonal_venue_advantages(self, venue_info: Dict, match_context: Dict) -> Dict:
        return {'advantage': 'moderate', 'factor': 1.02}
    
    def _optimize_crowd_presence(self, venue_info: Dict, match_context: Dict) -> Dict:
        return {'attendance_expected': '80%', 'impact': 'positive'}
    
    def _calculate_time_of_day_effects(self, match_context: Dict) -> Dict:
        return {'time_impact': 'minimal', 'factor': 1.0}
    
    def _calculate_weather_adaptations(self, weather_analysis: Dict, venue_info: Dict) -> Dict:
        return {'adaptation_needed': 'minimal', 'factor': 1.0}
    
    def _calculate_overall_dynamic_factor(self, real_time: Dict, seasonal: Dict, crowd: Dict, time: Dict, weather: Dict) -> float:
        return 1.0
    
    def _calculate_adjustment_confidence(self, real_time: Dict, weather: Dict) -> float:
        return 0.8
    
    def _compare_international_venue_performance(self, venue_info: Dict) -> Dict:
        return {'international_ranking': 'medium', 'comparison': 'favorable'}
    
    def _analyze_european_away_performance(self, away_team_id: int, venue_info: Dict) -> Dict:
        return {'european_experience': 'moderate', 'adaptation_score': 0.8}
    
    def _calculate_venue_difficulty_ranking(self, venue_info: Dict) -> Dict:
        return {'ranking': 50, 'category': 'medium_difficulty'}
    
    def _analyze_cultural_adaptation_factors(self, venue_info: Dict, away_team_id: int, match_context: Dict) -> Dict:
        return {'cultural_similarity': 'high', 'adaptation_ease': 0.9}
    
    def _generate_cross_league_insights(self, international: Dict, difficulty: Dict, cultural: Dict) -> List[str]:
        return ["Standard cross-league analysis applied"]
    
    def _calculate_goal_expectation_adjustments(self, home_advantage: Dict, venue_modeling: Dict, travel_impact: Dict) -> Dict:
        return {'home_goals_adjustment': 0.1, 'away_goals_adjustment': -0.05}
    
    def _calculate_win_probability_adjustments(self, home_boost: float, away_penalty: float) -> Dict:
        return {'home_win_boost': 0.05, 'away_win_penalty': 0.03}
    
    def _calculate_prediction_confidence_intervals(self, home_advantage: Dict, travel_impact: Dict) -> Dict:
        confidence = home_advantage.get('confidence_level', 0.7)
        return {'low': confidence - 0.1, 'high': confidence + 0.1}
    
    def _identify_optimal_weather_conditions(self, weather_analysis: Dict) -> Dict:
        return {'temperature': '15-20°C', 'wind': '<15 km/h', 'precipitation': 'none'}
    
    def _identify_optimal_timing_conditions(self, dynamic_adjustments: Dict) -> Dict:
        return {'time_of_day': '15:00-17:00', 'day_of_week': 'Saturday'}
    
    def _identify_optimal_crowd_conditions(self, venue_info: Dict) -> Dict:
        return {'attendance': '90-100%', 'atmosphere': 'electric'}
    
    def _identify_optimal_surface_conditions(self, venue_info: Dict) -> Dict:
        return {'surface_quality': 'excellent', 'moisture': 'optimal'}
    
    def _calculate_optimal_performance_score(self, weather: Dict, timing: Dict, crowd: Dict, surface: Dict) -> int:
        return 85  # Placeholder optimal score
    
    def _generate_optimal_condition_recommendations(self, weather: Dict, timing: Dict, crowd: Dict) -> List[str]:
        return ["Optimal conditions for high-quality football"]
    
    def _analyze_venue_h2h_records(self, home_team_id: int, away_team_id: int, 
                                  venue_info: Dict, historical_matches: List[Dict] = None) -> Dict[str, Any]:
        """Analyze historical venue head-to-head records"""
        if not historical_matches:
            return {'matches_played': 0, 'home_wins': 0, 'draws': 0, 'away_wins': 0}
        
        venue_h2h = [
            match for match in historical_matches
            if (match.get('home_team_id') == home_team_id and 
                match.get('away_team_id') == away_team_id)
        ]
        
        if not venue_h2h:
            return {'matches_played': 0, 'home_wins': 0, 'draws': 0, 'away_wins': 0}
        
        home_wins = sum(1 for match in venue_h2h 
                       if (match.get('home_score', 0) or 0) > (match.get('away_score', 0) or 0))
        draws = sum(1 for match in venue_h2h 
                   if (match.get('home_score', 0) or 0) == (match.get('away_score', 0) or 0))
        away_wins = len(venue_h2h) - home_wins - draws
        
        return {
            'matches_played': len(venue_h2h),
            'home_wins': home_wins,
            'draws': draws,
            'away_wins': away_wins,
            'home_win_percentage': home_wins / len(venue_h2h) if venue_h2h else 0,
            'recent_trend': self._analyze_recent_h2h_trend(venue_h2h[-5:]) if len(venue_h2h) >= 5 else 'insufficient_data'
        }
    
    def _analyze_team_venue_performance(self, home_team_id: int, away_team_id: int,
                                       venue_info: Dict, historical_matches: List[Dict] = None) -> Dict[str, Any]:
        """Analyze team-specific venue performance"""
        if not historical_matches:
            return {'home_team_venue_record': {}, 'away_team_venue_record': {}}
        
        # Home team venue performance
        home_venue_matches = [
            match for match in historical_matches
            if match.get('home_team_id') == home_team_id
        ]
        
        # Away team performance at this venue (as away team)
        away_venue_matches = [
            match for match in historical_matches
            if match.get('away_team_id') == away_team_id
        ]
        
        home_record = self._calculate_venue_record(home_venue_matches, 'home')
        away_record = self._calculate_venue_record(away_venue_matches, 'away')
        
        return {
            'home_team_venue_record': home_record,
            'away_team_venue_record': away_record,
            'venue_familiarity': self._assess_venue_familiarity(home_venue_matches, away_venue_matches)
        }
    
    def _calculate_overall_venue_factor(self, stadium_profile: Dict, surface_effects: Dict,
                                       geographic_effects: Dict, crowd_noise_impact: Dict) -> float:
        """Calculate overall venue factor combining all effects"""
        stadium_factor = stadium_profile.get('intimidation_factor', 0.75)
        surface_factor = surface_effects.get('characteristics', {}).get('technical_advantage', 1.0)
        geographic_factor = geographic_effects.get('geographic_advantage', 1.0)
        crowd_factor = crowd_noise_impact.get('psychological_pressure', 0.5)
        
        # Combine factors with appropriate weighting
        overall_factor = (
            stadium_factor * 0.3 +
            surface_factor * 0.2 +
            geographic_factor * 0.2 +
            (1.0 + crowd_factor * 0.1) * 0.3
        )
        
        return max(0.90, min(1.15, overall_factor))
    
    def _identify_architectural_advantages(self, stadium_data: Dict) -> List[str]:
        """Identify architectural advantages of the stadium"""
        advantages = []
        
        roof_type = stadium_data.get('roof_type', 'open')
        if roof_type == 'closed':
            advantages.append('Enclosed Design - Noise Amplification')
        elif roof_type == 'partial':
            advantages.append('Partial Roof - Weather Protection')
        
        capacity = stadium_data.get('capacity', 0)
        if capacity >= 80000:
            advantages.append('Mega Capacity - Intimidating Atmosphere')
        elif capacity >= 60000:
            advantages.append('Large Capacity - Strong Home Support')
        
        atmosphere_rating = stadium_data.get('atmosphere_rating', 7.0)
        if atmosphere_rating >= 9.0:
            advantages.append('Legendary Atmosphere')
        elif atmosphere_rating >= 8.5:
            advantages.append('Excellent Atmosphere')
        
        return advantages
    
    def _assess_historical_significance(self, stadium_data: Dict) -> str:
        """Assess historical significance of the stadium"""
        # This would be enhanced with actual historical data
        iconic_stadiums = ['old_trafford', 'anfield', 'camp_nou', 'bernabeu', 'san_siro']
        
        if stadium_data.get('id') in iconic_stadiums:
            return 'Iconic'
        
        capacity = stadium_data.get('capacity', 0)
        if capacity >= 75000:
            return 'High'
        elif capacity >= 50000:
            return 'Medium'
        else:
            return 'Standard'
    
    def _assess_modern_facilities(self, stadium_data: Dict) -> str:
        """Assess modern facilities quality"""
        # This would be enhanced with actual facility data
        atmosphere_rating = stadium_data.get('atmosphere_rating', 7.0)
        
        if atmosphere_rating >= 9.0:
            return 'Exceptional'
        elif atmosphere_rating >= 8.0:
            return 'Excellent'
        elif atmosphere_rating >= 7.0:
            return 'Good'
        else:
            return 'Standard'
    
    def _analyze_recent_h2h_trend(self, recent_matches: List[Dict]) -> str:
        """Analyze recent head-to-head trend"""
        if len(recent_matches) < 3:
            return 'insufficient_data'
        
        home_wins = sum(1 for match in recent_matches 
                       if (match.get('home_score', 0) or 0) > (match.get('away_score', 0) or 0))
        
        if home_wins >= len(recent_matches) * 0.7:
            return 'strong_home_advantage'
        elif home_wins <= len(recent_matches) * 0.3:
            return 'away_team_dominance'
        else:
            return 'balanced'
    
    def _calculate_venue_record(self, matches: List[Dict], team_type: str) -> Dict[str, Any]:
        """Calculate venue record for a team"""
        if not matches:
            return {'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'win_rate': 0.0}
        
        wins = 0
        draws = 0
        losses = 0
        
        for match in matches:
            if team_type == 'home':
                team_score = match.get('home_score', 0) or 0
                opponent_score = match.get('away_score', 0) or 0
            else:
                team_score = match.get('away_score', 0) or 0
                opponent_score = match.get('home_score', 0) or 0
            
            if team_score > opponent_score:
                wins += 1
            elif team_score == opponent_score:
                draws += 1
            else:
                losses += 1
        
        total_matches = len(matches)
        return {
            'matches': total_matches,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': wins / total_matches if total_matches > 0 else 0.0,
            'points_per_game': (wins * 3 + draws) / total_matches if total_matches > 0 else 0.0
        }
    
    def _assess_venue_familiarity(self, home_matches: List[Dict], away_matches: List[Dict]) -> Dict[str, Any]:
        """Assess venue familiarity for both teams"""
        home_familiarity = len(home_matches)
        away_familiarity = len(away_matches)
        
        # Calculate familiarity scores
        home_familiarity_score = min(1.0, home_familiarity / 20)  # Max familiarity at 20 matches
        away_familiarity_score = min(1.0, away_familiarity / 10)  # Max familiarity at 10 matches
        
        return {
            'home_team_familiarity': home_familiarity_score,
            'away_team_familiarity': away_familiarity_score,
            'familiarity_advantage': home_familiarity_score - away_familiarity_score,
            'home_team_matches_at_venue': home_familiarity,
            'away_team_matches_at_venue': away_familiarity
        }
    
    def _get_default_venue_analysis(self) -> Dict[str, Any]:
        """Get default venue analysis when errors occur"""
        return {
            'match_info': {
                'analysis_timestamp': datetime.now().isoformat()
            },
            'home_advantage_analysis': self._get_default_home_advantage(),
            'venue_difficulty_score': 50,
            'performance_predictions': {
                'home_boost': 1.1,
                'away_penalty': 0.95
            },
            'recommendations': ["Standard venue analysis applied"]
        }