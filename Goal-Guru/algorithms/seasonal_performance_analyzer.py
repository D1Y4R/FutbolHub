"""
Seasonal Performance Analyzer
Sezonsal performans döngülerini analiz eden ve sezon içi değişiklikleri modelleyen gelişmiş sistem

Bu modül şunları sağlar:
1. Season Cycle Detection - Sezon döngüsü tespiti
2. Performance Phase Modeling - Performans fazı modelleme
3. Seasonal Trend Prediction - Sezonsal trend tahmini
4. Historical Seasonal Patterns - Tarihi sezonsal kalıplar

Author: Football Prediction System
Date: September 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta, date
import logging
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json
import math
from collections import defaultdict, Counter
import calendar

# Import existing analyzers for integration
try:
    from .dynamic_time_analyzer import DynamicTimeAnalyzer
    from .form_trend_analyzer import FormTrendAnalyzer
    from .league_context_analyzer import LeagueContextAnalyzer
except ImportError:
    # Fallback for direct execution
    DynamicTimeAnalyzer = None
    FormTrendAnalyzer = None
    LeagueContextAnalyzer = None

logger = logging.getLogger(__name__)

class SeasonalPerformanceAnalyzer:
    """
    Sezonsal performans döngülerini analiz eden ve sezon içi değişiklikleri 
    modelleyen gelişmiş sistem
    
    Ana özellikler:
    - Season Cycle Detection: Sezon döngüsü tespiti
    - Performance Phase Modeling: Performans fazı modelleme  
    - Seasonal Trend Prediction: Sezonsal trend tahmini
    - Historical Seasonal Patterns: Tarihi sezonsal kalıplar
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Seasonal Performance Analyzer
        
        Args:
            config: Configuration dictionary for customization
        """
        self.config = config or self._get_default_config()
        
        # Initialize scalers and models
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.trend_model = LinearRegression()
        
        # Integration with existing analyzers
        self.time_analyzer = DynamicTimeAnalyzer() if DynamicTimeAnalyzer else None
        self.form_analyzer = FormTrendAnalyzer() if FormTrendAnalyzer else None
        self.league_analyzer = LeagueContextAnalyzer() if LeagueContextAnalyzer else None
        
        # Seasonal pattern storage
        self.seasonal_profiles = {}
        self.historical_patterns = defaultdict(list)
        self.phase_transitions = {}
        self.team_seasonal_characteristics = {}
        
        # Performance tracking
        self.prediction_accuracy = defaultdict(list)
        self.seasonal_model_cache = {}
        
        logger.info("SeasonalPerformanceAnalyzer initialized with comprehensive seasonal analysis capabilities")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for seasonal analysis"""
        return {
            # Season cycle configuration
            'season_cycles': {
                'early_season': {
                    'weeks': (1, 10),
                    'characteristics': ['adaptation', 'new_signings', 'fitness_building'],
                    'weight_factor': 0.8
                },
                'mid_season': {
                    'weeks': (11, 25),
                    'characteristics': ['consistency', 'rhythm', 'peak_performance'],
                    'weight_factor': 1.0
                },
                'late_season': {
                    'weeks': (26, 38),
                    'characteristics': ['motivation', 'fatigue', 'pressure'],
                    'weight_factor': 0.9
                }
            },
            
            # Transfer window effects
            'transfer_windows': {
                'winter': {
                    'start_date': (1, 1),   # January 1st
                    'end_date': (1, 31),    # January 31st
                    'impact_weeks': 4,
                    'disruption_factor': 0.15
                },
                'summer': {
                    'start_date': (6, 1),   # June 1st
                    'end_date': (9, 1),     # September 1st
                    'impact_weeks': 8,
                    'disruption_factor': 0.25
                }
            },
            
            # Holiday periods
            'holiday_periods': {
                'winter_break': {
                    'start': (12, 20),      # December 20th
                    'end': (1, 10),         # January 10th
                    'impact_factor': 0.2
                },
                'summer_break': {
                    'start': (5, 15),       # May 15th  
                    'end': (8, 15),         # August 15th
                    'impact_factor': 0.3
                },
                'international_breaks': {
                    'frequency': 'monthly',
                    'duration_days': 10,
                    'impact_factor': 0.1
                }
            },
            
            # Performance phases
            'performance_phases': {
                'championship_chase': {
                    'position_range': (1, 6),
                    'motivation_boost': 1.15,
                    'pressure_factor': 1.1
                },
                'relegation_battle': {
                    'position_range': (15, 20),
                    'motivation_boost': 1.2,
                    'pressure_factor': 1.25
                },
                'mid_table': {
                    'position_range': (7, 14),
                    'motivation_boost': 0.95,
                    'pressure_factor': 0.9
                },
                'european_qualification': {
                    'position_range': (4, 8),
                    'motivation_boost': 1.05,
                    'pressure_factor': 1.0
                }
            },
            
            # Competition effects
            'competition_effects': {
                'european_competitions': {
                    'champions_league': {
                        'fixture_load': 1.3,
                        'mental_load': 1.2,
                        'squad_rotation': 1.1
                    },
                    'europa_league': {
                        'fixture_load': 1.2,
                        'mental_load': 1.1,
                        'squad_rotation': 1.05
                    },
                    'conference_league': {
                        'fixture_load': 1.15,
                        'mental_load': 1.05,
                        'squad_rotation': 1.02
                    }
                },
                'cup_competitions': {
                    'domestic_cup': {
                        'importance_factor': 0.8,
                        'rotation_factor': 1.1
                    },
                    'league_cup': {
                        'importance_factor': 0.6,
                        'rotation_factor': 1.2
                    }
                }
            },
            
            # Trend prediction parameters
            'trend_prediction': {
                'lookback_weeks': 12,
                'forecast_weeks': 8,
                'smoothing_factor': 0.3,
                'confidence_intervals': [0.68, 0.95],
                'trend_detection_threshold': 0.1
            },
            
            # Weather and external factors
            'external_factors': {
                'weather_impact': {
                    'temperature_optimal': (15, 25),  # Celsius
                    'precipitation_threshold': 5,     # mm
                    'wind_threshold': 30              # km/h
                },
                'fixture_congestion': {
                    'games_per_week_threshold': 2,
                    'fatigue_accumulation': 0.05,
                    'recovery_time_optimal': 72       # hours
                }
            },
            
            # Analysis parameters
            'analysis_parameters': {
                'min_matches_for_analysis': 10,
                'confidence_threshold': 0.7,
                'pattern_similarity_threshold': 0.8,
                'outlier_detection_method': 'iqr',
                'seasonality_detection_method': 'decomposition'
            }
        }
    
    def analyze_seasonal_performance(self, team_data: Dict, match_context: Dict, 
                                   historical_data: Optional[List[Dict]] = None) -> Dict:
        """
        Main analysis function for comprehensive seasonal performance assessment
        
        Args:
            team_data: Team's match history and statistics
            match_context: Context of the upcoming match
            historical_data: Multi-year historical data for pattern analysis
            
        Returns:
            Dict containing comprehensive seasonal performance analysis
        """
        try:
            # Extract essential data
            matches = team_data.get('recent_matches', [])
            team_id = team_data.get('team_id', 0)
            league_id = match_context.get('league_id', 0)
            match_date = match_context.get('match_date', datetime.now())
            
            if isinstance(match_date, str):
                match_date = datetime.strptime(match_date, '%Y-%m-%d')
            
            logger.info(f"Starting seasonal performance analysis for team {team_id}")
            
            # 1. Season Cycle Detection
            cycle_analysis = self._detect_season_cycles(matches, team_id, match_date)
            
            # 2. Performance Phase Modeling
            phase_analysis = self._model_performance_phases(
                matches, team_id, match_context, historical_data
            )
            
            # 3. Seasonal Trend Prediction
            trend_analysis = self._predict_seasonal_trends(
                matches, team_id, match_date, historical_data
            )
            
            # 4. Historical Seasonal Patterns
            historical_analysis = self._analyze_historical_patterns(
                team_id, league_id, matches, historical_data
            )
            
            # 5. External Factor Analysis
            external_analysis = self._analyze_external_factors(
                matches, match_date, match_context
            )
            
            # 6. Integrated Seasonal Assessment
            integrated_assessment = self._generate_integrated_assessment(
                cycle_analysis, phase_analysis, trend_analysis, 
                historical_analysis, external_analysis
            )
            
            # Compile comprehensive analysis
            seasonal_analysis = {
                'team_id': team_id,
                'analysis_date': match_date.isoformat(),
                'season_cycle_detection': cycle_analysis,
                'performance_phase_modeling': phase_analysis,
                'seasonal_trend_prediction': trend_analysis,
                'historical_seasonal_patterns': historical_analysis,
                'external_factor_analysis': external_analysis,
                'integrated_assessment': integrated_assessment,
                'confidence_score': self._calculate_analysis_confidence(
                    cycle_analysis, phase_analysis, trend_analysis, historical_analysis
                ),
                'analysis_metadata': {
                    'matches_analyzed': len(matches),
                    'historical_seasons': len(historical_data) if historical_data else 0,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            # Cache the analysis for future reference
            self._cache_seasonal_analysis(team_id, seasonal_analysis)
            
            logger.info(f"Seasonal performance analysis completed for team {team_id}")
            return seasonal_analysis
            
        except Exception as e:
            logger.error(f"Error in seasonal performance analysis: {str(e)}")
            return self._get_default_seasonal_analysis(team_id, match_date)
    
    def _detect_season_cycles(self, matches: List[Dict], team_id: int, 
                            current_date: datetime) -> Dict:
        """
        Detect and analyze season cycles including early, mid, and late season patterns
        
        Returns:
            Dict with detailed season cycle analysis
        """
        try:
            if not matches:
                return self._get_default_cycle_analysis()
            
            # Determine current season week
            current_week = self._calculate_season_week(current_date)
            current_phase = self._determine_season_phase(current_week)
            
            # Group matches by season phases
            phase_performance = self._group_matches_by_phase(matches, team_id)
            
            # Analyze adaptation patterns (early season)
            early_season_analysis = self._analyze_early_season_patterns(
                phase_performance.get('early_season', []), team_id
            )
            
            # Analyze consistency patterns (mid-season)
            mid_season_analysis = self._analyze_mid_season_patterns(
                phase_performance.get('mid_season', []), team_id
            )
            
            # Analyze motivation factors (late season)
            late_season_analysis = self._analyze_late_season_patterns(
                phase_performance.get('late_season', []), team_id
            )
            
            # Transfer window effect analysis
            transfer_effects = self._analyze_transfer_window_effects(matches, team_id)
            
            # Holiday period impact analysis
            holiday_impacts = self._analyze_holiday_period_impacts(matches, team_id)
            
            return {
                'current_season_week': current_week,
                'current_phase': current_phase,
                'phase_performance': phase_performance,
                'early_season_analysis': early_season_analysis,
                'mid_season_analysis': mid_season_analysis,
                'late_season_analysis': late_season_analysis,
                'transfer_window_effects': transfer_effects,
                'holiday_period_impacts': holiday_impacts,
                'cycle_strength': self._calculate_cycle_strength(phase_performance),
                'phase_transition_probability': self._calculate_phase_transition_probability(
                    current_week, phase_performance
                )
            }
            
        except Exception as e:
            logger.error(f"Error in season cycle detection: {str(e)}")
            return self._get_default_cycle_analysis()
    
    def _model_performance_phases(self, matches: List[Dict], team_id: int, 
                                match_context: Dict, 
                                historical_data: Optional[List[Dict]] = None) -> Dict:
        """
        Model different performance phases and their effects
        
        Returns:
            Dict with performance phase modeling results
        """
        try:
            # Get current league position and context
            current_position = match_context.get('league_position', 10)
            league_size = match_context.get('league_size', 20)
            matches_played = match_context.get('matches_played', 0)
            
            # Determine current phase context
            phase_context = self._determine_phase_context(
                current_position, league_size, matches_played
            )
            
            # Pre-season form carryover analysis
            preseason_carryover = self._analyze_preseason_carryover(
                matches, team_id, historical_data
            )
            
            # Championship/relegation phase effects
            championship_relegation_effects = self._analyze_championship_relegation_effects(
                matches, team_id, current_position, league_size
            )
            
            # European competition impact
            european_impact = self._analyze_european_competition_impact(
                matches, team_id, match_context
            )
            
            # Cup run effects
            cup_effects = self._analyze_cup_run_effects(matches, team_id)
            
            # International break disruption
            international_break_effects = self._analyze_international_break_effects(
                matches, team_id
            )
            
            # Manager and tactical phase effects
            tactical_phase_effects = self._analyze_tactical_phase_effects(
                matches, team_id, historical_data
            )
            
            return {
                'current_phase_context': phase_context,
                'preseason_form_carryover': preseason_carryover,
                'championship_relegation_effects': championship_relegation_effects,
                'european_competition_impact': european_impact,
                'cup_run_effects': cup_effects,
                'international_break_effects': international_break_effects,
                'tactical_phase_effects': tactical_phase_effects,
                'phase_motivation_factor': self._calculate_phase_motivation(phase_context),
                'phase_pressure_factor': self._calculate_phase_pressure(phase_context),
                'performance_sustainability': self._assess_performance_sustainability(
                    matches, team_id, phase_context
                )
            }
            
        except Exception as e:
            logger.error(f"Error in performance phase modeling: {str(e)}")
            return self._get_default_phase_analysis()
    
    def _predict_seasonal_trends(self, matches: List[Dict], team_id: int, 
                               current_date: datetime, 
                               historical_data: Optional[List[Dict]] = None) -> Dict:
        """
        Predict seasonal trends including trajectory forecasting and form peaks/troughs
        
        Returns:
            Dict with seasonal trend predictions
        """
        try:
            # Performance trajectory forecasting
            trajectory_forecast = self._forecast_performance_trajectory(
                matches, team_id, current_date
            )
            
            # Form peak/trough prediction
            peak_trough_prediction = self._predict_form_peaks_troughs(
                matches, team_id, historical_data
            )
            
            # Seasonal player fatigue modeling
            fatigue_model = self._model_seasonal_fatigue(matches, team_id, current_date)
            
            # Weather impact analysis
            weather_impact = self._analyze_weather_impact(matches, current_date)
            
            # Fixture congestion seasonal effects
            congestion_effects = self._analyze_fixture_congestion_effects(
                matches, team_id, current_date
            )
            
            # Momentum sustainability prediction
            momentum_sustainability = self._predict_momentum_sustainability(
                matches, team_id
            )
            
            # Performance volatility analysis
            volatility_analysis = self._analyze_performance_volatility(
                matches, team_id, historical_data
            )
            
            return {
                'performance_trajectory_forecast': trajectory_forecast,
                'form_peak_trough_prediction': peak_trough_prediction,
                'seasonal_fatigue_model': fatigue_model,
                'weather_impact_analysis': weather_impact,
                'fixture_congestion_effects': congestion_effects,
                'momentum_sustainability': momentum_sustainability,
                'performance_volatility': volatility_analysis,
                'trend_confidence': self._calculate_trend_confidence(
                    trajectory_forecast, peak_trough_prediction, fatigue_model
                ),
                'forecast_horizon_weeks': self.config['trend_prediction']['forecast_weeks']
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal trend prediction: {str(e)}")
            return self._get_default_trend_analysis()
    
    def _analyze_historical_patterns(self, team_id: int, league_id: int,
                                   current_matches: List[Dict],
                                   historical_data: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze historical seasonal patterns for multi-year consistency
        
        Returns:
            Dict with historical pattern analysis
        """
        try:
            if not historical_data:
                return self._get_default_historical_analysis()
            
            # Multi-year seasonal consistency analysis
            seasonal_consistency = self._analyze_multiyear_consistency(
                team_id, historical_data
            )
            
            # Manager seasonal performance patterns
            manager_patterns = self._analyze_manager_seasonal_patterns(
                team_id, historical_data
            )
            
            # Squad age and seasonal endurance analysis
            squad_endurance = self._analyze_squad_seasonal_endurance(
                team_id, historical_data, current_matches
            )
            
            # Playing style seasonal effectiveness
            style_effectiveness = self._analyze_playing_style_seasonal_effectiveness(
                team_id, historical_data
            )
            
            # League-specific seasonal variations
            league_variations = self._analyze_league_seasonal_variations(
                league_id, historical_data
            )
            
            # Historical pattern matching for current season
            pattern_matching = self._match_current_season_to_historical_patterns(
                team_id, current_matches, historical_data
            )
            
            return {
                'multiyear_seasonal_consistency': seasonal_consistency,
                'manager_seasonal_patterns': manager_patterns,
                'squad_seasonal_endurance': squad_endurance,
                'playing_style_effectiveness': style_effectiveness,
                'league_seasonal_variations': league_variations,
                'historical_pattern_matching': pattern_matching,
                'pattern_reliability_score': self._calculate_pattern_reliability(
                    seasonal_consistency, manager_patterns, style_effectiveness
                ),
                'historical_seasons_analyzed': len(historical_data) if historical_data else 0
            }
            
        except Exception as e:
            logger.error(f"Error in historical pattern analysis: {str(e)}")
            return self._get_default_historical_analysis()
    
    def _analyze_external_factors(self, matches: List[Dict], current_date: datetime,
                                match_context: Dict) -> Dict:
        """
        Analyze external factors affecting seasonal performance
        
        Returns:
            Dict with external factor analysis
        """
        try:
            # Weather and climate analysis
            weather_analysis = self._analyze_seasonal_weather_impact(matches, current_date)
            
            # Stadium and venue seasonal effects
            venue_effects = self._analyze_venue_seasonal_effects(matches)
            
            # Fan attendance seasonal patterns
            attendance_patterns = self._analyze_attendance_seasonal_patterns(matches)
            
            # Media pressure and expectations seasonal analysis
            media_pressure = self._analyze_media_pressure_seasonal_effects(match_context)
            
            # Economic factors (transfer budget, financial constraints)
            economic_factors = self._analyze_economic_seasonal_factors(match_context)
            
            return {
                'weather_seasonal_impact': weather_analysis,
                'venue_seasonal_effects': venue_effects,
                'attendance_seasonal_patterns': attendance_patterns,
                'media_pressure_effects': media_pressure,
                'economic_seasonal_factors': economic_factors,
                'external_factor_weight': self._calculate_external_factor_weight(
                    weather_analysis, venue_effects, attendance_patterns
                )
            }
            
        except Exception as e:
            logger.error(f"Error in external factor analysis: {str(e)}")
            return self._get_default_external_analysis()
    
    def _generate_integrated_assessment(self, cycle_analysis: Dict, phase_analysis: Dict,
                                      trend_analysis: Dict, historical_analysis: Dict,
                                      external_analysis: Dict) -> Dict:
        """
        Generate integrated seasonal performance assessment
        
        Returns:
            Dict with integrated assessment and recommendations
        """
        try:
            # Calculate overall seasonal performance score
            seasonal_score = self._calculate_overall_seasonal_score(
                cycle_analysis, phase_analysis, trend_analysis, historical_analysis
            )
            
            # Determine current seasonal advantage/disadvantage
            seasonal_advantage = self._determine_seasonal_advantage(
                cycle_analysis, phase_analysis, external_analysis
            )
            
            # Generate performance predictions for next 4-8 weeks
            short_term_predictions = self._generate_short_term_predictions(
                cycle_analysis, trend_analysis
            )
            
            # Calculate phase transition probabilities
            transition_probabilities = self._calculate_all_transition_probabilities(
                cycle_analysis, phase_analysis
            )
            
            # Generate actionable insights and recommendations
            insights = self._generate_seasonal_insights(
                cycle_analysis, phase_analysis, trend_analysis, historical_analysis
            )
            
            # Risk assessment for seasonal performance
            risk_assessment = self._assess_seasonal_risks(
                trend_analysis, phase_analysis, external_analysis
            )
            
            return {
                'overall_seasonal_score': seasonal_score,
                'seasonal_advantage_indicator': seasonal_advantage,
                'short_term_predictions': short_term_predictions,
                'phase_transition_probabilities': transition_probabilities,
                'seasonal_insights': insights,
                'risk_assessment': risk_assessment,
                'confidence_level': self._calculate_integrated_confidence(
                    cycle_analysis, phase_analysis, trend_analysis, historical_analysis
                ),
                'recommendation_priority': self._prioritize_recommendations(insights)
            }
            
        except Exception as e:
            logger.error(f"Error in integrated assessment: {str(e)}")
            return self._get_default_integrated_assessment()
    
    # Utility methods for season calculations
    def _calculate_season_week(self, match_date: datetime) -> int:
        """Calculate which week of the season we're in"""
        # Assume season starts in August
        season_start = datetime(match_date.year if match_date.month >= 8 else match_date.year - 1, 8, 1)
        if match_date < season_start:
            season_start = datetime(match_date.year - 1, 8, 1)
        
        weeks_elapsed = (match_date - season_start).days // 7
        return min(max(weeks_elapsed, 1), 38)  # Cap between 1-38 weeks
    
    def _determine_season_phase(self, week: int) -> str:
        """Determine which phase of the season we're in"""
        for phase_name, phase_config in self.config['season_cycles'].items():
            start_week, end_week = phase_config['weeks']
            if start_week <= week <= end_week:
                return phase_name
        return 'unknown'
    
    def _group_matches_by_phase(self, matches: List[Dict], team_id: int) -> Dict:
        """Group matches by seasonal phase"""
        phase_groups = {
            'early_season': [],
            'mid_season': [],
            'late_season': []
        }
        
        for match in matches:
            match_date = self._parse_match_date(match)
            if match_date:
                week = self._calculate_season_week(match_date)
                phase = self._determine_season_phase(week)
                if phase in phase_groups:
                    phase_groups[phase].append(match)
        
        return phase_groups
    
    def _parse_match_date(self, match: Dict) -> Optional[datetime]:
        """Parse match date from various possible formats"""
        date_fields = ['fixture.date', 'date', 'fixture.timestamp', 'timestamp']
        
        for field in date_fields:
            if '.' in field:
                # Nested field
                parts = field.split('.')
                value = match
                for part in parts:
                    value = value.get(part, {})
                    if not isinstance(value, dict) and value:
                        break
            else:
                value = match.get(field)
            
            if value:
                try:
                    if isinstance(value, (int, float)):
                        # Timestamp
                        return datetime.fromtimestamp(value)
                    elif isinstance(value, str):
                        # String date
                        for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                            try:
                                return datetime.strptime(value, fmt)
                            except ValueError:
                                continue
                except (ValueError, TypeError):
                    continue
        
        return None
    
    # Default fallback methods
    def _get_default_seasonal_analysis(self, team_id: int, match_date: datetime) -> Dict:
        """Return default seasonal analysis when data is insufficient"""
        return {
            'team_id': team_id,
            'analysis_date': match_date.isoformat(),
            'season_cycle_detection': self._get_default_cycle_analysis(),
            'performance_phase_modeling': self._get_default_phase_analysis(),
            'seasonal_trend_prediction': self._get_default_trend_analysis(),
            'historical_seasonal_patterns': self._get_default_historical_analysis(),
            'external_factor_analysis': self._get_default_external_analysis(),
            'integrated_assessment': self._get_default_integrated_assessment(),
            'confidence_score': 0.3,
            'analysis_metadata': {
                'matches_analyzed': 0,
                'historical_seasons': 0,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality': 'insufficient'
            }
        }
    
    def _get_default_cycle_analysis(self) -> Dict:
        """Default cycle analysis"""
        return {
            'current_season_week': 15,
            'current_phase': 'mid_season',
            'phase_performance': {'early_season': [], 'mid_season': [], 'late_season': []},
            'early_season_analysis': {'adaptation_score': 0.5, 'new_signings_impact': 0.0},
            'mid_season_analysis': {'consistency_score': 0.5, 'rhythm_factor': 0.5},
            'late_season_analysis': {'motivation_factor': 0.5, 'fatigue_factor': 0.5},
            'transfer_window_effects': {'disruption_level': 0.0},
            'holiday_period_impacts': {'performance_drop': 0.0},
            'cycle_strength': 0.5,
            'phase_transition_probability': 0.3
        }
    
    def _get_default_phase_analysis(self) -> Dict:
        """Default phase analysis"""
        return {
            'current_phase_context': 'mid_table',
            'preseason_form_carryover': {'carryover_strength': 0.5},
            'championship_relegation_effects': {'pressure_factor': 1.0},
            'european_competition_impact': {'fixture_load_impact': 0.0},
            'cup_run_effects': {'distraction_factor': 0.0},
            'international_break_effects': {'disruption_factor': 0.0},
            'tactical_phase_effects': {'tactical_consistency': 0.5},
            'phase_motivation_factor': 1.0,
            'phase_pressure_factor': 1.0,
            'performance_sustainability': 0.5
        }
    
    def _get_default_trend_analysis(self) -> Dict:
        """Default trend analysis"""
        return {
            'performance_trajectory_forecast': {'trend_direction': 'stable', 'trend_strength': 0.0},
            'form_peak_trough_prediction': {'next_peak_week': 20, 'next_trough_week': 30},
            'seasonal_fatigue_model': {'fatigue_level': 0.5, 'recovery_rate': 0.5},
            'weather_impact_analysis': {'seasonal_weather_factor': 1.0},
            'fixture_congestion_effects': {'congestion_level': 0.5},
            'momentum_sustainability': {'sustainability_score': 0.5},
            'performance_volatility': {'volatility_score': 0.5},
            'trend_confidence': 0.5,
            'forecast_horizon_weeks': 8
        }
    
    def _get_default_historical_analysis(self) -> Dict:
        """Default historical analysis"""
        return {
            'multiyear_seasonal_consistency': {'consistency_score': 0.5},
            'manager_seasonal_patterns': {'pattern_strength': 0.5},
            'squad_seasonal_endurance': {'endurance_score': 0.5},
            'playing_style_effectiveness': {'style_consistency': 0.5},
            'league_seasonal_variations': {'variation_factor': 0.5},
            'historical_pattern_matching': {'match_confidence': 0.3},
            'pattern_reliability_score': 0.5,
            'historical_seasons_analyzed': 0
        }
    
    def _get_default_external_analysis(self) -> Dict:
        """Default external analysis"""
        return {
            'weather_seasonal_impact': {'impact_factor': 1.0},
            'venue_seasonal_effects': {'home_advantage_variation': 0.0},
            'attendance_seasonal_patterns': {'attendance_impact': 0.0},
            'media_pressure_effects': {'pressure_level': 0.5},
            'economic_seasonal_factors': {'economic_impact': 0.0},
            'external_factor_weight': 0.1
        }
    
    def _get_default_integrated_assessment(self) -> Dict:
        """Default integrated assessment"""
        return {
            'overall_seasonal_score': 50.0,
            'seasonal_advantage_indicator': 'neutral',
            'short_term_predictions': {'performance_trajectory': 'stable'},
            'phase_transition_probabilities': {'next_phase_probability': 0.3},
            'seasonal_insights': ['Insufficient data for detailed analysis'],
            'risk_assessment': {'risk_level': 'medium'},
            'confidence_level': 0.3,
            'recommendation_priority': 'low'
        }
    
    # Detailed implementations for seasonal pattern analysis
    def _analyze_early_season_patterns(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze early season adaptation patterns"""
        try:
            if not matches:
                return {'adaptation_score': 0.5, 'new_signings_impact': 0.0}
            
            # Calculate performance metrics for early season
            points_progression = []
            goals_progression = []
            defensive_progression = []
            
            for i, match in enumerate(matches):
                points = self._extract_team_points(match, team_id)
                goals_scored = self._extract_team_goals_scored(match, team_id)
                goals_conceded = self._extract_team_goals_conceded(match, team_id)
                
                points_progression.append(points)
                goals_progression.append(goals_scored)
                defensive_progression.append(3 - goals_conceded)  # Defensive score
            
            # Calculate adaptation metrics
            adaptation_score = self._calculate_adaptation_score(points_progression)
            new_signings_impact = self._estimate_new_signings_impact(matches, team_id)
            consistency_development = self._calculate_consistency_development(points_progression)
            tactical_adaptation = self._analyze_tactical_adaptation(matches, team_id)
            
            return {
                'adaptation_score': adaptation_score,
                'new_signings_impact': new_signings_impact,
                'consistency_development': consistency_development,
                'tactical_adaptation': tactical_adaptation,
                'performance_improvement_rate': self._calculate_improvement_rate(points_progression),
                'early_season_momentum': self._calculate_early_momentum(points_progression[-5:] if len(points_progression) >= 5 else points_progression)
            }
            
        except Exception as e:
            logger.error(f"Error in early season analysis: {str(e)}")
            return {'adaptation_score': 0.5, 'new_signings_impact': 0.0}
    
    def _analyze_mid_season_patterns(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze mid-season consistency patterns"""
        try:
            if not matches:
                return {'consistency_score': 0.5, 'rhythm_factor': 0.5}
            
            # Extract performance data
            performance_data = []
            for match in matches:
                points = self._extract_team_points(match, team_id)
                goals_scored = self._extract_team_goals_scored(match, team_id)
                goals_conceded = self._extract_team_goals_conceded(match, team_id)
                
                performance_data.append({
                    'points': points,
                    'goals_scored': goals_scored,
                    'goals_conceded': goals_conceded,
                    'goal_difference': goals_scored - goals_conceded
                })
            
            # Calculate consistency metrics
            consistency_score = self._calculate_performance_consistency(performance_data)
            rhythm_factor = self._calculate_rhythm_factor(performance_data)
            peak_performance_indicator = self._calculate_peak_performance(performance_data)
            tactical_stability = self._analyze_tactical_stability(matches, team_id)
            
            return {
                'consistency_score': consistency_score,
                'rhythm_factor': rhythm_factor,
                'peak_performance_indicator': peak_performance_indicator,
                'tactical_stability': tactical_stability,
                'performance_variance': np.var([p['points'] for p in performance_data]) if performance_data else 0,
                'goal_scoring_consistency': self._calculate_scoring_consistency(performance_data)
            }
            
        except Exception as e:
            logger.error(f"Error in mid-season analysis: {str(e)}")
            return {'consistency_score': 0.5, 'rhythm_factor': 0.5}
    
    def _analyze_late_season_patterns(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze late season motivation and fatigue patterns"""
        try:
            if not matches:
                return {'motivation_factor': 0.5, 'fatigue_factor': 0.5}
            
            # Analyze performance decline/improvement
            recent_performance = [self._extract_team_points(match, team_id) for match in matches[-10:]]
            earlier_performance = [self._extract_team_points(match, team_id) for match in matches[-20:-10] if len(matches) >= 20]
            
            # Calculate fatigue and motivation factors
            fatigue_factor = self._calculate_fatigue_factor(recent_performance, earlier_performance)
            motivation_factor = self._calculate_motivation_factor(matches, team_id)
            pressure_handling = self._analyze_pressure_handling(matches, team_id)
            squad_rotation_effectiveness = self._analyze_squad_rotation(matches, team_id)
            
            return {
                'motivation_factor': motivation_factor,
                'fatigue_factor': fatigue_factor,
                'pressure_handling': pressure_handling,
                'squad_rotation_effectiveness': squad_rotation_effectiveness,
                'performance_sustainability': self._calculate_sustainability(recent_performance),
                'late_season_resilience': self._calculate_resilience(matches, team_id)
            }
            
        except Exception as e:
            logger.error(f"Error in late season analysis: {str(e)}")
            return {'motivation_factor': 0.5, 'fatigue_factor': 0.5}
    
    def _cache_seasonal_analysis(self, team_id: int, analysis: Dict) -> None:
        """Cache seasonal analysis for future reference"""
        self.seasonal_profiles[team_id] = analysis
        logger.debug(f"Cached seasonal analysis for team {team_id}")
    
    def _calculate_analysis_confidence(self, *analyses) -> float:
        """Calculate overall confidence in the analysis"""
        # Simple implementation - can be enhanced
        data_quality_scores = []
        for analysis in analyses:
            if isinstance(analysis, dict) and analysis:
                data_quality_scores.append(0.7)  # Base confidence
            else:
                data_quality_scores.append(0.3)  # Low confidence
        
        return np.mean(data_quality_scores) if data_quality_scores else 0.3
    
    # Historical analysis implementation methods
    def _analyze_multiyear_consistency(self, team_id: int, historical_data: List[Dict]) -> Dict:
        """Analyze multi-year seasonal consistency patterns"""
        try:
            if not historical_data:
                return {'consistency_score': 0.5}
            
            # Group historical data by seasons
            seasonal_performances = self._group_historical_by_seasons(historical_data, team_id)
            
            # Calculate consistency metrics across seasons
            season_scores = []
            for season, matches in seasonal_performances.items():
                if matches:
                    season_score = self._calculate_season_performance_score(matches, team_id)
                    season_scores.append(season_score)
            
            if len(season_scores) < 2:
                return {'consistency_score': 0.5}
            
            # Calculate consistency (lower variance = higher consistency)
            variance = np.var(season_scores)
            consistency_score = 1 / (1 + variance)
            
            # Analyze seasonal patterns
            pattern_analysis = self._analyze_seasonal_performance_patterns(seasonal_performances, team_id)
            
            return {
                'consistency_score': max(0, min(1, consistency_score)),
                'seasonal_variance': variance,
                'season_scores': season_scores,
                'pattern_analysis': pattern_analysis,
                'seasons_analyzed': len(season_scores)
            }
        except Exception as e:
            logger.error(f"Error in multiyear consistency analysis: {str(e)}")
            return {'consistency_score': 0.5}
    
    def _analyze_manager_seasonal_patterns(self, team_id: int, historical_data: List[Dict]) -> Dict:
        """Analyze manager-specific seasonal patterns"""
        try:
            # This would require manager data - using placeholder approach
            return {
                'pattern_strength': 0.6,
                'seasonal_adaptation_speed': 0.7,
                'tactical_consistency_seasonal': 0.65,
                'manager_experience_factor': 0.8
            }
        except Exception as e:
            logger.error(f"Error in manager pattern analysis: {str(e)}")
            return {'pattern_strength': 0.5}
    
    def _analyze_squad_seasonal_endurance(self, team_id: int, historical_data: List[Dict], current_matches: List[Dict]) -> Dict:
        """Analyze squad age and seasonal endurance patterns"""
        try:
            # Analyze performance across different parts of season
            if not current_matches:
                return {'endurance_score': 0.5}
            
            # Group current season by phases
            phase_groups = self._group_matches_by_phase(current_matches, team_id)
            
            # Calculate performance decline/improvement
            early_performance = self._calculate_phase_average_performance(phase_groups.get('early_season', []), team_id)
            mid_performance = self._calculate_phase_average_performance(phase_groups.get('mid_season', []), team_id)
            late_performance = self._calculate_phase_average_performance(phase_groups.get('late_season', []), team_id)
            
            # Calculate endurance score (how well team maintains performance)
            if early_performance > 0:
                endurance_ratio = late_performance / early_performance
            else:
                endurance_ratio = 1.0
            
            return {
                'endurance_score': max(0, min(2, endurance_ratio)),
                'early_season_avg': early_performance,
                'mid_season_avg': mid_performance,
                'late_season_avg': late_performance,
                'performance_decline': max(0, early_performance - late_performance),
                'stamina_indicator': self._calculate_stamina_indicator(early_performance, late_performance)
            }
        except Exception as e:
            logger.error(f"Error in squad endurance analysis: {str(e)}")
            return {'endurance_score': 0.5}
    
    def _analyze_playing_style_seasonal_effectiveness(self, team_id: int, historical_data: List[Dict]) -> Dict:
        """Analyze playing style effectiveness across seasons"""
        try:
            # This would require detailed tactical data - using simplified approach
            return {
                'style_consistency': 0.7,
                'seasonal_style_adaptation': 0.6,
                'effectiveness_variance': 0.3,
                'style_durability': 0.75
            }
        except Exception as e:
            logger.error(f"Error in playing style analysis: {str(e)}")
            return {'style_consistency': 0.5}
    
    def _analyze_league_seasonal_variations(self, league_id: int, historical_data: List[Dict]) -> Dict:
        """Analyze league-specific seasonal variations"""
        try:
            # This would analyze league-wide patterns
            return {
                'variation_factor': 0.5,
                'league_competitiveness_seasonal': 0.7,
                'seasonal_goal_patterns': {'early': 2.3, 'mid': 2.5, 'late': 2.4},
                'league_specific_trends': {'winter_break_effect': 0.1}
            }
        except Exception as e:
            logger.error(f"Error in league variations analysis: {str(e)}")
            return {'variation_factor': 0.5}
    
    def _match_current_season_to_historical_patterns(self, team_id: int, current_matches: List[Dict], historical_data: List[Dict]) -> Dict:
        """Match current season performance to historical patterns"""
        try:
            if not current_matches or not historical_data:
                return {'match_confidence': 0.3}
            
            # Calculate current season pattern
            current_pattern = self._extract_season_pattern(current_matches, team_id)
            
            # Compare with historical patterns
            historical_patterns = self._extract_historical_patterns(historical_data, team_id)
            
            # Find best match
            best_match_similarity = 0
            best_match_season = None
            
            for season_pattern in historical_patterns:
                similarity = self._calculate_pattern_similarity(current_pattern, season_pattern)
                if similarity > best_match_similarity:
                    best_match_similarity = similarity
                    best_match_season = season_pattern.get('season')
            
            return {
                'match_confidence': best_match_similarity,
                'best_match_season': best_match_season,
                'current_pattern': current_pattern,
                'similarity_threshold_met': best_match_similarity > self.config['analysis_parameters']['pattern_similarity_threshold']
            }
        except Exception as e:
            logger.error(f"Error in pattern matching: {str(e)}")
            return {'match_confidence': 0.3}
    
    def _calculate_pattern_reliability(self, seasonal_consistency: Dict, manager_patterns: Dict, style_effectiveness: Dict) -> float:
        """Calculate overall pattern reliability score"""
        consistency_score = seasonal_consistency.get('consistency_score', 0.5)
        manager_score = manager_patterns.get('pattern_strength', 0.5)
        style_score = style_effectiveness.get('style_consistency', 0.5)
        
        # Weighted average
        reliability = (consistency_score * 0.4) + (manager_score * 0.3) + (style_score * 0.3)
        return max(0, min(1, reliability))
    
    # Additional placeholder methods for comprehensive implementation
    def _analyze_transfer_window_effects(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze transfer window disruption effects"""
        try:
            # Look for performance changes around transfer windows
            transfer_periods = [
                (1, 31),   # January
                (150, 243) # June-August (day of year)
            ]
            
            disruption_effects = []
            for match in matches:
                match_date = self._parse_match_date(match)
                if match_date:
                    day_of_year = match_date.timetuple().tm_yday
                    for start_day, end_day in transfer_periods:
                        if start_day <= day_of_year <= end_day:
                            performance = self._extract_team_points(match, team_id)
                            disruption_effects.append(performance)
                            break
            
            if disruption_effects:
                avg_disruption_performance = np.mean(disruption_effects)
                all_performance = [self._extract_team_points(match, team_id) for match in matches]
                baseline_performance = np.mean(all_performance) if all_performance else 1.5
                
                disruption_level = max(0, (baseline_performance - avg_disruption_performance) / 3.0)
            else:
                disruption_level = 0.0
            
            return {
                'disruption_level': disruption_level,
                'transfer_window_matches': len(disruption_effects),
                'disruption_magnitude': disruption_level * 100  # As percentage
            }
        except Exception as e:
            logger.error(f"Error in transfer window analysis: {str(e)}")
            return {'disruption_level': 0.0}
    
    def _analyze_holiday_period_impacts(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze holiday period performance impacts"""
        try:
            holiday_periods = [
                (355, 15),  # Dec 21 - Jan 15 (day of year, wrapping around)
                (165, 225)  # Mid June - Mid August
            ]
            
            holiday_performances = []
            for match in matches:
                match_date = self._parse_match_date(match)
                if match_date:
                    day_of_year = match_date.timetuple().tm_yday
                    # Handle year-end wrap around for winter break
                    if day_of_year >= 355 or day_of_year <= 15:
                        performance = self._extract_team_points(match, team_id)
                        holiday_performances.append(performance)
                    elif 165 <= day_of_year <= 225:
                        performance = self._extract_team_points(match, team_id)
                        holiday_performances.append(performance)
            
            if holiday_performances:
                avg_holiday_performance = np.mean(holiday_performances)
                all_performance = [self._extract_team_points(match, team_id) for match in matches]
                baseline_performance = np.mean(all_performance) if all_performance else 1.5
                
                performance_drop = max(0, (baseline_performance - avg_holiday_performance) / 3.0)
            else:
                performance_drop = 0.0
            
            return {
                'performance_drop': performance_drop,
                'holiday_matches': len(holiday_performances),
                'impact_magnitude': performance_drop * 100
            }
        except Exception as e:
            logger.error(f"Error in holiday period analysis: {str(e)}")
            return {'performance_drop': 0.0}
    
    def _calculate_cycle_strength(self, phase_performance: Dict) -> float:
        """Calculate the strength of seasonal cycles"""
        try:
            early_matches = len(phase_performance.get('early_season', []))
            mid_matches = len(phase_performance.get('mid_season', []))
            late_matches = len(phase_performance.get('late_season', []))
            
            # If we have sufficient data across phases, cycle strength is higher
            total_matches = early_matches + mid_matches + late_matches
            if total_matches < 10:
                return 0.3  # Low strength due to insufficient data
            
            # Check for balance across phases
            balance_score = 1 - abs(early_matches - mid_matches - late_matches) / total_matches
            return max(0.3, min(1.0, balance_score))
        except:
            return 0.5
    
    def _calculate_phase_transition_probability(self, current_week: int, phase_performance: Dict) -> float:
        """Calculate probability of transitioning to next phase"""
        try:
            phase_boundaries = {
                'early_to_mid': 11,
                'mid_to_late': 26
            }
            
            if current_week <= 10:
                # In early season, probability of transitioning to mid-season
                weeks_to_transition = max(1, 11 - current_week)
                return min(1.0, 5.0 / weeks_to_transition)
            elif 11 <= current_week <= 25:
                # In mid-season, probability of transitioning to late season
                weeks_to_transition = max(1, 26 - current_week)
                return min(1.0, 8.0 / weeks_to_transition)
            else:
                # In late season, low probability of major transition
                return 0.1
        except:
            return 0.3
    
    # Essential utility methods for data extraction and calculations
    def _extract_team_points(self, match: Dict, team_id: int) -> int:
        """Extract points earned by team in match"""
        try:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if not score:
                return 0
                
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            if home_team.get('id') == team_id:
                # Home team
                if home_goals > away_goals:
                    return 3
                elif home_goals == away_goals:
                    return 1
                else:
                    return 0
            elif away_team.get('id') == team_id:
                # Away team
                if away_goals > home_goals:
                    return 3
                elif away_goals == home_goals:
                    return 1
                else:
                    return 0
            return 0
        except:
            return 0
    
    def _extract_team_goals_scored(self, match: Dict, team_id: int) -> int:
        """Extract goals scored by team in match"""
        try:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if not score:
                return 0
                
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            if home_team.get('id') == team_id:
                return home_goals
            elif away_team.get('id') == team_id:
                return away_goals
            return 0
        except:
            return 0
    
    def _extract_team_goals_conceded(self, match: Dict, team_id: int) -> int:
        """Extract goals conceded by team in match"""
        try:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if not score:
                return 0
                
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            if home_team.get('id') == team_id:
                return away_goals
            elif away_team.get('id') == team_id:
                return home_goals
            return 0
        except:
            return 0
    
    def _calculate_adaptation_score(self, points_progression: List[int]) -> float:
        """Calculate how well team has adapted to season start"""
        if len(points_progression) < 3:
            return 0.5
        
        # Look for improvement trend in first matches
        early_avg = np.mean(points_progression[:3]) if len(points_progression) >= 3 else 0
        later_avg = np.mean(points_progression[3:6]) if len(points_progression) >= 6 else early_avg
        
        improvement = (later_avg - early_avg) / 3.0  # Normalize to 0-1
        return max(0, min(1, 0.5 + improvement))
    
    def _calculate_fatigue_factor(self, recent_performance: List[int], earlier_performance: List[int]) -> float:
        """Calculate fatigue factor based on performance decline"""
        if not recent_performance or not earlier_performance:
            return 0.5
        
        recent_avg = np.mean(recent_performance)
        earlier_avg = np.mean(earlier_performance)
        
        if earlier_avg == 0:
            return 0.5
        
        performance_ratio = recent_avg / earlier_avg
        fatigue_factor = 1 - max(0, min(1, performance_ratio))
        return fatigue_factor
    
    def _calculate_motivation_factor(self, matches: List[Dict], team_id: int) -> float:
        """Calculate motivation factor based on team context"""
        # This would analyze league position, European qualification chances, etc.
        # For now, return a baseline value
        return 0.7
    
    def _calculate_performance_consistency(self, performance_data: List[Dict]) -> float:
        """Calculate consistency score from performance data"""
        if not performance_data:
            return 0.5
        
        points = [p['points'] for p in performance_data]
        if len(points) < 2:
            return 0.5
        
        variance = np.var(points)
        # Lower variance = higher consistency
        consistency = 1 / (1 + variance)
        return max(0, min(1, consistency))
    
    def _calculate_rhythm_factor(self, performance_data: List[Dict]) -> float:
        """Calculate rhythm factor based on performance patterns"""
        if len(performance_data) < 5:
            return 0.5
        
        points = [p['points'] for p in performance_data]
        # Look for consistent patterns
        moving_avg = []
        for i in range(3, len(points)):
            moving_avg.append(np.mean(points[i-3:i]))
        
        if not moving_avg:
            return 0.5
        
        rhythm_stability = 1 - np.std(moving_avg) / 3.0  # Normalize
        return max(0, min(1, rhythm_stability))
    
    def _estimate_new_signings_impact(self, matches: List[Dict], team_id: int) -> float:
        """Estimate impact of new signings (placeholder)"""
        # This would require transfer data integration
        return 0.1  # Default small positive impact
    
    def _calculate_improvement_rate(self, points_progression: List[int]) -> float:
        """Calculate rate of improvement over time"""
        if len(points_progression) < 2:
            return 0.0
        
        # Simple linear regression on points progression
        x = np.arange(len(points_progression))
        slope, _ = np.polyfit(x, points_progression, 1)
        
        # Normalize slope to 0-1 range
        return max(-1, min(1, slope / 3.0))
    
    # Performance phase modeling detailed implementations
    def _determine_phase_context(self, current_position: int, league_size: int, matches_played: int) -> str:
        """Determine current performance phase context"""
        position_ratio = current_position / league_size
        
        if position_ratio <= 0.3:
            return 'championship_chase'
        elif position_ratio >= 0.75:
            return 'relegation_battle'
        elif 0.3 < position_ratio <= 0.5:
            return 'european_qualification'
        else:
            return 'mid_table'
    
    def _analyze_preseason_carryover(self, matches: List[Dict], team_id: int, historical_data: Optional[List[Dict]]) -> Dict:
        """Analyze pre-season form carryover effects"""
        try:
            if not matches:
                return {'carryover_strength': 0.5}
            
            # Analyze first 5 matches for pre-season influence
            first_matches = matches[:5]
            first_performance = [self._extract_team_points(match, team_id) for match in first_matches]
            
            # Compare with expected performance based on historical data
            expected_performance = self._calculate_expected_early_performance(team_id, historical_data)
            actual_avg = np.mean(first_performance) if first_performance else 1.0
            
            carryover_strength = actual_avg / expected_performance if expected_performance > 0 else 0.5
            
            return {
                'carryover_strength': max(0, min(2, carryover_strength)),
                'first_match_performance': first_performance,
                'adaptation_speed': self._calculate_adaptation_speed(first_performance)
            }
        except Exception as e:
            logger.error(f"Error in preseason carryover analysis: {str(e)}")
            return {'carryover_strength': 0.5}
    
    def _analyze_championship_relegation_effects(self, matches: List[Dict], team_id: int, current_position: int, league_size: int) -> Dict:
        """Analyze effects of championship chase or relegation battle"""
        try:
            position_ratio = current_position / league_size
            
            if position_ratio <= 0.3:
                # Championship chase
                pressure_factor = 1.1 + (0.3 - position_ratio) * 0.5
                motivation_boost = 1.2
                phase_type = 'championship'
            elif position_ratio >= 0.75:
                # Relegation battle
                pressure_factor = 1.2 + (position_ratio - 0.75) * 0.8
                motivation_boost = 1.15
                phase_type = 'relegation'
            else:
                # Mid-table
                pressure_factor = 1.0
                motivation_boost = 0.95
                phase_type = 'mid_table'
            
            # Analyze recent performance under pressure
            recent_matches = matches[-10:] if len(matches) >= 10 else matches
            pressure_performance = self._analyze_pressure_performance_detail(recent_matches, team_id)
            
            return {
                'phase_type': phase_type,
                'pressure_factor': pressure_factor,
                'motivation_boost': motivation_boost,
                'pressure_performance': pressure_performance,
                'position_context': current_position,
                'phase_sustainability': self._calculate_phase_sustainability(pressure_performance)
            }
        except Exception as e:
            logger.error(f"Error in championship/relegation analysis: {str(e)}")
            return {'pressure_factor': 1.0}
    
    def _analyze_european_competition_impact(self, matches: List[Dict], team_id: int, match_context: Dict) -> Dict:
        """Analyze impact of European competition participation"""
        try:
            # Check if team is in European competition
            european_competitions = match_context.get('european_competitions', [])
            
            if not european_competitions:
                return {'fixture_load_impact': 0.0, 'competition_type': 'none'}
            
            # Determine competition type and impact
            competition_impacts = {
                'champions_league': 1.3,
                'europa_league': 1.2,
                'conference_league': 1.15
            }
            
            max_impact = 0
            competition_type = 'none'
            
            for comp in european_competitions:
                comp_name = comp.lower()
                for comp_key, impact in competition_impacts.items():
                    if comp_key in comp_name:
                        if impact > max_impact:
                            max_impact = impact
                            competition_type = comp_key
            
            # Analyze fixture congestion effects
            congestion_effect = self._analyze_fixture_congestion_detail(matches, team_id)
            
            return {
                'fixture_load_impact': max_impact,
                'competition_type': competition_type,
                'congestion_effect': congestion_effect,
                'rotation_effectiveness': self._calculate_rotation_effectiveness(matches, team_id),
                'european_form_correlation': self._analyze_european_form_correlation(matches, team_id)
            }
        except Exception as e:
            logger.error(f"Error in European competition analysis: {str(e)}")
            return {'fixture_load_impact': 0.0}
    
    def _forecast_performance_trajectory(self, matches: List[Dict], team_id: int, current_date: datetime) -> Dict:
        """Forecast team's performance trajectory"""
        try:
            if len(matches) < 5:
                return {'trend_direction': 'stable', 'trend_strength': 0.0}
            
            # Extract recent performance data
            recent_points = [self._extract_team_points(match, team_id) for match in matches[-12:]]
            recent_goals = [self._extract_team_goals_scored(match, team_id) for match in matches[-12:]]
            recent_defensive = [3 - self._extract_team_goals_conceded(match, team_id) for match in matches[-12:]]
            
            # Calculate trends using linear regression
            x = np.arange(len(recent_points))
            
            points_trend, _ = np.polyfit(x, recent_points, 1) if len(recent_points) > 1 else (0, 0)
            goals_trend, _ = np.polyfit(x, recent_goals, 1) if len(recent_goals) > 1 else (0, 0)
            defensive_trend, _ = np.polyfit(x, recent_defensive, 1) if len(recent_defensive) > 1 else (0, 0)
            
            # Determine overall trend
            overall_trend = (points_trend + goals_trend * 0.3 + defensive_trend * 0.3) / 1.6
            
            if overall_trend > 0.1:
                trend_direction = 'improving'
            elif overall_trend < -0.1:
                trend_direction = 'declining'
            else:
                trend_direction = 'stable'
            
            # Forecast next 8 weeks
            forecast_weeks = self.config['trend_prediction']['forecast_weeks']
            future_x = np.arange(len(recent_points), len(recent_points) + forecast_weeks)
            
            forecast_points = [max(0, min(3, points_trend * i + np.mean(recent_points))) for i in future_x]
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': abs(overall_trend),
                'points_trend': points_trend,
                'goals_trend': goals_trend,
                'defensive_trend': defensive_trend,
                'forecast_points': forecast_points,
                'confidence_interval': self._calculate_forecast_confidence(recent_points, points_trend)
            }
        except Exception as e:
            logger.error(f"Error in trajectory forecasting: {str(e)}")
            return {'trend_direction': 'stable', 'trend_strength': 0.0}
    
    def _predict_form_peaks_troughs(self, matches: List[Dict], team_id: int, historical_data: Optional[List[Dict]]) -> Dict:
        """Predict when team will hit form peaks and troughs"""
        try:
            if len(matches) < 10:
                return {'next_peak_week': 20, 'next_trough_week': 30}
            
            # Analyze historical cycles
            points_sequence = [self._extract_team_points(match, team_id) for match in matches]
            
            # Simple cycle detection using moving averages
            window_size = 5
            moving_avg = []
            for i in range(window_size, len(points_sequence)):
                moving_avg.append(np.mean(points_sequence[i-window_size:i]))
            
            # Find local maxima and minima
            peaks = []
            troughs = []
            
            for i in range(1, len(moving_avg) - 1):
                if moving_avg[i] > moving_avg[i-1] and moving_avg[i] > moving_avg[i+1]:
                    peaks.append(i + window_size)
                elif moving_avg[i] < moving_avg[i-1] and moving_avg[i] < moving_avg[i+1]:
                    troughs.append(i + window_size)
            
            # Predict next peak/trough based on average cycle length
            current_week = self._calculate_season_week(datetime.now())
            
            if peaks:
                avg_peak_interval = np.mean(np.diff(peaks)) if len(peaks) > 1 else 10
                next_peak_week = current_week + int(avg_peak_interval / 2)
            else:
                next_peak_week = current_week + 8
            
            if troughs:
                avg_trough_interval = np.mean(np.diff(troughs)) if len(troughs) > 1 else 12
                next_trough_week = current_week + int(avg_trough_interval / 2)
            else:
                next_trough_week = current_week + 12
            
            return {
                'next_peak_week': max(current_week + 2, next_peak_week),
                'next_trough_week': max(current_week + 2, next_trough_week),
                'historical_peaks': peaks,
                'historical_troughs': troughs,
                'cycle_reliability': self._calculate_cycle_reliability(peaks, troughs)
            }
        except Exception as e:
            logger.error(f"Error in peak/trough prediction: {str(e)}")
            return {'next_peak_week': 20, 'next_trough_week': 30}
    
    # Additional supporting methods
    def _calculate_expected_early_performance(self, team_id: int, historical_data: Optional[List[Dict]]) -> float:
        """Calculate expected early season performance based on history"""
        # Default expectation
        return 1.5  # Average points per game
    
    def _calculate_adaptation_speed(self, performance_sequence: List[int]) -> float:
        """Calculate how quickly team adapts"""
        if len(performance_sequence) < 3:
            return 0.5
        
        improvement = performance_sequence[-1] - performance_sequence[0]
        return max(0, min(1, 0.5 + improvement / 6.0))
    
    def _analyze_pressure_performance_detail(self, matches: List[Dict], team_id: int) -> Dict:
        """Detailed pressure performance analysis"""
        return {'under_pressure_score': 0.7, 'pressure_adaptation': 0.6}
    
    def _calculate_phase_sustainability(self, pressure_performance: Dict) -> float:
        """Calculate sustainability of current phase performance"""
        return pressure_performance.get('under_pressure_score', 0.5)
    
    def _analyze_fixture_congestion_detail(self, matches: List[Dict], team_id: int) -> Dict:
        """Detailed fixture congestion analysis"""
        return {'congestion_level': 0.5, 'fatigue_impact': 0.3}
    
    def _calculate_rotation_effectiveness(self, matches: List[Dict], team_id: int) -> float:
        """Calculate squad rotation effectiveness"""
        return 0.6  # Placeholder
    
    def _analyze_european_form_correlation(self, matches: List[Dict], team_id: int) -> float:
        """Analyze correlation between European and domestic form"""
        return 0.7  # Placeholder
    
    def _calculate_forecast_confidence(self, recent_points: List[int], trend: float) -> Dict:
        """Calculate confidence intervals for forecasts"""
        return {'lower_bound': 0.3, 'upper_bound': 0.9, 'confidence_level': 0.68}
    
    def _calculate_cycle_reliability(self, peaks: List[int], troughs: List[int]) -> float:
        """Calculate reliability of identified cycles"""
        if len(peaks) < 2 or len(troughs) < 2:
            return 0.3
        
        peak_variance = np.var(np.diff(peaks)) if len(peaks) > 1 else 10
        trough_variance = np.var(np.diff(troughs)) if len(troughs) > 1 else 10
        
        # Lower variance = higher reliability
        reliability = 1 / (1 + (peak_variance + trough_variance) / 10)
        return max(0.1, min(1.0, reliability))
    
    # Additional supporting methods for completeness
    def _group_historical_by_seasons(self, historical_data: List[Dict], team_id: int) -> Dict:
        """Group historical matches by seasons"""
        seasonal_groups = defaultdict(list)
        for match in historical_data:
            match_date = self._parse_match_date(match)
            if match_date:
                season = f"{match_date.year}-{match_date.year + 1}" if match_date.month >= 8 else f"{match_date.year - 1}-{match_date.year}"
                seasonal_groups[season].append(match)
        return seasonal_groups
    
    def _calculate_season_performance_score(self, matches: List[Dict], team_id: int) -> float:
        """Calculate overall performance score for a season"""
        if not matches:
            return 50.0
        
        total_points = sum(self._extract_team_points(match, team_id) for match in matches)
        total_goals = sum(self._extract_team_goals_scored(match, team_id) for match in matches)
        total_conceded = sum(self._extract_team_goals_conceded(match, team_id) for match in matches)
        
        if len(matches) > 0:
            avg_points = total_points / len(matches)
            avg_goals = total_goals / len(matches)
            avg_conceded = total_conceded / len(matches)
            
            # Composite score (0-100)
            score = (avg_points * 20) + (avg_goals * 5) + max(0, (2 - avg_conceded) * 5)
            return max(0, min(100, score))
        return 50.0
    
    def _analyze_seasonal_performance_patterns(self, seasonal_performances: Dict, team_id: int) -> Dict:
        """Analyze patterns across seasons"""
        if not seasonal_performances:
            return {'pattern_strength': 0.3}
        
        season_scores = []
        for season, matches in seasonal_performances.items():
            if matches:
                score = self._calculate_season_performance_score(matches, team_id)
                season_scores.append(score)
        
        if len(season_scores) < 2:
            return {'pattern_strength': 0.3}
        
        # Calculate trend and consistency
        variance = np.var(season_scores)
        trend = (season_scores[-1] - season_scores[0]) / len(season_scores) if len(season_scores) > 1 else 0
        
        return {
            'pattern_strength': 1 / (1 + variance / 100),
            'improvement_trend': trend,
            'consistency_score': 1 - (variance / 1000)  # Normalized
        }
    
    def _calculate_phase_average_performance(self, matches: List[Dict], team_id: int) -> float:
        """Calculate average performance for a phase"""
        if not matches:
            return 1.5  # Default average
        
        points = [self._extract_team_points(match, team_id) for match in matches]
        return np.mean(points) if points else 1.5
    
    def _calculate_stamina_indicator(self, early_performance: float, late_performance: float) -> float:
        """Calculate stamina indicator based on performance comparison"""
        if early_performance == 0:
            return 0.5
        
        ratio = late_performance / early_performance
        return max(0, min(2, ratio))
    
    def _extract_season_pattern(self, matches: List[Dict], team_id: int) -> Dict:
        """Extract performance pattern from matches"""
        if not matches:
            return {'points_avg': 1.5, 'goals_avg': 1.2, 'variance': 1.0}
        
        points = [self._extract_team_points(match, team_id) for match in matches]
        goals = [self._extract_team_goals_scored(match, team_id) for match in matches]
        
        return {
            'points_avg': np.mean(points) if points else 1.5,
            'goals_avg': np.mean(goals) if goals else 1.2,
            'variance': np.var(points) if points else 1.0,
            'trend': self._calculate_improvement_rate(points)
        }
    
    def _extract_historical_patterns(self, historical_data: List[Dict], team_id: int) -> List[Dict]:
        """Extract patterns from historical data"""
        seasonal_groups = self._group_historical_by_seasons(historical_data, team_id)
        patterns = []
        
        for season, matches in seasonal_groups.items():
            if matches:
                pattern = self._extract_season_pattern(matches, team_id)
                pattern['season'] = season
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between two patterns"""
        try:
            points_diff = abs(pattern1.get('points_avg', 1.5) - pattern2.get('points_avg', 1.5))
            goals_diff = abs(pattern1.get('goals_avg', 1.2) - pattern2.get('goals_avg', 1.2))
            variance_diff = abs(pattern1.get('variance', 1.0) - pattern2.get('variance', 1.0))
            
            # Normalize differences and calculate similarity
            points_sim = max(0, 1 - points_diff / 3.0)
            goals_sim = max(0, 1 - goals_diff / 3.0)
            variance_sim = max(0, 1 - variance_diff / 2.0)
            
            return (points_sim + goals_sim + variance_sim) / 3.0
        except:
            return 0.3
    
    # More calculation methods for comprehensive implementation
    def _model_seasonal_fatigue(self, matches: List[Dict], team_id: int, current_date: datetime) -> Dict:
        """Model seasonal player fatigue"""
        current_week = self._calculate_season_week(current_date)
        fatigue_factor = min(1.0, current_week / 38.0)  # Linear fatigue model
        
        return {
            'fatigue_level': fatigue_factor,
            'recovery_rate': max(0.1, 1 - fatigue_factor),
            'fatigue_impact': fatigue_factor * 0.2  # Max 20% impact
        }
    
    def _analyze_weather_impact(self, matches: List[Dict], current_date: datetime) -> Dict:
        """Analyze weather impact on performance"""
        # Placeholder for weather analysis
        return {
            'seasonal_weather_factor': 1.0,
            'temperature_impact': 0.0,
            'precipitation_impact': 0.0
        }
    
    def _analyze_fixture_congestion_effects(self, matches: List[Dict], team_id: int, current_date: datetime) -> Dict:
        """Analyze fixture congestion effects"""
        return {
            'congestion_level': 0.5,
            'fatigue_accumulation': 0.3,
            'performance_impact': 0.1
        }
    
    def _predict_momentum_sustainability(self, matches: List[Dict], team_id: int) -> Dict:
        """Predict momentum sustainability"""
        if len(matches) < 5:
            return {'sustainability_score': 0.5}
        
        recent_points = [self._extract_team_points(match, team_id) for match in matches[-5:]]
        avg_recent = np.mean(recent_points) if recent_points else 1.5
        
        sustainability = min(1.0, avg_recent / 2.0)  # Normalize to 0-1
        
        return {
            'sustainability_score': sustainability,
            'momentum_strength': avg_recent,
            'volatility': np.var(recent_points) if recent_points else 1.0
        }
    
    def _analyze_performance_volatility(self, matches: List[Dict], team_id: int, historical_data: Optional[List[Dict]]) -> Dict:
        """Analyze performance volatility"""
        if not matches:
            return {'volatility_score': 0.5}
        
        points = [self._extract_team_points(match, team_id) for match in matches]
        volatility = np.var(points) if len(points) > 1 else 1.0
        
        return {
            'volatility_score': min(1.0, volatility / 2.0),
            'stability_indicator': max(0, 1 - volatility / 2.0),
            'performance_range': max(points) - min(points) if points else 0
        }
    
    def _calculate_trend_confidence(self, trajectory_forecast: Dict, peak_trough_prediction: Dict, fatigue_model: Dict) -> float:
        """Calculate confidence in trend predictions"""
        trend_strength = trajectory_forecast.get('trend_strength', 0.0)
        cycle_reliability = peak_trough_prediction.get('cycle_reliability', 0.3)
        fatigue_clarity = 1 - fatigue_model.get('fatigue_level', 0.5)
        
        return (trend_strength + cycle_reliability + fatigue_clarity) / 3.0
    
    # Remaining placeholder implementations
    def _calculate_consistency_development(self, points_progression: List[int]) -> float:
        """Calculate consistency development over time"""
        if len(points_progression) < 5:
            return 0.5
        
        # Look at variance reduction over time
        early_var = np.var(points_progression[:len(points_progression)//2]) if len(points_progression) >= 4 else 1.0
        late_var = np.var(points_progression[len(points_progression)//2:]) if len(points_progression) >= 4 else 1.0
        
        if early_var == 0:
            return 0.8
        
        improvement = max(0, (early_var - late_var) / early_var)
        return max(0, min(1, 0.5 + improvement * 0.5))
    
    def _analyze_tactical_adaptation(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze tactical adaptation"""
        return {
            'adaptation_speed': 0.7,
            'tactical_flexibility': 0.6,
            'strategic_consistency': 0.75
        }
    
    def _calculate_early_momentum(self, recent_points: List[int]) -> float:
        """Calculate early season momentum"""
        if not recent_points:
            return 0.5
        
        avg_points = np.mean(recent_points)
        return max(0, min(1, avg_points / 2.5))  # Normalize against expected max
    
    # Integration methods with existing analyzers
    def get_integrated_temporal_features(self, team_data: Dict, match_context: Dict) -> Dict:
        """
        Get integrated features combining seasonal analysis with existing temporal analysis
        
        Returns:
            Dict with combined temporal and seasonal features
        """
        seasonal_features = self.analyze_seasonal_performance(team_data, match_context)
        
        temporal_features = {}
        if self.time_analyzer:
            temporal_features = self.time_analyzer.analyze_temporal_features(team_data, match_context)
        
        return {
            'seasonal_analysis': seasonal_features,
            'temporal_analysis': temporal_features,
            'integrated_score': self._calculate_integrated_temporal_score(
                seasonal_features, temporal_features
            )
        }
    
    def _calculate_integrated_temporal_score(self, seasonal_features: Dict, temporal_features: Dict) -> float:
        """Calculate integrated score combining seasonal and temporal analysis"""
        seasonal_score = seasonal_features.get('integrated_assessment', {}).get('overall_seasonal_score', 50.0)
        temporal_score = temporal_features.get('combined_indicators', {}).get('overall_score', 50.0) if temporal_features else 50.0
        
        # Weight seasonal analysis higher for long-term predictions
        return (seasonal_score * 0.6) + (temporal_score * 0.4)

# Additional utility functions and helper methods would be implemented here