"""
Dynamic Time-Weighted Features Analyzer
Implements sophisticated temporal analysis for football predictions

Provides:
1. Exponential Decay Weighting (last 30 days)
2. Seasonal Form Curve Fitting
3. Weekly Performance Analysis
4. Temporal Pattern Recognition

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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
import math
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class DynamicTimeAnalyzer:
    """
    Advanced temporal analysis system for football predictions
    
    Implements time-weighted feature engineering with:
    - Exponential decay weighting
    - Seasonal form curves
    - Weekly performance patterns
    - Temporal pattern recognition
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Dynamic Time Analyzer
        
        Args:
            config: Configuration dictionary for customization
        """
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()
        
        # Internal state for pattern learning
        self.seasonal_patterns = {}
        self.weekly_patterns = {}
        self.temporal_clusters = {}
        self.league_decay_rates = {}
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        
        logger.info("DynamicTimeAnalyzer initialized with advanced temporal features")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for temporal analysis"""
        return {
            'decay_window_days': 30,
            'base_decay_rate': 0.95,
            'league_specific_decay': True,
            'seasonal_periods': {
                'season_start': (8, 10),    # August-October
                'mid_season': (11, 2),      # November-February  
                'season_end': (3, 5),       # March-May
                'transfer_windows': [(1, 1), (6, 9)],  # January, June-September
                'holiday_periods': [(12, 1), (7, 8)]   # Dec-Jan, July-August
            },
            'weekly_analysis': {
                'recovery_days': 3,
                'performance_weights': {
                    'weekend': 1.1,
                    'midweek': 0.9,
                    'monday': 0.85,
                    'tuesday': 0.9,
                    'wednesday': 0.95,
                    'thursday': 0.9,
                    'friday': 1.0,
                    'saturday': 1.1,
                    'sunday': 1.05
                }
            },
            'pattern_recognition': {
                'min_pattern_length': 5,
                'confidence_threshold': 0.7,
                'similarity_threshold': 0.8
            }
        }
    
    def analyze_temporal_features(self, team_data: Dict, match_context: Dict) -> Dict:
        """
        Main analysis function that generates all temporal features
        
        Args:
            team_data: Team's match history and statistics
            match_context: Context of the upcoming match
            
        Returns:
            Dict containing all temporal features and indicators
        """
        try:
            # Extract match data and context
            matches = team_data.get('recent_matches', [])
            team_id = team_data.get('team_id', 0)
            league_id = match_context.get('league_id', 0)
            match_date = match_context.get('match_date', datetime.now())
            
            if isinstance(match_date, str):
                match_date = datetime.strptime(match_date, '%Y-%m-%d')
            
            # 1. Exponential Decay Weighting Analysis
            decay_features = self._analyze_exponential_decay(matches, team_id, league_id, match_date)
            
            # 2. Seasonal Form Curve Fitting
            seasonal_features = self._analyze_seasonal_patterns(matches, team_id, match_date)
            
            # 3. Weekly Performance Analysis
            weekly_features = self._analyze_weekly_patterns(matches, team_id, match_date)
            
            # 4. Temporal Pattern Recognition
            pattern_features = self._recognize_temporal_patterns(matches, team_id, match_context)
            
            # 5. Combined temporal indicators
            combined_features = self._generate_combined_indicators(
                decay_features, seasonal_features, weekly_features, pattern_features
            )
            
            # Compile all features
            temporal_features = {
                'exponential_decay': decay_features,
                'seasonal_analysis': seasonal_features,
                'weekly_patterns': weekly_features,
                'temporal_patterns': pattern_features,
                'combined_indicators': combined_features
            }
            
            logger.info(f"Generated {len(temporal_features)} temporal feature categories for team {team_id}")
            return temporal_features
            
        except Exception as e:
            logger.error(f"Error in temporal feature analysis: {str(e)}")
            return self._get_default_temporal_features()
    
    def _analyze_exponential_decay(self, matches: List[Dict], team_id: int, 
                                 league_id: int, match_date: datetime) -> Dict:
        """
        Implement exponential decay weighting for recent matches
        
        Focuses on last 30 days with configurable decay rates per league
        """
        try:
            if not matches:
                return self._get_default_decay_features()
            
            # Get league-specific decay rate
            decay_rate = self._get_league_decay_rate(league_id)
            window_days = self.config['decay_window_days']
            cutoff_date = match_date - timedelta(days=window_days)
            
            # Filter matches within window and add time weights
            weighted_matches = []
            total_weight = 0
            
            for match in matches:
                match_date_obj = self._parse_match_date(match)
                if match_date_obj and match_date_obj >= cutoff_date:
                    days_ago = (match_date - match_date_obj).days
                    weight = decay_rate ** days_ago
                    
                    weighted_matches.append({
                        'match': match,
                        'weight': weight,
                        'days_ago': days_ago,
                        'recency_factor': weight
                    })
                    total_weight += weight
            
            if not weighted_matches:
                return self._get_default_decay_features()
            
            # Calculate weighted performance metrics
            weighted_metrics = self._calculate_weighted_metrics(weighted_matches, total_weight, team_id)
            
            # Performance trend detection
            trend_analysis = self._detect_performance_trends(weighted_matches, team_id)
            
            # Recent form strength
            recent_strength = self._calculate_recent_strength(weighted_matches, team_id)
            
            return {
                'weighted_performance': weighted_metrics,
                'trend_analysis': trend_analysis,
                'recent_strength': recent_strength,
                'decay_rate_used': decay_rate,
                'matches_analyzed': len(weighted_matches),
                'total_weight': total_weight,
                'time_weighted_score': weighted_metrics.get('overall_score', 50)
            }
            
        except Exception as e:
            logger.error(f"Error in exponential decay analysis: {str(e)}")
            return self._get_default_decay_features()
    
    def _analyze_seasonal_patterns(self, matches: List[Dict], team_id: int, 
                                 current_date: datetime) -> Dict:
        """
        Analyze seasonal performance patterns and curve fitting
        
        Detects season start effects, mid-season patterns, end-season motivation
        """
        try:
            # Group matches by seasonal periods
            seasonal_groups = self._group_matches_by_season(matches, current_date)
            
            # Analyze each seasonal period
            season_analysis = {}
            for period, period_matches in seasonal_groups.items():
                if period_matches:
                    season_analysis[period] = self._analyze_seasonal_period(
                        period_matches, team_id, period
                    )
            
            # Fit seasonal performance curve
            curve_params = self._fit_seasonal_curve(seasonal_groups, team_id)
            
            # Detect season start/end effects
            season_effects = self._detect_season_effects(seasonal_groups, team_id)
            
            # Holiday period analysis
            holiday_effects = self._analyze_holiday_effects(matches, team_id, current_date)
            
            # Current seasonal adjustment
            current_adjustment = self._calculate_current_seasonal_adjustment(
                current_date, season_analysis, curve_params
            )
            
            return {
                'seasonal_periods': season_analysis,
                'curve_parameters': curve_params,
                'season_effects': season_effects,
                'holiday_effects': holiday_effects,
                'current_adjustment_factor': current_adjustment,
                'seasonal_form_score': self._calculate_seasonal_form_score(season_analysis),
                'predicted_seasonal_performance': self._predict_seasonal_performance(
                    current_date, curve_params
                )
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal pattern analysis: {str(e)}")
            return self._get_default_seasonal_features()
    
    def _analyze_weekly_patterns(self, matches: List[Dict], team_id: int, 
                               match_date: datetime) -> Dict:
        """
        Analyze weekly performance cycles and day-of-week effects
        
        Examines weekend vs midweek performance, recovery patterns
        """
        try:
            # Group matches by day of week
            day_performance = defaultdict(list)
            weekly_cycles = []
            
            for match in matches:
                match_date_obj = self._parse_match_date(match)
                if match_date_obj:
                    day_name = match_date_obj.strftime('%A').lower()
                    performance = self._extract_match_performance(match, team_id)
                    
                    day_performance[day_name].append(performance)
                    
                    # Analyze weekly cycles (time between matches)
                    weekly_cycles.append({
                        'date': match_date_obj,
                        'day': day_name,
                        'performance': performance
                    })
            
            # Calculate day-of-week performance
            day_analysis = {}
            for day, performances in day_performance.items():
                if performances:
                    day_analysis[day] = {
                        'avg_performance': np.mean(performances),
                        'consistency': 1.0 - np.std(performances) if len(performances) > 1 else 1.0,
                        'sample_size': len(performances),
                        'performance_weight': self.config['weekly_analysis']['performance_weights'].get(day, 1.0)
                    }
            
            # Weekend vs Midweek analysis
            weekend_midweek = self._analyze_weekend_vs_midweek(day_analysis)
            
            # Recovery time analysis
            recovery_analysis = self._analyze_recovery_patterns(weekly_cycles, team_id)
            
            # Weekly rhythm detection
            rhythm_analysis = self._detect_weekly_rhythm(weekly_cycles, team_id)
            
            # Calculate optimal timing recommendations
            timing_recommendations = self._generate_timing_recommendations(
                day_analysis, recovery_analysis, match_date
            )
            
            return {
                'day_of_week_performance': day_analysis,
                'weekend_vs_midweek': weekend_midweek,
                'recovery_patterns': recovery_analysis,
                'weekly_rhythm': rhythm_analysis,
                'timing_recommendations': timing_recommendations,
                'weekly_advantage_score': self._calculate_weekly_advantage_score(
                    day_analysis, match_date.strftime('%A').lower()
                )
            }
            
        except Exception as e:
            logger.error(f"Error in weekly pattern analysis: {str(e)}")
            return self._get_default_weekly_features()
    
    def _recognize_temporal_patterns(self, matches: List[Dict], team_id: int, 
                                   match_context: Dict) -> Dict:
        """
        Advanced temporal pattern recognition
        
        Detects monthly cycles, opponent-specific patterns, manager effects
        """
        try:
            # Monthly performance cycles
            monthly_patterns = self._analyze_monthly_cycles(matches, team_id)
            
            # Opponent-specific temporal patterns
            opponent_patterns = self._analyze_opponent_patterns(matches, team_id, match_context)
            
            # Manager effect timeline analysis
            manager_effects = self._analyze_manager_effects(matches, team_id, match_context)
            
            # Transfer window impact analysis
            transfer_effects = self._analyze_transfer_window_effects(matches, team_id)
            
            # Pattern clustering and recognition
            pattern_clusters = self._cluster_temporal_patterns(matches, team_id)
            
            # Predict upcoming pattern
            pattern_prediction = self._predict_upcoming_pattern(
                monthly_patterns, opponent_patterns, manager_effects, match_context
            )
            
            return {
                'monthly_cycles': monthly_patterns,
                'opponent_specific': opponent_patterns,
                'manager_effects': manager_effects,
                'transfer_window_effects': transfer_effects,
                'pattern_clusters': pattern_clusters,
                'pattern_prediction': pattern_prediction,
                'temporal_advantage_indicators': self._calculate_temporal_advantages(
                    monthly_patterns, opponent_patterns, manager_effects
                )
            }
            
        except Exception as e:
            logger.error(f"Error in temporal pattern recognition: {str(e)}")
            return self._get_default_pattern_features()
    
    def _generate_combined_indicators(self, decay_features: Dict, seasonal_features: Dict,
                                    weekly_features: Dict, pattern_features: Dict) -> Dict:
        """
        Generate combined temporal indicators and performance curves
        """
        try:
            # Overall temporal score (0-100)
            temporal_scores = [
                decay_features.get('time_weighted_score', 50),
                seasonal_features.get('seasonal_form_score', 50),
                weekly_features.get('weekly_advantage_score', 50),
                pattern_features.get('temporal_advantage_indicators', {}).get('overall_score', 50)
            ]
            overall_temporal_score = np.mean(temporal_scores)
            
            # Performance prediction curves
            performance_curves = self._generate_performance_curves(
                decay_features, seasonal_features, weekly_features, pattern_features
            )
            
            # Temporal advantage indicators
            advantage_indicators = {
                'time_advantage': self._calculate_time_advantage(decay_features),
                'seasonal_advantage': seasonal_features.get('current_adjustment_factor', 1.0),
                'weekly_advantage': weekly_features.get('weekly_advantage_score', 50) / 50,
                'pattern_advantage': pattern_features.get('pattern_prediction', {}).get('confidence', 0.5)
            }
            
            # Optimal timing recommendations
            timing_recommendations = self._generate_optimal_timing(
                weekly_features, seasonal_features, pattern_features
            )
            
            return {
                'overall_temporal_score': overall_temporal_score,
                'performance_curves': performance_curves,
                'advantage_indicators': advantage_indicators,
                'timing_recommendations': timing_recommendations,
                'temporal_momentum': self._calculate_temporal_momentum(decay_features),
                'confidence_level': self._calculate_confidence_level(
                    decay_features, seasonal_features, weekly_features, pattern_features
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating combined indicators: {str(e)}")
            return self._get_default_combined_features()
    
    # Helper methods for calculations
    
    def _get_league_decay_rate(self, league_id: int) -> float:
        """Get decay rate for specific league"""
        if league_id in self.league_decay_rates:
            return self.league_decay_rates[league_id]
        
        # Default decay rate based on league characteristics
        base_rate = self.config['base_decay_rate']
        
        # League-specific adjustments (can be learned from data)
        league_adjustments = {
            39: 0.96,   # Premier League (higher volatility)
            140: 0.95,  # La Liga
            78: 0.94,   # Bundesliga
            135: 0.95,  # Serie A
            61: 0.93,   # Ligue 1
        }
        
        adjusted_rate = league_adjustments.get(league_id, base_rate)
        self.league_decay_rates[league_id] = adjusted_rate
        return adjusted_rate
    
    def _parse_match_date(self, match: Dict) -> Optional[datetime]:
        """Parse match date from various formats"""
        try:
            # Try different date formats
            date_fields = ['date', 'fixture_date', 'match_date']
            for field in date_fields:
                if field in match and match[field]:
                    date_str = match[field]
                    if isinstance(date_str, str):
                        # Try different formats
                        formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y']
                        for fmt in formats:
                            try:
                                return datetime.strptime(date_str, fmt)
                            except ValueError:
                                continue
            
            # Try fixture timestamp
            if 'fixture' in match and 'timestamp' in match['fixture']:
                timestamp = match['fixture']['timestamp']
                return datetime.fromtimestamp(timestamp)
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not parse match date: {str(e)}")
            return None
    
    def _calculate_weighted_metrics(self, weighted_matches: List[Dict], 
                                  total_weight: float, team_id: int) -> Dict:
        """Calculate weighted performance metrics"""
        if not weighted_matches or total_weight == 0:
            return {'overall_score': 50, 'goals_per_game': 1.0, 'points_per_game': 1.0}
        
        weighted_goals = 0
        weighted_points = 0
        weighted_performance = 0
        
        for item in weighted_matches:
            match = item['match']
            weight = item['weight']
            
            # Extract performance metrics
            goals = self._extract_goals_scored(match, team_id)
            points = self._extract_points_earned(match, team_id)
            performance = self._extract_match_performance(match, team_id)
            
            weighted_goals += goals * weight
            weighted_points += points * weight
            weighted_performance += performance * weight
        
        return {
            'goals_per_game': weighted_goals / total_weight,
            'points_per_game': weighted_points / total_weight,
            'overall_score': (weighted_performance / total_weight) * 100,
            'performance_variance': np.var([item['match'] for item in weighted_matches])
        }
    
    def _detect_performance_trends(self, weighted_matches: List[Dict], team_id: int) -> Dict:
        """Detect performance trends in recent matches"""
        if len(weighted_matches) < 3:
            return {'trend': 'stable', 'strength': 0.0, 'confidence': 0.0}
        
        # Sort by recency (most recent first)
        sorted_matches = sorted(weighted_matches, key=lambda x: x['days_ago'])
        
        # Extract performance scores
        performances = []
        for item in sorted_matches:
            performance = self._extract_match_performance(item['match'], team_id)
            performances.append(performance)
        
        # Calculate trend using linear regression
        x = np.arange(len(performances))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, performances)
        
        # Determine trend direction and strength
        if abs(slope) < 0.1:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving'
        else:
            trend = 'declining'
        
        return {
            'trend': trend,
            'strength': abs(slope),
            'confidence': abs(r_value),
            'slope': slope,
            'recent_performance': performances[-3:] if len(performances) >= 3 else performances
        }
    
    def _calculate_recent_strength(self, weighted_matches: List[Dict], team_id: int) -> Dict:
        """Calculate recent form strength with multiple indicators"""
        if not weighted_matches:
            return {'strength_score': 50, 'momentum': 0.0, 'consistency': 0.5}
        
        # Recent performance scores
        recent_scores = []
        for item in weighted_matches[-5:]:  # Last 5 matches
            score = self._extract_match_performance(item['match'], team_id)
            recent_scores.append(score)
        
        # Calculate strength indicators
        avg_score = np.mean(recent_scores) if recent_scores else 50
        consistency = 1.0 - (np.std(recent_scores) / 100) if len(recent_scores) > 1 else 0.5
        
        # Momentum calculation (recent vs older)
        if len(weighted_matches) >= 6:
            recent_avg = np.mean(recent_scores[-3:])
            older_avg = np.mean([self._extract_match_performance(item['match'], team_id) 
                               for item in weighted_matches[-6:-3]])
            momentum = (recent_avg - older_avg) / 100
        else:
            momentum = 0.0
        
        return {
            'strength_score': avg_score,
            'momentum': momentum,
            'consistency': consistency,
            'form_quality': avg_score * consistency
        }
    
    def _group_matches_by_season(self, matches: List[Dict], current_date: datetime) -> Dict:
        """Group matches by seasonal periods"""
        seasonal_groups = {
            'season_start': [],
            'mid_season': [],
            'season_end': [],
            'transfer_windows': [],
            'holiday_periods': []
        }
        
        for match in matches:
            match_date = self._parse_match_date(match)
            if not match_date:
                continue
            
            month = match_date.month
            
            # Categorize by seasonal periods
            if month in range(8, 11):  # August-October
                seasonal_groups['season_start'].append(match)
            elif month in [11, 12, 1, 2]:  # November-February
                seasonal_groups['mid_season'].append(match)
            elif month in range(3, 6):  # March-May
                seasonal_groups['season_end'].append(match)
            
            # Transfer windows
            if month in [1, 6, 7, 8, 9]:
                seasonal_groups['transfer_windows'].append(match)
            
            # Holiday periods
            if month in [12, 1, 7, 8]:
                seasonal_groups['holiday_periods'].append(match)
        
        return seasonal_groups
    
    def _analyze_seasonal_period(self, period_matches: List[Dict], 
                               team_id: int, period: str) -> Dict:
        """Analyze performance in a specific seasonal period"""
        if not period_matches:
            return {'avg_performance': 50, 'matches_count': 0, 'trend': 'stable'}
        
        performances = []
        for match in period_matches:
            performance = self._extract_match_performance(match, team_id)
            performances.append(performance)
        
        return {
            'avg_performance': np.mean(performances),
            'std_performance': np.std(performances) if len(performances) > 1 else 0,
            'matches_count': len(period_matches),
            'best_performance': max(performances),
            'worst_performance': min(performances),
            'consistency': 1.0 - (np.std(performances) / 100) if len(performances) > 1 else 1.0
        }
    
    def _fit_seasonal_curve(self, seasonal_groups: Dict, team_id: int) -> Dict:
        """Fit polynomial curve to seasonal performance"""
        try:
            # Collect performance data with time indices
            time_points = []
            performances = []
            
            # Map seasonal periods to time indices (0-12 months)
            period_mapping = {
                'season_start': [8, 9, 10],  # Aug-Oct
                'mid_season': [11, 0, 1, 2], # Nov-Feb (0=Dec)
                'season_end': [3, 4, 5]      # Mar-May
            }
            
            for period, months in period_mapping.items():
                if period in seasonal_groups and seasonal_groups[period]:
                    period_performance = np.mean([
                        self._extract_match_performance(match, team_id) 
                        for match in seasonal_groups[period]
                    ])
                    
                    for month in months:
                        time_points.append(month)
                        performances.append(period_performance)
            
            if len(time_points) < 3:
                return {'fitted': False, 'curve_type': 'insufficient_data'}
            
            # Fit polynomial curve (degree 2)
            time_array = np.array(time_points)
            perf_array = np.array(performances)
            
            coefficients = np.polyfit(time_array, perf_array, 2)
            
            return {
                'fitted': True,
                'curve_type': 'polynomial',
                'coefficients': coefficients.tolist(),
                'r_squared': self._calculate_r_squared(time_array, perf_array, coefficients),
                'peak_month': self._find_curve_peak(coefficients),
                'seasonal_variation': np.std(performances)
            }
            
        except Exception as e:
            logger.warning(f"Could not fit seasonal curve: {str(e)}")
            return {'fitted': False, 'curve_type': 'error'}
    
    def _detect_season_effects(self, seasonal_groups: Dict, team_id: int) -> Dict:
        """Detect specific season start/end effects"""
        effects = {}
        
        # Season start effect
        if seasonal_groups.get('season_start'):
            start_performances = [
                self._extract_match_performance(match, team_id) 
                for match in seasonal_groups['season_start']
            ]
            effects['season_start'] = {
                'avg_performance': np.mean(start_performances),
                'is_slow_starter': np.mean(start_performances) < 45,
                'is_fast_starter': np.mean(start_performances) > 55,
                'adaptation_time': len(start_performances) // 3  # Rough estimate
            }
        
        # Season end effect
        if seasonal_groups.get('season_end'):
            end_performances = [
                self._extract_match_performance(match, team_id) 
                for match in seasonal_groups['season_end']
            ]
            effects['season_end'] = {
                'avg_performance': np.mean(end_performances),
                'motivation_level': 'high' if np.mean(end_performances) > 50 else 'low',
                'consistency': 1.0 - np.std(end_performances) / 100 if len(end_performances) > 1 else 1.0
            }
        
        return effects
    
    def _analyze_holiday_effects(self, matches: List[Dict], team_id: int, 
                               current_date: datetime) -> Dict:
        """Analyze performance during holiday periods"""
        holiday_matches = []
        regular_matches = []
        
        for match in matches:
            match_date = self._parse_match_date(match)
            if not match_date:
                continue
            
            month = match_date.month
            
            # Holiday periods: December-January, July-August
            if month in [12, 1, 7, 8]:
                holiday_matches.append(match)
            else:
                regular_matches.append(match)
        
        holiday_performance = np.mean([
            self._extract_match_performance(match, team_id) 
            for match in holiday_matches
        ]) if holiday_matches else 50
        
        regular_performance = np.mean([
            self._extract_match_performance(match, team_id) 
            for match in regular_matches
        ]) if regular_matches else 50
        
        return {
            'holiday_performance': holiday_performance,
            'regular_performance': regular_performance,
            'holiday_effect': holiday_performance - regular_performance,
            'holiday_matches_count': len(holiday_matches),
            'is_holiday_sensitive': abs(holiday_performance - regular_performance) > 5
        }
    
    def _calculate_current_seasonal_adjustment(self, current_date: datetime,
                                             season_analysis: Dict, 
                                             curve_params: Dict) -> float:
        """Calculate seasonal adjustment factor for current date"""
        try:
            month = current_date.month
            
            # Use fitted curve if available
            if curve_params.get('fitted', False) and 'coefficients' in curve_params:
                coeffs = curve_params['coefficients']
                predicted_performance = np.polyval(coeffs, month)
                baseline_performance = 50  # Neutral baseline
                adjustment_factor = predicted_performance / baseline_performance
                
                # Clamp adjustment factor to reasonable range
                return max(0.8, min(1.2, adjustment_factor))
            
            # Fallback to period-based adjustment
            if month in [8, 9, 10] and 'season_start' in season_analysis:
                period_perf = season_analysis['season_start'].get('avg_performance', 50)
            elif month in [11, 12, 1, 2] and 'mid_season' in season_analysis:
                period_perf = season_analysis['mid_season'].get('avg_performance', 50)
            elif month in [3, 4, 5] and 'season_end' in season_analysis:
                period_perf = season_analysis['season_end'].get('avg_performance', 50)
            else:
                return 1.0  # Neutral adjustment
            
            return max(0.8, min(1.2, period_perf / 50))
            
        except Exception as e:
            logger.warning(f"Could not calculate seasonal adjustment: {str(e)}")
            return 1.0
    
    def _calculate_seasonal_form_score(self, season_analysis: Dict) -> float:
        """Calculate overall seasonal form score"""
        if not season_analysis:
            return 50.0
        
        scores = []
        weights = []
        
        for period, data in season_analysis.items():
            if isinstance(data, dict) and 'avg_performance' in data:
                scores.append(data['avg_performance'])
                # Weight by number of matches
                weight = data.get('matches_count', 1)
                weights.append(weight)
        
        if not scores:
            return 50.0
        
        # Weighted average
        return np.average(scores, weights=weights)
    
    def _predict_seasonal_performance(self, current_date: datetime, 
                                    curve_params: Dict) -> Dict:
        """Predict seasonal performance based on fitted curve"""
        if not curve_params.get('fitted', False):
            return {'prediction': 50.0, 'confidence': 0.0}
        
        try:
            month = current_date.month
            coeffs = curve_params.get('coefficients', [0, 0, 50])
            
            # Predict performance for current month
            predicted_perf = np.polyval(coeffs, month)
            
            # Calculate confidence based on R-squared
            confidence = curve_params.get('r_squared', 0.0)
            
            return {
                'prediction': max(0, min(100, predicted_perf)),
                'confidence': confidence,
                'trend_direction': 'improving' if coeffs[0] > 0 else 'declining' if coeffs[0] < 0 else 'stable'
            }
            
        except Exception as e:
            logger.warning(f"Could not predict seasonal performance: {str(e)}")
            return {'prediction': 50.0, 'confidence': 0.0}
    
    def _analyze_weekend_vs_midweek(self, day_analysis: Dict) -> Dict:
        """Analyze weekend vs midweek performance differences"""
        weekend_days = ['friday', 'saturday', 'sunday']
        midweek_days = ['monday', 'tuesday', 'wednesday', 'thursday']
        
        weekend_performances = []
        midweek_performances = []
        
        for day, data in day_analysis.items():
            if day in weekend_days:
                weekend_performances.append(data.get('avg_performance', 50))
            elif day in midweek_days:
                midweek_performances.append(data.get('avg_performance', 50))
        
        weekend_avg = np.mean(weekend_performances) if weekend_performances else 50
        midweek_avg = np.mean(midweek_performances) if midweek_performances else 50
        
        return {
            'weekend_performance': weekend_avg,
            'midweek_performance': midweek_avg,
            'difference': weekend_avg - midweek_avg,
            'weekend_advantage': weekend_avg > midweek_avg,
            'effect_size': abs(weekend_avg - midweek_avg) / 10  # Normalize to 0-10
        }
    
    def _analyze_recovery_patterns(self, weekly_cycles: List[Dict], team_id: int) -> Dict:
        """Analyze recovery time patterns between matches"""
        if len(weekly_cycles) < 2:
            return {'avg_recovery_days': 7, 'optimal_recovery': 7, 'recovery_effect': 0.0}
        
        # Sort by date
        sorted_cycles = sorted(weekly_cycles, key=lambda x: x['date'])
        
        recovery_data = []
        for i in range(1, len(sorted_cycles)):
            prev_match = sorted_cycles[i-1]
            curr_match = sorted_cycles[i]
            
            # Calculate days between matches
            days_between = (curr_match['date'] - prev_match['date']).days
            curr_performance = curr_match['performance']
            
            recovery_data.append({
                'days_recovery': days_between,
                'performance': curr_performance
            })
        
        if not recovery_data:
            return {'avg_recovery_days': 7, 'optimal_recovery': 7, 'recovery_effect': 0.0}
        
        # Group by recovery days and analyze performance
        recovery_groups = defaultdict(list)
        for item in recovery_data:
            recovery_groups[item['days_recovery']].append(item['performance'])
        
        # Find optimal recovery time
        best_recovery = 7  # Default
        best_performance = 0
        
        for days, performances in recovery_groups.items():
            if len(performances) >= 2:  # Need sufficient sample
                avg_perf = np.mean(performances)
                if avg_perf > best_performance:
                    best_performance = avg_perf
                    best_recovery = days
        
        return {
            'avg_recovery_days': np.mean([item['days_recovery'] for item in recovery_data]),
            'optimal_recovery': best_recovery,
            'recovery_effect': self._calculate_recovery_effect(recovery_groups),
            'recovery_consistency': self._calculate_recovery_consistency(recovery_groups)
        }
    
    def _detect_weekly_rhythm(self, weekly_cycles: List[Dict], team_id: int) -> Dict:
        """Detect weekly performance rhythms and patterns"""
        if not weekly_cycles:
            return {'rhythm_detected': False, 'rhythm_strength': 0.0}
        
        # Extract performance by day of week
        day_performances = defaultdict(list)
        for cycle in weekly_cycles:
            day = cycle['day']
            performance = cycle['performance']
            day_performances[day].append(performance)
        
        # Calculate rhythm strength (consistency across days)
        day_averages = []
        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            if day in day_performances and day_performances[day]:
                day_averages.append(np.mean(day_performances[day]))
            else:
                day_averages.append(50)  # Neutral default
        
        # Rhythm strength based on variance
        rhythm_variance = np.var(day_averages)
        rhythm_strength = min(1.0, rhythm_variance / 100)  # Normalize
        
        # Detect patterns
        rhythm_pattern = self._classify_rhythm_pattern(day_averages)
        
        return {
            'rhythm_detected': rhythm_strength > 0.3,
            'rhythm_strength': rhythm_strength,
            'rhythm_pattern': rhythm_pattern,
            'day_averages': dict(zip(
                ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                day_averages
            ))
        }
    
    def _generate_timing_recommendations(self, day_analysis: Dict, 
                                       recovery_analysis: Dict, 
                                       match_date: datetime) -> Dict:
        """Generate optimal timing recommendations"""
        match_day = match_date.strftime('%A').lower()
        
        # Day-based recommendation
        day_performance = day_analysis.get(match_day, {}).get('avg_performance', 50)
        day_recommendation = "optimal" if day_performance > 55 else "suboptimal" if day_performance < 45 else "neutral"
        
        # Recovery-based recommendation
        optimal_recovery = recovery_analysis.get('optimal_recovery', 7)
        recovery_recommendation = f"Optimal recovery time: {optimal_recovery} days"
        
        # Combined recommendation
        overall_score = (day_performance + 50) / 2  # Blend with neutral baseline
        
        return {
            'match_day_recommendation': day_recommendation,
            'day_performance_score': day_performance,
            'recovery_recommendation': recovery_recommendation,
            'overall_timing_score': overall_score,
            'timing_advice': self._generate_timing_advice(day_performance, optimal_recovery, match_day)
        }
    
    def _calculate_weekly_advantage_score(self, day_analysis: Dict, match_day: str) -> float:
        """Calculate weekly advantage score for specific match day"""
        if match_day not in day_analysis:
            return 50.0  # Neutral score
        
        day_data = day_analysis[match_day]
        base_performance = day_data.get('avg_performance', 50)
        consistency = day_data.get('consistency', 0.5)
        sample_size = day_data.get('sample_size', 1)
        
        # Weight by sample size and consistency
        confidence = min(1.0, sample_size / 5) * consistency
        weighted_score = base_performance * confidence + 50 * (1 - confidence)
        
        return max(0, min(100, weighted_score))
    
    def _analyze_monthly_cycles(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze monthly performance cycles"""
        monthly_performance = defaultdict(list)
        
        for match in matches:
            match_date = self._parse_match_date(match)
            if match_date:
                month = match_date.month
                performance = self._extract_match_performance(match, team_id)
                monthly_performance[month].append(performance)
        
        # Calculate monthly averages
        monthly_averages = {}
        for month, performances in monthly_performance.items():
            if performances:
                monthly_averages[month] = {
                    'avg_performance': np.mean(performances),
                    'consistency': 1.0 - np.std(performances) / 100 if len(performances) > 1 else 1.0,
                    'sample_size': len(performances)
                }
        
        # Detect cyclical patterns
        cycle_analysis = self._detect_monthly_cycles(monthly_averages)
        
        return {
            'monthly_averages': monthly_averages,
            'cycle_analysis': cycle_analysis,
            'best_months': self._find_best_months(monthly_averages),
            'worst_months': self._find_worst_months(monthly_averages)
        }
    
    def _analyze_opponent_patterns(self, matches: List[Dict], team_id: int, 
                                 match_context: Dict) -> Dict:
        """Analyze opponent-specific temporal patterns"""
        opponent_id = match_context.get('opponent_id')
        if not opponent_id:
            return {'patterns_found': False}
        
        # Find historical matches against this opponent
        opponent_matches = []
        for match in matches:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            
            if ((home_team.get('id') == team_id and away_team.get('id') == opponent_id) or
                (away_team.get('id') == team_id and home_team.get('id') == opponent_id)):
                opponent_matches.append(match)
        
        if len(opponent_matches) < 2:
            return {'patterns_found': False, 'insufficient_data': True}
        
        # Analyze temporal patterns in H2H matches
        temporal_patterns = self._analyze_h2h_temporal_patterns(opponent_matches, team_id)
        
        return {
            'patterns_found': True,
            'h2h_matches_count': len(opponent_matches),
            'temporal_patterns': temporal_patterns,
            'performance_trend': self._calculate_h2h_trend(opponent_matches, team_id)
        }
    
    def _analyze_manager_effects(self, matches: List[Dict], team_id: int, 
                               match_context: Dict) -> Dict:
        """Analyze manager effect timeline"""
        # This would require manager change data
        # For now, return basic analysis
        return {
            'manager_changes_detected': False,
            'current_manager_tenure': 'unknown',
            'performance_under_manager': 50.0,
            'honeymoon_effect': False
        }
    
    def _analyze_transfer_window_effects(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze transfer window impact periods"""
        transfer_months = [1, 6, 7, 8, 9]  # January and summer window
        
        transfer_matches = []
        regular_matches = []
        
        for match in matches:
            match_date = self._parse_match_date(match)
            if match_date and match_date.month in transfer_months:
                transfer_matches.append(match)
            else:
                regular_matches.append(match)
        
        transfer_performance = np.mean([
            self._extract_match_performance(match, team_id)
            for match in transfer_matches
        ]) if transfer_matches else 50
        
        regular_performance = np.mean([
            self._extract_match_performance(match, team_id)
            for match in regular_matches
        ]) if regular_matches else 50
        
        return {
            'transfer_window_performance': transfer_performance,
            'regular_period_performance': regular_performance,
            'transfer_effect': transfer_performance - regular_performance,
            'transfer_matches_count': len(transfer_matches),
            'is_transfer_sensitive': abs(transfer_performance - regular_performance) > 5
        }
    
    def _cluster_temporal_patterns(self, matches: List[Dict], team_id: int) -> Dict:
        """Cluster temporal patterns for pattern recognition"""
        # Extract temporal features for clustering
        temporal_features = []
        for match in matches[:20]:  # Last 20 matches
            match_date = self._parse_match_date(match)
            if match_date:
                performance = self._extract_match_performance(match, team_id)
                features = [
                    match_date.month,
                    match_date.weekday(),
                    performance,
                    match_date.day  # Day of month
                ]
                temporal_features.append(features)
        
        if len(temporal_features) < 5:
            return {'clusters_found': False, 'insufficient_data': True}
        
        try:
            # Perform clustering
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(temporal_features)
            
            # Use 3 clusters as default
            kmeans = KMeans(n_clusters=min(3, len(temporal_features)//2), random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            return {
                'clusters_found': True,
                'n_clusters': len(set(cluster_labels)),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'cluster_labels': cluster_labels.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Could not perform temporal clustering: {str(e)}")
            return {'clusters_found': False, 'error': str(e)}
    
    def _predict_upcoming_pattern(self, monthly_patterns: Dict, opponent_patterns: Dict,
                                manager_effects: Dict, match_context: Dict) -> Dict:
        """Predict upcoming temporal pattern"""
        match_date = match_context.get('match_date', datetime.now())
        if isinstance(match_date, str):
            match_date = datetime.strptime(match_date, '%Y-%m-%d')
        
        month = match_date.month
        
        # Monthly pattern prediction
        monthly_pred = 50.0  # Default
        if 'monthly_averages' in monthly_patterns and month in monthly_patterns['monthly_averages']:
            monthly_pred = monthly_patterns['monthly_averages'][month]['avg_performance']
        
        # Opponent pattern prediction
        opponent_pred = 50.0  # Default
        if opponent_patterns.get('patterns_found', False):
            opponent_pred = opponent_patterns.get('temporal_patterns', {}).get('avg_performance', 50.0)
        
        # Combine predictions
        combined_prediction = np.mean([monthly_pred, opponent_pred])
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(monthly_patterns, opponent_patterns)
        
        return {
            'predicted_performance': combined_prediction,
            'confidence': confidence,
            'monthly_component': monthly_pred,
            'opponent_component': opponent_pred,
            'pattern_type': self._classify_pattern_type(combined_prediction, confidence)
        }
    
    def _calculate_temporal_advantages(self, monthly_patterns: Dict, opponent_patterns: Dict,
                                     manager_effects: Dict) -> Dict:
        """Calculate various temporal advantage indicators"""
        advantages = {}
        
        # Monthly advantage
        if 'monthly_averages' in monthly_patterns:
            best_month_perf = max(
                [data['avg_performance'] for data in monthly_patterns['monthly_averages'].values()],
                default=50
            )
            advantages['monthly_peak_performance'] = best_month_perf
        
        # Opponent advantage
        if opponent_patterns.get('patterns_found', False):
            h2h_trend = opponent_patterns.get('performance_trend', {})
            advantages['h2h_trend_advantage'] = h2h_trend.get('trend_strength', 0.0)
        
        # Overall temporal advantage score
        advantage_scores = [v for v in advantages.values() if isinstance(v, (int, float))]
        overall_score = np.mean(advantage_scores) if advantage_scores else 50
        
        advantages['overall_score'] = overall_score
        
        return advantages
    
    def _generate_performance_curves(self, decay_features: Dict, seasonal_features: Dict,
                                   weekly_features: Dict, pattern_features: Dict) -> Dict:
        """Generate performance prediction curves"""
        # Create time series for next 30 days
        future_dates = [datetime.now() + timedelta(days=i) for i in range(30)]
        
        performance_curve = []
        for date in future_dates:
            # Combine different temporal factors
            decay_factor = decay_features.get('time_weighted_score', 50) / 50
            seasonal_factor = seasonal_features.get('current_adjustment_factor', 1.0)
            
            # Day of week factor
            day_name = date.strftime('%A').lower()
            weekly_data = weekly_features.get('day_of_week_performance', {})
            weekly_factor = weekly_data.get(day_name, {}).get('avg_performance', 50) / 50
            
            # Combined prediction
            predicted_performance = 50 * decay_factor * seasonal_factor * weekly_factor
            performance_curve.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_performance': max(0, min(100, predicted_performance))
            })
        
        return {
            'daily_predictions': performance_curve,
            'trend_direction': self._determine_curve_trend(performance_curve),
            'volatility': self._calculate_curve_volatility(performance_curve)
        }
    
    def _calculate_time_advantage(self, decay_features: Dict) -> float:
        """Calculate time-based advantage factor"""
        recent_strength = decay_features.get('recent_strength', {})
        momentum = recent_strength.get('momentum', 0.0)
        consistency = recent_strength.get('consistency', 0.5)
        
        # Time advantage is higher with positive momentum and high consistency
        time_advantage = 0.5 + (momentum * 0.3) + (consistency * 0.2)
        return max(0.0, min(1.0, time_advantage))
    
    def _generate_optimal_timing(self, weekly_features: Dict, seasonal_features: Dict,
                               pattern_features: Dict) -> Dict:
        """Generate optimal timing recommendations"""
        # Find best day of week
        day_performances = weekly_features.get('day_of_week_performance', {})
        best_day = max(day_performances.keys(), 
                      key=lambda x: day_performances[x].get('avg_performance', 0),
                      default='saturday')
        
        # Find best seasonal period
        seasonal_periods = seasonal_features.get('seasonal_periods', {})
        best_season = max(seasonal_periods.keys(),
                         key=lambda x: seasonal_periods[x].get('avg_performance', 0),
                         default='mid_season')
        
        return {
            'optimal_day_of_week': best_day,
            'optimal_seasonal_period': best_season,
            'current_timing_score': self._calculate_current_timing_score(
                weekly_features, seasonal_features
            ),
            'timing_advice': f"Best performance typically on {best_day.title()} during {best_season.replace('_', ' ')}"
        }
    
    def _calculate_temporal_momentum(self, decay_features: Dict) -> float:
        """Calculate temporal momentum indicator"""
        trend_analysis = decay_features.get('trend_analysis', {})
        recent_strength = decay_features.get('recent_strength', {})
        
        trend_strength = trend_analysis.get('strength', 0.0)
        momentum = recent_strength.get('momentum', 0.0)
        
        # Combine trend and momentum
        temporal_momentum = (trend_strength + momentum) / 2
        return max(-1.0, min(1.0, temporal_momentum))
    
    def _calculate_confidence_level(self, decay_features: Dict, seasonal_features: Dict,
                                  weekly_features: Dict, pattern_features: Dict) -> float:
        """Calculate overall confidence level for temporal analysis"""
        confidence_factors = []
        
        # Decay analysis confidence
        if decay_features.get('matches_analyzed', 0) >= 5:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Seasonal analysis confidence
        if seasonal_features.get('curve_parameters', {}).get('fitted', False):
            r_squared = seasonal_features['curve_parameters'].get('r_squared', 0.0)
            confidence_factors.append(r_squared)
        else:
            confidence_factors.append(0.3)
        
        # Weekly analysis confidence
        day_analysis = weekly_features.get('day_of_week_performance', {})
        if len(day_analysis) >= 5:  # At least 5 days with data
            avg_sample_size = np.mean([data.get('sample_size', 1) for data in day_analysis.values()])
            confidence_factors.append(min(1.0, avg_sample_size / 5))
        else:
            confidence_factors.append(0.4)
        
        # Pattern analysis confidence
        if pattern_features.get('pattern_prediction', {}).get('confidence', 0) > 0:
            confidence_factors.append(pattern_features['pattern_prediction']['confidence'])
        else:
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors)
    
    # Helper methods for match data extraction
    
    def _extract_match_performance(self, match: Dict, team_id: int) -> float:
        """Extract normalized performance score (0-100) from match"""
        try:
            goals_scored = self._extract_goals_scored(match, team_id)
            goals_conceded = self._extract_goals_conceded(match, team_id)
            points = self._extract_points_earned(match, team_id)
            
            # Normalize to 0-100 scale
            # Goals: 0-5 goals -> 0-50 points
            goal_score = min(50, goals_scored * 10)
            
            # Defense: 0 conceded = 25 points, 1 = 20, 2 = 15, etc.
            defense_score = max(0, 25 - goals_conceded * 5)
            
            # Result: Win = 25, Draw = 15, Loss = 0
            result_score = points * 25 / 3 if points > 0 else 0
            
            total_score = goal_score + defense_score + result_score
            return min(100, max(0, total_score))
            
        except Exception as e:
            logger.warning(f"Could not extract match performance: {str(e)}")
            return 50.0  # Neutral performance
    
    def _extract_goals_scored(self, match: Dict, team_id: int) -> int:
        """Extract goals scored by team"""
        try:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if home_team.get('id') == team_id:
                return score.get('home', 0) or 0
            elif away_team.get('id') == team_id:
                return score.get('away', 0) or 0
            
            # Fallback to direct fields
            return match.get('goals_scored', 0)
            
        except Exception:
            return 0
    
    def _extract_goals_conceded(self, match: Dict, team_id: int) -> int:
        """Extract goals conceded by team"""
        try:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if home_team.get('id') == team_id:
                return score.get('away', 0) or 0
            elif away_team.get('id') == team_id:
                return score.get('home', 0) or 0
            
            # Fallback to direct fields
            return match.get('goals_conceded', 0)
            
        except Exception:
            return 0
    
    def _extract_points_earned(self, match: Dict, team_id: int) -> int:
        """Extract points earned (3 for win, 1 for draw, 0 for loss)"""
        try:
            goals_scored = self._extract_goals_scored(match, team_id)
            goals_conceded = self._extract_goals_conceded(match, team_id)
            
            if goals_scored > goals_conceded:
                return 3  # Win
            elif goals_scored == goals_conceded:
                return 1  # Draw
            else:
                return 0  # Loss
                
        except Exception:
            return 0
    
    # Default feature sets for error handling
    
    def _get_default_temporal_features(self) -> Dict:
        """Return default temporal features when analysis fails"""
        return {
            'exponential_decay': self._get_default_decay_features(),
            'seasonal_analysis': self._get_default_seasonal_features(),
            'weekly_patterns': self._get_default_weekly_features(),
            'temporal_patterns': self._get_default_pattern_features(),
            'combined_indicators': self._get_default_combined_features()
        }
    
    def _get_default_decay_features(self) -> Dict:
        """Default exponential decay features"""
        return {
            'weighted_performance': {'overall_score': 50, 'goals_per_game': 1.0, 'points_per_game': 1.0},
            'trend_analysis': {'trend': 'stable', 'strength': 0.0, 'confidence': 0.0},
            'recent_strength': {'strength_score': 50, 'momentum': 0.0, 'consistency': 0.5},
            'decay_rate_used': self.config['base_decay_rate'],
            'matches_analyzed': 0,
            'total_weight': 0.0,
            'time_weighted_score': 50.0
        }
    
    def _get_default_seasonal_features(self) -> Dict:
        """Default seasonal features"""
        return {
            'seasonal_periods': {},
            'curve_parameters': {'fitted': False, 'curve_type': 'none'},
            'season_effects': {},
            'holiday_effects': {'holiday_effect': 0.0, 'is_holiday_sensitive': False},
            'current_adjustment_factor': 1.0,
            'seasonal_form_score': 50.0,
            'predicted_seasonal_performance': {'prediction': 50.0, 'confidence': 0.0}
        }
    
    def _get_default_weekly_features(self) -> Dict:
        """Default weekly features"""
        return {
            'day_of_week_performance': {},
            'weekend_vs_midweek': {'difference': 0.0, 'weekend_advantage': False},
            'recovery_patterns': {'avg_recovery_days': 7, 'optimal_recovery': 7},
            'weekly_rhythm': {'rhythm_detected': False, 'rhythm_strength': 0.0},
            'timing_recommendations': {'match_day_recommendation': 'neutral'},
            'weekly_advantage_score': 50.0
        }
    
    def _get_default_pattern_features(self) -> Dict:
        """Default pattern features"""
        return {
            'monthly_cycles': {'monthly_averages': {}},
            'opponent_specific': {'patterns_found': False},
            'manager_effects': {'manager_changes_detected': False},
            'transfer_window_effects': {'transfer_effect': 0.0},
            'pattern_clusters': {'clusters_found': False},
            'pattern_prediction': {'predicted_performance': 50.0, 'confidence': 0.0},
            'temporal_advantage_indicators': {'overall_score': 50.0}
        }
    
    def _get_default_combined_features(self) -> Dict:
        """Default combined features"""
        return {
            'overall_temporal_score': 50.0,
            'performance_curves': {'daily_predictions': [], 'trend_direction': 'stable'},
            'advantage_indicators': {'time_advantage': 0.5, 'seasonal_advantage': 1.0},
            'timing_recommendations': {'optimal_day_of_week': 'saturday'},
            'temporal_momentum': 0.0,
            'confidence_level': 0.5
        }
    
    # Additional helper methods
    
    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> float:
        """Calculate R-squared for polynomial fit"""
        try:
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        except:
            return 0.0
    
    def _find_curve_peak(self, coeffs: np.ndarray) -> int:
        """Find peak month from polynomial coefficients"""
        try:
            # For quadratic ax^2 + bx + c, peak is at -b/(2a)
            if len(coeffs) >= 3 and coeffs[0] != 0:
                peak = -coeffs[1] / (2 * coeffs[0])
                return int(np.clip(peak, 1, 12))
            return 6  # Default to June
        except:
            return 6
    
    def _calculate_recovery_effect(self, recovery_groups: Dict) -> float:
        """Calculate recovery time effect on performance"""
        try:
            recovery_effects = []
            for days, performances in recovery_groups.items():
                if len(performances) >= 2:
                    avg_perf = np.mean(performances)
                    recovery_effects.append((days, avg_perf))
            
            if len(recovery_effects) < 2:
                return 0.0
            
            # Calculate correlation between recovery days and performance
            days_list, perf_list = zip(*recovery_effects)
            correlation, _ = stats.pearsonr(days_list, perf_list)
            return correlation
            
        except:
            return 0.0
    
    def _calculate_recovery_consistency(self, recovery_groups: Dict) -> float:
        """Calculate consistency of recovery effects"""
        try:
            consistencies = []
            for days, performances in recovery_groups.items():
                if len(performances) > 1:
                    consistency = 1.0 - (np.std(performances) / 100)
                    consistencies.append(consistency)
            
            return np.mean(consistencies) if consistencies else 0.5
        except:
            return 0.5
    
    def _classify_rhythm_pattern(self, day_averages: List[float]) -> str:
        """Classify weekly rhythm pattern"""
        try:
            weekend_avg = np.mean([day_averages[4], day_averages[5], day_averages[6]])  # Fri-Sun
            midweek_avg = np.mean([day_averages[0], day_averages[1], day_averages[2], day_averages[3]])  # Mon-Thu
            
            if weekend_avg > midweek_avg + 5:
                return 'weekend_strong'
            elif midweek_avg > weekend_avg + 5:
                return 'midweek_strong'
            else:
                return 'balanced'
        except:
            return 'unknown'
    
    def _generate_timing_advice(self, day_performance: float, optimal_recovery: int, match_day: str) -> str:
        """Generate textual timing advice"""
        advice = []
        
        if day_performance > 55:
            advice.append(f"{match_day.title()} is a strong day for this team")
        elif day_performance < 45:
            advice.append(f"{match_day.title()} tends to be challenging for this team")
        
        if optimal_recovery != 7:
            advice.append(f"Team performs best with {optimal_recovery} days rest")
        
        return ". ".join(advice) if advice else "No specific timing patterns detected"
    
    def _detect_monthly_cycles(self, monthly_averages: Dict) -> Dict:
        """Detect cyclical patterns in monthly performance"""
        try:
            if len(monthly_averages) < 6:  # Need at least 6 months
                return {'cycle_detected': False, 'insufficient_data': True}
            
            # Extract performance values in month order
            performances = []
            months = sorted(monthly_averages.keys())
            
            for month in months:
                performances.append(monthly_averages[month]['avg_performance'])
            
            # Simple cycle detection using autocorrelation
            if len(performances) >= 8:
                # Check for 6-month and 3-month cycles
                autocorr_6 = np.corrcoef(performances[:-6], performances[6:])[0,1] if len(performances) > 6 else 0
                autocorr_3 = np.corrcoef(performances[:-3], performances[3:])[0,1] if len(performances) > 3 else 0
                
                cycle_strength = max(abs(autocorr_6), abs(autocorr_3))
                
                return {
                    'cycle_detected': cycle_strength > 0.5,
                    'cycle_strength': cycle_strength,
                    'cycle_type': '6_month' if abs(autocorr_6) > abs(autocorr_3) else '3_month'
                }
            
            return {'cycle_detected': False, 'insufficient_data': True}
            
        except Exception as e:
            logger.warning(f"Could not detect monthly cycles: {str(e)}")
            return {'cycle_detected': False, 'error': str(e)}
    
    def _find_best_months(self, monthly_averages: Dict) -> List[int]:
        """Find best performing months"""
        if not monthly_averages:
            return []
        
        sorted_months = sorted(monthly_averages.items(), 
                             key=lambda x: x[1]['avg_performance'], 
                             reverse=True)
        
        return [month for month, _ in sorted_months[:3]]  # Top 3 months
    
    def _find_worst_months(self, monthly_averages: Dict) -> List[int]:
        """Find worst performing months"""
        if not monthly_averages:
            return []
        
        sorted_months = sorted(monthly_averages.items(), 
                             key=lambda x: x[1]['avg_performance'])
        
        return [month for month, _ in sorted_months[:3]]  # Bottom 3 months
    
    def _analyze_h2h_temporal_patterns(self, opponent_matches: List[Dict], team_id: int) -> Dict:
        """Analyze temporal patterns in head-to-head matches"""
        if not opponent_matches:
            return {'avg_performance': 50.0, 'pattern_detected': False}
        
        performances = []
        for match in opponent_matches:
            performance = self._extract_match_performance(match, team_id)
            performances.append(performance)
        
        return {
            'avg_performance': np.mean(performances),
            'consistency': 1.0 - np.std(performances) / 100 if len(performances) > 1 else 1.0,
            'pattern_detected': len(performances) >= 3,
            'recent_h2h_trend': np.mean(performances[-3:]) if len(performances) >= 3 else np.mean(performances)
        }
    
    def _calculate_h2h_trend(self, opponent_matches: List[Dict], team_id: int) -> Dict:
        """Calculate head-to-head performance trend"""
        if len(opponent_matches) < 3:
            return {'trend': 'stable', 'trend_strength': 0.0}
        
        # Sort by date
        sorted_matches = sorted(opponent_matches, 
                              key=lambda x: self._parse_match_date(x) or datetime.min)
        
        performances = [self._extract_match_performance(match, team_id) for match in sorted_matches]
        
        # Linear regression for trend
        x = np.arange(len(performances))
        slope, _, r_value, _, _ = stats.linregress(x, performances)
        
        if abs(slope) < 0.5:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving'
        else:
            trend = 'declining'
        
        return {
            'trend': trend,
            'trend_strength': abs(slope),
            'confidence': abs(r_value)
        }
    
    def _calculate_pattern_confidence(self, monthly_patterns: Dict, opponent_patterns: Dict) -> float:
        """Calculate confidence in pattern predictions"""
        confidence_factors = []
        
        # Monthly pattern confidence
        if 'monthly_averages' in monthly_patterns:
            sample_sizes = [data.get('sample_size', 1) for data in monthly_patterns['monthly_averages'].values()]
            avg_sample_size = np.mean(sample_sizes) if sample_sizes else 1
            confidence_factors.append(min(1.0, avg_sample_size / 5))
        
        # Opponent pattern confidence
        if opponent_patterns.get('patterns_found', False):
            h2h_matches = opponent_patterns.get('h2h_matches_count', 0)
            confidence_factors.append(min(1.0, h2h_matches / 5))
        
        return np.mean(confidence_factors) if confidence_factors else 0.3
    
    def _classify_pattern_type(self, predicted_performance: float, confidence: float) -> str:
        """Classify the type of temporal pattern"""
        if confidence < 0.3:
            return 'uncertain'
        elif predicted_performance > 60:
            return 'favorable'
        elif predicted_performance < 40:
            return 'unfavorable'
        else:
            return 'neutral'
    
    def _determine_curve_trend(self, performance_curve: List[Dict]) -> str:
        """Determine trend direction of performance curve"""
        try:
            values = [item['predicted_performance'] for item in performance_curve]
            
            if len(values) < 2:
                return 'stable'
            
            # Simple trend calculation
            start_avg = np.mean(values[:5])
            end_avg = np.mean(values[-5:])
            
            diff = end_avg - start_avg
            
            if diff > 2:
                return 'improving'
            elif diff < -2:
                return 'declining'
            else:
                return 'stable'
                
        except:
            return 'stable'
    
    def _calculate_curve_volatility(self, performance_curve: List[Dict]) -> float:
        """Calculate volatility of performance curve"""
        try:
            values = [item['predicted_performance'] for item in performance_curve]
            return np.std(values) if len(values) > 1 else 0.0
        except:
            return 0.0
    
    def _calculate_current_timing_score(self, weekly_features: Dict, seasonal_features: Dict) -> float:
        """Calculate current timing score based on today's date"""
        try:
            today = datetime.now()
            day_name = today.strftime('%A').lower()
            
            # Day score
            day_analysis = weekly_features.get('day_of_week_performance', {})
            day_score = day_analysis.get(day_name, {}).get('avg_performance', 50)
            
            # Seasonal score
            seasonal_adjustment = seasonal_features.get('current_adjustment_factor', 1.0)
            seasonal_score = 50 * seasonal_adjustment
            
            # Combined score
            combined_score = (day_score + seasonal_score) / 2
            return max(0, min(100, combined_score))
            
        except:
            return 50.0
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance scores for temporal features"""
        return {
            'exponential_decay_weight': 0.35,
            'seasonal_patterns_weight': 0.25, 
            'weekly_patterns_weight': 0.20,
            'temporal_patterns_weight': 0.20,
            'total_features_generated': len(getattr(self, 'all_features', [])) if hasattr(self, 'all_features') else 0
        }
    
    def save_temporal_patterns(self, filepath: str) -> bool:
        """Save learned temporal patterns to file"""
        try:
            patterns_data = {
                'seasonal_patterns': self.seasonal_patterns,
                'weekly_patterns': self.weekly_patterns,
                'temporal_clusters': self.temporal_clusters,
                'league_decay_rates': self.league_decay_rates,
                'config': self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
            
            logger.info(f"Temporal patterns saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save temporal patterns: {str(e)}")
            return False
    
    def load_temporal_patterns(self, filepath: str) -> bool:
        """Load temporal patterns from file"""
        try:
            with open(filepath, 'r') as f:
                patterns_data = json.load(f)
            
            self.seasonal_patterns = patterns_data.get('seasonal_patterns', {})
            self.weekly_patterns = patterns_data.get('weekly_patterns', {})
            self.temporal_clusters = patterns_data.get('temporal_clusters', {})
            self.league_decay_rates = patterns_data.get('league_decay_rates', {})
            
            # Update config if present
            if 'config' in patterns_data:
                self.config.update(patterns_data['config'])
            
            logger.info(f"Temporal patterns loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load temporal patterns: {str(e)}")
            return False