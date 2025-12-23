"""
Feature Engineering Module for Football Prediction System
Implements advanced feature engineering for Phase 3.2
Enhanced with Dynamic Time-Weighted Features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import json

# Import Dynamic Time Analyzer for advanced temporal features
from .dynamic_time_analyzer import DynamicTimeAnalyzer

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for football predictions"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.selected_features = []
        self.interaction_pairs = []
        self.temporal_features = []
        self.league_specific_features = {}
        
        # Initialize Dynamic Time Analyzer for advanced temporal features
        self.dynamic_time_analyzer = DynamicTimeAnalyzer()
        
        logger.info("FeatureEngineer initialized with Dynamic Time-Weighted Features")
    
    def engineer_features(self, home_data: Dict, away_data: Dict, match_context: Dict) -> Dict:
        """
        Create comprehensive feature set for prediction models
        
        Args:
            home_data: Home team data including form, stats, etc.
            away_data: Away team data including form, stats, etc.
            match_context: Match context (time, league, etc.)
            
        Returns:
            Dict containing engineered features
        """
        try:
            features = {}
            
            # 1. Basic statistical features
            features.update(self._create_basic_features(home_data, away_data))
            
            # 2. Advanced statistical features
            features.update(self._create_advanced_statistical_features(home_data, away_data))
            
            # 3. Interaction features
            features.update(self._create_interaction_features(home_data, away_data))
            
            # 4. Basic temporal features
            features.update(self._create_temporal_features(match_context))
            
            # 4.1. Advanced Dynamic Time-Weighted Features
            features.update(self._create_dynamic_temporal_features(home_data, away_data, match_context))
            
            # 5. League-specific features
            league_id = match_context.get('league_id', 0)
            if league_id:
                features.update(self._create_league_specific_features(int(league_id), home_data, away_data))
            
            # 6. Form-based engineered features
            features.update(self._create_form_engineered_features(home_data, away_data))
            
            # 7. Momentum and psychological features
            features.update(self._create_momentum_features(home_data, away_data))
            
            # 8. Head-to-head engineered features
            features.update(self._create_h2h_engineered_features(match_context.get('h2h_data', {})))
            
            # 9. Contextual and situational features
            features.update(self._create_contextual_features(home_data, away_data, match_context))
            
            # 10. Composite and ratio features
            features.update(self._create_composite_features(features))
            
            # Store feature names for importance analysis
            self.all_features = list(features.keys())
            
            return features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return self._get_default_features()
    
    def _create_basic_features(self, home_data: Dict, away_data: Dict) -> Dict:
        """Create basic statistical features"""
        features = {}
        
        # Goals features
        features['home_goals_scored_avg'] = home_data.get('goals_for_avg', 0)
        features['home_goals_conceded_avg'] = home_data.get('goals_against_avg', 0)
        features['away_goals_scored_avg'] = away_data.get('goals_for_avg', 0)
        features['away_goals_conceded_avg'] = away_data.get('goals_against_avg', 0)
        
        # Points and form
        features['home_points_per_game'] = home_data.get('points_per_game', 0)
        features['away_points_per_game'] = away_data.get('points_per_game', 0)
        features['home_win_rate'] = home_data.get('win_rate', 0)
        features['away_win_rate'] = away_data.get('win_rate', 0)
        
        # Recent form (last 5 games)
        features['home_recent_ppg'] = home_data.get('recent_form', {}).get('points_per_game', 0)
        features['away_recent_ppg'] = away_data.get('recent_form', {}).get('points_per_game', 0)
        
        return features
    
    def _create_advanced_statistical_features(self, home_data: Dict, away_data: Dict) -> Dict:
        """Create advanced statistical features"""
        features = {}
        
        # Expected goals (xG) based features
        home_xg = home_data.get('xG', home_data.get('goals_for_avg', 0))
        away_xg = away_data.get('xG', away_data.get('goals_for_avg', 0))
        home_xga = home_data.get('xGA', home_data.get('goals_against_avg', 0))
        away_xga = away_data.get('xGA', away_data.get('goals_against_avg', 0))
        
        features['home_xg'] = home_xg
        features['away_xg'] = away_xg
        features['home_xga'] = home_xga
        features['away_xga'] = away_xga
        
        # xG differences and ratios
        features['xg_difference'] = home_xg - away_xg
        features['xga_difference'] = home_xga - away_xga
        features['home_xg_ratio'] = home_xg / (home_xg + away_xg + 0.001)
        features['defensive_balance'] = (home_xga + away_xga) / (home_xg + away_xg + 0.001)
        
        # Shot-based features
        home_shots = home_data.get('shots_per_game', 0)
        away_shots = away_data.get('shots_per_game', 0)
        home_shots_on_target = home_data.get('shots_on_target_per_game', 0)
        away_shots_on_target = away_data.get('shots_on_target_per_game', 0)
        
        features['home_shot_accuracy'] = home_shots_on_target / (home_shots + 0.001)
        features['away_shot_accuracy'] = away_shots_on_target / (away_shots + 0.001)
        features['shot_volume_diff'] = home_shots - away_shots
        
        # Conversion rates
        features['home_goal_conversion'] = home_xg / (home_shots + 0.001)
        features['away_goal_conversion'] = away_xg / (away_shots + 0.001)
        
        # Defensive metrics
        features['home_defensive_actions'] = home_data.get('defensive_actions_per_game', 0)
        features['away_defensive_actions'] = away_data.get('defensive_actions_per_game', 0)
        features['home_clean_sheet_rate'] = home_data.get('clean_sheet_rate', 0)
        features['away_clean_sheet_rate'] = away_data.get('clean_sheet_rate', 0)
        
        # Possession and passing
        home_possession = home_data.get('possession_avg', 50)
        away_possession = away_data.get('possession_avg', 50)
        features['possession_differential'] = home_possession - away_possession
        features['home_pass_accuracy'] = home_data.get('pass_accuracy', 0)
        features['away_pass_accuracy'] = away_data.get('pass_accuracy', 0)
        
        # Set pieces
        features['home_corners_per_game'] = home_data.get('corners_per_game', 0)
        features['away_corners_per_game'] = away_data.get('corners_per_game', 0)
        features['home_fouls_per_game'] = home_data.get('fouls_per_game', 0)
        features['away_fouls_per_game'] = away_data.get('fouls_per_game', 0)
        
        return features
    
    def _create_interaction_features(self, home_data: Dict, away_data: Dict) -> Dict:
        """Create interaction features between teams"""
        features = {}
        
        # Attack vs Defense interactions
        home_attack = home_data.get('xG', home_data.get('goals_for_avg', 0))
        away_defense = away_data.get('xGA', away_data.get('goals_against_avg', 0))
        away_attack = away_data.get('xG', away_data.get('goals_for_avg', 0))
        home_defense = home_data.get('xGA', home_data.get('goals_against_avg', 0))
        
        features['home_attack_vs_away_defense'] = home_attack * away_defense
        features['away_attack_vs_home_defense'] = away_attack * home_defense
        features['attack_defense_balance'] = (home_attack + away_attack) / (home_defense + away_defense + 0.001)
        
        # Form interactions
        home_form = home_data.get('form_score', 50) / 100
        away_form = away_data.get('form_score', 50) / 100
        features['form_product'] = home_form * away_form
        features['form_ratio'] = home_form / (away_form + 0.001)
        features['form_difference_squared'] = (home_form - away_form) ** 2
        
        # Momentum interactions
        home_momentum = home_data.get('momentum', 0)
        away_momentum = away_data.get('momentum', 0)
        features['momentum_clash'] = abs(home_momentum - away_momentum)
        features['momentum_product'] = home_momentum * away_momentum
        
        # Style clash features
        home_tempo = home_data.get('tempo_score', 0.5)
        away_tempo = away_data.get('tempo_score', 0.5)
        features['tempo_mismatch'] = abs(home_tempo - away_tempo)
        
        home_pressing = home_data.get('pressing_intensity', 0.5)
        away_pressing = away_data.get('pressing_intensity', 0.5)
        features['pressing_differential'] = home_pressing - away_pressing
        
        # Strength vs weakness interactions
        home_home_strength = home_data.get('home_strength', 0.5)
        away_away_strength = away_data.get('away_strength', 0.5)
        features['venue_strength_interaction'] = home_home_strength * (1 - away_away_strength)
        
        # Goal expectation interactions
        features['total_goals_expected'] = home_attack + away_attack
        features['goals_variance'] = abs(home_attack - away_attack)
        features['defensive_solidity'] = 1 / (home_defense + away_defense + 0.1)
        
        return features
    
    def _create_temporal_features(self, match_context: Dict) -> Dict:
        """Create temporal features based on match timing"""
        features = {}
        
        # Extract match datetime
        match_datetime = match_context.get('datetime')
        if isinstance(match_datetime, str):
            try:
                match_datetime = datetime.fromisoformat(match_datetime.replace('Z', '+00:00'))
            except:
                match_datetime = datetime.now()
        elif not match_datetime:
            match_datetime = datetime.now()
        
        # Time of day features
        hour = match_datetime.hour
        features['match_hour'] = hour
        features['is_evening_match'] = 1 if 18 <= hour <= 22 else 0
        features['is_afternoon_match'] = 1 if 14 <= hour < 18 else 0
        features['is_night_match'] = 1 if hour >= 22 or hour < 2 else 0
        
        # Day of week features
        day_of_week = match_datetime.weekday()
        features['day_of_week'] = day_of_week
        features['is_weekend'] = 1 if day_of_week >= 5 else 0
        features['is_midweek'] = 1 if 1 <= day_of_week <= 3 else 0
        
        # Season progress features
        month = match_datetime.month
        features['month'] = month
        features['is_season_start'] = 1 if month in [8, 9] else 0
        features['is_season_end'] = 1 if month in [4, 5] else 0
        features['is_winter'] = 1 if month in [12, 1, 2] else 0
        
        # Season progress percentage (assuming Aug-May season)
        if month >= 8:
            season_progress = (month - 8) / 10
        else:
            season_progress = (month + 4) / 10
        features['season_progress'] = min(1.0, max(0.0, season_progress))
        
        # Holiday period features
        features['is_holiday_period'] = 1 if month == 12 and 20 <= match_datetime.day <= 31 else 0
        
        # Days since last match (if available)
        home_last_match = match_context.get('home_last_match_date')
        away_last_match = match_context.get('away_last_match_date')
        
        if home_last_match:
            if isinstance(home_last_match, str):
                home_last_match = datetime.fromisoformat(home_last_match.replace('Z', '+00:00'))
            home_days_rest = (match_datetime - home_last_match).days
            features['home_days_rest'] = home_days_rest
            features['home_short_rest'] = 1 if home_days_rest < 4 else 0
            features['home_long_rest'] = 1 if home_days_rest > 7 else 0
        
        if away_last_match:
            if isinstance(away_last_match, str):
                away_last_match = datetime.fromisoformat(away_last_match.replace('Z', '+00:00'))
            away_days_rest = (match_datetime - away_last_match).days
            features['away_days_rest'] = away_days_rest
            features['away_short_rest'] = 1 if away_days_rest < 4 else 0
            features['away_long_rest'] = 1 if away_days_rest > 7 else 0
        
        # Match importance temporal features
        features['is_crucial_period'] = 1 if season_progress > 0.7 or season_progress < 0.1 else 0
        
        return features
    
    def _create_dynamic_temporal_features(self, home_data: Dict, away_data: Dict, match_context: Dict) -> Dict:
        """
        Create advanced dynamic time-weighted features using DynamicTimeAnalyzer
        
        Implements:
        - Exponential decay weighting (last 30 days)
        - Seasonal form curve fitting
        - Weekly performance analysis  
        - Temporal pattern recognition
        """
        try:
            dynamic_features = {}
            
            # Analyze temporal features for both teams
            home_temporal = self.dynamic_time_analyzer.analyze_temporal_features(home_data, match_context)
            away_temporal = self.dynamic_time_analyzer.analyze_temporal_features(away_data, match_context)
            
            # 1. Exponential Decay Weighted Features
            home_decay = home_temporal.get('exponential_decay', {})
            away_decay = away_temporal.get('exponential_decay', {})
            
            # Time-weighted performance scores
            dynamic_features['home_time_weighted_score'] = home_decay.get('time_weighted_score', 50.0)
            dynamic_features['away_time_weighted_score'] = away_decay.get('time_weighted_score', 50.0)
            dynamic_features['time_weighted_advantage'] = (
                home_decay.get('time_weighted_score', 50.0) - away_decay.get('time_weighted_score', 50.0)
            )
            
            # Recent form strength and momentum
            home_recent = home_decay.get('recent_strength', {})
            away_recent = away_decay.get('recent_strength', {})
            dynamic_features['home_temporal_momentum'] = home_recent.get('momentum', 0.0)
            dynamic_features['away_temporal_momentum'] = away_recent.get('momentum', 0.0)
            dynamic_features['home_form_consistency'] = home_recent.get('consistency', 0.5)
            dynamic_features['away_form_consistency'] = away_recent.get('consistency', 0.5)
            
            # Performance trends
            home_trend = home_decay.get('trend_analysis', {})
            away_trend = away_decay.get('trend_analysis', {})
            dynamic_features['home_performance_trend'] = 1 if home_trend.get('trend') == 'improving' else -1 if home_trend.get('trend') == 'declining' else 0
            dynamic_features['away_performance_trend'] = 1 if away_trend.get('trend') == 'improving' else -1 if away_trend.get('trend') == 'declining' else 0
            dynamic_features['trend_difference'] = dynamic_features['home_performance_trend'] - dynamic_features['away_performance_trend']
            
            # 2. Seasonal Adjustment Features
            home_seasonal = home_temporal.get('seasonal_analysis', {})
            away_seasonal = away_temporal.get('seasonal_analysis', {})
            
            # Seasonal adjustment factors
            dynamic_features['home_seasonal_adjustment'] = home_seasonal.get('current_adjustment_factor', 1.0)
            dynamic_features['away_seasonal_adjustment'] = away_seasonal.get('current_adjustment_factor', 1.0)
            dynamic_features['seasonal_advantage'] = (
                home_seasonal.get('current_adjustment_factor', 1.0) - away_seasonal.get('current_adjustment_factor', 1.0)
            )
            
            # Seasonal form scores
            dynamic_features['home_seasonal_form'] = home_seasonal.get('seasonal_form_score', 50.0)
            dynamic_features['away_seasonal_form'] = away_seasonal.get('seasonal_form_score', 50.0)
            
            # Holiday and season effects
            home_holiday = home_seasonal.get('holiday_effects', {})
            away_holiday = away_seasonal.get('holiday_effects', {})
            dynamic_features['home_holiday_effect'] = home_holiday.get('holiday_effect', 0.0)
            dynamic_features['away_holiday_effect'] = away_holiday.get('holiday_effect', 0.0)
            
            # Season effects (start/end performance)
            home_effects = home_seasonal.get('season_effects', {})
            away_effects = away_seasonal.get('season_effects', {})
            if 'season_start' in home_effects:
                dynamic_features['home_season_start_effect'] = 1 if home_effects['season_start'].get('is_fast_starter', False) else -1 if home_effects['season_start'].get('is_slow_starter', False) else 0
            if 'season_start' in away_effects:
                dynamic_features['away_season_start_effect'] = 1 if away_effects['season_start'].get('is_fast_starter', False) else -1 if away_effects['season_start'].get('is_slow_starter', False) else 0
            
            # 3. Weekly Performance Features
            home_weekly = home_temporal.get('weekly_patterns', {})
            away_weekly = away_temporal.get('weekly_patterns', {})
            
            # Weekly advantage scores
            dynamic_features['home_weekly_advantage'] = home_weekly.get('weekly_advantage_score', 50.0)
            dynamic_features['away_weekly_advantage'] = away_weekly.get('weekly_advantage_score', 50.0)
            dynamic_features['weekly_advantage_diff'] = (
                home_weekly.get('weekly_advantage_score', 50.0) - away_weekly.get('weekly_advantage_score', 50.0)
            )
            
            # Weekend vs midweek performance
            home_weekend = home_weekly.get('weekend_vs_midweek', {})
            away_weekend = away_weekly.get('weekend_vs_midweek', {})
            dynamic_features['home_weekend_advantage'] = 1 if home_weekend.get('weekend_advantage', False) else 0
            dynamic_features['away_weekend_advantage'] = 1 if away_weekend.get('weekend_advantage', False) else 0
            dynamic_features['home_weekend_effect'] = home_weekend.get('difference', 0.0)
            dynamic_features['away_weekend_effect'] = away_weekend.get('difference', 0.0)
            
            # Recovery patterns
            home_recovery = home_weekly.get('recovery_patterns', {})
            away_recovery = away_weekly.get('recovery_patterns', {})
            dynamic_features['home_optimal_recovery'] = home_recovery.get('optimal_recovery', 7)
            dynamic_features['away_optimal_recovery'] = away_recovery.get('optimal_recovery', 7)
            dynamic_features['home_recovery_effect'] = home_recovery.get('recovery_effect', 0.0)
            dynamic_features['away_recovery_effect'] = away_recovery.get('recovery_effect', 0.0)
            
            # Weekly rhythm detection
            home_rhythm = home_weekly.get('weekly_rhythm', {})
            away_rhythm = away_weekly.get('weekly_rhythm', {})
            dynamic_features['home_rhythm_detected'] = 1 if home_rhythm.get('rhythm_detected', False) else 0
            dynamic_features['away_rhythm_detected'] = 1 if away_rhythm.get('rhythm_detected', False) else 0
            dynamic_features['home_rhythm_strength'] = home_rhythm.get('rhythm_strength', 0.0)
            dynamic_features['away_rhythm_strength'] = away_rhythm.get('rhythm_strength', 0.0)
            
            # 4. Temporal Pattern Recognition Features
            home_patterns = home_temporal.get('temporal_patterns', {})
            away_patterns = away_temporal.get('temporal_patterns', {})
            
            # Monthly cycle performance
            home_monthly = home_patterns.get('monthly_cycles', {})
            away_monthly = away_patterns.get('monthly_cycles', {})
            
            # Best/worst months indicators
            current_month = match_context.get('match_date', datetime.now())
            if isinstance(current_month, str):
                current_month = datetime.strptime(current_month, '%Y-%m-%d')
            month_num = current_month.month
            
            home_best_months = home_monthly.get('best_months', [])
            away_best_months = away_monthly.get('best_months', [])
            dynamic_features['home_in_best_month'] = 1 if month_num in home_best_months else 0
            dynamic_features['away_in_best_month'] = 1 if month_num in away_best_months else 0
            
            home_worst_months = home_monthly.get('worst_months', [])
            away_worst_months = away_monthly.get('worst_months', [])
            dynamic_features['home_in_worst_month'] = 1 if month_num in home_worst_months else 0
            dynamic_features['away_in_worst_month'] = 1 if month_num in away_worst_months else 0
            
            # Opponent-specific patterns
            home_opponent = home_patterns.get('opponent_specific', {})
            away_opponent = away_patterns.get('opponent_specific', {})
            dynamic_features['home_h2h_patterns_found'] = 1 if home_opponent.get('patterns_found', False) else 0
            dynamic_features['away_h2h_patterns_found'] = 1 if away_opponent.get('patterns_found', False) else 0
            
            if home_opponent.get('patterns_found', False):
                h2h_temporal = home_opponent.get('temporal_patterns', {})
                dynamic_features['home_h2h_performance'] = h2h_temporal.get('avg_performance', 50.0)
            
            # Transfer window effects
            home_transfer = home_patterns.get('transfer_window_effects', {})
            away_transfer = away_patterns.get('transfer_window_effects', {})
            dynamic_features['home_transfer_effect'] = home_transfer.get('transfer_effect', 0.0)
            dynamic_features['away_transfer_effect'] = away_transfer.get('transfer_effect', 0.0)
            dynamic_features['home_transfer_sensitive'] = 1 if home_transfer.get('is_transfer_sensitive', False) else 0
            dynamic_features['away_transfer_sensitive'] = 1 if away_transfer.get('is_transfer_sensitive', False) else 0
            
            # Pattern predictions
            home_prediction = home_patterns.get('pattern_prediction', {})
            away_prediction = away_patterns.get('pattern_prediction', {})
            dynamic_features['home_pattern_confidence'] = home_prediction.get('confidence', 0.0)
            dynamic_features['away_pattern_confidence'] = away_prediction.get('confidence', 0.0)
            
            # 5. Combined Temporal Indicators
            home_combined = home_temporal.get('combined_indicators', {})
            away_combined = away_temporal.get('combined_indicators', {})
            
            # Overall temporal scores
            dynamic_features['home_overall_temporal_score'] = home_combined.get('overall_temporal_score', 50.0)
            dynamic_features['away_overall_temporal_score'] = away_combined.get('overall_temporal_score', 50.0)
            dynamic_features['temporal_score_advantage'] = (
                home_combined.get('overall_temporal_score', 50.0) - away_combined.get('overall_temporal_score', 50.0)
            )
            
            # Temporal momentum
            dynamic_features['home_temporal_momentum_combined'] = home_combined.get('temporal_momentum', 0.0)
            dynamic_features['away_temporal_momentum_combined'] = away_combined.get('temporal_momentum', 0.0)
            dynamic_features['momentum_differential'] = (
                home_combined.get('temporal_momentum', 0.0) - away_combined.get('temporal_momentum', 0.0)
            )
            
            # Confidence levels
            dynamic_features['home_temporal_confidence'] = home_combined.get('confidence_level', 0.5)
            dynamic_features['away_temporal_confidence'] = away_combined.get('confidence_level', 0.5)
            dynamic_features['avg_temporal_confidence'] = (
                home_combined.get('confidence_level', 0.5) + away_combined.get('confidence_level', 0.5)
            ) / 2
            
            # Advantage indicators
            home_advantages = home_combined.get('advantage_indicators', {})
            away_advantages = away_combined.get('advantage_indicators', {})
            
            dynamic_features['home_time_advantage'] = home_advantages.get('time_advantage', 0.5)
            dynamic_features['away_time_advantage'] = away_advantages.get('time_advantage', 0.5)
            dynamic_features['home_seasonal_advantage_indicator'] = home_advantages.get('seasonal_advantage', 1.0)
            dynamic_features['away_seasonal_advantage_indicator'] = away_advantages.get('seasonal_advantage', 1.0)
            
            # 6. Performance Prediction Curves
            home_curves = home_combined.get('performance_curves', {})
            away_curves = away_combined.get('performance_curves', {})
            
            dynamic_features['home_performance_trend_direction'] = 1 if home_curves.get('trend_direction') == 'improving' else -1 if home_curves.get('trend_direction') == 'declining' else 0
            dynamic_features['away_performance_trend_direction'] = 1 if away_curves.get('trend_direction') == 'improving' else -1 if away_curves.get('trend_direction') == 'declining' else 0
            dynamic_features['home_performance_volatility'] = home_curves.get('volatility', 0.0)
            dynamic_features['away_performance_volatility'] = away_curves.get('volatility', 0.0)
            
            # 7. Timing Recommendations
            home_timing = home_combined.get('timing_recommendations', {})
            away_timing = away_combined.get('timing_recommendations', {})
            
            dynamic_features['home_timing_score'] = home_timing.get('current_timing_score', 50.0)
            dynamic_features['away_timing_score'] = away_timing.get('current_timing_score', 50.0)
            dynamic_features['timing_advantage'] = (
                home_timing.get('current_timing_score', 50.0) - away_timing.get('current_timing_score', 50.0)
            )
            
            # 8. Composite Features
            # Overall temporal advantage (composite score)
            temporal_advantages = [
                dynamic_features.get('time_weighted_advantage', 0.0) / 10,  # Normalize
                dynamic_features.get('seasonal_advantage', 0.0),
                dynamic_features.get('weekly_advantage_diff', 0.0) / 10,  # Normalize
                dynamic_features.get('momentum_differential', 0.0) * 10,  # Scale up
                dynamic_features.get('timing_advantage', 0.0) / 10  # Normalize
            ]
            dynamic_features['composite_temporal_advantage'] = np.mean(temporal_advantages)
            
            # Temporal stability score
            home_stability = (
                dynamic_features.get('home_form_consistency', 0.5) +
                dynamic_features.get('home_temporal_confidence', 0.5) +
                (1.0 - dynamic_features.get('home_performance_volatility', 0.0) / 10)
            ) / 3
            
            away_stability = (
                dynamic_features.get('away_form_consistency', 0.5) +
                dynamic_features.get('away_temporal_confidence', 0.5) +
                (1.0 - dynamic_features.get('away_performance_volatility', 0.0) / 10)
            ) / 3
            
            dynamic_features['home_temporal_stability'] = home_stability
            dynamic_features['away_temporal_stability'] = away_stability
            dynamic_features['stability_advantage'] = home_stability - away_stability
            
            # Feature importance scores
            feature_importance = self.dynamic_time_analyzer.get_feature_importance()
            dynamic_features['temporal_feature_count'] = feature_importance.get('total_features_generated', 0)
            
            logger.info(f"Generated {len(dynamic_features)} dynamic temporal features")
            
            return dynamic_features
            
        except Exception as e:
            logger.error(f"Error creating dynamic temporal features: {str(e)}")
            return self._get_default_dynamic_temporal_features()
    
    def _get_default_dynamic_temporal_features(self) -> Dict:
        """Return default dynamic temporal features when analysis fails"""
        return {
            'home_time_weighted_score': 50.0,
            'away_time_weighted_score': 50.0,
            'time_weighted_advantage': 0.0,
            'home_temporal_momentum': 0.0,
            'away_temporal_momentum': 0.0,
            'home_seasonal_adjustment': 1.0,
            'away_seasonal_adjustment': 1.0,
            'seasonal_advantage': 0.0,
            'home_weekly_advantage': 50.0,
            'away_weekly_advantage': 50.0,
            'weekly_advantage_diff': 0.0,
            'home_overall_temporal_score': 50.0,
            'away_overall_temporal_score': 50.0,
            'temporal_score_advantage': 0.0,
            'composite_temporal_advantage': 0.0,
            'home_temporal_stability': 0.5,
            'away_temporal_stability': 0.5,
            'stability_advantage': 0.0,
            'avg_temporal_confidence': 0.3
        }
    
    def _create_league_specific_features(self, league_id: int, home_data: Dict, away_data: Dict) -> Dict:
        """Create features specific to league characteristics"""
        features = {}
        
        # League characteristics mapping
        league_characteristics = {
            # High-scoring leagues
            39: {'scoring_level': 'high', 'tempo': 'fast', 'physicality': 'medium'},    # Premier League
            140: {'scoring_level': 'high', 'tempo': 'fast', 'physicality': 'low'},      # La Liga
            78: {'scoring_level': 'very_high', 'tempo': 'very_fast', 'physicality': 'medium'},  # Bundesliga
            
            # Low-scoring leagues
            135: {'scoring_level': 'low', 'tempo': 'slow', 'physicality': 'low'},       # Serie A
            61: {'scoring_level': 'low', 'tempo': 'medium', 'physicality': 'high'},     # Ligue 1
            
            # Medium-scoring leagues
            88: {'scoring_level': 'medium', 'tempo': 'medium', 'physicality': 'medium'}, # Eredivisie
            94: {'scoring_level': 'medium', 'tempo': 'slow', 'physicality': 'medium'},   # Primeira Liga
            203: {'scoring_level': 'medium', 'tempo': 'medium', 'physicality': 'high'},  # Super Lig
        }
        
        # Get league characteristics or use default
        league_chars = league_characteristics.get(league_id, {
            'scoring_level': 'medium',
            'tempo': 'medium',
            'physicality': 'medium'
        })
        
        # Encode league characteristics
        scoring_levels = {'very_low': 0, 'low': 0.25, 'medium': 0.5, 'high': 0.75, 'very_high': 1.0}
        tempo_levels = {'very_slow': 0, 'slow': 0.25, 'medium': 0.5, 'fast': 0.75, 'very_fast': 1.0}
        physicality_levels = {'low': 0, 'medium': 0.5, 'high': 1.0}
        
        features['league_scoring_tendency'] = scoring_levels.get(league_chars['scoring_level'], 0.5)
        features['league_tempo'] = tempo_levels.get(league_chars['tempo'], 0.5)
        features['league_physicality'] = physicality_levels.get(league_chars['physicality'], 0.5)
        
        # Team adaptation to league style
        home_goals_avg = home_data.get('goals_for_avg', 0)
        away_goals_avg = away_data.get('goals_for_avg', 0)
        league_avg_goals = 2.5 * features['league_scoring_tendency'] + 1.5
        
        features['home_league_adaptation'] = home_goals_avg / (league_avg_goals + 0.001)
        features['away_league_adaptation'] = away_goals_avg / (league_avg_goals + 0.001)
        
        # League-specific tactical features
        if league_id in [39, 78]:  # Premier League, Bundesliga
            features['high_pressing_league'] = 1
            features['counter_attack_importance'] = 0.8
        elif league_id in [135, 94]:  # Serie A, Primeira Liga
            features['tactical_league'] = 1
            features['possession_importance'] = 0.9
        else:
            features['balanced_league'] = 1
            features['flexibility_importance'] = 0.7
        
        # Home advantage by league
        home_advantage_by_league = {
            39: 0.58,   # Premier League
            140: 0.61,  # La Liga
            78: 0.55,   # Bundesliga
            135: 0.59,  # Serie A
            61: 0.60,   # Ligue 1
            203: 0.65,  # Super Lig (traditionally high)
        }
        features['league_home_advantage'] = home_advantage_by_league.get(league_id, 0.6)
        
        # League competitiveness (affects unpredictability)
        league_competitiveness = {
            39: 0.8,   # Premier League (very competitive)
            78: 0.7,   # Bundesliga
            140: 0.6,  # La Liga (top-heavy)
            61: 0.5,   # Ligue 1 (PSG dominance)
        }
        features['league_competitiveness'] = league_competitiveness.get(league_id, 0.6)
        
        return features
    
    def _create_form_engineered_features(self, home_data: Dict, away_data: Dict) -> Dict:
        """Create engineered features from form data"""
        features = {}
        
        # Form trajectory features
        home_form = home_data.get('form_analysis', {})
        away_form = away_data.get('form_analysis', {})
        
        # Momentum features
        home_momentum = home_form.get('momentum_shifts', {}).get('momentum_change', 0)
        away_momentum = away_form.get('momentum_shifts', {}).get('momentum_change', 0)
        
        features['momentum_differential'] = home_momentum - away_momentum
        features['momentum_conflict'] = 1 if (home_momentum > 0.5 and away_momentum < -0.5) else 0
        features['both_improving'] = 1 if (home_momentum > 0 and away_momentum > 0) else 0
        features['both_declining'] = 1 if (home_momentum < 0 and away_momentum < 0) else 0
        
        # Form volatility
        home_volatility = home_form.get('momentum_shifts', {}).get('volatility', 0)
        away_volatility = away_form.get('momentum_shifts', {}).get('volatility', 0)
        features['form_volatility_sum'] = home_volatility + away_volatility
        features['volatility_mismatch'] = abs(home_volatility - away_volatility)
        
        # Streak features
        home_streak = home_form.get('streak_analysis', {}).get('current_streak', {})
        away_streak = away_form.get('streak_analysis', {}).get('current_streak', {})
        
        features['home_on_win_streak'] = 1 if home_streak.get('type') == 'W' and home_streak.get('length', 0) >= 3 else 0
        features['away_on_win_streak'] = 1 if away_streak.get('type') == 'W' and away_streak.get('length', 0) >= 3 else 0
        features['home_on_losing_streak'] = 1 if home_streak.get('type') == 'L' and home_streak.get('length', 0) >= 2 else 0
        features['away_on_losing_streak'] = 1 if away_streak.get('type') == 'L' and away_streak.get('length', 0) >= 2 else 0
        
        # Pressure performance
        home_pressure = home_form.get('pressure_performance', {})
        away_pressure = away_form.get('pressure_performance', {})
        
        features['home_clutch_factor'] = home_pressure.get('clutch_factor', 0)
        features['away_clutch_factor'] = away_pressure.get('clutch_factor', 0)
        features['mental_edge'] = home_pressure.get('mental_strength', 0.5) - away_pressure.get('mental_strength', 0.5)
        
        # Form windows comparison
        home_windows = home_form.get('rolling_windows', {})
        away_windows = away_form.get('rolling_windows', {})
        
        # Short vs long form
        home_short_ppg = home_windows.get('short', {}).get('points_per_game', 0)
        home_long_ppg = home_windows.get('long', {}).get('points_per_game', 0)
        away_short_ppg = away_windows.get('short', {}).get('points_per_game', 0)
        away_long_ppg = away_windows.get('long', {}).get('points_per_game', 0)
        
        features['home_form_improvement'] = home_short_ppg - home_long_ppg
        features['away_form_improvement'] = away_short_ppg - away_long_ppg
        features['form_trend_differential'] = features['home_form_improvement'] - features['away_form_improvement']
        
        return features
    
    def _create_momentum_features(self, home_data: Dict, away_data: Dict) -> Dict:
        """Create psychological and momentum features"""
        features = {}
        
        # Confidence features
        home_confidence = home_data.get('confidence_score', 0.5)
        away_confidence = away_data.get('confidence_score', 0.5)
        
        features['confidence_differential'] = home_confidence - away_confidence
        features['confidence_product'] = home_confidence * away_confidence
        features['low_confidence_match'] = 1 if (home_confidence < 0.3 and away_confidence < 0.3) else 0
        
        # Motivation features
        home_motivation = home_data.get('motivation_level', 0.5)
        away_motivation = away_data.get('motivation_level', 0.5)
        
        features['motivation_differential'] = home_motivation - away_motivation
        features['high_stakes_match'] = 1 if (home_motivation > 0.8 or away_motivation > 0.8) else 0
        features['motivation_mismatch'] = abs(home_motivation - away_motivation)
        
        # Psychological pressure
        home_position = home_data.get('league_position', 10)
        away_position = away_data.get('league_position', 10)
        
        features['position_differential'] = away_position - home_position  # Lower is better
        features['top_vs_bottom'] = 1 if abs(home_position - away_position) > 10 else 0
        features['relegation_battle'] = 1 if (home_position > 15 or away_position > 15) else 0
        features['title_race'] = 1 if (home_position <= 3 or away_position <= 3) else 0
        
        # Recent results impact
        home_last_results = home_data.get('last_5_results', [])
        away_last_results = away_data.get('last_5_results', [])
        
        if home_last_results:
            home_recent_wins = sum(1 for r in home_last_results if r == 'W')
            home_recent_losses = sum(1 for r in home_last_results if r == 'L')
            features['home_recent_win_rate'] = home_recent_wins / len(home_last_results)
            features['home_momentum_score'] = (home_recent_wins - home_recent_losses) / len(home_last_results)
        
        if away_last_results:
            away_recent_wins = sum(1 for r in away_last_results if r == 'W')
            away_recent_losses = sum(1 for r in away_last_results if r == 'L')
            features['away_recent_win_rate'] = away_recent_wins / len(away_last_results)
            features['away_momentum_score'] = (away_recent_wins - away_recent_losses) / len(away_last_results)
        
        return features
    
    def _create_h2h_engineered_features(self, h2h_data: Dict) -> Dict:
        """Create engineered features from head-to-head data"""
        features = {}
        
        if not h2h_data or not isinstance(h2h_data, dict):
            # Return default H2H features
            features['h2h_matches_played'] = 0
            features['h2h_home_dominance'] = 0.5
            features['h2h_goals_avg'] = 2.5
            features['h2h_btts_rate'] = 0.5
            return features
        
        # Basic H2H stats
        matches = h2h_data.get('matches', [])
        features['h2h_matches_played'] = len(matches)
        
        if matches:
            home_wins = 0
            away_wins = 0
            draws = 0
            total_goals = 0
            btts_count = 0
            over_2_5_count = 0
            
            for match in matches[:10]:  # Last 10 H2H matches
                home_score = match.get('home_score', 0)
                away_score = match.get('away_score', 0)
                
                if home_score > away_score:
                    home_wins += 1
                elif away_score > home_score:
                    away_wins += 1
                else:
                    draws += 1
                
                total_goals += (home_score + away_score)
                if home_score > 0 and away_score > 0:
                    btts_count += 1
                if home_score + away_score > 2.5:
                    over_2_5_count += 1
            
            num_matches = len(matches[:10])
            features['h2h_home_win_rate'] = home_wins / num_matches
            features['h2h_away_win_rate'] = away_wins / num_matches
            features['h2h_draw_rate'] = draws / num_matches
            features['h2h_home_dominance'] = (home_wins - away_wins) / num_matches
            features['h2h_goals_avg'] = total_goals / num_matches
            features['h2h_btts_rate'] = btts_count / num_matches
            features['h2h_over_2_5_rate'] = over_2_5_count / num_matches
            
            # Trend in H2H
            if len(matches) >= 5:
                recent_home_wins = sum(1 for m in matches[:3] if m.get('home_score', 0) > m.get('away_score', 0))
                older_home_wins = sum(1 for m in matches[3:6] if m.get('home_score', 0) > m.get('away_score', 0))
                features['h2h_trend'] = (recent_home_wins - older_home_wins) / 3
            
            # Goal patterns
            home_goals_list = [m.get('home_score', 0) for m in matches[:5]]
            away_goals_list = [m.get('away_score', 0) for m in matches[:5]]
            
            if home_goals_list:
                features['h2h_home_goals_trend'] = np.mean(home_goals_list[:2]) - np.mean(home_goals_list[2:])
                features['h2h_away_goals_trend'] = np.mean(away_goals_list[:2]) - np.mean(away_goals_list[2:])
        else:
            # No H2H history
            features['h2h_home_win_rate'] = 0.33
            features['h2h_away_win_rate'] = 0.33
            features['h2h_draw_rate'] = 0.34
            features['h2h_home_dominance'] = 0
            features['h2h_goals_avg'] = 2.5
            features['h2h_btts_rate'] = 0.5
            features['h2h_over_2_5_rate'] = 0.5
        
        return features
    
    def _create_contextual_features(self, home_data: Dict, away_data: Dict, match_context: Dict) -> Dict:
        """Create contextual and situational features"""
        features = {}
        
        # Derby and rivalry features
        features['is_derby'] = 1 if match_context.get('is_derby', False) else 0
        features['rivalry_intensity'] = match_context.get('rivalry_intensity', 0)
        
        # Cup vs League
        competition_type = match_context.get('competition_type', 'league')
        features['is_cup_match'] = 1 if 'cup' in competition_type.lower() else 0
        features['is_knockout'] = 1 if match_context.get('is_knockout', False) else 0
        
        # Match importance
        features['match_importance'] = match_context.get('importance_score', 0.5)
        features['must_win_home'] = 1 if home_data.get('must_win_situation', False) else 0
        features['must_win_away'] = 1 if away_data.get('must_win_situation', False) else 0
        
        # Manager factors
        home_manager_exp = home_data.get('manager_experience', 0.5)
        away_manager_exp = away_data.get('manager_experience', 0.5)
        features['manager_experience_diff'] = home_manager_exp - away_manager_exp
        
        # Squad depth and rotation
        features['home_squad_rotation'] = home_data.get('expected_rotation', 0)
        features['away_squad_rotation'] = away_data.get('expected_rotation', 0)
        features['fatigue_differential'] = home_data.get('fatigue_score', 0) - away_data.get('fatigue_score', 0)
        
        # Weather impact (if available)
        weather = match_context.get('weather', {})
        features['temperature'] = weather.get('temperature', 20) / 40  # Normalize
        features['is_rainy'] = 1 if weather.get('condition') == 'rain' else 0
        features['wind_speed'] = weather.get('wind_speed', 0) / 50  # Normalize
        
        # Attendance and crowd factors
        features['expected_attendance_ratio'] = match_context.get('attendance_ratio', 0.5)
        features['hostile_atmosphere'] = 1 if features['expected_attendance_ratio'] > 0.9 and features['is_derby'] else 0
        
        return features
    
    def _create_composite_features(self, existing_features: Dict) -> Dict:
        """Create composite features from existing features"""
        features = {}
        
        # Offensive power index
        if 'home_xg' in existing_features and 'home_shot_accuracy' in existing_features:
            features['home_offensive_index'] = (
                existing_features['home_xg'] * 0.4 +
                existing_features['home_goals_scored_avg'] * 0.3 +
                existing_features['home_shot_accuracy'] * 0.3
            )
        
        if 'away_xg' in existing_features and 'away_shot_accuracy' in existing_features:
            features['away_offensive_index'] = (
                existing_features['away_xg'] * 0.4 +
                existing_features['away_goals_scored_avg'] * 0.3 +
                existing_features['away_shot_accuracy'] * 0.3
            )
        
        # Defensive solidity index
        if 'home_xga' in existing_features and 'home_clean_sheet_rate' in existing_features:
            features['home_defensive_index'] = (
                (1 - existing_features['home_xga'] / 3) * 0.5 +
                existing_features['home_clean_sheet_rate'] * 0.5
            )
        
        if 'away_xga' in existing_features and 'away_clean_sheet_rate' in existing_features:
            features['away_defensive_index'] = (
                (1 - existing_features['away_xga'] / 3) * 0.5 +
                existing_features['away_clean_sheet_rate'] * 0.5
            )
        
        # Overall strength index
        if 'home_offensive_index' in features and 'home_defensive_index' in features:
            features['home_strength_index'] = (
                features['home_offensive_index'] * 0.4 +
                features['home_defensive_index'] * 0.3 +
                existing_features.get('home_points_per_game', 0) / 3 * 0.3
            )
        
        if 'away_offensive_index' in features and 'away_defensive_index' in features:
            features['away_strength_index'] = (
                features['away_offensive_index'] * 0.4 +
                features['away_defensive_index'] * 0.3 +
                existing_features.get('away_points_per_game', 0) / 3 * 0.3
            )
        
        # Match balance features
        if 'home_strength_index' in features and 'away_strength_index' in features:
            features['strength_differential'] = features['home_strength_index'] - features['away_strength_index']
            features['competitive_balance'] = 1 - abs(features['strength_differential'])
        
        # Goal expectation features
        if 'total_goals_expected' in existing_features:
            features['high_scoring_potential'] = 1 if existing_features['total_goals_expected'] > 3 else 0
            features['low_scoring_potential'] = 1 if existing_features['total_goals_expected'] < 2 else 0
        
        # Form and momentum composite
        if 'home_form_improvement' in existing_features and 'home_momentum_score' in existing_features:
            features['home_composite_momentum'] = (
                existing_features.get('home_form_improvement', 0) * 0.5 +
                existing_features.get('home_momentum_score', 0) * 0.5
            )
        
        if 'away_form_improvement' in existing_features and 'away_momentum_score' in existing_features:
            features['away_composite_momentum'] = (
                existing_features.get('away_form_improvement', 0) * 0.5 +
                existing_features.get('away_momentum_score', 0) * 0.5
            )
        
        return features
    
    def select_features(self, features: Dict, target: np.ndarray = None, method: str = 'importance', k: int = 50) -> Dict:
        """
        Select most important features using various methods
        
        Args:
            features: Dictionary of all features
            target: Target variable for supervised selection
            method: Selection method ('importance', 'univariate', 'pca')
            k: Number of features to select
            
        Returns:
            Dictionary of selected features
        """
        if not features:
            return features
            
        # Convert to numpy array
        feature_names = list(features.keys())
        feature_values = np.array(list(features.values())).reshape(1, -1)
        
        selected_features = {}
        
        if method == 'importance' and self.feature_importance:
            # Select by stored importance scores
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature_name, _ in sorted_features[:k]:
                if feature_name in features:
                    selected_features[feature_name] = features[feature_name]
        
        elif method == 'univariate' and target is not None:
            # Univariate feature selection
            selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_names)))
            # Ensure target is numpy array
            if not isinstance(target, np.ndarray):
                target = np.array(target)
            selector.fit(feature_values, target.reshape(-1))
            selected_indices = selector.get_support(indices=True)
            
            if selected_indices is not None:
                for idx in selected_indices:
                    feature_name = feature_names[idx]
                    selected_features[feature_name] = features[feature_name]
        
        elif method == 'variance':
            # Remove low variance features
            variances = []
            for feature_name in feature_names:
                # Calculate variance from historical data if available
                variance = self._calculate_feature_variance(feature_name)
                variances.append((feature_name, variance))
            
            # Sort by variance and select top k
            sorted_variances = sorted(variances, key=lambda x: x[1], reverse=True)
            for feature_name, _ in sorted_variances[:k]:
                selected_features[feature_name] = features[feature_name]
        
        else:
            # If no specific method or fallback, return top k features
            for i, feature_name in enumerate(feature_names[:k]):
                selected_features[feature_name] = features[feature_name]
        
        self.selected_features = list(selected_features.keys())
        return selected_features
    
    def calculate_feature_importance(self, features: Dict, predictions: Dict, actual_outcomes: Dict = None) -> Dict:
        """
        Calculate and update feature importance scores
        
        Args:
            features: Features used for prediction
            predictions: Model predictions
            actual_outcomes: Actual match outcomes (if available)
            
        Returns:
            Updated feature importance scores
        """
        if not actual_outcomes or not features:
            # Use heuristic importance based on feature characteristics
            for feature_name, feature_value in (features or {}).items():
                if feature_name not in self.feature_importance:
                    self.feature_importance[feature_name] = 0.5
                
                # Update importance based on feature value extremity
                if isinstance(feature_value, (int, float)):
                    # Features with extreme values are often important
                    extremity = abs(feature_value - 0.5) if 0 <= feature_value <= 1 else abs(feature_value) / 10
                    self.feature_importance[feature_name] = (
                        0.9 * self.feature_importance[feature_name] + 
                        0.1 * extremity
                    )
        else:
            # Calculate importance based on prediction accuracy
            prediction_error = self._calculate_prediction_error(predictions, actual_outcomes)
            
            # Use permutation importance concept
            base_error = prediction_error
            
            for feature_name in features:
                # Simulate feature permutation by setting to mean
                permuted_features = features.copy()
                permuted_features[feature_name] = 0.5  # Neutral value
                
                # Calculate error with permuted feature
                # This is simplified - in practice would re-run prediction
                permuted_error = base_error * (1 + abs(features[feature_name] - 0.5))
                
                # Importance is proportional to error increase
                importance = (permuted_error - base_error) / (base_error + 0.001)
                
                if feature_name not in self.feature_importance:
                    self.feature_importance[feature_name] = importance
                else:
                    # Exponential moving average
                    self.feature_importance[feature_name] = (
                        0.9 * self.feature_importance[feature_name] + 
                        0.1 * importance
                    )
        
        return self.feature_importance
    
    def get_feature_explanations(self, features: Dict, top_n: int = 10) -> List[Dict]:
        """
        Get human-readable explanations for top features
        
        Args:
            features: Dictionary of features
            top_n: Number of top features to explain
            
        Returns:
            List of feature explanations
        """
        explanations = []
        
        # Sort features by importance
        if self.feature_importance:
            sorted_features = sorted(
                [(name, value) for name, value in features.items() if name in self.feature_importance],
                key=lambda x: self.feature_importance[x[0]],
                reverse=True
            )
        else:
            sorted_features = list(features.items())[:top_n]
        
        # Feature explanation templates
        feature_explanations = {
            'home_offensive_index': "Home team's attacking strength (xG, goals, shot accuracy)",
            'away_offensive_index': "Away team's attacking strength (xG, goals, shot accuracy)",
            'strength_differential': "Overall quality difference between teams",
            'momentum_differential': "Current form momentum difference",
            'h2h_home_dominance': "Historical dominance in head-to-head matches",
            'venue_strength_interaction': "Home advantage vs away team's away form",
            'form_trend_differential': "Recent form improvement comparison",
            'total_goals_expected': "Expected total goals based on team stats",
            'league_scoring_tendency': "League's typical scoring level",
            'is_derby': "Local rivalry match with heightened intensity",
            'season_progress': "Stage of the season (early/middle/late)",
            'confidence_differential': "Team confidence level difference",
            'both_improving': "Both teams showing improving form",
            'h2h_goals_avg': "Average goals in previous meetings",
            'competitive_balance': "How evenly matched the teams are"
        }
        
        for feature_name, feature_value in sorted_features[:top_n]:
            explanation = {
                'feature': feature_name,
                'value': feature_value,
                'importance': self.feature_importance.get(feature_name, 0.5),
                'description': feature_explanations.get(feature_name, feature_name.replace('_', ' ').title())
            }
            
            # Add impact direction
            if isinstance(feature_value, (int, float)):
                if feature_value > 0.6:
                    explanation['impact'] = 'positive'
                elif feature_value < 0.4:
                    explanation['impact'] = 'negative'
                else:
                    explanation['impact'] = 'neutral'
            else:
                explanation['impact'] = 'categorical'
            
            explanations.append(explanation)
        
        return explanations
    
    def _calculate_feature_variance(self, feature_name: str) -> float:
        """Calculate variance for a feature based on historical data"""
        # Simplified - in practice would use historical feature values
        # For now, use feature name patterns to estimate variance
        
        high_variance_patterns = ['momentum', 'form', 'streak', 'recent', 'trend']
        low_variance_patterns = ['league', 'is_', 'venue_advantage']
        
        for pattern in high_variance_patterns:
            if pattern in feature_name.lower():
                return 0.8
        
        for pattern in low_variance_patterns:
            if pattern in feature_name.lower():
                return 0.2
        
        return 0.5  # Default medium variance
    
    def _calculate_prediction_error(self, predictions: Dict, actual_outcomes: Dict) -> float:
        """Calculate prediction error"""
        # Simplified error calculation
        predicted_home_win = predictions.get('home_win_probability', 0.33)
        actual_result = actual_outcomes.get('result', 'draw')
        
        if actual_result == 'home_win':
            error = 1 - predicted_home_win
        elif actual_result == 'away_win':
            error = predicted_home_win
        else:  # draw
            error = abs(predicted_home_win - 0.33)
        
        return error
    
    def _get_default_features(self) -> Dict:
        """Return default features when engineering fails"""
        return {
            'home_goals_scored_avg': 1.5,
            'home_goals_conceded_avg': 1.0,
            'away_goals_scored_avg': 1.2,
            'away_goals_conceded_avg': 1.3,
            'home_points_per_game': 1.5,
            'away_points_per_game': 1.2,
            'home_win_rate': 0.4,
            'away_win_rate': 0.35,
            'strength_differential': 0.1,
            'total_goals_expected': 2.5,
            'form_trend_differential': 0,
            'momentum_differential': 0,
            'h2h_home_dominance': 0,
            'league_scoring_tendency': 0.5,
            'season_progress': 0.5,
            'is_derby': 0,
            'match_importance': 0.5
        }
    
    def save_feature_importance(self, filepath: str = 'models/feature_importance.json'):
        """Save feature importance scores to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            logger.info(f"Feature importance saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save feature importance: {str(e)}")
    
    def load_feature_importance(self, filepath: str = 'models/feature_importance.json'):
        """Load feature importance scores from file"""
        try:
            with open(filepath, 'r') as f:
                self.feature_importance = json.load(f)
            logger.info(f"Feature importance loaded from {filepath}")
        except Exception as e:
            logger.warning(f"Failed to load feature importance: {str(e)}")
            self.feature_importance = {}