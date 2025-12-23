"""
League-Specific Normalization Engine
Liga özgü karakteristiklere göre tahmin kalibrasyonu yapan gelişmiş normalizasyon sistemi

Bu modül şunları sağlar:
1. League Characteristic Profiling - Liga profil oluşturma
2. Statistical Normalization System - İstatistik normalizasyon sistemi
3. Seasonal Calibration Engine - Sezonsal kalibrasyon motoru
4. Dynamic League Intelligence - Dinamik liga zekası

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

# Import existing league analysis modules
from .league_context_analyzer import LeagueContextAnalyzer
from .league_strength_analyzer import LeagueStrengthAnalyzer
from .dynamic_time_analyzer import DynamicTimeAnalyzer

logger = logging.getLogger(__name__)

class LeagueNormalizationEngine:
    """
    Her liga özgü karakteristiklere göre tahmin kalibrasyonu yapan 
    gelişmiş normalizasyon sistemi
    
    Ana bileşenler:
    - League Characteristic Profiling
    - Statistical Normalization System  
    - Seasonal Calibration Engine
    - Dynamic League Intelligence
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the League Normalization Engine
        
        Args:
            config: Configuration dictionary for customization
        """
        self.config = config or self._get_default_config()
        
        # Initialize existing analyzers
        self.league_context_analyzer = LeagueContextAnalyzer()
        self.league_strength_analyzer = LeagueStrengthAnalyzer()
        self.dynamic_time_analyzer = DynamicTimeAnalyzer()
        
        # Scalers for different metrics
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # League profiles database
        self.league_profiles = {}
        self.seasonal_patterns = {}
        self.normalization_factors = {}
        self.competitive_balance_cache = {}
        
        # Dynamic intelligence tracking
        self.league_trends = defaultdict(list)
        self.meta_evolution = defaultdict(dict)
        self.strength_assessments = {}
        
        logger.info("League Normalization Engine initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for the normalization engine"""
        return {
            'min_matches_for_profile': 50,
            'seasonal_windows': {
                'early_season': (0, 10),
                'mid_season': (11, 25), 
                'late_season': (26, 38)
            },
            'transfer_windows': {
                'winter': (1, 31),    # January
                'summer': (150, 243)  # June-August (day of year)
            },
            'holiday_periods': {
                'winter_break': (355, 15),   # Dec 21 - Jan 15
                'summer_break': (165, 225)   # Mid June - Mid August
            },
            'normalization_methods': {
                'z_score': True,
                'min_max': True,
                'percentile': True,
                'league_relative': True
            },
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            },
            'outlier_detection': {
                'method': 'iqr',
                'factor': 1.5
            }
        }
    
    def generate_comprehensive_profile(self, league_id: int, league_name: str, 
                                     recent_matches: List[Dict], 
                                     historical_data: Optional[List[Dict]] = None) -> Dict:
        """
        Generate comprehensive league profile with all characteristics
        
        Args:
            league_id: League identifier
            league_name: League name
            recent_matches: Recent matches (current season)
            historical_data: Historical data for pattern analysis
            
        Returns:
            Comprehensive league profile
        """
        try:
            logger.info(f"Generating comprehensive profile for league: {league_name}")
            
            # 1. Basic league context analysis
            basic_context = self.league_context_analyzer.analyze_league_context(
                league_name, recent_matches
            )
            
            # 2. League characteristic profiling
            characteristics = self._profile_league_characteristics(
                recent_matches, historical_data
            )
            
            # 3. Competitive balance metrics
            competitive_balance = self._calculate_competitive_balance(recent_matches)
            
            # 4. Scoring pattern analysis
            scoring_patterns = self._analyze_scoring_patterns(recent_matches)
            
            # 5. Seasonal effects analysis
            seasonal_effects = self._analyze_seasonal_effects(
                recent_matches, historical_data
            )
            
            # 6. Meta characteristics
            meta_characteristics = self._analyze_meta_characteristics(recent_matches)
            
            # Compile comprehensive profile
            profile = {
                'league_id': league_id,
                'league_name': league_name,
                'basic_context': basic_context,
                'characteristics': characteristics,
                'competitive_balance': competitive_balance,
                'scoring_patterns': scoring_patterns,
                'seasonal_effects': seasonal_effects,
                'meta_characteristics': meta_characteristics,
                'profile_confidence': self._calculate_profile_confidence(recent_matches),
                'last_updated': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(recent_matches)
            }
            
            # Cache the profile
            self.league_profiles[league_id] = profile
            
            logger.info(f"League profile generated successfully for {league_name}")
            return profile
            
        except Exception as e:
            logger.error(f"Error generating league profile: {str(e)}")
            return self._get_default_profile(league_id, league_name)
    
    def _profile_league_characteristics(self, recent_matches: List[Dict], 
                                       historical_data: Optional[List[Dict]] = None) -> Dict:
        """
        Profile detailed league characteristics
        
        Returns:
            Dict with detailed league characteristics
        """
        if not recent_matches:
            return self._get_default_characteristics()
        
        # Goals per match analysis
        goals_analysis = self._analyze_goals_per_match(recent_matches)
        
        # Home advantage analysis
        home_advantage = self._analyze_home_advantage(recent_matches)
        
        # Red card analysis
        red_card_analysis = self._analyze_red_cards(recent_matches)
        
        # Match tempo analysis
        tempo_analysis = self._analyze_match_tempo(recent_matches)
        
        # Result distribution
        result_distribution = self._analyze_result_distribution(recent_matches)
        
        return {
            'goals_analysis': goals_analysis,
            'home_advantage': home_advantage,
            'red_card_analysis': red_card_analysis,
            'tempo_analysis': tempo_analysis,
            'result_distribution': result_distribution,
            'league_style': self._determine_league_style(recent_matches)
        }
    
    def _analyze_goals_per_match(self, matches: List[Dict]) -> Dict:
        """Analyze goals per match with detailed breakdown"""
        if not matches:
            return {'avg_goals': 2.5, 'std_goals': 1.5}
        
        total_goals = []
        home_goals = []
        away_goals = []
        
        for match in matches:
            if match.get('status') in ['FINISHED', 'FT', 'AET', 'PEN']:
                h_goals = match.get('home_score', 0) or 0
                a_goals = match.get('away_score', 0) or 0
                
                total_goals.append(h_goals + a_goals)
                home_goals.append(h_goals)
                away_goals.append(a_goals)
        
        if not total_goals:
            return {'avg_goals': 2.5, 'std_goals': 1.5}
        
        return {
            'avg_goals': np.mean(total_goals),
            'std_goals': np.std(total_goals),
            'median_goals': np.median(total_goals),
            'avg_home_goals': np.mean(home_goals),
            'avg_away_goals': np.mean(away_goals),
            'goal_variance': np.var(total_goals),
            'high_scoring_rate': len([g for g in total_goals if g >= 4]) / len(total_goals),
            'low_scoring_rate': len([g for g in total_goals if g <= 1]) / len(total_goals),
            'goals_distribution': self._calculate_goal_distribution(total_goals)
        }
    
    def _analyze_home_advantage(self, matches: List[Dict]) -> Dict:
        """Analyze home advantage with multiple metrics"""
        if not matches:
            return {'coefficient': 1.1, 'win_rate': 0.45}
        
        home_results = {'wins': 0, 'draws': 0, 'losses': 0}
        home_goals_for = []
        home_goals_against = []
        
        for match in matches:
            if match.get('status') in ['FINISHED', 'FT', 'AET', 'PEN']:
                h_goals = match.get('home_score', 0) or 0
                a_goals = match.get('away_score', 0) or 0
                
                home_goals_for.append(h_goals)
                home_goals_against.append(a_goals)
                
                if h_goals > a_goals:
                    home_results['wins'] += 1
                elif h_goals == a_goals:
                    home_results['draws'] += 1
                else:
                    home_results['losses'] += 1
        
        total_matches = sum(home_results.values())
        if total_matches == 0:
            return {'coefficient': 1.1, 'win_rate': 0.45}
        
        home_win_rate = home_results['wins'] / total_matches
        home_points_per_game = (home_results['wins'] * 3 + home_results['draws']) / total_matches
        
        # Calculate home advantage coefficient
        expected_points = 1.5  # Expected points per game for neutral venue
        home_advantage_points = home_points_per_game - expected_points
        coefficient = 1.0 + (home_advantage_points / 3.0)
        
        return {
            'coefficient': max(0.95, min(1.25, coefficient)),
            'win_rate': home_win_rate,
            'draw_rate': home_results['draws'] / total_matches,
            'loss_rate': home_results['losses'] / total_matches,
            'points_per_game': home_points_per_game,
            'goal_advantage': np.mean(home_goals_for) - np.mean(home_goals_against) if home_goals_for else 0,
            'strength': self._classify_home_advantage(home_win_rate)
        }
    
    def _calculate_competitive_balance(self, matches: List[Dict]) -> Dict:
        """Calculate competitive balance metrics for the league"""
        if not matches:
            return {'hhi_index': 0.5, 'balance_score': 0.5}
        
        # Team performance tracking
        team_stats = defaultdict(lambda: {'points': 0, 'matches': 0, 'wins': 0})
        
        for match in matches:
            if match.get('status') in ['FINISHED', 'FT', 'AET', 'PEN']:
                home_team = match.get('home_team_id') or match.get('home_team')
                away_team = match.get('away_team_id') or match.get('away_team')
                h_goals = match.get('home_score', 0) or 0
                a_goals = match.get('away_score', 0) or 0
                
                # Update home team stats
                team_stats[home_team]['matches'] += 1
                if h_goals > a_goals:
                    team_stats[home_team]['points'] += 3
                    team_stats[home_team]['wins'] += 1
                elif h_goals == a_goals:
                    team_stats[home_team]['points'] += 1
                
                # Update away team stats
                team_stats[away_team]['matches'] += 1
                if a_goals > h_goals:
                    team_stats[away_team]['points'] += 3
                    team_stats[away_team]['wins'] += 1
                elif a_goals == h_goals:
                    team_stats[away_team]['points'] += 1
        
        if not team_stats:
            return {'hhi_index': 0.5, 'balance_score': 0.5}
        
        # Calculate Herfindahl-Hirschman Index for competitive balance
        total_points = sum(stats['points'] for stats in team_stats.values())
        if total_points == 0:
            return {'hhi_index': 0.5, 'balance_score': 0.5}
        
        hhi = sum((stats['points'] / total_points) ** 2 for stats in team_stats.values())
        
        # Normalize HHI to 0-1 scale (lower = more competitive)
        normalized_hhi = (hhi - (1 / len(team_stats))) / (1 - (1 / len(team_stats)))
        balance_score = 1 - normalized_hhi  # Higher score = more balanced
        
        # Additional balance metrics
        win_rates = [stats['wins'] / max(stats['matches'], 1) for stats in team_stats.values()]
        points_per_game = [stats['points'] / max(stats['matches'], 1) for stats in team_stats.values()]
        
        return {
            'hhi_index': normalized_hhi,
            'balance_score': balance_score,
            'win_rate_std': np.std(win_rates),
            'points_std': np.std(points_per_game),
            'balance_category': self._classify_competitive_balance(balance_score),
            'parity_level': 1 - np.std(points_per_game) / 3 if points_per_game else 0.5
        }
    
    def _analyze_scoring_patterns(self, matches: List[Dict]) -> Dict:
        """Analyze scoring patterns including timing and distribution"""
        if not matches:
            return self._get_default_scoring_patterns()
        
        # Basic scoring analysis
        goals_by_half = {'first_half': [], 'second_half': []}
        match_types = {'both_score': 0, 'one_team_scores': 0, 'no_goals': 0}
        goal_margins = []
        
        for match in matches:
            if match.get('status') in ['FINISHED', 'FT', 'AET', 'PEN']:
                h_goals = match.get('home_score', 0) or 0
                a_goals = match.get('away_score', 0) or 0
                total_goals = h_goals + a_goals
                
                # Goal margin analysis
                goal_margins.append(abs(h_goals - a_goals))
                
                # Match type classification
                if h_goals > 0 and a_goals > 0:
                    match_types['both_score'] += 1
                elif total_goals > 0:
                    match_types['one_team_scores'] += 1
                else:
                    match_types['no_goals'] += 1
                
                # Estimate half-time scoring (simplified)
                # In real implementation, this would use actual half-time data
                first_half_goals = int(total_goals * 0.55)  # Typically 55% in first half
                second_half_goals = total_goals - first_half_goals
                
                goals_by_half['first_half'].append(first_half_goals)
                goals_by_half['second_half'].append(second_half_goals)
        
        total_matches = len([m for m in matches if m.get('status') in ['FINISHED', 'FT']])
        if total_matches == 0:
            return self._get_default_scoring_patterns()
        
        return {
            'first_half_avg': np.mean(goals_by_half['first_half']),
            'second_half_avg': np.mean(goals_by_half['second_half']),
            'both_teams_score_rate': match_types['both_score'] / total_matches,
            'clean_sheet_rate': match_types['one_team_scores'] / total_matches,
            'no_goal_rate': match_types['no_goals'] / total_matches,
            'avg_goal_margin': np.mean(goal_margins) if goal_margins else 1.0,
            'close_match_rate': len([m for m in goal_margins if m <= 1]) / len(goal_margins) if goal_margins else 0.5,
            'blowout_rate': len([m for m in goal_margins if m >= 3]) / len(goal_margins) if goal_margins else 0.1
        }
    
    def normalize_team_performance(self, team_stats: Dict, league_profile: Dict, 
                                 normalization_method: str = 'z_score') -> Dict:
        """
        Normalize team performance relative to league averages
        
        Args:
            team_stats: Raw team statistics
            league_profile: League profile for normalization
            normalization_method: Method to use ('z_score', 'min_max', 'percentile')
            
        Returns:
            Normalized team statistics (0-100 scale)
        """
        try:
            if normalization_method == 'z_score':
                return self._normalize_z_score(team_stats, league_profile)
            elif normalization_method == 'min_max':
                return self._normalize_min_max(team_stats, league_profile)
            elif normalization_method == 'percentile':
                return self._normalize_percentile(team_stats, league_profile)
            elif normalization_method == 'league_relative':
                return self._normalize_league_relative(team_stats, league_profile)
            else:
                logger.warning(f"Unknown normalization method: {normalization_method}")
                return self._normalize_z_score(team_stats, league_profile)
                
        except Exception as e:
            logger.error(f"Error in team performance normalization: {str(e)}")
            return self._get_default_normalized_stats()
    
    def _normalize_z_score(self, team_stats: Dict, league_profile: Dict) -> Dict:
        """Normalize using Z-score method"""
        league_avgs = league_profile.get('characteristics', {}).get('goals_analysis', {})
        
        normalized = {}
        
        # Goals metrics
        team_goals_avg = team_stats.get('goals_for_avg', 0)
        league_goals_avg = league_avgs.get('avg_goals', 2.5) / 2  # Per team
        league_goals_std = league_avgs.get('std_goals', 1.5) / 2
        
        goals_z_score = (team_goals_avg - league_goals_avg) / max(league_goals_std, 0.1)
        normalized['attack_rating'] = self._z_score_to_100_scale(goals_z_score)
        
        # Defense metrics
        team_goals_against = team_stats.get('goals_against_avg', 0)
        defense_z_score = -(team_goals_against - league_goals_avg) / max(league_goals_std, 0.1)
        normalized['defense_rating'] = self._z_score_to_100_scale(defense_z_score)
        
        # Overall rating
        normalized['overall_rating'] = (normalized['attack_rating'] + normalized['defense_rating']) / 2
        
        # Form rating
        form_score = team_stats.get('form_score', 50)
        normalized['form_rating'] = min(100, max(0, form_score))
        
        # Home/Away specific ratings
        if team_stats.get('is_home'):
            home_advantage = league_profile.get('characteristics', {}).get('home_advantage', {}).get('coefficient', 1.1)
            normalized['venue_adjusted_rating'] = normalized['overall_rating'] * home_advantage
        else:
            normalized['venue_adjusted_rating'] = normalized['overall_rating'] * 0.95
        
        return normalized
    
    def _z_score_to_100_scale(self, z_score: float) -> float:
        """Convert Z-score to 0-100 scale"""
        # Z-score of -3 to +3 maps to 0-100 scale
        normalized = 50 + (z_score * 16.67)  # 50/3 = 16.67
        return max(0, min(100, normalized))
    
    def calculate_cross_league_comparison(self, team1_stats: Dict, team1_league: int,
                                        team2_stats: Dict, team2_league: int) -> Dict:
        """
        Calculate cross-league performance comparison
        
        Args:
            team1_stats: First team's statistics
            team1_league: First team's league ID
            team2_stats: Second team's statistics  
            team2_league: Second team's league ID
            
        Returns:
            Cross-league comparison analysis
        """
        try:
            # Get league profiles
            league1_profile = self.league_profiles.get(team1_league, {})
            league2_profile = self.league_profiles.get(team2_league, {})
            
            if not league1_profile or not league2_profile:
                logger.warning("Missing league profiles for cross-league comparison")
                return self._get_default_cross_league_comparison()
            
            # Normalize both teams to their respective leagues
            team1_normalized = self.normalize_team_performance(team1_stats, league1_profile)
            team2_normalized = self.normalize_team_performance(team2_stats, league2_profile)
            
            # Calculate league difficulty adjustment
            league_strength_diff = self._calculate_league_strength_difference(
                league1_profile, league2_profile
            )
            
            # Adjust ratings based on league strength
            adjusted_team1_rating = team1_normalized['overall_rating'] * league_strength_diff['league1_factor']
            adjusted_team2_rating = team2_normalized['overall_rating'] * league_strength_diff['league2_factor']
            
            # Calculate comparison metrics
            strength_difference = adjusted_team1_rating - adjusted_team2_rating
            confidence = self._calculate_comparison_confidence(league1_profile, league2_profile)
            
            return {
                'team1_normalized_rating': team1_normalized['overall_rating'],
                'team2_normalized_rating': team2_normalized['overall_rating'],
                'team1_adjusted_rating': adjusted_team1_rating,
                'team2_adjusted_rating': adjusted_team2_rating,
                'strength_difference': strength_difference,
                'league_strength_diff': league_strength_diff,
                'comparison_confidence': confidence,
                'recommendation': self._generate_cross_league_recommendation(
                    strength_difference, confidence, league_strength_diff
                )
            }
            
        except Exception as e:
            logger.error(f"Error in cross-league comparison: {str(e)}")
            return self._get_default_cross_league_comparison()
    
    def apply_seasonal_calibration(self, base_prediction: Dict, match_context: Dict, 
                                 league_profile: Dict) -> Dict:
        """
        Apply seasonal calibration to base prediction
        
        Args:
            base_prediction: Base prediction before calibration
            match_context: Match context (date, round, etc.)
            league_profile: League profile with seasonal patterns
            
        Returns:
            Seasonally calibrated prediction
        """
        try:
            calibrated_prediction = base_prediction.copy()
            
            # Get seasonal factors
            seasonal_factors = self._get_seasonal_factors(match_context, league_profile)
            
            # Apply calibrations
            for factor_type, factor_value in seasonal_factors.items():
                if factor_type == 'goal_expectation_factor':
                    # Adjust goal expectations
                    calibrated_prediction['home_goals'] *= factor_value
                    calibrated_prediction['away_goals'] *= factor_value
                    
                elif factor_type == 'home_advantage_factor':
                    # Adjust home advantage
                    if 'home_win_prob' in calibrated_prediction:
                        # Boost home win probability
                        home_boost = (factor_value - 1.0) * 0.1
                        calibrated_prediction['home_win_prob'] = min(0.9, 
                            calibrated_prediction['home_win_prob'] + home_boost)
                        
                elif factor_type == 'upset_probability_factor':
                    # Adjust upset probabilities
                    if factor_value > 1.0:  # Higher upset chance
                        # Flatten probabilities slightly
                        probs = [calibrated_prediction.get(k, 0.33) for k in ['home_win_prob', 'draw_prob', 'away_win_prob']]
                        avg_prob = np.mean(probs)
                        adjustment = (factor_value - 1.0) * 0.1
                        
                        calibrated_prediction['home_win_prob'] = probs[0] - adjustment if probs[0] > avg_prob else probs[0] + adjustment
                        calibrated_prediction['away_win_prob'] = probs[2] - adjustment if probs[2] > avg_prob else probs[2] + adjustment
                        calibrated_prediction['draw_prob'] = 1 - calibrated_prediction['home_win_prob'] - calibrated_prediction['away_win_prob']
            
            # Ensure probabilities sum to 1
            total_prob = (calibrated_prediction.get('home_win_prob', 0) + 
                         calibrated_prediction.get('draw_prob', 0) + 
                         calibrated_prediction.get('away_win_prob', 0))
            
            if total_prob > 0:
                calibrated_prediction['home_win_prob'] /= total_prob
                calibrated_prediction['draw_prob'] /= total_prob  
                calibrated_prediction['away_win_prob'] /= total_prob
            
            # Add calibration metadata
            calibrated_prediction['seasonal_calibration'] = {
                'applied_factors': seasonal_factors,
                'calibration_confidence': self._calculate_calibration_confidence(seasonal_factors),
                'seasonal_period': self._determine_seasonal_period(match_context)
            }
            
            return calibrated_prediction
            
        except Exception as e:
            logger.error(f"Error in seasonal calibration: {str(e)}")
            return base_prediction
    
    def detect_league_trends(self, league_id: int, recent_matches: List[Dict], 
                           historical_data: Optional[List[Dict]] = None) -> Dict:
        """
        Detect real-time league trends and emerging patterns
        
        Args:
            league_id: League identifier
            recent_matches: Recent matches data
            historical_data: Historical data for comparison
            
        Returns:
            Detected trends and patterns
        """
        try:
            trends = {}
            
            # Goal scoring trends
            trends['scoring_trends'] = self._detect_scoring_trends(recent_matches, historical_data)
            
            # Result pattern trends
            trends['result_trends'] = self._detect_result_pattern_trends(recent_matches)
            
            # Competitive balance trends
            trends['balance_trends'] = self._detect_balance_trends(recent_matches, historical_data)
            
            # Home advantage trends
            trends['home_advantage_trends'] = self._detect_home_advantage_trends(recent_matches, historical_data)
            
            # Meta evolution patterns
            trends['meta_evolution'] = self._detect_meta_evolution(recent_matches, historical_data)
            
            # Store trends for historical tracking
            self.league_trends[league_id].append({
                'timestamp': datetime.now().isoformat(),
                'trends': trends
            })
            
            # Keep only last 10 trend analyses
            if len(self.league_trends[league_id]) > 10:
                self.league_trends[league_id] = self.league_trends[league_id][-10:]
            
            return trends
            
        except Exception as e:
            logger.error(f"Error detecting league trends: {str(e)}")
            return self._get_default_trends()
    
    def assess_comparative_strength(self, leagues: List[Tuple[int, Dict]]) -> Dict:
        """
        Assess comparative strength between multiple leagues
        
        Args:
            leagues: List of (league_id, league_profile) tuples
            
        Returns:
            Comparative strength assessment
        """
        try:
            if len(leagues) < 2:
                return {'error': 'Need at least 2 leagues for comparison'}
            
            # Extract strength metrics for each league
            league_strengths = {}
            
            for league_id, profile in leagues:
                league_strengths[league_id] = self._extract_strength_metrics(profile)
            
            # Calculate relative strengths
            strength_rankings = self._calculate_strength_rankings(league_strengths)
            
            # Generate pairwise comparisons
            pairwise_comparisons = self._generate_pairwise_comparisons(league_strengths)
            
            # Calculate overall strength tiers
            strength_tiers = self._calculate_strength_tiers(league_strengths)
            
            return {
                'league_rankings': strength_rankings,
                'pairwise_comparisons': pairwise_comparisons,
                'strength_tiers': strength_tiers,
                'assessment_confidence': self._calculate_assessment_confidence(leagues),
                'methodology': 'Multi-factor comparative analysis',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in comparative strength assessment: {str(e)}")
            return {'error': str(e)}
    
    # Helper methods for default values and error handling
    
    def _get_default_profile(self, league_id: int, league_name: str) -> Dict:
        """Get default league profile when data is insufficient"""
        return {
            'league_id': league_id,
            'league_name': league_name,
            'basic_context': self.league_context_analyzer._get_default_context(),
            'characteristics': self._get_default_characteristics(),
            'competitive_balance': {'hhi_index': 0.5, 'balance_score': 0.5},
            'scoring_patterns': self._get_default_scoring_patterns(),
            'seasonal_effects': {},
            'meta_characteristics': {},
            'profile_confidence': 0.3,
            'last_updated': datetime.now().isoformat(),
            'data_quality': 'insufficient'
        }
    
    def _get_default_characteristics(self) -> Dict:
        """Get default league characteristics"""
        return {
            'goals_analysis': {'avg_goals': 2.5, 'std_goals': 1.5},
            'home_advantage': {'coefficient': 1.1, 'win_rate': 0.45},
            'red_card_analysis': {'frequency': 0.3, 'impact': 0.15},
            'tempo_analysis': {'pace': 'medium'},
            'result_distribution': {'home_win': 0.45, 'draw': 0.25, 'away_win': 0.30},
            'league_style': 'balanced'
        }
    
    def _get_default_scoring_patterns(self) -> Dict:
        """Get default scoring patterns"""
        return {
            'first_half_avg': 1.4,
            'second_half_avg': 1.1,
            'both_teams_score_rate': 0.55,
            'clean_sheet_rate': 0.40,
            'no_goal_rate': 0.05,
            'avg_goal_margin': 1.2,
            'close_match_rate': 0.65,
            'blowout_rate': 0.10
        }
    
    def _get_default_normalized_stats(self) -> Dict:
        """Get default normalized statistics"""
        return {
            'attack_rating': 50,
            'defense_rating': 50,
            'overall_rating': 50,
            'form_rating': 50,
            'venue_adjusted_rating': 50
        }
    
    def _get_default_cross_league_comparison(self) -> Dict:
        """Get default cross-league comparison"""
        return {
            'team1_normalized_rating': 50,
            'team2_normalized_rating': 50,
            'team1_adjusted_rating': 50,
            'team2_adjusted_rating': 50,
            'strength_difference': 0,
            'league_strength_diff': {'league1_factor': 1.0, 'league2_factor': 1.0},
            'comparison_confidence': 0.5,
            'recommendation': 'Insufficient data for reliable comparison'
        }
    
    def _get_default_trends(self) -> Dict:
        """Get default trends when analysis fails"""
        return {
            'scoring_trends': {'trend': 'stable', 'confidence': 0.5},
            'result_trends': {'pattern': 'normal', 'confidence': 0.5},
            'balance_trends': {'direction': 'stable', 'confidence': 0.5},
            'home_advantage_trends': {'trend': 'stable', 'confidence': 0.5},
            'meta_evolution': {'status': 'no_change', 'confidence': 0.5}
        }
    
    # Additional helper methods would be implemented here...
    # (Due to length constraints, showing main structure)
    
    def get_league_profile(self, league_id: int) -> Optional[Dict]:
        """Get cached league profile"""
        return self.league_profiles.get(league_id)
    
    def update_league_profile(self, league_id: int, new_data: List[Dict]) -> bool:
        """Update existing league profile with new data"""
        try:
            if league_id in self.league_profiles:
                profile = self.league_profiles[league_id]
                # Update profile with new data
                # Implementation would merge new data with existing profile
                logger.info(f"Updated league profile for league {league_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating league profile: {str(e)}")
            return False
    
    def get_normalization_summary(self) -> Dict:
        """Get summary of all normalization activities"""
        return {
            'total_leagues_profiled': len(self.league_profiles),
            'normalization_methods_available': list(self.config['normalization_methods'].keys()),
            'last_activity': datetime.now().isoformat(),
            'engine_status': 'active'
        }

    def _analyze_red_cards(self, matches: List[Dict]) -> Dict:
        """Analyze red card frequency and impact"""
        if not matches:
            return {'frequency': 0.3, 'impact': 0.15, 'avg_per_match': 0.3}
        
        red_cards_total = 0
        matches_with_cards = 0
        total_finished_matches = 0
        
        for match in matches:
            if match.get('status') in ['FINISHED', 'FT', 'AET', 'PEN']:
                total_finished_matches += 1
                # Estimate red cards from match data (would be actual data in real implementation)
                home_goals = match.get('home_score', 0) or 0
                away_goals = match.get('away_score', 0) or 0
                goal_difference = abs(home_goals - away_goals)
                
                # Heuristic: larger goal differences might indicate more cards
                estimated_cards = min(2, goal_difference * 0.2) if goal_difference >= 3 else 0.1
                red_cards_total += estimated_cards
                
                if estimated_cards > 0:
                    matches_with_cards += 1
        
        if total_finished_matches == 0:
            return {'frequency': 0.3, 'impact': 0.15, 'avg_per_match': 0.3}
        
        frequency = red_cards_total / total_finished_matches
        impact_factor = min(0.3, frequency * 0.5)  # Impact on match outcome
        
        return {
            'frequency': frequency,
            'impact': impact_factor,
            'avg_per_match': frequency,
            'matches_with_cards_rate': matches_with_cards / total_finished_matches,
            'severity_level': 'high' if frequency > 0.5 else 'medium' if frequency > 0.2 else 'low'
        }
    
    def _analyze_match_tempo(self, matches: List[Dict]) -> Dict:
        """Analyze match tempo and pace"""
        if not matches:
            return {'pace': 'medium', 'tempo_score': 0.5, 'avg_goals_per_minute': 0.026}
        
        total_goals = []
        for match in matches:
            if match.get('status') in ['FINISHED', 'FT', 'AET', 'PEN']:
                home_goals = match.get('home_score', 0) or 0
                away_goals = match.get('away_score', 0) or 0
                total_goals.append(home_goals + away_goals)
        
        if not total_goals:
            return {'pace': 'medium', 'tempo_score': 0.5, 'avg_goals_per_minute': 0.026}
        
        avg_goals = np.mean(total_goals)
        goals_per_minute = avg_goals / 90  # 90 minutes per match
        
        # Classify tempo based on goals per minute
        if goals_per_minute > 0.035:
            pace = 'fast'
            tempo_score = 0.8
        elif goals_per_minute > 0.025:
            pace = 'medium'
            tempo_score = 0.5
        else:
            pace = 'slow'
            tempo_score = 0.3
        
        return {
            'pace': pace,
            'tempo_score': tempo_score,
            'avg_goals_per_minute': goals_per_minute,
            'tempo_classification': pace,
            'intensity_level': tempo_score
        }
    
    def _analyze_result_distribution(self, matches: List[Dict]) -> Dict:
        """Analyze result distribution patterns"""
        if not matches:
            return {'home_win': 0.45, 'draw': 0.25, 'away_win': 0.30}
        
        results = {'home_win': 0, 'draw': 0, 'away_win': 0}
        total = 0
        
        for match in matches:
            if match.get('status') in ['FINISHED', 'FT']:
                h_goals = match.get('home_score', 0) or 0
                a_goals = match.get('away_score', 0) or 0
                
                if h_goals > a_goals:
                    results['home_win'] += 1
                elif h_goals == a_goals:
                    results['draw'] += 1
                else:
                    results['away_win'] += 1
                total += 1
        
        if total > 0:
            return {k: v/total for k, v in results.items()}
        return {'home_win': 0.45, 'draw': 0.25, 'away_win': 0.30}
    
    def _determine_league_style(self, matches: List[Dict]) -> str:
        """Determine overall league playing style"""
        # This would analyze tactical patterns, possession styles, etc.
        return 'balanced'  # Simplified
    
    def _calculate_goal_distribution(self, total_goals: List[int]) -> Dict:
        """Calculate goal distribution statistics"""
        if not total_goals:
            return {}
        
        distribution = Counter(total_goals)
        total_matches = len(total_goals)
        
        return {str(k): v/total_matches for k, v in distribution.items()}
    
    def _classify_home_advantage(self, win_rate: float) -> str:
        """Classify home advantage strength"""
        if win_rate >= 0.55:
            return 'strong'
        elif win_rate >= 0.45:
            return 'moderate'
        else:
            return 'weak'
    
    def _classify_competitive_balance(self, balance_score: float) -> str:
        """Classify competitive balance level"""
        if balance_score >= 0.7:
            return 'very_balanced'
        elif balance_score >= 0.5:
            return 'balanced'
        elif balance_score >= 0.3:
            return 'moderately_imbalanced'
        else:
            return 'imbalanced'
    
    def _analyze_seasonal_effects(self, recent_matches: List[Dict], 
                                 historical_data: Optional[List[Dict]] = None) -> Dict:
        """Analyze seasonal effects on league performance"""
        if not recent_matches:
            return {}
        
        # Group matches by month to analyze seasonal patterns
        monthly_stats = defaultdict(lambda: {'goals': [], 'results': []})
        
        for match in recent_matches:
            if match.get('status') in ['FINISHED', 'FT', 'AET', 'PEN']:
                match_date = match.get('date') or match.get('match_date')
                if match_date:
                    try:
                        if isinstance(match_date, str):
                            match_date = datetime.strptime(match_date.split('T')[0], '%Y-%m-%d')
                        month = match_date.month
                        
                        home_goals = match.get('home_score', 0) or 0
                        away_goals = match.get('away_score', 0) or 0
                        total_goals = home_goals + away_goals
                        
                        monthly_stats[month]['goals'].append(total_goals)
                        
                        if home_goals > away_goals:
                            result = 'H'
                        elif away_goals > home_goals:
                            result = 'A'
                        else:
                            result = 'D'
                        monthly_stats[month]['results'].append(result)
                    except:
                        continue
        
        # Calculate seasonal patterns
        seasonal_patterns = {}
        for month, stats in monthly_stats.items():
            if stats['goals']:
                seasonal_patterns[month] = {
                    'avg_goals': np.mean(stats['goals']),
                    'home_win_rate': stats['results'].count('H') / len(stats['results']),
                    'away_win_rate': stats['results'].count('A') / len(stats['results']),
                    'draw_rate': stats['results'].count('D') / len(stats['results']),
                    'matches_count': len(stats['goals'])
                }
        
        return {
            'monthly_patterns': seasonal_patterns,
            'season_start_effect': self._calculate_season_start_effect(monthly_stats),
            'winter_break_effect': self._calculate_winter_break_effect(monthly_stats),
            'end_season_effect': self._calculate_end_season_effect(monthly_stats)
        }
    
    def _analyze_meta_characteristics(self, matches: List[Dict]) -> Dict:
        """Analyze meta characteristics of the league"""
        if not matches:
            return {}
        
        # Analyze volatility and predictability
        results = []
        goal_differences = []
        
        for match in matches:
            if match.get('status') in ['FINISHED', 'FT', 'AET', 'PEN']:
                home_goals = match.get('home_score', 0) or 0
                away_goals = match.get('away_score', 0) or 0
                
                goal_differences.append(abs(home_goals - away_goals))
                
                if home_goals > away_goals:
                    results.append(1)  # Home win
                elif away_goals > home_goals:
                    results.append(-1)  # Away win
                else:
                    results.append(0)  # Draw
        
        if not results:
            return {}
        
        volatility = np.std(goal_differences) if goal_differences else 1.0
        predictability = 1 - (volatility / 3.0)  # Normalize to 0-1 scale
        
        return {
            'volatility': volatility,
            'predictability': max(0, min(1, predictability)),
            'avg_goal_difference': np.mean(goal_differences) if goal_differences else 1.0,
            'result_entropy': self._calculate_result_entropy(results),
            'upset_frequency': self._calculate_upset_frequency(matches)
        }
    
    def _calculate_profile_confidence(self, matches: List[Dict]) -> float:
        """Calculate confidence level of the league profile"""
        if not matches:
            return 0.0
        
        finished_matches = len([m for m in matches if m.get('status') in ['FINISHED', 'FT']])
        min_matches = self.config['min_matches_for_profile']
        
        if finished_matches >= min_matches:
            return 1.0
        elif finished_matches >= min_matches * 0.5:
            return 0.7
        elif finished_matches >= min_matches * 0.25:
            return 0.5
        else:
            return 0.3
    
    def _assess_data_quality(self, matches: List[Dict]) -> str:
        """Assess the quality of match data"""
        if not matches:
            return 'no_data'
        
        total_matches = len(matches)
        finished_matches = len([m for m in matches if m.get('status') in ['FINISHED', 'FT']])
        
        if finished_matches == 0:
            return 'no_finished_matches'
        
        completion_rate = finished_matches / total_matches
        
        if completion_rate >= 0.9:
            return 'excellent'
        elif completion_rate >= 0.7:
            return 'good'
        elif completion_rate >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _normalize_min_max(self, team_stats: Dict, league_profile: Dict) -> Dict:
        """Normalize using Min-Max method"""
        # Implementation for min-max normalization
        normalized = {}
        
        # For min-max normalization, we need league min/max values
        # This is a simplified implementation
        goals_avg = team_stats.get('goals_for_avg', 0)
        normalized['attack_rating'] = min(100, max(0, (goals_avg / 4.0) * 100))
        
        goals_against = team_stats.get('goals_against_avg', 0)
        normalized['defense_rating'] = min(100, max(0, 100 - (goals_against / 4.0) * 100))
        
        normalized['overall_rating'] = (normalized['attack_rating'] + normalized['defense_rating']) / 2
        normalized['form_rating'] = team_stats.get('form_score', 50)
        normalized['venue_adjusted_rating'] = normalized['overall_rating']
        
        return normalized
    
    def _normalize_percentile(self, team_stats: Dict, league_profile: Dict) -> Dict:
        """Normalize using percentile method"""
        # Percentile normalization based on league distribution
        normalized = {}
        
        # This would use actual league percentile data in full implementation
        goals_avg = team_stats.get('goals_for_avg', 0)
        league_avg_goals = league_profile.get('characteristics', {}).get('goals_analysis', {}).get('avg_goals', 2.5) / 2
        
        # Simple percentile approximation
        if goals_avg >= league_avg_goals * 1.5:
            normalized['attack_rating'] = 90
        elif goals_avg >= league_avg_goals * 1.2:
            normalized['attack_rating'] = 75
        elif goals_avg >= league_avg_goals:
            normalized['attack_rating'] = 60
        elif goals_avg >= league_avg_goals * 0.8:
            normalized['attack_rating'] = 40
        else:
            normalized['attack_rating'] = 25
        
        # Similar for defense (inverse logic)
        goals_against = team_stats.get('goals_against_avg', 0)
        if goals_against <= league_avg_goals * 0.5:
            normalized['defense_rating'] = 90
        elif goals_against <= league_avg_goals * 0.8:
            normalized['defense_rating'] = 75
        elif goals_against <= league_avg_goals:
            normalized['defense_rating'] = 60
        elif goals_against <= league_avg_goals * 1.2:
            normalized['defense_rating'] = 40
        else:
            normalized['defense_rating'] = 25
        
        normalized['overall_rating'] = (normalized['attack_rating'] + normalized['defense_rating']) / 2
        normalized['form_rating'] = team_stats.get('form_score', 50)
        normalized['venue_adjusted_rating'] = normalized['overall_rating']
        
        return normalized
    
    def _normalize_league_relative(self, team_stats: Dict, league_profile: Dict) -> Dict:
        """Normalize relative to league characteristics"""
        normalized = {}
        
        league_chars = league_profile.get('characteristics', {})
        goals_analysis = league_chars.get('goals_analysis', {})
        
        # Relative to league average with league-specific adjustments
        league_avg_goals = goals_analysis.get('avg_goals', 2.5) / 2
        league_std = goals_analysis.get('std_goals', 1.5) / 2
        
        team_goals = team_stats.get('goals_for_avg', 0)
        team_goals_against = team_stats.get('goals_against_avg', 0)
        
        # Calculate relative performance
        attack_relative = (team_goals - league_avg_goals) / max(league_std, 0.1)
        defense_relative = -(team_goals_against - league_avg_goals) / max(league_std, 0.1)
        
        # Convert to 0-100 scale with league context
        normalized['attack_rating'] = 50 + (attack_relative * 20)
        normalized['defense_rating'] = 50 + (defense_relative * 20)
        
        # Apply league-specific adjustments
        league_type = league_profile.get('basic_context', {}).get('league_type', 'medium_scoring')
        if league_type == 'high_scoring':
            normalized['attack_rating'] *= 1.1
        elif league_type == 'low_scoring':
            normalized['defense_rating'] *= 1.1
        
        normalized['overall_rating'] = (normalized['attack_rating'] + normalized['defense_rating']) / 2
        normalized['form_rating'] = team_stats.get('form_score', 50)
        normalized['venue_adjusted_rating'] = normalized['overall_rating']
        
        # Ensure 0-100 range
        for key in normalized:
            normalized[key] = max(0, min(100, normalized[key]))
        
        return normalized
    
    def _calculate_league_strength_difference(self, league1_profile: Dict, league2_profile: Dict) -> Dict:
        """Calculate strength difference between two leagues"""
        # Extract strength indicators from profiles
        league1_goals = league1_profile.get('characteristics', {}).get('goals_analysis', {}).get('avg_goals', 2.5)
        league2_goals = league2_profile.get('characteristics', {}).get('goals_analysis', {}).get('avg_goals', 2.5)
        
        league1_balance = league1_profile.get('competitive_balance', {}).get('balance_score', 0.5)
        league2_balance = league2_profile.get('competitive_balance', {}).get('balance_score', 0.5)
        
        league1_quality = league1_profile.get('basic_context', {}).get('league_quality', 'unknown')
        league2_quality = league2_profile.get('basic_context', {}).get('league_quality', 'unknown')
        
        # Quality score mapping
        quality_scores = {'elite': 1.0, 'high': 0.85, 'medium': 0.7, 'low': 0.55, 'unknown': 0.6}
        
        league1_quality_score = quality_scores.get(league1_quality, 0.6)
        league2_quality_score = quality_scores.get(league2_quality, 0.6)
        
        # Calculate composite strength scores
        league1_strength = (league1_quality_score * 0.6 + league1_balance * 0.2 + 
                           min(league1_goals / 3.5, 1.0) * 0.2)
        league2_strength = (league2_quality_score * 0.6 + league2_balance * 0.2 + 
                           min(league2_goals / 3.5, 1.0) * 0.2)
        
        # Calculate adjustment factors
        if league1_strength > league2_strength:
            league1_factor = 1.0 + (league1_strength - league2_strength) * 0.3
            league2_factor = 1.0 - (league1_strength - league2_strength) * 0.3
        else:
            league1_factor = 1.0 - (league2_strength - league1_strength) * 0.3
            league2_factor = 1.0 + (league2_strength - league1_strength) * 0.3
        
        return {
            'league1_strength': league1_strength,
            'league2_strength': league2_strength,
            'league1_factor': max(0.7, min(1.3, league1_factor)),
            'league2_factor': max(0.7, min(1.3, league2_factor)),
            'strength_difference': abs(league1_strength - league2_strength)
        }
    
    def _calculate_comparison_confidence(self, league1_profile: Dict, league2_profile: Dict) -> float:
        """Calculate confidence in cross-league comparison"""
        conf1 = league1_profile.get('profile_confidence', 0.5)
        conf2 = league2_profile.get('profile_confidence', 0.5)
        
        # Combined confidence is minimum of both
        combined_confidence = min(conf1, conf2)
        
        # Reduce confidence if leagues are very different types
        league1_type = league1_profile.get('basic_context', {}).get('league_type', 'medium_scoring')
        league2_type = league2_profile.get('basic_context', {}).get('league_type', 'medium_scoring')
        
        if league1_type != league2_type:
            combined_confidence *= 0.8
        
        return combined_confidence
    
    def _generate_cross_league_recommendation(self, strength_diff: float, confidence: float, 
                                            league_strength_diff: Dict) -> str:
        """Generate recommendation for cross-league comparison"""
        if confidence < 0.5:
            return "Insufficient data for reliable cross-league comparison"
        
        if abs(strength_diff) > 20:
            stronger_league = 1 if strength_diff > 0 else 2
            return f"Significant strength difference detected. League {stronger_league} appears considerably stronger."
        elif abs(strength_diff) > 10:
            stronger_league = 1 if strength_diff > 0 else 2
            return f"Moderate strength difference. League {stronger_league} has advantage."
        else:
            return "Leagues appear to be of similar strength level. Form and individual quality more important."
    
    def _get_seasonal_factors(self, match_context: Dict, league_profile: Dict) -> Dict:
        """Get seasonal adjustment factors"""
        factors = {}
        
        match_date = match_context.get('match_date', datetime.now())
        if isinstance(match_date, str):
            match_date = datetime.strptime(match_date.split('T')[0], '%Y-%m-%d')
        
        month = match_date.month
        day_of_year = match_date.timetuple().tm_yday
        
        # Season period effects
        if month in [8, 9, 10]:  # Early season
            factors['goal_expectation_factor'] = 1.05  # Slightly more goals
            factors['home_advantage_factor'] = 1.1     # Stronger home advantage
            factors['upset_probability_factor'] = 1.1  # More upsets
        elif month in [11, 12, 1, 2]:  # Mid season
            factors['goal_expectation_factor'] = 1.0   # Normal
            factors['home_advantage_factor'] = 1.0     # Normal
            factors['upset_probability_factor'] = 1.0  # Normal
        elif month in [3, 4, 5]:  # End season
            factors['goal_expectation_factor'] = 0.95  # Slightly fewer goals
            factors['home_advantage_factor'] = 0.95    # Weaker home advantage
            factors['upset_probability_factor'] = 0.9  # Fewer upsets
        
        # Transfer window effects
        if month == 1 or month in [6, 7, 8]:  # Transfer windows
            factors['upset_probability_factor'] = factors.get('upset_probability_factor', 1.0) * 1.05
        
        # Holiday period effects
        if day_of_year in range(355, 366) or day_of_year in range(1, 15):  # Winter holidays
            factors['goal_expectation_factor'] = factors.get('goal_expectation_factor', 1.0) * 0.95
        
        return factors
    
    def _calculate_calibration_confidence(self, seasonal_factors: Dict) -> float:
        """Calculate confidence in seasonal calibration"""
        # More factors applied = lower confidence (more uncertainty)
        factor_count = len(seasonal_factors)
        base_confidence = 0.8
        
        # Reduce confidence for each additional factor
        confidence_reduction = (factor_count - 1) * 0.1
        
        return max(0.3, base_confidence - confidence_reduction)
    
    def _determine_seasonal_period(self, match_context: Dict) -> str:
        """Determine which seasonal period the match is in"""
        match_date = match_context.get('match_date', datetime.now())
        if isinstance(match_date, str):
            match_date = datetime.strptime(match_date.split('T')[0], '%Y-%m-%d')
        
        month = match_date.month
        
        if month in [8, 9, 10]:
            return 'early_season'
        elif month in [11, 12, 1, 2]:
            return 'mid_season'
        elif month in [3, 4, 5]:
            return 'late_season'
        else:
            return 'off_season'
    
    # Additional helper methods for trend detection and intelligence
    
    def _detect_scoring_trends(self, recent_matches: List[Dict], historical_data: Optional[List[Dict]] = None) -> Dict:
        """Detect scoring trends in the league"""
        if not recent_matches:
            return {'trend': 'stable', 'confidence': 0.5}
        
        # Analyze recent scoring compared to historical
        recent_goals = []
        for match in recent_matches[-20:]:  # Last 20 matches
            if match.get('status') in ['FINISHED', 'FT']:
                home_goals = match.get('home_score', 0) or 0
                away_goals = match.get('away_score', 0) or 0
                recent_goals.append(home_goals + away_goals)
        
        if not recent_goals:
            return {'trend': 'stable', 'confidence': 0.5}
        
        recent_avg = np.mean(recent_goals)
        
        # Compare with historical if available
        if historical_data:
            historical_goals = []
            for match in historical_data:
                if match.get('status') in ['FINISHED', 'FT']:
                    home_goals = match.get('home_score', 0) or 0
                    away_goals = match.get('away_score', 0) or 0
                    historical_goals.append(home_goals + away_goals)
            
            if historical_goals:
                historical_avg = np.mean(historical_goals)
                change = (recent_avg - historical_avg) / historical_avg
                
                if change > 0.1:
                    trend = 'increasing'
                elif change < -0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                confidence = min(0.9, 0.5 + abs(change))
            else:
                trend = 'stable'
                confidence = 0.5
        else:
            # Analyze trend within recent matches
            if len(recent_goals) >= 10:
                first_half = recent_goals[:len(recent_goals)//2]
                second_half = recent_goals[len(recent_goals)//2:]
                
                first_avg = np.mean(first_half)
                second_avg = np.mean(second_half)
                
                change = (second_avg - first_avg) / first_avg if first_avg > 0 else 0
                
                if change > 0.1:
                    trend = 'increasing'
                elif change < -0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                confidence = min(0.8, 0.4 + abs(change))
            else:
                trend = 'stable'
                confidence = 0.3
        
        return {
            'trend': trend,
            'confidence': confidence,
            'recent_average': recent_avg,
            'trend_strength': abs(change) if 'change' in locals() else 0
        }
    
    def _detect_result_pattern_trends(self, recent_matches: List[Dict]) -> Dict:
        """Detect trends in result patterns"""
        if not recent_matches:
            return {'pattern': 'normal', 'confidence': 0.5}
        
        results = []
        for match in recent_matches:
            if match.get('status') in ['FINISHED', 'FT']:
                home_goals = match.get('home_score', 0) or 0
                away_goals = match.get('away_score', 0) or 0
                
                if home_goals > away_goals:
                    results.append('H')
                elif away_goals > home_goals:
                    results.append('A')
                else:
                    results.append('D')
        
        if len(results) < 10:
            return {'pattern': 'normal', 'confidence': 0.3}
        
        # Analyze result distribution
        home_wins = results.count('H') / len(results)
        away_wins = results.count('A') / len(results)
        draws = results.count('D') / len(results)
        
        # Detect patterns
        if home_wins > 0.6:
            pattern = 'home_dominated'
        elif away_wins > 0.4:
            pattern = 'away_strong'
        elif draws > 0.35:
            pattern = 'draw_heavy'
        else:
            pattern = 'normal'
        
        # Calculate confidence based on sample size and deviation from normal
        expected_home = 0.45
        expected_away = 0.30
        expected_draw = 0.25
        
        deviation = (abs(home_wins - expected_home) + 
                    abs(away_wins - expected_away) + 
                    abs(draws - expected_draw)) / 3
        
        confidence = min(0.9, 0.3 + deviation * 2)
        
        return {
            'pattern': pattern,
            'confidence': confidence,
            'home_win_rate': home_wins,
            'away_win_rate': away_wins,
            'draw_rate': draws
        }
    
    def _detect_balance_trends(self, recent_matches: List[Dict], historical_data: Optional[List[Dict]] = None) -> Dict:
        """Detect trends in competitive balance"""
        if not recent_matches:
            return {'direction': 'stable', 'confidence': 0.5}
        
        # Calculate recent competitive balance
        recent_balance = self._calculate_competitive_balance(recent_matches)
        
        if historical_data:
            historical_balance = self._calculate_competitive_balance(historical_data)
            
            recent_score = recent_balance.get('balance_score', 0.5)
            historical_score = historical_balance.get('balance_score', 0.5)
            
            change = recent_score - historical_score
            
            if change > 0.1:
                direction = 'more_balanced'
            elif change < -0.1:
                direction = 'less_balanced'
            else:
                direction = 'stable'
            
            confidence = min(0.9, 0.5 + abs(change) * 2)
        else:
            direction = 'stable'
            confidence = 0.4
        
        return {
            'direction': direction,
            'confidence': confidence,
            'recent_balance_score': recent_balance.get('balance_score', 0.5),
            'change_magnitude': abs(change) if 'change' in locals() else 0
        }
    
    def _detect_home_advantage_trends(self, recent_matches: List[Dict], historical_data: Optional[List[Dict]] = None) -> Dict:
        """Detect trends in home advantage"""
        if not recent_matches:
            return {'trend': 'stable', 'confidence': 0.5}
        
        recent_home_adv = self._analyze_home_advantage(recent_matches)
        
        if historical_data:
            historical_home_adv = self._analyze_home_advantage(historical_data)
            
            recent_rate = recent_home_adv.get('win_rate', 0.45)
            historical_rate = historical_home_adv.get('win_rate', 0.45)
            
            change = recent_rate - historical_rate
            
            if change > 0.05:
                trend = 'increasing'
            elif change < -0.05:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            confidence = min(0.9, 0.5 + abs(change) * 5)
        else:
            trend = 'stable'
            confidence = 0.4
        
        return {
            'trend': trend,
            'confidence': confidence,
            'recent_win_rate': recent_home_adv.get('win_rate', 0.45),
            'change': change if 'change' in locals() else 0
        }
    
    def _detect_meta_evolution(self, recent_matches: List[Dict], historical_data: Optional[List[Dict]] = None) -> Dict:
        """Detect meta evolution patterns in the league"""
        if not recent_matches:
            return {'status': 'no_change', 'confidence': 0.5}
        
        recent_meta = self._analyze_meta_characteristics(recent_matches)
        
        if historical_data:
            historical_meta = self._analyze_meta_characteristics(historical_data)
            
            recent_volatility = recent_meta.get('volatility', 1.0)
            historical_volatility = historical_meta.get('volatility', 1.0)
            
            volatility_change = abs(recent_volatility - historical_volatility) / historical_volatility if historical_volatility > 0 else 0
            
            if volatility_change > 0.2:
                status = 'significant_change'
            elif volatility_change > 0.1:
                status = 'moderate_change'
            else:
                status = 'stable'
            
            confidence = min(0.9, 0.3 + volatility_change)
        else:
            status = 'insufficient_data'
            confidence = 0.3
        
        return {
            'status': status,
            'confidence': confidence,
            'volatility_change': volatility_change if 'volatility_change' in locals() else 0,
            'recent_volatility': recent_meta.get('volatility', 1.0)
        }
    
    def _extract_strength_metrics(self, profile: Dict) -> Dict:
        """Extract strength metrics from league profile"""
        characteristics = profile.get('characteristics', {})
        goals_analysis = characteristics.get('goals_analysis', {})
        competitive_balance = profile.get('competitive_balance', {})
        
        return {
            'avg_goals': goals_analysis.get('avg_goals', 2.5),
            'goal_variance': goals_analysis.get('goal_variance', 2.0),
            'balance_score': competitive_balance.get('balance_score', 0.5),
            'home_advantage': characteristics.get('home_advantage', {}).get('coefficient', 1.1),
            'quality_rating': self._calculate_quality_rating(profile),
            'predictability': profile.get('meta_characteristics', {}).get('predictability', 0.5)
        }
    
    def _calculate_strength_rankings(self, league_strengths: Dict) -> List[Dict]:
        """Calculate strength rankings for leagues"""
        rankings = []
        
        for league_id, metrics in league_strengths.items():
            # Calculate composite strength score
            strength_score = (
                metrics['quality_rating'] * 0.4 +
                metrics['balance_score'] * 0.2 +
                min(metrics['avg_goals'] / 3.5, 1.0) * 0.2 +
                metrics['predictability'] * 0.1 +
                (metrics['home_advantage'] - 1.0) * 0.1
            )
            
            rankings.append({
                'league_id': league_id,
                'strength_score': strength_score,
                'metrics': metrics
            })
        
        # Sort by strength score descending
        rankings.sort(key=lambda x: x['strength_score'], reverse=True)
        
        # Add rank numbers
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def _generate_pairwise_comparisons(self, league_strengths: Dict) -> Dict:
        """Generate pairwise comparisons between leagues"""
        comparisons = {}
        league_ids = list(league_strengths.keys())
        
        for i, league1 in enumerate(league_ids):
            for j, league2 in enumerate(league_ids[i+1:], i+1):
                metrics1 = league_strengths[league1]
                metrics2 = league_strengths[league2]
                
                # Calculate difference in key metrics
                goal_diff = metrics1['avg_goals'] - metrics2['avg_goals']
                balance_diff = metrics1['balance_score'] - metrics2['balance_score']
                quality_diff = metrics1['quality_rating'] - metrics2['quality_rating']
                
                # Overall strength difference
                strength_diff = (quality_diff * 0.6 + balance_diff * 0.2 + 
                               min(goal_diff / 2.0, 0.5) * 0.2)
                
                comparison_key = f"{league1}_vs_{league2}"
                comparisons[comparison_key] = {
                    'league1_id': league1,
                    'league2_id': league2,
                    'strength_difference': strength_diff,
                    'goal_difference': goal_diff,
                    'balance_difference': balance_diff,
                    'quality_difference': quality_diff,
                    'stronger_league': league1 if strength_diff > 0 else league2,
                    'confidence': min(0.9, 0.5 + abs(strength_diff))
                }
        
        return comparisons
    
    def _calculate_strength_tiers(self, league_strengths: Dict) -> Dict:
        """Calculate strength tiers for leagues"""
        if not league_strengths:
            return {}
        
        # Calculate quality ratings for all leagues
        quality_ratings = [metrics['quality_rating'] for metrics in league_strengths.values()]
        
        if len(quality_ratings) < 2:
            # Single league - put in medium tier
            return {list(league_strengths.keys())[0]: 'medium'}
        
        # Use quartiles to define tiers
        q1 = np.percentile(quality_ratings, 25)
        q3 = np.percentile(quality_ratings, 75)
        
        tiers = {}
        for league_id, metrics in league_strengths.items():
            quality = metrics['quality_rating']
            
            if quality >= q3:
                tiers[league_id] = 'elite'
            elif quality >= np.median(quality_ratings):
                tiers[league_id] = 'high'
            elif quality >= q1:
                tiers[league_id] = 'medium'
            else:
                tiers[league_id] = 'low'
        
        return tiers
    
    def _calculate_assessment_confidence(self, leagues: List[Tuple[int, Dict]]) -> float:
        """Calculate confidence in comparative assessment"""
        if len(leagues) < 2:
            return 0.0
        
        # Average confidence across all league profiles
        confidences = [profile.get('profile_confidence', 0.5) for _, profile in leagues]
        avg_confidence = np.mean(confidences)
        
        # Reduce confidence if sample size is small
        if len(leagues) < 5:
            avg_confidence *= 0.8
        
        return avg_confidence
    
    def _calculate_quality_rating(self, profile: Dict) -> float:
        """Calculate overall quality rating for a league"""
        # Extract key indicators
        basic_context = profile.get('basic_context', {})
        characteristics = profile.get('characteristics', {})
        competitive_balance = profile.get('competitive_balance', {})
        
        # Quality mapping from league context
        quality_map = {'elite': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.4, 'unknown': 0.5}
        base_quality = quality_map.get(basic_context.get('league_quality', 'unknown'), 0.5)
        
        # Adjust based on competitive balance (more balanced = higher quality)
        balance_score = competitive_balance.get('balance_score', 0.5)
        balance_adjustment = (balance_score - 0.5) * 0.2
        
        # Adjust based on goals average (moderate scoring preferred)
        goals_analysis = characteristics.get('goals_analysis', {})
        avg_goals = goals_analysis.get('avg_goals', 2.5)
        
        # Optimal goals per match around 2.5-3.0
        if 2.3 <= avg_goals <= 3.2:
            goals_adjustment = 0.1
        elif 2.0 <= avg_goals <= 3.5:
            goals_adjustment = 0.05
        else:
            goals_adjustment = -0.05
        
        quality_rating = base_quality + balance_adjustment + goals_adjustment
        return max(0.0, min(1.0, quality_rating))
    
    def _calculate_season_start_effect(self, monthly_stats: Dict) -> Dict:
        """Calculate season start effects"""
        # Analyze August-October data
        start_months = [8, 9, 10]
        start_data = {month: stats for month, stats in monthly_stats.items() if month in start_months}
        
        if not start_data:
            return {'effect': 'none', 'confidence': 0.0}
        
        # Calculate average goals in start period
        all_goals = []
        for month_data in start_data.values():
            all_goals.extend(month_data['goals'])
        
        if not all_goals:
            return {'effect': 'none', 'confidence': 0.0}
        
        start_avg = np.mean(all_goals)
        
        # Compare with other periods
        other_months = [month for month in monthly_stats.keys() if month not in start_months]
        other_goals = []
        for month in other_months:
            other_goals.extend(monthly_stats[month]['goals'])
        
        if other_goals:
            other_avg = np.mean(other_goals)
            diff = start_avg - other_avg
            
            if diff > 0.3:
                effect = 'high_scoring'
            elif diff < -0.3:
                effect = 'low_scoring'
            else:
                effect = 'normal'
            
            confidence = min(0.9, 0.3 + abs(diff))
        else:
            effect = 'normal'
            confidence = 0.3
        
        return {
            'effect': effect,
            'confidence': confidence,
            'avg_goals': start_avg,
            'difference': diff if 'diff' in locals() else 0
        }
    
    def _calculate_winter_break_effect(self, monthly_stats: Dict) -> Dict:
        """Calculate winter break effects"""
        winter_months = [12, 1]
        winter_data = {month: stats for month, stats in monthly_stats.items() if month in winter_months}
        
        if not winter_data:
            return {'effect': 'none', 'confidence': 0.0}
        
        # Calculate average goals in winter period
        all_goals = []
        for month_data in winter_data.values():
            all_goals.extend(month_data['goals'])
        
        if not all_goals:
            return {'effect': 'none', 'confidence': 0.0}
        
        winter_avg = np.mean(all_goals)
        
        # Compare with other periods
        other_months = [month for month in monthly_stats.keys() if month not in winter_months]
        other_goals = []
        for month in other_months:
            other_goals.extend(monthly_stats[month]['goals'])
        
        if other_goals:
            other_avg = np.mean(other_goals)
            diff = winter_avg - other_avg
            
            if diff > 0.2:
                effect = 'high_scoring'
            elif diff < -0.2:
                effect = 'low_scoring'
            else:
                effect = 'normal'
            
            confidence = min(0.8, 0.3 + abs(diff))
        else:
            effect = 'normal'
            confidence = 0.3
        
        return {
            'effect': effect,
            'confidence': confidence,
            'avg_goals': winter_avg,
            'difference': diff if 'diff' in locals() else 0
        }
    
    def _calculate_end_season_effect(self, monthly_stats: Dict) -> Dict:
        """Calculate end season effects"""
        end_months = [4, 5]
        end_data = {month: stats for month, stats in monthly_stats.items() if month in end_months}
        
        if not end_data:
            return {'effect': 'none', 'confidence': 0.0}
        
        # Calculate average goals in end period
        all_goals = []
        for month_data in end_data.values():
            all_goals.extend(month_data['goals'])
        
        if not all_goals:
            return {'effect': 'none', 'confidence': 0.0}
        
        end_avg = np.mean(all_goals)
        
        # Compare with other periods
        other_months = [month for month in monthly_stats.keys() if month not in end_months]
        other_goals = []
        for month in other_months:
            other_goals.extend(monthly_stats[month]['goals'])
        
        if other_goals:
            other_avg = np.mean(other_goals)
            diff = end_avg - other_avg
            
            if diff > 0.2:
                effect = 'high_intensity'
            elif diff < -0.2:
                effect = 'low_intensity'
            else:
                effect = 'normal'
            
            confidence = min(0.8, 0.3 + abs(diff))
        else:
            effect = 'normal'
            confidence = 0.3
        
        return {
            'effect': effect,
            'confidence': confidence,
            'avg_goals': end_avg,
            'difference': diff if 'diff' in locals() else 0
        }
    
    def _calculate_result_entropy(self, results: List[int]) -> float:
        """Calculate entropy of results for predictability measure"""
        if not results:
            return 1.0
        
        # Count occurrences of each result type
        result_counts = Counter(results)
        total = len(results)
        
        # Calculate entropy
        entropy = 0
        for count in result_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize entropy (max entropy for 3 outcomes is log2(3) ≈ 1.585)
        max_entropy = math.log2(3)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _calculate_upset_frequency(self, matches: List[Dict]) -> float:
        """Calculate frequency of upset results"""
        if not matches:
            return 0.1
        
        upsets = 0
        total = 0
        
        for match in matches:
            if match.get('status') in ['FINISHED', 'FT', 'AET', 'PEN']:
                home_goals = match.get('home_score', 0) or 0
                away_goals = match.get('away_score', 0) or 0
                
                total += 1
                
                # Simple heuristic: away win or large goal difference could indicate upset
                if away_goals > home_goals or abs(home_goals - away_goals) >= 3:
                    upsets += 1
        
        return upsets / total if total > 0 else 0.1