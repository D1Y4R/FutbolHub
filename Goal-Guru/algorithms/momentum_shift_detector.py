"""
Advanced Momentum Shift Detector for Football Prediction System
Implements sophisticated momentum change detection and predictive modeling

Key Features:
1. Statistical Changepoint Detection (PELT/CUSUM algorithms)
2. Momentum Pattern Recognition (winning/losing streaks, confidence shifts)  
3. Predictive Momentum Modeling (trajectory prediction, sustainability assessment)
4. Context-Aware Analysis (league-specific patterns, manager effects)

Author: Football Prediction System
Date: September 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from scipy import stats, signal
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import math
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MomentumShiftDetector:
    """
    Advanced momentum shift detection system for football predictions
    
    Implements state-of-the-art changepoint detection algorithms and
    predictive momentum modeling for tactical advantage prediction
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Momentum Shift Detector
        
        Args:
            config: Configuration dictionary for customization
        """
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()
        
        # Internal state for pattern learning
        self.momentum_patterns = {}
        self.shift_histories = defaultdict(list)
        self.league_momentum_profiles = {}
        self.manager_effect_patterns = {}
        
        # Performance tracking for continuous learning
        self.prediction_accuracy = defaultdict(list)
        self.shift_detection_history = []
        
        # Real-time momentum tracking
        self.real_time_momentum = {}
        self.momentum_trajectories = {}
        
        logger.info("MomentumShiftDetector initialized with advanced algorithms")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for momentum shift detection"""
        return {
            # Changepoint detection parameters
            'changepoint_detection': {
                'pelt_penalty': 2.0,        # PELT algorithm penalty parameter
                'cusum_threshold': 5.0,     # CUSUM detection threshold
                'min_segment_length': 3,    # Minimum matches between changepoints
                'max_changepoints': 10,     # Maximum changepoints to detect
                'confidence_level': 0.95    # Statistical confidence level
            },
            
            # Pattern recognition parameters
            'pattern_recognition': {
                'min_streak_length': 3,     # Minimum streak to consider significant
                'confidence_threshold': 0.7, # Pattern confidence threshold
                'similarity_threshold': 0.8, # Pattern similarity threshold
                'volatility_window': 10,    # Window for volatility calculation
                'momentum_memory': 20       # Historical momentum memory
            },
            
            # Predictive modeling parameters
            'predictive_modeling': {
                'prediction_horizon': 5,    # Matches to predict ahead
                'decay_rate': 0.9,         # Momentum decay rate
                'sustainability_threshold': 0.6, # Sustainability threshold
                'recovery_estimation_window': 15, # Window for recovery estimation
                'peak_detection_sensitivity': 0.8 # Peak detection sensitivity
            },
            
            # Context-aware analysis parameters
            'context_analysis': {
                'league_adaptation_rate': 0.1, # Rate of league pattern learning
                'venue_weight_difference': 0.15, # Home/away momentum difference
                'manager_effect_window': 10,   # Matches to analyze manager effect
                'transfer_window_impact': 0.2, # Transfer window momentum impact
                'pressure_match_multiplier': 1.5 # Multiplier for high-pressure matches
            },
            
            # Output configuration
            'output_config': {
                'momentum_score_range': (0, 100), # Momentum score range
                'trend_strength_levels': 5,       # Number of trend strength levels
                'shift_probability_precision': 0.01, # Shift probability precision
                'historical_depth': 50           # Historical data depth
            }
        }
    
    def detect_momentum_shifts(self, team_data: Dict, match_context: Dict) -> Dict:
        """
        Main momentum shift detection function
        
        Args:
            team_data: Team's comprehensive match and performance data
            match_context: Context of upcoming match and environment
            
        Returns:
            Dict containing complete momentum shift analysis
        """
        try:
            # Extract and prepare data
            matches = team_data.get('recent_matches', [])
            team_id = team_data.get('team_id', 0)
            league_id = match_context.get('league_id', 0)
            match_date = match_context.get('match_date', datetime.now())
            
            if isinstance(match_date, str):
                match_date = datetime.strptime(match_date, '%Y-%m-%d')
            
            # 1. Statistical Changepoint Detection
            changepoint_analysis = self._detect_statistical_changepoints(matches, team_id)
            
            # 2. Momentum Pattern Recognition  
            pattern_analysis = self._recognize_momentum_patterns(matches, team_id, match_context)
            
            # 3. Predictive Momentum Modeling
            predictive_analysis = self._model_momentum_trajectory(matches, team_id, match_context)
            
            # 4. Context-Aware Analysis
            context_analysis = self._analyze_contextual_factors(matches, team_id, match_context)
            
            # 5. Current Momentum Assessment
            current_momentum = self._assess_current_momentum(
                changepoint_analysis, pattern_analysis, predictive_analysis, context_analysis
            )
            
            # 6. Shift Probability Calculation
            shift_probabilities = self._calculate_shift_probabilities(
                changepoint_analysis, pattern_analysis, predictive_analysis, match_context
            )
            
            # 7. Historical Shift Points Identification
            historical_shifts = self._identify_historical_shifts(matches, team_id)
            
            # Compile comprehensive analysis
            momentum_analysis = {
                'current_momentum_score': current_momentum['score'],
                'momentum_direction': current_momentum['direction'],
                'trend_strength': current_momentum['strength'],
                'changepoint_analysis': changepoint_analysis,
                'pattern_analysis': pattern_analysis,
                'predictive_analysis': predictive_analysis,
                'context_analysis': context_analysis,
                'shift_probabilities': shift_probabilities,
                'historical_shifts': historical_shifts,
                'confidence_level': current_momentum['confidence'],
                'analysis_metadata': {
                    'team_id': team_id,
                    'analysis_date': match_date.isoformat(),
                    'matches_analyzed': len(matches),
                    'detection_algorithms_used': ['PELT', 'CUSUM', 'MovingVariance', 'PatternRecognition']
                }
            }
            
            # Update real-time tracking
            self._update_real_time_tracking(team_id, momentum_analysis)
            
            logger.info(f"Momentum shift analysis completed for team {team_id}")
            logger.info(f"Current momentum score: {current_momentum['score']:.1f}, Direction: {current_momentum['direction']}")
            
            return momentum_analysis
            
        except Exception as e:
            logger.error(f"Error in momentum shift detection: {str(e)}")
            return self._get_default_momentum_analysis()
    
    def _detect_statistical_changepoints(self, matches: List[Dict], team_id: int) -> Dict:
        """
        Implement multiple statistical changepoint detection algorithms
        
        Uses PELT, CUSUM, and moving window variance detection
        """
        try:
            if len(matches) < 5:
                return self._get_default_changepoint_analysis()
            
            # Extract performance time series
            performance_series = self._extract_performance_timeseries(matches, team_id)
            if len(performance_series) < 5:
                return self._get_default_changepoint_analysis()
            
            # 1. PELT (Pruned Exact Linear Time) Algorithm
            pelt_changepoints = self._pelt_algorithm(performance_series)
            
            # 2. CUSUM (Cumulative Sum) Algorithm
            cusum_changepoints = self._cusum_algorithm(performance_series)
            
            # 3. Moving Window Variance Detection
            variance_changepoints = self._moving_variance_detection(performance_series)
            
            # 4. Bayesian Changepoint Detection
            bayesian_changepoints = self._bayesian_changepoint_detection(performance_series)
            
            # Combine and validate changepoints
            combined_changepoints = self._combine_changepoint_results(
                pelt_changepoints, cusum_changepoints, variance_changepoints, bayesian_changepoints
            )
            
            # Analyze changepoint characteristics
            changepoint_characteristics = self._analyze_changepoint_characteristics(
                combined_changepoints, performance_series
            )
            
            # Calculate confidence scores
            detection_confidence = self._calculate_detection_confidence(
                pelt_changepoints, cusum_changepoints, variance_changepoints, bayesian_changepoints
            )
            
            return {
                'detected_changepoints': combined_changepoints,
                'changepoint_characteristics': changepoint_characteristics,
                'algorithm_results': {
                    'pelt': pelt_changepoints,
                    'cusum': cusum_changepoints,
                    'variance': variance_changepoints,
                    'bayesian': bayesian_changepoints
                },
                'detection_confidence': detection_confidence,
                'performance_series': performance_series,
                'statistical_summary': self._calculate_statistical_summary(performance_series)
            }
            
        except Exception as e:
            logger.error(f"Error in statistical changepoint detection: {str(e)}")
            return self._get_default_changepoint_analysis()
    
    def _pelt_algorithm(self, data: List[float]) -> List[int]:
        """
        Implement PELT (Pruned Exact Linear Time) changepoint detection
        
        Detects changes in mean and variance with optimal computational complexity
        """
        try:
            n = len(data)
            if n < 3:
                return []
            
            data_array = np.array(data)
            penalty = self.config['changepoint_detection']['pelt_penalty']
            min_segment = self.config['changepoint_detection']['min_segment_length']
            
            # Cost function for PELT (negative log-likelihood for normal distribution)
            def cost_function(segment_data):
                if len(segment_data) <= 1:
                    return float('inf')
                mean = np.mean(segment_data)
                variance = np.var(segment_data)
                if variance <= 1e-10:  # Avoid log(0)
                    variance = 1e-10
                return len(segment_data) * (np.log(2 * np.pi * variance) + 1)
            
            # Dynamic programming for PELT
            cost = np.full(n + 1, float('inf'))
            cost[0] = 0
            changepoints = []
            
            for t in range(min_segment, n + 1):
                candidates = []
                for s in range(max(0, t - 50), t - min_segment + 1):  # Limit search window
                    if cost[s] != float('inf'):
                        segment_cost = cost_function(data_array[s:t])
                        total_cost = cost[s] + segment_cost + penalty
                        candidates.append((total_cost, s))
                
                if candidates:
                    best_cost, best_s = min(candidates)
                    cost[t] = best_cost
                    
                    # Track changepoints
                    if t == n:  # Final step - reconstruct changepoints
                        current = n
                        while current > 0:
                            for candidate_cost, candidate_s in candidates:
                                if abs(candidate_cost - cost[current]) < 1e-10:
                                    if candidate_s > 0:
                                        changepoints.append(candidate_s)
                                    current = candidate_s
                                    break
                            else:
                                break
            
            changepoints = sorted(list(set(changepoints)))
            # Filter out changepoints too close to each other
            filtered_changepoints = []
            for cp in changepoints:
                if not filtered_changepoints or cp - filtered_changepoints[-1] >= min_segment:
                    filtered_changepoints.append(cp)
            
            return filtered_changepoints[:self.config['changepoint_detection']['max_changepoints']]
            
        except Exception as e:
            logger.warning(f"PELT algorithm error: {str(e)}")
            return []
    
    def _cusum_algorithm(self, data: List[float]) -> List[int]:
        """
        Implement CUSUM (Cumulative Sum) changepoint detection
        
        Detects changes in mean level using cumulative sum statistics
        """
        try:
            if len(data) < 3:
                return []
            
            data_array = np.array(data)
            n = len(data_array)
            threshold = self.config['changepoint_detection']['cusum_threshold']
            
            # Calculate mean and standard deviation
            overall_mean = np.mean(data_array)
            overall_std = np.std(data_array)
            
            if overall_std <= 1e-10:
                return []
            
            # CUSUM statistics
            cusum_pos = np.zeros(n)
            cusum_neg = np.zeros(n)
            changepoints = []
            
            for i in range(1, n):
                # Positive CUSUM (detects upward shifts)
                cusum_pos[i] = max(0, cusum_pos[i-1] + (data_array[i] - overall_mean) - overall_std/2)
                
                # Negative CUSUM (detects downward shifts)
                cusum_neg[i] = max(0, cusum_neg[i-1] - (data_array[i] - overall_mean) - overall_std/2)
                
                # Check for changepoints
                if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                    changepoints.append(i)
                    # Reset CUSUM statistics
                    cusum_pos[i] = 0
                    cusum_neg[i] = 0
            
            # Filter out consecutive changepoints
            min_distance = self.config['changepoint_detection']['min_segment_length']
            filtered_changepoints = []
            for cp in changepoints:
                if not filtered_changepoints or cp - filtered_changepoints[-1] >= min_distance:
                    filtered_changepoints.append(cp)
            
            return filtered_changepoints[:self.config['changepoint_detection']['max_changepoints']]
            
        except Exception as e:
            logger.warning(f"CUSUM algorithm error: {str(e)}")
            return []
    
    def _moving_variance_detection(self, data: List[float]) -> List[int]:
        """
        Detect changepoints using moving window variance analysis
        
        Identifies points where performance volatility changes significantly
        """
        try:
            if len(data) < 6:
                return []
            
            data_array = np.array(data)
            window_size = min(5, len(data) // 3)
            changepoints = []
            
            # Calculate moving variance
            variances = []
            for i in range(len(data_array) - window_size + 1):
                window_data = data_array[i:i + window_size]
                variance = np.var(window_data)
                variances.append(variance)
            
            if len(variances) < 3:
                return []
            
            variances = np.array(variances)
            
            # Detect significant variance changes
            variance_changes = np.abs(np.diff(variances))
            variance_threshold = np.percentile(variance_changes, 75)  # Top 25% of changes
            
            for i, change in enumerate(variance_changes):
                if change > variance_threshold:
                    changepoint = i + window_size // 2  # Position at center of window
                    changepoints.append(changepoint)
            
            # Filter changepoints
            min_distance = self.config['changepoint_detection']['min_segment_length']
            filtered_changepoints = []
            for cp in changepoints:
                if not filtered_changepoints or cp - filtered_changepoints[-1] >= min_distance:
                    filtered_changepoints.append(cp)
            
            return filtered_changepoints[:self.config['changepoint_detection']['max_changepoints']]
            
        except Exception as e:
            logger.warning(f"Moving variance detection error: {str(e)}")
            return []
    
    def _bayesian_changepoint_detection(self, data: List[float]) -> List[int]:
        """
        Implement Bayesian changepoint detection
        
        Uses Bayesian inference to detect changepoints with uncertainty quantification
        """
        try:
            if len(data) < 4:
                return []
            
            data_array = np.array(data)
            n = len(data_array)
            
            # Simple Bayesian changepoint detection using probability ratios
            changepoints = []
            min_segment = self.config['changepoint_detection']['min_segment_length']
            
            for t in range(min_segment, n - min_segment + 1):
                # Split data at potential changepoint
                before = data_array[:t]
                after = data_array[t:]
                
                # Calculate likelihood ratio
                if len(before) > 0 and len(after) > 0:
                    # Full data likelihood (no changepoint)
                    full_mean = np.mean(data_array)
                    full_var = np.var(data_array)
                    if full_var <= 1e-10:
                        full_var = 1e-10
                    
                    full_likelihood = -0.5 * n * np.log(2 * np.pi * full_var) - \
                                    0.5 * np.sum((data_array - full_mean)**2) / full_var
                    
                    # Split data likelihood (with changepoint)
                    before_mean = np.mean(before)
                    before_var = np.var(before) if len(before) > 1 else 1e-10
                    if before_var <= 1e-10:
                        before_var = 1e-10
                    
                    after_mean = np.mean(after)
                    after_var = np.var(after) if len(after) > 1 else 1e-10
                    if after_var <= 1e-10:
                        after_var = 1e-10
                    
                    before_likelihood = -0.5 * len(before) * np.log(2 * np.pi * before_var) - \
                                      0.5 * np.sum((before - before_mean)**2) / before_var
                    
                    after_likelihood = -0.5 * len(after) * np.log(2 * np.pi * after_var) - \
                                     0.5 * np.sum((after - after_mean)**2) / after_var
                    
                    split_likelihood = before_likelihood + after_likelihood
                    
                    # Log Bayes factor
                    log_bayes_factor = split_likelihood - full_likelihood
                    
                    # Decision threshold (log Bayes factor > 1 indicates strong evidence)
                    if log_bayes_factor > 1.0:
                        changepoints.append(t)
            
            # Filter changepoints
            filtered_changepoints = []
            for cp in changepoints:
                if not filtered_changepoints or cp - filtered_changepoints[-1] >= min_segment:
                    filtered_changepoints.append(cp)
            
            return filtered_changepoints[:self.config['changepoint_detection']['max_changepoints']]
            
        except Exception as e:
            logger.warning(f"Bayesian changepoint detection error: {str(e)}")
            return []
    
    def _recognize_momentum_patterns(self, matches: List[Dict], team_id: int, 
                                   match_context: Dict) -> Dict:
        """
        Advanced momentum pattern recognition
        
        Identifies winning/losing streaks, confidence shifts, and team dynamics
        """
        try:
            if not matches:
                return self._get_default_pattern_analysis()
            
            # Extract performance and result patterns
            performance_data = self._extract_comprehensive_performance_data(matches, team_id)
            
            # 1. Winning/Losing Streak Analysis
            streak_analysis = self._analyze_streaks_advanced(performance_data)
            
            # 2. Performance Inflection Point Detection
            inflection_analysis = self._detect_performance_inflections(performance_data)
            
            # 3. Confidence Level Shifts Detection
            confidence_analysis = self._analyze_confidence_shifts(performance_data, matches, team_id)
            
            # 4. Team Dynamics Change Detection
            dynamics_analysis = self._detect_team_dynamics_changes(performance_data, matches, team_id)
            
            # 5. Gradual vs Sudden Momentum Shifts
            shift_type_analysis = self._classify_momentum_shift_types(performance_data)
            
            # 6. Pattern Clustering and Classification
            pattern_clusters = self._cluster_momentum_patterns(performance_data)
            
            # 7. Recurring Pattern Detection
            recurring_patterns = self._detect_recurring_patterns(performance_data, team_id)
            
            return {
                'streak_analysis': streak_analysis,
                'inflection_points': inflection_analysis,
                'confidence_shifts': confidence_analysis,
                'team_dynamics_changes': dynamics_analysis,
                'shift_type_classification': shift_type_analysis,
                'pattern_clusters': pattern_clusters,
                'recurring_patterns': recurring_patterns,
                'pattern_strength_score': self._calculate_pattern_strength(
                    streak_analysis, inflection_analysis, confidence_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error in momentum pattern recognition: {str(e)}")
            return self._get_default_pattern_analysis()
    
    def _model_momentum_trajectory(self, matches: List[Dict], team_id: int, 
                                 match_context: Dict) -> Dict:
        """
        Predictive momentum modeling and trajectory forecasting
        
        Predicts future momentum changes and sustainability
        """
        try:
            if len(matches) < 3:
                return self._get_default_predictive_analysis()
            
            performance_data = self._extract_comprehensive_performance_data(matches, team_id)
            
            # 1. Future Momentum Trajectory Prediction
            trajectory_prediction = self._predict_momentum_trajectory(performance_data, match_context)
            
            # 2. Momentum Sustainability Assessment
            sustainability_analysis = self._assess_momentum_sustainability(performance_data)
            
            # 3. Recovery Time Estimation
            recovery_analysis = self._estimate_recovery_time(performance_data)
            
            # 4. Peak Performance Window Prediction
            peak_prediction = self._predict_peak_performance_windows(performance_data, match_context)
            
            # 5. Momentum Decay Rate Modeling
            decay_modeling = self._model_momentum_decay_rate(performance_data)
            
            # 6. Confidence Intervals for Predictions
            prediction_intervals = self._calculate_prediction_intervals(
                trajectory_prediction, sustainability_analysis, recovery_analysis
            )
            
            return {
                'trajectory_prediction': trajectory_prediction,
                'sustainability_assessment': sustainability_analysis,
                'recovery_time_estimation': recovery_analysis,
                'peak_performance_prediction': peak_prediction,
                'momentum_decay_modeling': decay_modeling,
                'prediction_confidence_intervals': prediction_intervals,
                'prediction_accuracy_score': self._calculate_prediction_accuracy_score(performance_data)
            }
            
        except Exception as e:
            logger.error(f"Error in predictive momentum modeling: {str(e)}")
            return self._get_default_predictive_analysis()
    
    def _analyze_contextual_factors(self, matches: List[Dict], team_id: int, 
                                  match_context: Dict) -> Dict:
        """
        Context-aware momentum analysis
        
        Analyzes league-specific patterns, venue effects, and external factors
        """
        try:
            league_id = match_context.get('league_id', 0)
            venue = match_context.get('venue', 'unknown')
            opponent_id = match_context.get('opponent_id', 0)
            
            # 1. League-Specific Momentum Patterns
            league_analysis = self._analyze_league_momentum_patterns(matches, team_id, league_id)
            
            # 2. Home/Away Momentum Differences
            venue_analysis = self._analyze_venue_momentum_effects(matches, team_id, venue)
            
            # 3. Manager Change Momentum Effects
            manager_analysis = self._analyze_manager_change_effects(matches, team_id, match_context)
            
            # 4. Transfer Window Momentum Impacts
            transfer_analysis = self._analyze_transfer_window_impacts(matches, team_id, match_context)
            
            # 5. Derby/Pressure Match Momentum Variations
            pressure_analysis = self._analyze_pressure_match_effects(matches, team_id, match_context)
            
            # 6. Opponent-Specific Momentum Patterns
            opponent_analysis = self._analyze_opponent_specific_patterns(matches, team_id, opponent_id)
            
            # 7. Seasonal and Calendar Effects
            seasonal_analysis = self._analyze_seasonal_momentum_effects(matches, team_id, match_context)
            
            return {
                'league_specific_patterns': league_analysis,
                'venue_momentum_effects': venue_analysis,
                'manager_change_effects': manager_analysis,
                'transfer_window_impacts': transfer_analysis,
                'pressure_match_effects': pressure_analysis,
                'opponent_specific_patterns': opponent_analysis,
                'seasonal_effects': seasonal_analysis,
                'contextual_adjustment_factor': self._calculate_contextual_adjustment_factor(
                    league_analysis, venue_analysis, manager_analysis, pressure_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error in contextual factor analysis: {str(e)}")
            return self._get_default_context_analysis()
    
    # Helper Methods for Statistical Analysis
    
    def _extract_performance_timeseries(self, matches: List[Dict], team_id: int) -> List[float]:
        """Extract normalized performance time series from matches"""
        try:
            performance_series = []
            
            for match in matches:
                # Extract basic match performance
                home_team = match.get('teams', {}).get('home', {})
                away_team = match.get('teams', {}).get('away', {})
                score = match.get('score', {}).get('fulltime', {})
                
                if not score:
                    continue
                
                home_goals = score.get('home', 0) or 0
                away_goals = score.get('away', 0) or 0
                
                # Calculate performance score (0-1 scale)
                if home_team.get('id') == team_id:
                    # Home team performance
                    if home_goals > away_goals:
                        performance = 1.0  # Win
                    elif home_goals == away_goals:
                        performance = 0.5  # Draw
                    else:
                        performance = 0.0  # Loss
                    
                    # Adjust for goal difference
                    goal_diff = home_goals - away_goals
                    performance += goal_diff * 0.1  # Small adjustment for goal difference
                    
                elif away_team.get('id') == team_id:
                    # Away team performance
                    if away_goals > home_goals:
                        performance = 1.0  # Win
                    elif away_goals == home_goals:
                        performance = 0.5  # Draw
                    else:
                        performance = 0.0  # Loss
                    
                    # Adjust for goal difference
                    goal_diff = away_goals - home_goals
                    performance += goal_diff * 0.1
                else:
                    continue
                
                # Normalize to [0, 1] range
                performance = max(0.0, min(1.0, performance))
                performance_series.append(performance)
            
            # Reverse to get chronological order (oldest first)
            return performance_series[::-1]
            
        except Exception as e:
            logger.error(f"Error extracting performance time series: {str(e)}")
            return []
    
    def _combine_changepoint_results(self, pelt: List[int], cusum: List[int], 
                                   variance: List[int], bayesian: List[int]) -> List[int]:
        """Combine results from multiple changepoint detection algorithms"""
        try:
            all_changepoints = set(pelt + cusum + variance + bayesian)
            min_distance = self.config['changepoint_detection']['min_segment_length']
            
            # Remove changepoints that are too close
            sorted_changepoints = sorted(all_changepoints)
            filtered_changepoints = []
            
            for cp in sorted_changepoints:
                if not filtered_changepoints or cp - filtered_changepoints[-1] >= min_distance:
                    filtered_changepoints.append(cp)
            
            return filtered_changepoints[:self.config['changepoint_detection']['max_changepoints']]
            
        except Exception as e:
            logger.error(f"Error combining changepoint results: {str(e)}")
            return []
    
    def _calculate_detection_confidence(self, pelt: List[int], cusum: List[int], 
                                      variance: List[int], bayesian: List[int]) -> float:
        """Calculate confidence in changepoint detection"""
        try:
            all_algorithms = [pelt, cusum, variance, bayesian]
            total_detections = sum(len(alg_result) for alg_result in all_algorithms)
            
            if total_detections == 0:
                return 0.0
            
            # Calculate agreement between algorithms
            all_changepoints = set()
            for alg_result in all_algorithms:
                all_changepoints.update(alg_result)
            
            agreement_scores = []
            for cp in all_changepoints:
                agreements = sum(1 for alg_result in all_algorithms 
                               if any(abs(cp - detected_cp) <= 1 for detected_cp in alg_result))
                agreement_scores.append(agreements / len(all_algorithms))
            
            return np.mean(agreement_scores) if agreement_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating detection confidence: {str(e)}")
            return 0.0
    
    # Default response methods
    
    def _get_default_momentum_analysis(self) -> Dict:
        """Return default momentum analysis when errors occur"""
        return {
            'current_momentum_score': 50.0,
            'momentum_direction': 'stable',
            'trend_strength': 0.0,
            'changepoint_analysis': self._get_default_changepoint_analysis(),
            'pattern_analysis': self._get_default_pattern_analysis(),
            'predictive_analysis': self._get_default_predictive_analysis(),
            'context_analysis': self._get_default_context_analysis(),
            'shift_probabilities': {'next_1_matches': 0.1, 'next_5_matches': 0.3},
            'historical_shifts': [],
            'confidence_level': 0.0,
            'analysis_metadata': {
                'team_id': 0,
                'analysis_date': datetime.now().isoformat(),
                'matches_analyzed': 0,
                'detection_algorithms_used': []
            }
        }
    
    def _get_default_changepoint_analysis(self) -> Dict:
        """Return default changepoint analysis"""
        return {
            'detected_changepoints': [],
            'changepoint_characteristics': [],
            'algorithm_results': {'pelt': [], 'cusum': [], 'variance': [], 'bayesian': []},
            'detection_confidence': 0.0,
            'performance_series': [],
            'statistical_summary': {'mean': 0.5, 'std': 0.0, 'variance': 0.0}
        }
    
    def _get_default_pattern_analysis(self) -> Dict:
        """Return default pattern analysis"""
        return {
            'streak_analysis': {'current_streak': {'type': 'none', 'length': 0}},
            'inflection_points': [],
            'confidence_shifts': [],
            'team_dynamics_changes': [],
            'shift_type_classification': {'gradual': 0, 'sudden': 0},
            'pattern_clusters': [],
            'recurring_patterns': [],
            'pattern_strength_score': 0.0
        }
    
    def _get_default_predictive_analysis(self) -> Dict:
        """Return default predictive analysis"""
        return {
            'trajectory_prediction': {'predicted_scores': [50.0] * 5},
            'sustainability_assessment': {'current_sustainability': 0.5},
            'recovery_time_estimation': {'estimated_recovery_matches': 5},
            'peak_performance_prediction': {'next_peak_probability': 0.2},
            'momentum_decay_modeling': {'decay_rate': 0.9},
            'prediction_confidence_intervals': {'lower': [40.0] * 5, 'upper': [60.0] * 5},
            'prediction_accuracy_score': 0.0
        }
    
    def _get_default_context_analysis(self) -> Dict:
        """Return default context analysis"""
        return {
            'league_specific_patterns': {'adjustment_factor': 1.0},
            'venue_momentum_effects': {'home_advantage': 0.0, 'away_challenge': 0.0},
            'manager_change_effects': {'recent_change': False, 'effect_strength': 0.0},
            'transfer_window_impacts': {'recent_transfers': False, 'impact_strength': 0.0},
            'pressure_match_effects': {'is_pressure_match': False, 'pressure_multiplier': 1.0},
            'opponent_specific_patterns': {'historical_performance': 0.5},
            'seasonal_effects': {'current_season_effect': 0.0},
            'contextual_adjustment_factor': 1.0
        }
    
    # Placeholder methods for advanced algorithms (to be implemented)
    
    def _analyze_changepoint_characteristics(self, changepoints: List[int], 
                                           performance_series: List[float]) -> List[Dict]:
        """Analyze characteristics of detected changepoints"""
        # Implementation placeholder
        return []
    
    def _calculate_statistical_summary(self, performance_series: List[float]) -> Dict:
        """Calculate statistical summary of performance series"""
        try:
            if not performance_series:
                return {'mean': 0.5, 'std': 0.0, 'variance': 0.0}
            
            array = np.array(performance_series)
            return {
                'mean': float(np.mean(array)),
                'std': float(np.std(array)),
                'variance': float(np.var(array)),
                'min': float(np.min(array)),
                'max': float(np.max(array)),
                'median': float(np.median(array))
            }
        except Exception as e:
            logger.error(f"Error calculating statistical summary: {str(e)}")
            return {'mean': 0.5, 'std': 0.0, 'variance': 0.0}
    
    def _extract_comprehensive_performance_data(self, matches: List[Dict], team_id: int) -> Dict:
        """Extract comprehensive performance data for pattern analysis"""
        # Implementation placeholder
        return {'performance_scores': self._extract_performance_timeseries(matches, team_id)}
    
    def _assess_current_momentum(self, changepoint_analysis: Dict, pattern_analysis: Dict,
                               predictive_analysis: Dict, context_analysis: Dict) -> Dict:
        """Assess current momentum score and characteristics"""
        try:
            # Base momentum score from performance series
            performance_series = changepoint_analysis.get('performance_series', [50.0])
            if performance_series:
                recent_performance = np.mean(performance_series[-5:]) * 100  # Last 5 matches
            else:
                recent_performance = 50.0
            
            # Adjust for pattern strength
            pattern_strength = pattern_analysis.get('pattern_strength_score', 0.0)
            momentum_score = recent_performance + (pattern_strength * 10)
            
            # Apply contextual adjustments
            context_factor = context_analysis.get('contextual_adjustment_factor', 1.0)
            momentum_score *= context_factor
            
            # Normalize to 0-100 range
            momentum_score = max(0.0, min(100.0, momentum_score))
            
            # Determine direction and strength
            if momentum_score > 60:
                direction = 'positive'
                strength = (momentum_score - 60) / 40
            elif momentum_score < 40:
                direction = 'negative'  
                strength = (40 - momentum_score) / 40
            else:
                direction = 'stable'
                strength = 0.0
            
            # Calculate confidence
            detection_confidence = changepoint_analysis.get('detection_confidence', 0.0)
            pattern_confidence = pattern_analysis.get('pattern_strength_score', 0.0) / 100
            confidence = (detection_confidence + pattern_confidence) / 2
            
            return {
                'score': momentum_score,
                'direction': direction,
                'strength': strength,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error assessing current momentum: {str(e)}")
            return {'score': 50.0, 'direction': 'stable', 'strength': 0.0, 'confidence': 0.0}
    
    def _calculate_shift_probabilities(self, changepoint_analysis: Dict, pattern_analysis: Dict,
                                     predictive_analysis: Dict, match_context: Dict) -> Dict:
        """Calculate probabilities of momentum shifts in upcoming matches"""
        try:
            # Base probability from recent changepoint detection frequency
            recent_changepoints = len(changepoint_analysis.get('detected_changepoints', []))
            total_matches = len(changepoint_analysis.get('performance_series', []))
            
            if total_matches > 0:
                base_probability = min(0.5, recent_changepoints / total_matches)
            else:
                base_probability = 0.1
            
            # Adjust for pattern volatility
            pattern_strength = pattern_analysis.get('pattern_strength_score', 0.0)
            volatility_adjustment = pattern_strength / 200  # Scale down
            
            # Calculate probabilities for different horizons
            horizons = [1, 2, 3, 4, 5]
            probabilities = {}
            
            for horizon in horizons:
                # Probability increases with horizon but with diminishing returns
                horizon_prob = base_probability * (1 + np.log(horizon) * volatility_adjustment)
                horizon_prob = max(0.01, min(0.8, horizon_prob))  # Clamp between 1% and 80%
                probabilities[f'next_{horizon}_matches'] = horizon_prob
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error calculating shift probabilities: {str(e)}")
            return {'next_1_matches': 0.1, 'next_2_matches': 0.15, 'next_3_matches': 0.2,
                   'next_4_matches': 0.25, 'next_5_matches': 0.3}
    
    def _identify_historical_shifts(self, matches: List[Dict], team_id: int) -> List[Dict]:
        """Identify and characterize historical momentum shifts"""
        try:
            performance_series = self._extract_performance_timeseries(matches, team_id)
            if len(performance_series) < 5:
                return []
            
            # Simple momentum shift detection based on performance changes
            shifts = []
            window_size = 3
            
            for i in range(window_size, len(performance_series) - window_size):
                before_window = performance_series[i-window_size:i]
                after_window = performance_series[i:i+window_size]
                
                before_avg = np.mean(before_window)
                after_avg = np.mean(after_window)
                
                change_magnitude = abs(after_avg - before_avg)
                
                # Significant shift threshold
                if change_magnitude > 0.3:  # 30% change in performance
                    shift_type = 'positive' if after_avg > before_avg else 'negative'
                    shifts.append({
                        'match_index': i,
                        'shift_type': shift_type,
                        'magnitude': change_magnitude,
                        'before_performance': before_avg,
                        'after_performance': after_avg,
                        'confidence': min(1.0, change_magnitude / 0.5)
                    })
            
            return shifts[-10:]  # Return last 10 significant shifts
            
        except Exception as e:
            logger.error(f"Error identifying historical shifts: {str(e)}")
            return []
    
    def _update_real_time_tracking(self, team_id: int, momentum_analysis: Dict):
        """Update real-time momentum tracking for the team"""
        try:
            current_time = datetime.now()
            
            self.real_time_momentum[team_id] = {
                'last_updated': current_time.isoformat(),
                'current_score': momentum_analysis['current_momentum_score'],
                'direction': momentum_analysis['momentum_direction'],
                'trend_strength': momentum_analysis['trend_strength'],
                'confidence': momentum_analysis['confidence_level']
            }
            
            # Keep trajectory history
            if team_id not in self.momentum_trajectories:
                self.momentum_trajectories[team_id] = deque(maxlen=50)
            
            self.momentum_trajectories[team_id].append({
                'timestamp': current_time.isoformat(),
                'momentum_score': momentum_analysis['current_momentum_score'],
                'direction': momentum_analysis['momentum_direction']
            })
            
        except Exception as e:
            logger.error(f"Error updating real-time tracking: {str(e)}")
    
    # Advanced Pattern Recognition Implementation
    def _analyze_streaks_advanced(self, performance_data: Dict) -> Dict:
        """
        Advanced streak analysis with momentum impact assessment
        
        Analyzes winning/losing streaks, unbeaten runs, and their impact on momentum
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 3:
                return {'current_streak': {'type': 'none', 'length': 0}}
            
            # Convert performance scores to results (W/D/L)
            results = []
            for score in performance_scores:
                if score > 0.75:
                    results.append('W')
                elif score > 0.25:
                    results.append('D')
                else:
                    results.append('L')
            
            # Analyze current streak
            current_streak = self._get_current_streak_advanced(results)
            
            # Analyze historical streaks
            all_streaks = self._get_all_streaks(results)
            
            # Calculate streak momentum impact
            streak_momentum_impact = self._calculate_streak_momentum_impact(current_streak, all_streaks)
            
            # Analyze unbeaten/winless runs
            unbeaten_analysis = self._analyze_unbeaten_runs(results)
            
            # Streak stability analysis
            streak_stability = self._analyze_streak_stability(all_streaks)
            
            # Predict streak continuation probability
            continuation_probability = self._predict_streak_continuation(current_streak, all_streaks)
            
            return {
                'current_streak': current_streak,
                'streak_momentum_impact': streak_momentum_impact,
                'historical_streaks': {
                    'longest_winning': max([s['length'] for s in all_streaks if s['type'] == 'W'] + [0]),
                    'longest_losing': max([s['length'] for s in all_streaks if s['type'] == 'L'] + [0]),
                    'longest_unbeaten': unbeaten_analysis['longest_unbeaten'],
                    'current_unbeaten': unbeaten_analysis['current_unbeaten']
                },
                'streak_patterns': all_streaks[-10:],  # Last 10 streaks
                'streak_stability_score': streak_stability,
                'continuation_probability': continuation_probability,
                'momentum_boost_factor': self._calculate_momentum_boost_factor(current_streak)
            }
            
        except Exception as e:
            logger.error(f"Error in advanced streak analysis: {str(e)}")
            return {'current_streak': {'type': 'none', 'length': 0}}
    
    def _detect_performance_inflections(self, performance_data: Dict) -> List[Dict]:
        """
        Detect performance inflection points (local maxima/minima)
        
        Identifies points where momentum direction changes significantly
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 5:
                return []
            
            scores = np.array(performance_scores)
            inflection_points = []
            
            # Use scipy to find peaks and valleys
            peaks, peak_properties = signal.find_peaks(scores, height=0.6, distance=2)
            valleys, valley_properties = signal.find_peaks(-scores, height=-0.4, distance=2)
            
            # Process peaks (performance highs)
            for peak_idx in peaks:
                if 2 <= peak_idx <= len(scores) - 3:  # Ensure we have context
                    inflection_points.append({
                        'index': int(peak_idx),
                        'type': 'peak',
                        'value': float(scores[peak_idx]),
                        'prominence': self._calculate_prominence(scores, peak_idx),
                        'context_before': float(np.mean(scores[max(0, peak_idx-3):peak_idx])),
                        'context_after': float(np.mean(scores[peak_idx+1:min(len(scores), peak_idx+4)])),
                        'momentum_shift_strength': self._calculate_momentum_shift_strength(scores, peak_idx)
                    })
            
            # Process valleys (performance lows)
            for valley_idx in valleys:
                if 2 <= valley_idx <= len(scores) - 3:
                    inflection_points.append({
                        'index': int(valley_idx),
                        'type': 'valley',
                        'value': float(scores[valley_idx]),
                        'prominence': self._calculate_prominence(-scores, valley_idx),
                        'context_before': float(np.mean(scores[max(0, valley_idx-3):valley_idx])),
                        'context_after': float(np.mean(scores[valley_idx+1:min(len(scores), valley_idx+4)])),
                        'momentum_shift_strength': self._calculate_momentum_shift_strength(scores, valley_idx)
                    })
            
            # Sort by index (chronological order)
            inflection_points.sort(key=lambda x: x['index'])
            
            # Add turning point classification
            for point in inflection_points:
                point['turning_point_strength'] = self._classify_turning_point_strength(point)
                point['recovery_pattern'] = self._analyze_recovery_pattern(scores, point)
            
            return inflection_points
            
        except Exception as e:
            logger.error(f"Error detecting performance inflections: {str(e)}")
            return []
    
    def _analyze_confidence_shifts(self, performance_data: Dict, matches: List[Dict], team_id: int) -> List[Dict]:
        """
        Analyze confidence level shifts based on performance patterns
        
        Detects psychological momentum changes reflected in performance consistency
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 6:
                return []
            
            confidence_shifts = []
            window_size = 3
            
            # Calculate rolling variance (confidence indicator)
            rolling_variance = []
            rolling_mean = []
            
            for i in range(len(performance_scores) - window_size + 1):
                window = performance_scores[i:i + window_size]
                rolling_variance.append(np.var(window))
                rolling_mean.append(np.mean(window))
            
            # Detect significant variance changes (confidence shifts)
            variance_changes = np.diff(rolling_variance)
            mean_changes = np.diff(rolling_mean)
            
            for i, (var_change, mean_change) in enumerate(zip(variance_changes, mean_changes)):
                # Significant confidence shift criteria
                if abs(var_change) > np.std(variance_changes) * 1.5:  # 1.5 sigma threshold
                    shift_type = self._classify_confidence_shift(var_change, mean_change)
                    
                    # Get match context if available
                    match_context = self._get_match_context(matches, i + window_size, team_id)
                    
                    confidence_shifts.append({
                        'index': i + window_size,
                        'shift_type': shift_type,
                        'variance_change': float(var_change),
                        'performance_change': float(mean_change),
                        'confidence_level_before': self._calculate_confidence_level(rolling_variance[i]),
                        'confidence_level_after': self._calculate_confidence_level(rolling_variance[i + 1]),
                        'shift_magnitude': float(abs(var_change)),
                        'match_context': match_context,
                        'psychological_impact': self._assess_psychological_impact(var_change, mean_change)
                    })
            
            # Filter significant shifts only
            significant_shifts = [shift for shift in confidence_shifts 
                                if shift['shift_magnitude'] > np.percentile([s['shift_magnitude'] for s in confidence_shifts], 60)]
            
            return significant_shifts[-8:]  # Return last 8 significant shifts
            
        except Exception as e:
            logger.error(f"Error analyzing confidence shifts: {str(e)}")
            return []
    
    def _detect_team_dynamics_changes(self, performance_data: Dict, matches: List[Dict], team_id: int) -> List[Dict]:
        """
        Detect changes in team dynamics based on performance patterns and external factors
        
        Identifies systemic changes in team behavior and cohesion
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 8:
                return []
            
            dynamics_changes = []
            
            # 1. Formation/Tactical Changes Detection
            tactical_changes = self._detect_tactical_changes(performance_scores, matches, team_id)
            
            # 2. Cohesion Index Changes
            cohesion_changes = self._detect_cohesion_changes(performance_scores)
            
            # 3. Performance Consistency Changes
            consistency_changes = self._detect_consistency_changes(performance_scores)
            
            # 4. Pressure Response Changes
            pressure_response_changes = self._detect_pressure_response_changes(performance_scores, matches, team_id)
            
            # Combine all dynamics changes
            all_changes = (tactical_changes + cohesion_changes + 
                          consistency_changes + pressure_response_changes)
            
            # Sort by significance and recency
            all_changes.sort(key=lambda x: (x['significance'], -x['index']), reverse=True)
            
            return all_changes[:6]  # Return top 6 most significant changes
            
        except Exception as e:
            logger.error(f"Error detecting team dynamics changes: {str(e)}")
            return []
    
    def _classify_momentum_shift_types(self, performance_data: Dict) -> Dict:
        """
        Classify momentum shifts as gradual or sudden based on rate of change
        
        Analyzes the temporal characteristics of performance changes
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 5:
                return {'gradual': 0, 'sudden': 0}
            
            scores = np.array(performance_scores)
            
            # Calculate first and second derivatives
            first_derivative = np.diff(scores)
            second_derivative = np.diff(first_derivative)
            
            gradual_shifts = 0
            sudden_shifts = 0
            
            # Analyze each significant change
            for i, change in enumerate(first_derivative):
                if abs(change) > 0.3:  # Significant change threshold
                    # Check if change is sudden (high second derivative) or gradual
                    if i < len(second_derivative):
                        acceleration = abs(second_derivative[i])
                        
                        if acceleration > 0.2:  # High acceleration = sudden shift
                            sudden_shifts += 1
                        else:  # Low acceleration = gradual shift
                            gradual_shifts += 1
            
            # Calculate shift characteristics
            total_shifts = gradual_shifts + sudden_shifts
            if total_shifts > 0:
                gradual_ratio = gradual_shifts / total_shifts
                sudden_ratio = sudden_shifts / total_shifts
            else:
                gradual_ratio = sudden_ratio = 0.0
            
            # Analyze shift patterns
            shift_patterns = self._analyze_shift_patterns(first_derivative, second_derivative)
            
            return {
                'gradual': gradual_shifts,
                'sudden': sudden_shifts,
                'gradual_ratio': gradual_ratio,
                'sudden_ratio': sudden_ratio,
                'total_significant_shifts': total_shifts,
                'shift_patterns': shift_patterns,
                'average_shift_magnitude': float(np.mean(np.abs(first_derivative))) if len(first_derivative) > 0 else 0.0,
                'shift_volatility': float(np.std(first_derivative)) if len(first_derivative) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error classifying momentum shift types: {str(e)}")
            return {'gradual': 0, 'sudden': 0}
    
    def _cluster_momentum_patterns(self, performance_data: Dict) -> List[Dict]:
        """
        Cluster similar momentum patterns using machine learning
        
        Groups similar performance trajectories for pattern recognition
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 10:
                return []
            
            # Create feature vectors for clustering
            window_size = 5
            feature_vectors = []
            
            for i in range(len(performance_scores) - window_size + 1):
                window = performance_scores[i:i + window_size]
                
                # Feature extraction from window
                features = [
                    np.mean(window),           # Average performance
                    np.std(window),            # Variability
                    np.max(window) - np.min(window),  # Range
                    np.sum(np.diff(window) > 0),      # Upward moves
                    np.sum(np.diff(window) < 0),      # Downward moves
                    window[-1] - window[0],    # Overall change
                    np.mean(np.diff(window))   # Average change rate
                ]
                
                feature_vectors.append(features)
                
            if len(feature_vectors) < 3:
                return []
            
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_vectors)
            
            # Determine optimal number of clusters
            max_clusters = min(5, len(feature_vectors) // 2)
            if max_clusters < 2:
                return []
            
            best_k = 2
            best_score = -1
            
            for k in range(2, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(normalized_features)
                    score = silhouette_score(normalized_features, cluster_labels)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                except:
                    continue
            
            # Final clustering with best k
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_features)
            
            # Analyze clusters
            clusters = []
            for cluster_id in range(best_k):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_features = [feature_vectors[i] for i in cluster_indices]
                
                if cluster_features:
                    cluster_analysis = self._analyze_cluster_characteristics(cluster_features, cluster_indices)
                    clusters.append({
                        'cluster_id': cluster_id,
                        'size': len(cluster_indices),
                        'characteristics': cluster_analysis,
                        'representative_windows': cluster_indices.tolist(),
                        'centroid': kmeans.cluster_centers_[cluster_id].tolist()
                    })
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering momentum patterns: {str(e)}")
            return []
    
    def _detect_recurring_patterns(self, performance_data: Dict, team_id: int) -> List[Dict]:
        """
        Detect recurring momentum patterns in team performance
        
        Identifies cyclical or repeated momentum behaviors
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 12:
                return []
            
            recurring_patterns = []
            
            # 1. Seasonal patterns (if enough data)
            if len(performance_scores) >= 20:
                seasonal_patterns = self._detect_seasonal_patterns(performance_scores)
                recurring_patterns.extend(seasonal_patterns)
            
            # 2. Cyclical patterns (repeating sequences)
            cyclical_patterns = self._detect_cyclical_patterns(performance_scores)
            recurring_patterns.extend(cyclical_patterns)
            
            # 3. Response patterns (consistent reactions to situations)
            response_patterns = self._detect_response_patterns(performance_scores)
            recurring_patterns.extend(response_patterns)
            
            # 4. Recovery patterns (consistent recovery behaviors)
            recovery_patterns = self._detect_recovery_patterns(performance_scores)
            recurring_patterns.extend(recovery_patterns)
            
            # Filter and rank patterns by strength and frequency
            significant_patterns = [p for p in recurring_patterns if p['confidence'] > 0.6]
            significant_patterns.sort(key=lambda x: x['strength'], reverse=True)
            
            return significant_patterns[:5]  # Return top 5 patterns
            
        except Exception as e:
            logger.error(f"Error detecting recurring patterns: {str(e)}")
            return []
    
    def _calculate_pattern_strength(self, streak_analysis: Dict, inflection_analysis: List[Dict], 
                                  confidence_analysis: List[Dict]) -> float:
        """
        Calculate overall pattern strength score (0-100)
        
        Combines multiple pattern indicators into a single strength measure
        """
        try:
            strength_components = []
            
            # Streak strength component
            current_streak = streak_analysis.get('current_streak', {})
            streak_length = current_streak.get('length', 0)
            streak_momentum = streak_analysis.get('streak_momentum_impact', 0.0)
            streak_strength = min(30.0, streak_length * 5 + streak_momentum * 10)
            strength_components.append(streak_strength)
            
            # Inflection point strength component
            recent_inflections = [inf for inf in inflection_analysis if inf.get('index', 0) >= len(inflection_analysis) - 5]
            if recent_inflections:
                inflection_strength = sum(inf.get('momentum_shift_strength', 0.0) for inf in recent_inflections)
                inflection_strength = min(25.0, inflection_strength * 5)
            else:
                inflection_strength = 0.0
            strength_components.append(inflection_strength)
            
            # Confidence shift strength component
            recent_confidence_shifts = [conf for conf in confidence_analysis if conf.get('index', 0) >= len(confidence_analysis) - 5]
            if recent_confidence_shifts:
                confidence_strength = sum(conf.get('shift_magnitude', 0.0) for conf in recent_confidence_shifts)
                confidence_strength = min(20.0, confidence_strength * 10)
            else:
                confidence_strength = 0.0
            strength_components.append(confidence_strength)
            
            # Consistency component
            streak_stability = streak_analysis.get('streak_stability_score', 0.5)
            consistency_strength = streak_stability * 15.0
            strength_components.append(consistency_strength)
            
            # Momentum boost factor
            momentum_boost = streak_analysis.get('momentum_boost_factor', 1.0)
            boost_strength = (momentum_boost - 1.0) * 10.0
            strength_components.append(boost_strength)
            
            # Calculate weighted average
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Prioritize streak and inflections
            total_strength = sum(comp * weight for comp, weight in zip(strength_components, weights))
            
            # Normalize to 0-100 scale
            return max(0.0, min(100.0, total_strength))
            
        except Exception as e:
            logger.error(f"Error calculating pattern strength: {str(e)}")
            return 0.0
    
    def _predict_momentum_trajectory(self, performance_data: Dict, match_context: Dict) -> Dict:
        """
        Predict future momentum trajectory using multiple forecasting methods
        
        Combines autoregressive models, exponential smoothing, and pattern-based prediction
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 5:
                return {'predicted_scores': [50.0] * 5, 'confidence': 0.1}
            
            scores = np.array(performance_scores)
            prediction_horizon = self.config['predictive_modeling']['prediction_horizon']
            
            # 1. Autoregressive prediction (AR model)
            ar_predictions = self._autoregressive_prediction(scores, prediction_horizon)
            
            # 2. Exponential smoothing prediction
            exponential_predictions = self._exponential_smoothing_prediction(scores, prediction_horizon)
            
            # 3. Pattern-based prediction
            pattern_predictions = self._pattern_based_prediction(scores, prediction_horizon)
            
            # 4. Trend-based prediction
            trend_predictions = self._trend_based_prediction(scores, prediction_horizon)
            
            # Combine predictions with weights
            weights = [0.3, 0.25, 0.25, 0.2]  # AR, Exponential, Pattern, Trend
            combined_predictions = []
            
            for i in range(prediction_horizon):
                prediction = (
                    ar_predictions[i] * weights[0] +
                    exponential_predictions[i] * weights[1] +
                    pattern_predictions[i] * weights[2] +
                    trend_predictions[i] * weights[3]
                )
                combined_predictions.append(max(0.0, min(1.0, prediction)))
            
            # Convert to 0-100 scale
            trajectory_scores = [p * 100 for p in combined_predictions]
            
            # Calculate trajectory characteristics
            trajectory_trend = self._analyze_trajectory_trend(combined_predictions)
            trajectory_volatility = np.std(combined_predictions)
            trajectory_confidence = self._calculate_trajectory_confidence(
                ar_predictions, exponential_predictions, pattern_predictions, trend_predictions
            )
            
            return {
                'predicted_scores': trajectory_scores,
                'trajectory_trend': trajectory_trend,
                'trajectory_volatility': float(trajectory_volatility),
                'confidence': trajectory_confidence,
                'method_predictions': {
                    'autoregressive': [p * 100 for p in ar_predictions],
                    'exponential_smoothing': [p * 100 for p in exponential_predictions],
                    'pattern_based': [p * 100 for p in pattern_predictions],
                    'trend_based': [p * 100 for p in trend_predictions]
                },
                'prediction_metadata': {
                    'input_length': len(scores),
                    'horizon': prediction_horizon,
                    'methods_used': 4
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting momentum trajectory: {str(e)}")
            return {'predicted_scores': [50.0] * 5, 'confidence': 0.1}
    
    def _assess_momentum_sustainability(self, performance_data: Dict) -> Dict:
        """
        Assess sustainability of current momentum based on historical patterns
        
        Analyzes momentum decay patterns and strength indicators
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 8:
                return {'current_sustainability': 0.5, 'sustainability_factors': {}}
            
            scores = np.array(performance_scores)
            current_momentum = scores[-5:] if len(scores) >= 5 else scores
            
            # 1. Momentum strength analysis
            momentum_strength = self._calculate_momentum_strength(current_momentum)
            
            # 2. Historical sustainability analysis
            historical_sustainability = self._analyze_historical_sustainability(scores)
            
            # 3. Volatility-based sustainability
            volatility_sustainability = self._assess_volatility_sustainability(current_momentum)
            
            # 4. Trend consistency analysis
            trend_consistency = self._analyze_trend_consistency(scores)
            
            # 5. Peak/valley distance analysis
            peak_valley_sustainability = self._assess_peak_valley_sustainability(scores)
            
            # Combine sustainability factors
            sustainability_factors = {
                'momentum_strength': momentum_strength,
                'historical_pattern': historical_sustainability,
                'volatility_stability': volatility_sustainability,
                'trend_consistency': trend_consistency,
                'peak_valley_balance': peak_valley_sustainability
            }
            
            # Weighted average sustainability score
            weights = [0.25, 0.2, 0.2, 0.2, 0.15]
            overall_sustainability = sum(
                factor * weight for factor, weight in zip(sustainability_factors.values(), weights)
            )
            
            # Sustainability classification
            if overall_sustainability > 0.75:
                sustainability_class = 'highly_sustainable'
            elif overall_sustainability > 0.6:
                sustainability_class = 'moderately_sustainable'
            elif overall_sustainability > 0.4:
                sustainability_class = 'weakly_sustainable'
            else:
                sustainability_class = 'unsustainable'
            
            # Predict sustainability duration
            sustainability_duration = self._predict_sustainability_duration(
                current_momentum, historical_sustainability, momentum_strength
            )
            
            return {
                'current_sustainability': overall_sustainability,
                'sustainability_class': sustainability_class,
                'sustainability_factors': sustainability_factors,
                'predicted_duration_matches': sustainability_duration,
                'sustainability_confidence': self._calculate_sustainability_confidence(sustainability_factors),
                'risk_factors': self._identify_sustainability_risk_factors(sustainability_factors)
            }
            
        except Exception as e:
            logger.error(f"Error assessing momentum sustainability: {str(e)}")
            return {'current_sustainability': 0.5}
    
    def _estimate_recovery_time(self, performance_data: Dict) -> Dict:
        """
        Estimate recovery time after negative momentum phases
        
        Analyzes historical recovery patterns and current momentum state
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 10:
                return {'estimated_recovery_matches': 5, 'confidence': 0.1}
            
            scores = np.array(performance_scores)
            
            # 1. Identify current momentum state
            current_state = self._identify_momentum_state(scores[-5:])
            
            # 2. Find historical recovery patterns
            historical_recoveries = self._find_historical_recovery_patterns(scores)
            
            # 3. Analyze recovery speed factors
            recovery_factors = self._analyze_recovery_factors(scores, current_state)
            
            # 4. Calculate base recovery time
            if current_state['type'] == 'negative':
                base_recovery_time = self._calculate_base_recovery_time(
                    current_state['severity'], historical_recoveries
                )
            else:
                # Not in negative momentum, no recovery needed
                return {
                    'estimated_recovery_matches': 0,
                    'current_state': 'no_recovery_needed',
                    'confidence': 1.0
                }
            
            # 5. Adjust for current factors
            adjusted_recovery_time = self._adjust_recovery_time(
                base_recovery_time, recovery_factors, current_state
            )
            
            # 6. Calculate recovery confidence
            recovery_confidence = self._calculate_recovery_confidence(
                historical_recoveries, recovery_factors, current_state
            )
            
            # 7. Identify recovery accelerators and inhibitors
            recovery_accelerators = self._identify_recovery_accelerators(recovery_factors)
            recovery_inhibitors = self._identify_recovery_inhibitors(recovery_factors)
            
            return {
                'estimated_recovery_matches': int(round(adjusted_recovery_time)),
                'recovery_confidence': recovery_confidence,
                'current_momentum_state': current_state,
                'recovery_factors': recovery_factors,
                'historical_recovery_data': {
                    'average_recovery_time': np.mean([r['duration'] for r in historical_recoveries]) if historical_recoveries else 5.0,
                    'fastest_recovery': min([r['duration'] for r in historical_recoveries]) if historical_recoveries else 1,
                    'slowest_recovery': max([r['duration'] for r in historical_recoveries]) if historical_recoveries else 10,
                    'recovery_success_rate': len([r for r in historical_recoveries if r['successful']]) / len(historical_recoveries) if historical_recoveries else 0.5
                },
                'recovery_accelerators': recovery_accelerators,
                'recovery_inhibitors': recovery_inhibitors
            }
            
        except Exception as e:
            logger.error(f"Error estimating recovery time: {str(e)}")
            return {'estimated_recovery_matches': 5}
    
    def _predict_peak_performance_windows(self, performance_data: Dict, match_context: Dict) -> Dict:
        """
        Predict upcoming peak performance windows
        
        Identifies optimal timing for peak performance based on patterns
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 12:
                return {'next_peak_probability': 0.2, 'confidence': 0.1}
            
            scores = np.array(performance_scores)
            
            # 1. Identify historical peak patterns
            historical_peaks = self._identify_historical_peaks(scores)
            
            # 2. Analyze peak cycles and timing
            peak_cycles = self._analyze_peak_cycles(historical_peaks, scores)
            
            # 3. Calculate current cycle position
            cycle_position = self._calculate_current_cycle_position(scores, peak_cycles)
            
            # 4. Predict next peak timing
            next_peak_predictions = self._predict_next_peak_timing(peak_cycles, cycle_position)
            
            # 5. Assess peak readiness factors
            peak_readiness = self._assess_peak_readiness_factors(scores, match_context)
            
            # 6. Calculate peak probabilities for next 5 matches
            peak_probabilities = []
            for i in range(1, 6):
                probability = self._calculate_peak_probability(
                    i, next_peak_predictions, cycle_position, peak_readiness
                )
                peak_probabilities.append(probability)
            
            # 7. Identify optimal peak window
            optimal_window = self._identify_optimal_peak_window(peak_probabilities, peak_readiness)
            
            return {
                'next_peak_probability': max(peak_probabilities) if peak_probabilities else 0.2,
                'peak_probabilities_by_match': peak_probabilities,
                'optimal_peak_window': optimal_window,
                'peak_readiness_score': peak_readiness['overall_score'],
                'cycle_analysis': {
                    'current_cycle_position': cycle_position,
                    'historical_cycle_length': peak_cycles.get('average_cycle_length', 8),
                    'time_since_last_peak': peak_cycles.get('time_since_last_peak', 5)
                },
                'peak_factors': peak_readiness,
                'confidence': self._calculate_peak_prediction_confidence(historical_peaks, peak_cycles)
            }
            
        except Exception as e:
            logger.error(f"Error predicting peak performance windows: {str(e)}")
            return {'next_peak_probability': 0.2}
    
    def _model_momentum_decay_rate(self, performance_data: Dict) -> Dict:
        """
        Model momentum decay rate using exponential and polynomial models
        
        Analyzes how momentum decays over time for different momentum types
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 10:
                return {'decay_rate': 0.9, 'model_type': 'default'}
            
            scores = np.array(performance_scores)
            
            # 1. Identify momentum peaks and decays
            momentum_phases = self._identify_momentum_phases(scores)
            
            # 2. Model different types of decay
            decay_models = {}
            
            # Positive momentum decay
            positive_phases = [p for p in momentum_phases if p['type'] == 'positive']
            if positive_phases:
                decay_models['positive'] = self._model_phase_decay(positive_phases, scores)
            
            # Negative momentum decay (recovery)
            negative_phases = [p for p in momentum_phases if p['type'] == 'negative']
            if negative_phases:
                decay_models['negative'] = self._model_phase_decay(negative_phases, scores)
            
            # Neutral momentum decay
            neutral_phases = [p for p in momentum_phases if p['type'] == 'neutral']
            if neutral_phases:
                decay_models['neutral'] = self._model_phase_decay(neutral_phases, scores)
            
            # 3. Determine overall decay characteristics
            overall_decay_rate = self._calculate_overall_decay_rate(decay_models)
            dominant_decay_pattern = self._identify_dominant_decay_pattern(decay_models)
            
            # 4. Model momentum half-life
            momentum_half_life = self._calculate_momentum_half_life(decay_models)
            
            # 5. Predict decay for current momentum
            current_momentum = scores[-3:] if len(scores) >= 3 else scores
            current_decay_prediction = self._predict_current_decay(
                current_momentum, decay_models, overall_decay_rate
            )
            
            return {
                'decay_rate': overall_decay_rate,
                'decay_models': decay_models,
                'dominant_pattern': dominant_decay_pattern,
                'momentum_half_life': momentum_half_life,
                'current_decay_prediction': current_decay_prediction,
                'decay_consistency': self._calculate_decay_consistency(decay_models),
                'model_reliability': self._assess_decay_model_reliability(momentum_phases)
            }
            
        except Exception as e:
            logger.error(f"Error modeling momentum decay rate: {str(e)}")
            return {'decay_rate': 0.9}
    
    def _calculate_prediction_intervals(self, trajectory: Dict, sustainability: Dict, recovery: Dict) -> Dict:
        """
        Calculate prediction confidence intervals using uncertainty quantification
        
        Provides uncertainty bounds for momentum predictions
        """
        try:
            predicted_scores = trajectory.get('predicted_scores', [50.0] * 5)
            trajectory_confidence = trajectory.get('confidence', 0.5)
            sustainability_confidence = sustainability.get('sustainability_confidence', 0.5)
            recovery_confidence = recovery.get('recovery_confidence', 0.5)
            
            # Calculate overall prediction confidence
            overall_confidence = (trajectory_confidence + sustainability_confidence + recovery_confidence) / 3.0
            
            # Calculate uncertainty based on multiple factors
            uncertainties = []
            for i, score in enumerate(predicted_scores):
                # Base uncertainty increases with prediction horizon
                horizon_uncertainty = 5.0 + (i * 2.0)
                
                # Confidence-based uncertainty
                confidence_uncertainty = (1.0 - overall_confidence) * 15.0
                
                # Score-based uncertainty (more uncertain at extremes)
                score_uncertainty = 0.0
                if score > 80:
                    score_uncertainty = (score - 80) * 0.2
                elif score < 20:
                    score_uncertainty = (20 - score) * 0.2
                
                total_uncertainty = horizon_uncertainty + confidence_uncertainty + score_uncertainty
                uncertainties.append(total_uncertainty)
            
            # Calculate confidence intervals (95% confidence)
            confidence_level = 0.95
            z_score = 1.96  # 95% confidence interval
            
            lower_bounds = []
            upper_bounds = []
            
            for score, uncertainty in zip(predicted_scores, uncertainties):
                margin = z_score * uncertainty
                lower_bound = max(0.0, score - margin)
                upper_bound = min(100.0, score + margin)
                
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)
            
            # Calculate interval widths
            interval_widths = [upper - lower for upper, lower in zip(upper_bounds, lower_bounds)]
            
            return {
                'confidence_level': confidence_level,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'interval_widths': interval_widths,
                'overall_confidence': overall_confidence,
                'uncertainty_sources': {
                    'trajectory_uncertainty': 1.0 - trajectory_confidence,
                    'sustainability_uncertainty': 1.0 - sustainability_confidence,
                    'recovery_uncertainty': 1.0 - recovery_confidence,
                    'horizon_uncertainty': 'increases_with_time'
                },
                'prediction_reliability': self._assess_prediction_reliability(
                    overall_confidence, interval_widths
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating prediction intervals: {str(e)}")
            return {'lower': [40.0] * 5, 'upper': [60.0] * 5}
    
    def _calculate_prediction_accuracy_score(self, performance_data: Dict) -> float:
        """
        Calculate prediction accuracy score based on historical performance
        
        Uses backtesting to evaluate prediction model accuracy
        """
        try:
            performance_scores = performance_data.get('performance_scores', [])
            if len(performance_scores) < 15:
                return 0.0
            
            scores = np.array(performance_scores)
            prediction_errors = []
            
            # Backtest predictions on historical data
            for i in range(10, len(scores) - 5):  # Leave room for predictions
                # Use data up to point i to predict next 5 points
                historical_data = scores[:i]
                actual_future = scores[i:i+5]
                
                # Make prediction using simplified version of trajectory model
                predicted_future = self._backtest_trajectory_prediction(historical_data)
                
                # Calculate prediction errors
                if len(predicted_future) == len(actual_future):
                    errors = np.abs(predicted_future - actual_future)
                    prediction_errors.extend(errors)
            
            if not prediction_errors:
                return 0.0
            
            # Calculate accuracy metrics
            mean_error = np.mean(prediction_errors)
            max_error = np.max(prediction_errors)
            error_std = np.std(prediction_errors)
            
            # Convert to accuracy score (0-1)
            # Lower error = higher accuracy
            accuracy_score = 1.0 / (1.0 + mean_error / 20.0)  # Normalize by expected max error
            
            # Penalty for high variability in errors
            consistency_penalty = error_std / 100.0
            accuracy_score = max(0.0, accuracy_score - consistency_penalty)
            
            return min(1.0, accuracy_score)
            
        except Exception as e:
            logger.error(f"Error calculating prediction accuracy score: {str(e)}")
            return 0.0
    
    # Context analysis placeholder methods
    def _analyze_league_momentum_patterns(self, matches: List[Dict], team_id: int, league_id: int) -> Dict:
        """Analyze league-specific momentum patterns"""
        return {'adjustment_factor': 1.0}
    
    def _analyze_venue_momentum_effects(self, matches: List[Dict], team_id: int, venue: str) -> Dict:
        """Analyze venue-specific momentum effects"""
        return {'home_advantage': 0.0, 'away_challenge': 0.0}
    
    def _analyze_manager_change_effects(self, matches: List[Dict], team_id: int, match_context: Dict) -> Dict:
        """Analyze manager change effects on momentum"""
        return {'recent_change': False, 'effect_strength': 0.0}
    
    def _analyze_transfer_window_impacts(self, matches: List[Dict], team_id: int, match_context: Dict) -> Dict:
        """Analyze transfer window impacts on momentum"""
        return {'recent_transfers': False, 'impact_strength': 0.0}
    
    def _analyze_pressure_match_effects(self, matches: List[Dict], team_id: int, match_context: Dict) -> Dict:
        """Analyze pressure match effects on momentum"""
        return {'is_pressure_match': False, 'pressure_multiplier': 1.0}
    
    def _analyze_opponent_specific_patterns(self, matches: List[Dict], team_id: int, opponent_id: int) -> Dict:
        """Analyze opponent-specific momentum patterns"""
        return {'historical_performance': 0.5}
    
    def _analyze_seasonal_momentum_effects(self, matches: List[Dict], team_id: int, match_context: Dict) -> Dict:
        """Analyze seasonal momentum effects"""
        return {'current_season_effect': 0.0}
    
    def _calculate_contextual_adjustment_factor(self, league_analysis: Dict, venue_analysis: Dict,
                                              manager_analysis: Dict, pressure_analysis: Dict) -> float:
        """Calculate overall contextual adjustment factor"""
        try:
            factors = [
                league_analysis.get('adjustment_factor', 1.0),
                1.0 + venue_analysis.get('home_advantage', 0.0) - venue_analysis.get('away_challenge', 0.0),
                1.0 + manager_analysis.get('effect_strength', 0.0),
                pressure_analysis.get('pressure_multiplier', 1.0)
            ]
            
            # Geometric mean for combining multiplicative factors
            adjustment_factor = np.prod(factors) ** (1.0 / len(factors))
            return max(0.5, min(2.0, adjustment_factor))  # Clamp to reasonable range
            
        except Exception as e:
            logger.error(f"Error calculating contextual adjustment factor: {str(e)}")
            return 1.0
    
    # Helper Methods for Pattern Recognition
    
    def _get_current_streak_advanced(self, results: List[str]) -> Dict:
        """Get detailed current streak information"""
        try:
            if not results:
                return {'type': 'none', 'length': 0}
            
            current_result = results[-1]
            streak_length = 1
            
            # Count consecutive results of same type
            for i in range(len(results) - 2, -1, -1):
                if results[i] == current_result:
                    streak_length += 1
                else:
                    break
            
            # Calculate streak quality (consistency within streak)
            streak_quality = self._calculate_streak_quality(results, streak_length)
            
            return {
                'type': current_result,
                'length': streak_length,
                'quality': streak_quality,
                'significance': self._calculate_streak_significance(current_result, streak_length),
                'momentum_value': self._calculate_streak_momentum_value(current_result, streak_length, streak_quality)
            }
            
        except Exception as e:
            logger.error(f"Error getting current streak: {str(e)}")
            return {'type': 'none', 'length': 0}
    
    def _get_all_streaks(self, results: List[str]) -> List[Dict]:
        """Get all streaks in the results history"""
        try:
            if not results:
                return []
            
            streaks = []
            current_type = results[0]
            current_length = 1
            start_index = 0
            
            for i in range(1, len(results)):
                if results[i] == current_type:
                    current_length += 1
                else:
                    # End of streak
                    if current_length >= 2:  # Only count streaks of 2+
                        streaks.append({
                            'type': current_type,
                            'length': current_length,
                            'start_index': start_index,
                            'end_index': i - 1,
                            'quality': self._calculate_streak_quality(results[start_index:i], current_length)
                        })
                    
                    # Start new streak
                    current_type = results[i]
                    current_length = 1
                    start_index = i
            
            # Handle final streak
            if current_length >= 2:
                streaks.append({
                    'type': current_type,
                    'length': current_length,
                    'start_index': start_index,
                    'end_index': len(results) - 1,
                    'quality': self._calculate_streak_quality(results[start_index:], current_length)
                })
            
            return streaks
            
        except Exception as e:
            logger.error(f"Error getting all streaks: {str(e)}")
            return []
    
    def _calculate_streak_momentum_impact(self, current_streak: Dict, all_streaks: List[Dict]) -> float:
        """Calculate momentum impact of current streak"""
        try:
            if not current_streak or current_streak.get('length', 0) == 0:
                return 0.0
            
            streak_type = current_streak.get('type', 'D')
            streak_length = current_streak.get('length', 0)
            streak_quality = current_streak.get('quality', 0.5)
            
            # Base impact based on streak type and length
            if streak_type == 'W':
                base_impact = min(0.8, streak_length * 0.15)  # Positive impact
            elif streak_type == 'L':
                base_impact = -min(0.8, streak_length * 0.15)  # Negative impact
            else:  # Draw
                base_impact = 0.05 * streak_length  # Small positive impact
            
            # Adjust for streak quality
            quality_multiplier = 0.5 + (streak_quality * 0.5)
            
            # Adjust for historical context
            if all_streaks:
                avg_streak_length = np.mean([s['length'] for s in all_streaks])
                if streak_length > avg_streak_length:
                    historical_multiplier = 1.2
                else:
                    historical_multiplier = 0.9
            else:
                historical_multiplier = 1.0
            
            return base_impact * quality_multiplier * historical_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating streak momentum impact: {str(e)}")
            return 0.0
    
    def _analyze_unbeaten_runs(self, results: List[str]) -> Dict:
        """Analyze unbeaten and winless runs"""
        try:
            if not results:
                return {'longest_unbeaten': 0, 'current_unbeaten': 0, 'longest_winless': 0, 'current_winless': 0}
            
            # Current unbeaten run
            current_unbeaten = 0
            for result in reversed(results):
                if result in ['W', 'D']:
                    current_unbeaten += 1
                else:
                    break
            
            # Current winless run
            current_winless = 0
            for result in reversed(results):
                if result in ['D', 'L']:
                    current_winless += 1
                else:
                    break
            
            # Longest unbeaten run
            longest_unbeaten = 0
            current_count = 0
            for result in results:
                if result in ['W', 'D']:
                    current_count += 1
                    longest_unbeaten = max(longest_unbeaten, current_count)
                else:
                    current_count = 0
            
            # Longest winless run
            longest_winless = 0
            current_count = 0
            for result in results:
                if result in ['D', 'L']:
                    current_count += 1
                    longest_winless = max(longest_winless, current_count)
                else:
                    current_count = 0
            
            return {
                'longest_unbeaten': longest_unbeaten,
                'current_unbeaten': current_unbeaten,
                'longest_winless': longest_winless,
                'current_winless': current_winless,
                'unbeaten_momentum': self._calculate_unbeaten_momentum(current_unbeaten, longest_unbeaten),
                'pressure_index': self._calculate_pressure_index(current_winless, longest_winless)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing unbeaten runs: {str(e)}")
            return {'longest_unbeaten': 0, 'current_unbeaten': 0}
    
    def _analyze_streak_stability(self, all_streaks: List[Dict]) -> float:
        """Analyze stability of streak patterns"""
        try:
            if len(all_streaks) < 3:
                return 0.5
            
            # Calculate streak length variance
            streak_lengths = [s['length'] for s in all_streaks]
            length_variance = np.var(streak_lengths)
            length_stability = 1.0 / (1.0 + length_variance)
            
            # Calculate streak type distribution
            streak_types = [s['type'] for s in all_streaks]
            type_counts = {t: streak_types.count(t) for t in set(streak_types)}
            type_distribution = list(type_counts.values())
            type_stability = 1.0 - np.std(type_distribution) / (np.mean(type_distribution) + 1e-6)
            
            # Calculate quality consistency
            qualities = [s.get('quality', 0.5) for s in all_streaks]
            quality_variance = np.var(qualities)
            quality_stability = 1.0 / (1.0 + quality_variance * 2)
            
            # Combined stability score
            stability_score = (length_stability + type_stability + quality_stability) / 3.0
            return max(0.0, min(1.0, stability_score))
            
        except Exception as e:
            logger.error(f"Error analyzing streak stability: {str(e)}")
            return 0.5
    
    def _predict_streak_continuation(self, current_streak: Dict, all_streaks: List[Dict]) -> float:
        """Predict probability of streak continuation"""
        try:
            if not current_streak or current_streak.get('length', 0) == 0:
                return 0.1
            
            streak_type = current_streak.get('type', 'D')
            streak_length = current_streak.get('length', 0)
            
            # Base probability decreases with streak length
            base_probability = 0.8 * (0.9 ** streak_length)
            
            # Adjust based on historical patterns
            if all_streaks:
                similar_streaks = [s for s in all_streaks 
                                 if s['type'] == streak_type and s['length'] >= streak_length]
                
                if similar_streaks:
                    # How often did similar streaks continue?
                    continued_count = sum(1 for s in similar_streaks if s['length'] > streak_length)
                    historical_factor = continued_count / len(similar_streaks)
                    base_probability = (base_probability + historical_factor) / 2.0
            
            # Adjust for streak quality
            streak_quality = current_streak.get('quality', 0.5)
            quality_factor = 0.5 + (streak_quality * 0.5)
            
            return base_probability * quality_factor
            
        except Exception as e:
            logger.error(f"Error predicting streak continuation: {str(e)}")
            return 0.1
    
    def _calculate_momentum_boost_factor(self, current_streak: Dict) -> float:
        """Calculate momentum boost factor from current streak"""
        try:
            if not current_streak:
                return 1.0
            
            streak_type = current_streak.get('type', 'D')
            streak_length = current_streak.get('length', 0)
            streak_quality = current_streak.get('quality', 0.5)
            
            if streak_type == 'W':
                # Winning streak boost
                boost = 1.0 + (streak_length * 0.05) + (streak_quality * 0.1)
                return min(1.5, boost)  # Cap at 1.5x
            elif streak_type == 'L':
                # Losing streak penalty
                penalty = 1.0 - (streak_length * 0.04) - (streak_quality * 0.08)
                return max(0.6, penalty)  # Floor at 0.6x
            else:
                # Draw streak - small boost
                return 1.0 + (streak_length * 0.02)
            
        except Exception as e:
            logger.error(f"Error calculating momentum boost factor: {str(e)}")
            return 1.0
    
    def _calculate_prominence(self, scores: np.ndarray, index: int) -> float:
        """Calculate prominence of a peak or valley"""
        try:
            if index <= 0 or index >= len(scores) - 1:
                return 0.0
            
            peak_value = scores[index]
            
            # Find left base
            left_base = peak_value
            for i in range(index - 1, -1, -1):
                if scores[i] < left_base:
                    left_base = scores[i]
                if scores[i] > peak_value:
                    break
            
            # Find right base
            right_base = peak_value
            for i in range(index + 1, len(scores)):
                if scores[i] < right_base:
                    right_base = scores[i]
                if scores[i] > peak_value:
                    break
            
            # Prominence is difference from highest base
            base = max(left_base, right_base)
            return float(abs(peak_value - base))
            
        except Exception as e:
            logger.error(f"Error calculating prominence: {str(e)}")
            return 0.0
    
    def _calculate_momentum_shift_strength(self, scores: np.ndarray, index: int) -> float:
        """Calculate strength of momentum shift at given index"""
        try:
            if index < 2 or index >= len(scores) - 2:
                return 0.0
            
            # Compare before and after windows
            before_window = scores[max(0, index-2):index]
            after_window = scores[index:min(len(scores), index+3)]
            
            if len(before_window) == 0 or len(after_window) == 0:
                return 0.0
            
            before_avg = np.mean(before_window)
            after_avg = np.mean(after_window)
            
            # Shift strength is normalized difference
            shift_strength = abs(after_avg - before_avg)
            
            # Consider volatility in the calculation
            before_var = np.var(before_window) if len(before_window) > 1 else 0.0
            after_var = np.var(after_window) if len(after_window) > 1 else 0.0
            avg_volatility = (before_var + after_var) / 2.0
            
            # Adjust strength for volatility (more volatile = less significant shift)
            if avg_volatility > 0:
                adjusted_strength = shift_strength / (1.0 + avg_volatility)
            else:
                adjusted_strength = shift_strength
            
            return float(min(1.0, adjusted_strength * 2.0))  # Scale to 0-1
            
        except Exception as e:
            logger.error(f"Error calculating momentum shift strength: {str(e)}")
            return 0.0
    
    def _classify_turning_point_strength(self, point: Dict) -> str:
        """Classify turning point strength based on characteristics"""
        try:
            prominence = point.get('prominence', 0.0)
            shift_strength = point.get('momentum_shift_strength', 0.0)
            
            combined_strength = (prominence + shift_strength) / 2.0
            
            if combined_strength > 0.7:
                return 'strong'
            elif combined_strength > 0.4:
                return 'moderate'
            elif combined_strength > 0.2:
                return 'weak'
            else:
                return 'minimal'
                
        except Exception as e:
            logger.error(f"Error classifying turning point strength: {str(e)}")
            return 'minimal'
    
    def _analyze_recovery_pattern(self, scores: np.ndarray, point: Dict) -> Dict:
        """Analyze recovery pattern after a turning point"""
        try:
            index = point.get('index', 0)
            point_type = point.get('type', 'unknown')
            
            if index >= len(scores) - 3:
                return {'recovery_speed': 'unknown', 'recovery_strength': 0.0}
            
            # Analyze next 3-5 points for recovery pattern
            recovery_window = scores[index:min(len(scores), index + 5)]
            
            if len(recovery_window) < 2:
                return {'recovery_speed': 'unknown', 'recovery_strength': 0.0}
            
            # Calculate recovery metrics
            initial_value = recovery_window[0]
            final_value = recovery_window[-1]
            recovery_change = final_value - initial_value
            
            # Determine expected recovery direction
            if point_type == 'valley':
                expected_direction = 1  # Should recover upward
            else:  # peak
                expected_direction = -1  # Should decline from peak
            
            # Check if recovery is in expected direction
            correct_direction = (recovery_change * expected_direction) > 0
            
            # Calculate recovery speed (how quickly it changes)
            recovery_gradient = np.diff(recovery_window)
            recovery_speed = np.mean(np.abs(recovery_gradient))
            
            # Classify recovery speed
            if recovery_speed > 0.3:
                speed_class = 'fast'
            elif recovery_speed > 0.15:
                speed_class = 'moderate'
            else:
                speed_class = 'slow'
            
            return {
                'recovery_speed': speed_class,
                'recovery_strength': float(abs(recovery_change)),
                'correct_direction': correct_direction,
                'recovery_consistency': float(1.0 - np.std(recovery_gradient)) if len(recovery_gradient) > 0 else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error analyzing recovery pattern: {str(e)}")
            return {'recovery_speed': 'unknown', 'recovery_strength': 0.0}
    
    def _classify_confidence_shift(self, variance_change: float, mean_change: float) -> str:
        """Classify type of confidence shift"""
        try:
            if variance_change > 0:
                if mean_change > 0:
                    return 'volatile_improvement'  # Getting better but inconsistent
                elif mean_change < -0.1:
                    return 'collapse'  # Performance declining with increasing volatility
                else:
                    return 'increased_volatility'  # More inconsistent but similar average
            else:
                if mean_change > 0.1:
                    return 'confident_improvement'  # Getting better and more consistent
                elif mean_change < 0:
                    return 'stable_decline'  # Declining but consistently
                else:
                    return 'stabilization'  # More consistent performance
                    
        except Exception as e:
            logger.error(f"Error classifying confidence shift: {str(e)}")
            return 'unknown'
    
    def _get_match_context(self, matches: List[Dict], index: int, team_id: int) -> Dict:
        """Get context information for a specific match"""
        try:
            if not matches or index >= len(matches):
                return {'context': 'unknown'}
            
            match = matches[-(index + 1)]  # Reverse indexing
            
            # Extract basic context
            context = {
                'opponent_id': None,
                'venue': 'unknown',
                'competition': 'unknown',
                'importance': 'normal'
            }
            
            # Get opponent
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            
            if home_team.get('id') == team_id:
                context['opponent_id'] = away_team.get('id')
                context['venue'] = 'home'
            elif away_team.get('id') == team_id:
                context['opponent_id'] = home_team.get('id')
                context['venue'] = 'away'
            
            # Get competition info
            league = match.get('league', {})
            context['competition'] = league.get('name', 'unknown')
            
            # Assess match importance (simplified)
            if 'cup' in context['competition'].lower() or 'final' in context['competition'].lower():
                context['importance'] = 'high'
            elif 'champion' in context['competition'].lower():
                context['importance'] = 'very_high'
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting match context: {str(e)}")
            return {'context': 'unknown'}
    
    def _calculate_confidence_level(self, variance: float) -> float:
        """Calculate confidence level from performance variance"""
        try:
            # Lower variance = higher confidence
            # Map variance (0-1) to confidence (0-1) inversely
            confidence = 1.0 / (1.0 + variance * 5.0)
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence level: {str(e)}")
            return 0.5
    
    def _assess_psychological_impact(self, variance_change: float, mean_change: float) -> float:
        """Assess psychological impact of confidence shift"""
        try:
            # High variance change or significant mean change = high psychological impact
            impact = abs(variance_change) * 2.0 + abs(mean_change) * 1.5
            return min(1.0, impact)
            
        except Exception as e:
            logger.error(f"Error assessing psychological impact: {str(e)}")
            return 0.0
    
    # Helper methods for team dynamics analysis
    def _detect_tactical_changes(self, performance_scores: List[float], matches: List[Dict], team_id: int) -> List[Dict]:
        """Detect tactical/formation changes from performance patterns"""
        try:
            if len(performance_scores) < 6:
                return []
            
            changes = []
            window_size = 3
            
            # Look for sudden systematic changes in performance pattern
            for i in range(window_size, len(performance_scores) - window_size):
                before_window = performance_scores[i-window_size:i]
                after_window = performance_scores[i:i+window_size]
                
                before_pattern = self._extract_performance_pattern(before_window)
                after_pattern = self._extract_performance_pattern(after_window)
                
                # Check for significant pattern change
                pattern_similarity = self._calculate_pattern_similarity(before_pattern, after_pattern)
                
                if pattern_similarity < 0.6:  # Significant change
                    changes.append({
                        'index': i,
                        'type': 'tactical_change',
                        'significance': 1.0 - pattern_similarity,
                        'before_pattern': before_pattern,
                        'after_pattern': after_pattern
                    })
            
            return changes
            
        except Exception as e:
            logger.error(f"Error detecting tactical changes: {str(e)}")
            return []
    
    def _detect_cohesion_changes(self, performance_scores: List[float]) -> List[Dict]:
        """Detect changes in team cohesion from performance consistency"""
        try:
            if len(performance_scores) < 8:
                return []
            
            changes = []
            window_size = 4
            
            for i in range(window_size, len(performance_scores) - window_size):
                before_window = performance_scores[i-window_size:i]
                after_window = performance_scores[i:i+window_size]
                
                before_consistency = 1.0 - np.std(before_window)
                after_consistency = 1.0 - np.std(after_window)
                
                consistency_change = after_consistency - before_consistency
                
                if abs(consistency_change) > 0.3:  # Significant change
                    change_type = 'improved_cohesion' if consistency_change > 0 else 'decreased_cohesion'
                    changes.append({
                        'index': i,
                        'type': change_type,
                        'significance': abs(consistency_change),
                        'consistency_before': before_consistency,
                        'consistency_after': after_consistency
                    })
            
            return changes
            
        except Exception as e:
            logger.error(f"Error detecting cohesion changes: {str(e)}")
            return []
    
    def _detect_consistency_changes(self, performance_scores: List[float]) -> List[Dict]:
        """Detect changes in performance consistency"""
        try:
            if len(performance_scores) < 6:
                return []
            
            changes = []
            
            # Rolling variance analysis
            window_size = 3
            variances = []
            
            for i in range(len(performance_scores) - window_size + 1):
                window = performance_scores[i:i + window_size]
                variances.append(np.var(window))
            
            # Detect significant variance changes
            variance_changes = np.diff(variances)
            threshold = np.std(variance_changes) * 1.5
            
            for i, change in enumerate(variance_changes):
                if abs(change) > threshold:
                    change_type = 'decreased_consistency' if change > 0 else 'improved_consistency'
                    changes.append({
                        'index': i + window_size,
                        'type': change_type,
                        'significance': abs(change) / threshold,
                        'variance_change': change
                    })
            
            return changes
            
        except Exception as e:
            logger.error(f"Error detecting consistency changes: {str(e)}")
            return []
    
    def _detect_pressure_response_changes(self, performance_scores: List[float], matches: List[Dict], team_id: int) -> List[Dict]:
        """Detect changes in how team responds to pressure situations"""
        try:
            # This would require identifying pressure matches and analyzing performance
            # For now, return simplified implementation
            return []
            
        except Exception as e:
            logger.error(f"Error detecting pressure response changes: {str(e)}")
            return []
    
    # Additional helper methods
    def _calculate_streak_quality(self, results: List[str], length: int) -> float:
        """Calculate quality/consistency of a streak"""
        try:
            if not results or length == 0:
                return 0.0
            
            # Quality based on dominance within streak type
            streak_type = results[0]
            consistency = sum(1 for r in results if r == streak_type) / len(results)
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating streak quality: {str(e)}")
            return 0.0
    
    def _calculate_streak_significance(self, streak_type: str, length: int) -> float:
        """Calculate significance of a streak based on type and length"""
        try:
            base_significance = {
                'W': 0.8,  # Winning streaks are highly significant
                'L': 0.7,  # Losing streaks are also significant
                'D': 0.3   # Draw streaks are less significant
            }.get(streak_type, 0.1)
            
            # Significance increases with length (diminishing returns)
            length_factor = min(2.0, 1.0 + np.log(length) / 3.0)
            
            return base_significance * length_factor
            
        except Exception as e:
            logger.error(f"Error calculating streak significance: {str(e)}")
            return 0.0
    
    def _calculate_streak_momentum_value(self, streak_type: str, length: int, quality: float) -> float:
        """Calculate momentum value of a streak"""
        try:
            type_values = {'W': 1.0, 'D': 0.1, 'L': -0.8}
            base_value = type_values.get(streak_type, 0.0)
            
            # Value increases with length and quality
            momentum_value = base_value * (1.0 + np.log(length) / 4.0) * quality
            
            return momentum_value
            
        except Exception as e:
            logger.error(f"Error calculating streak momentum value: {str(e)}")
            return 0.0
    
    def _calculate_unbeaten_momentum(self, current: int, longest: int) -> float:
        """Calculate momentum from unbeaten run"""
        try:
            if longest == 0:
                return 0.0
            
            # Momentum increases with current run, especially if approaching record
            progress_ratio = current / longest if longest > 0 else 0.0
            base_momentum = min(0.5, current * 0.05)
            
            # Bonus if approaching or exceeding record
            if progress_ratio > 0.8:
                record_bonus = (progress_ratio - 0.8) * 0.5
                return base_momentum + record_bonus
            
            return base_momentum
            
        except Exception as e:
            logger.error(f"Error calculating unbeaten momentum: {str(e)}")
            return 0.0
    
    def _calculate_pressure_index(self, current_winless: int, longest_winless: int) -> float:
        """Calculate pressure index from winless run"""
        try:
            if current_winless == 0:
                return 0.0
            
            # Pressure increases with length of winless run
            base_pressure = min(1.0, current_winless * 0.1)
            
            # Extra pressure if approaching negative record
            if longest_winless > 0:
                progress_ratio = current_winless / longest_winless
                if progress_ratio > 0.7:
                    record_pressure = (progress_ratio - 0.7) * 0.5
                    return base_pressure + record_pressure
            
            return base_pressure
            
        except Exception as e:
            logger.error(f"Error calculating pressure index: {str(e)}")
            return 0.0
    
    def _extract_performance_pattern(self, window: List[float]) -> Dict:
        """Extract performance pattern characteristics from a window"""
        try:
            if not window:
                return {'mean': 0.0, 'trend': 0.0, 'volatility': 0.0}
            
            return {
                'mean': float(np.mean(window)),
                'trend': float(np.polyfit(range(len(window)), window, 1)[0]) if len(window) > 1 else 0.0,
                'volatility': float(np.std(window)),
                'range': float(np.max(window) - np.min(window)) if len(window) > 1 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error extracting performance pattern: {str(e)}")
            return {'mean': 0.0, 'trend': 0.0, 'volatility': 0.0}
    
    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between two performance patterns"""
        try:
            # Compare key pattern features
            mean_diff = abs(pattern1.get('mean', 0.0) - pattern2.get('mean', 0.0))
            trend_diff = abs(pattern1.get('trend', 0.0) - pattern2.get('trend', 0.0))
            volatility_diff = abs(pattern1.get('volatility', 0.0) - pattern2.get('volatility', 0.0))
            
            # Normalize differences and calculate similarity
            mean_similarity = 1.0 - min(1.0, mean_diff)
            trend_similarity = 1.0 - min(1.0, trend_diff * 2.0)
            volatility_similarity = 1.0 - min(1.0, volatility_diff)
            
            # Weighted average
            overall_similarity = (mean_similarity * 0.4 + trend_similarity * 0.3 + volatility_similarity * 0.3)
            
            return overall_similarity
            
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {str(e)}")
            return 0.0
    
    def _analyze_cluster_characteristics(self, cluster_features: List[List[float]], cluster_indices: np.ndarray) -> Dict:
        """Analyze characteristics of a momentum pattern cluster"""
        try:
            if not cluster_features:
                return {}
            
            feature_array = np.array(cluster_features)
            
            return {
                'avg_performance': float(np.mean(feature_array[:, 0])),
                'avg_variability': float(np.mean(feature_array[:, 1])),
                'avg_range': float(np.mean(feature_array[:, 2])),
                'trend_bias': float(np.mean(feature_array[:, 6])),
                'size': len(cluster_features),
                'temporal_distribution': cluster_indices.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cluster characteristics: {str(e)}")
            return {}
    
    # Pattern detection helper methods
    def _detect_seasonal_patterns(self, performance_scores: List[float]) -> List[Dict]:
        """Detect seasonal performance patterns"""
        # Implementation placeholder - would require date information
        return []
    
    def _detect_cyclical_patterns(self, performance_scores: List[float]) -> List[Dict]:
        """Detect cyclical patterns in performance"""
        # Implementation placeholder - would use autocorrelation analysis
        return []
    
    def _detect_response_patterns(self, performance_scores: List[float]) -> List[Dict]:
        """Detect consistent response patterns"""
        # Implementation placeholder
        return []
    
    def _detect_recovery_patterns(self, performance_scores: List[float]) -> List[Dict]:
        """Detect recovery patterns after poor performance"""
        try:
            patterns = []
            
            # Find low points and analyze recovery
            for i in range(2, len(performance_scores) - 3):
                if (performance_scores[i] < 0.3 and  # Low performance
                    performance_scores[i] < performance_scores[i-1] and 
                    performance_scores[i] < performance_scores[i+1]):
                    
                    # Analyze recovery in next 3 matches
                    recovery_window = performance_scores[i+1:i+4]
                    if recovery_window:
                        recovery_strength = np.mean(recovery_window) - performance_scores[i]
                        if recovery_strength > 0.2:  # Significant recovery
                            patterns.append({
                                'type': 'recovery_pattern',
                                'start_index': i,
                                'recovery_strength': float(recovery_strength),
                                'confidence': min(1.0, recovery_strength * 2.0),
                                'strength': recovery_strength
                            })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting recovery patterns: {str(e)}")
            return []
    
    def _analyze_shift_patterns(self, first_derivative: np.ndarray, second_derivative: np.ndarray) -> Dict:
        """Analyze patterns in momentum shifts"""
        try:
            if len(first_derivative) == 0:
                return {}
            
            # Analyze shift characteristics
            positive_shifts = np.sum(first_derivative > 0.1)
            negative_shifts = np.sum(first_derivative < -0.1)
            sudden_changes = np.sum(np.abs(second_derivative) > 0.2) if len(second_derivative) > 0 else 0
            
            return {
                'positive_shifts': int(positive_shifts),
                'negative_shifts': int(negative_shifts),
                'sudden_changes': int(sudden_changes),
                'shift_frequency': float((positive_shifts + negative_shifts) / len(first_derivative)),
                'avg_shift_magnitude': float(np.mean(np.abs(first_derivative))),
                'momentum_volatility': float(np.std(first_derivative))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing shift patterns: {str(e)}")
            return {}
    
    # Critical Helper Methods for Predictive Modeling
    
    def _autoregressive_prediction(self, scores: np.ndarray, horizon: int) -> List[float]:
        """
        Autoregressive prediction using AR model
        
        Predicts future values based on linear combination of past values
        """
        try:
            if len(scores) < 3:
                return [np.mean(scores)] * horizon if len(scores) > 0 else [0.5] * horizon
            
            # Simple AR(2) model for stability
            if len(scores) >= 3:
                # Calculate AR coefficients using least squares
                X = []
                y = []
                
                for i in range(2, len(scores)):
                    X.append([scores[i-1], scores[i-2], 1])  # AR(2) + intercept
                    y.append(scores[i])
                
                if len(X) > 0:
                    X = np.array(X)
                    y = np.array(y)
                    
                    # Solve normal equations
                    try:
                        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                    except:
                        # Fallback to simple average
                        return [np.mean(scores[-3:])] * horizon
                    
                    # Make predictions
                    predictions = []
                    last_values = [scores[-2], scores[-1]]
                    
                    for _ in range(horizon):
                        next_pred = coeffs[0] * last_values[-1] + coeffs[1] * last_values[-2] + coeffs[2]
                        next_pred = max(0.0, min(1.0, next_pred))  # Clamp to valid range
                        predictions.append(next_pred)
                        
                        # Update last values for next prediction
                        last_values = [last_values[-1], next_pred]
                    
                    return predictions
                else:
                    return [np.mean(scores[-3:])] * horizon
            else:
                return [np.mean(scores)] * horizon
                
        except Exception as e:
            logger.error(f"Error in autoregressive prediction: {str(e)}")
            return [np.mean(scores[-3:]) if len(scores) >= 3 else 0.5] * horizon
    
    def _exponential_smoothing_prediction(self, scores: np.ndarray, horizon: int) -> List[float]:
        """
        Exponential smoothing prediction with trend
        
        Uses Holt's method for trend-adjusted exponential smoothing
        """
        try:
            if len(scores) < 2:
                return [scores[0] if len(scores) > 0 else 0.5] * horizon
            
            # Holt's exponential smoothing parameters
            alpha = 0.3  # Level smoothing
            beta = 0.1   # Trend smoothing
            
            # Initialize level and trend
            level = scores[0]
            trend = scores[1] - scores[0] if len(scores) > 1 else 0.0
            
            # Update level and trend for each observation
            for i in range(1, len(scores)):
                prev_level = level
                level = alpha * scores[i] + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
            
            # Make predictions
            predictions = []
            for h in range(1, horizon + 1):
                forecast = level + h * trend
                forecast = max(0.0, min(1.0, forecast))  # Clamp to valid range
                predictions.append(forecast)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in exponential smoothing prediction: {str(e)}")
            return [np.mean(scores[-3:]) if len(scores) >= 3 else 0.5] * horizon
    
    def _pattern_based_prediction(self, scores: np.ndarray, horizon: int) -> List[float]:
        """
        Pattern-based prediction using historical pattern matching
        
        Finds similar historical patterns and uses them for prediction
        """
        try:
            if len(scores) < 6:
                return [np.mean(scores[-3:]) if len(scores) >= 3 else 0.5] * horizon
            
            # Use last 3 values as pattern to match
            current_pattern = scores[-3:]
            pattern_length = len(current_pattern)
            
            # Find similar patterns in history
            similar_patterns = []
            
            for i in range(pattern_length, len(scores) - horizon):
                historical_pattern = scores[i-pattern_length:i]
                
                # Calculate pattern similarity
                similarity = self._calculate_pattern_similarity_simple(current_pattern, historical_pattern)
                
                if similarity > 0.7:  # High similarity threshold
                    # Get what happened after this pattern
                    after_pattern = scores[i:i+horizon] if i+horizon <= len(scores) else scores[i:]
                    if len(after_pattern) > 0:
                        similar_patterns.append({
                            'similarity': similarity,
                            'after_pattern': after_pattern,
                            'weight': similarity ** 2  # Square for emphasis on very similar patterns
                        })
            
            if similar_patterns:
                # Weighted average of similar patterns
                predictions = []
                total_weight = sum(p['weight'] for p in similar_patterns)
                
                max_length = max(len(p['after_pattern']) for p in similar_patterns)
                
                for h in range(min(horizon, max_length)):
                    weighted_sum = 0.0
                    weight_sum = 0.0
                    
                    for pattern in similar_patterns:
                        if h < len(pattern['after_pattern']):
                            weighted_sum += pattern['after_pattern'][h] * pattern['weight']
                            weight_sum += pattern['weight']
                    
                    if weight_sum > 0:
                        prediction = weighted_sum / weight_sum
                    else:
                        prediction = np.mean(current_pattern)
                    
                    prediction = max(0.0, min(1.0, prediction))
                    predictions.append(prediction)
                
                # Fill remaining horizon with trend
                while len(predictions) < horizon:
                    if len(predictions) >= 2:
                        trend = predictions[-1] - predictions[-2]
                        next_val = predictions[-1] + trend
                    else:
                        next_val = predictions[-1] if predictions else np.mean(current_pattern)
                    
                    next_val = max(0.0, min(1.0, next_val))
                    predictions.append(next_val)
                
                return predictions[:horizon]
            else:
                # No similar patterns found, use trend extrapolation
                if len(scores) >= 2:
                    trend = np.mean(np.diff(scores[-3:]))  # Average trend of last changes
                    predictions = []
                    last_value = scores[-1]
                    
                    for h in range(horizon):
                        next_val = last_value + (h + 1) * trend
                        next_val = max(0.0, min(1.0, next_val))
                        predictions.append(next_val)
                    
                    return predictions
                else:
                    return [np.mean(scores)] * horizon
                    
        except Exception as e:
            logger.error(f"Error in pattern-based prediction: {str(e)}")
            return [np.mean(scores[-3:]) if len(scores) >= 3 else 0.5] * horizon
    
    def _trend_based_prediction(self, scores: np.ndarray, horizon: int) -> List[float]:
        """
        Trend-based prediction using linear regression
        
        Fits a linear trend and extrapolates for prediction
        """
        try:
            if len(scores) < 3:
                return [scores[-1] if len(scores) > 0 else 0.5] * horizon
            
            # Use recent data for trend estimation (last 8 points or all if less)
            recent_data = scores[-8:] if len(scores) >= 8 else scores
            x = np.arange(len(recent_data))
            
            # Fit linear trend
            try:
                coeffs = np.polyfit(x, recent_data, 1)
                slope = coeffs[0]
                intercept = coeffs[1]
            except:
                # Fallback to simple average
                return [np.mean(recent_data)] * horizon
            
            # Make predictions
            predictions = []
            start_x = len(recent_data)
            
            for h in range(horizon):
                x_pred = start_x + h
                y_pred = slope * x_pred + intercept
                y_pred = max(0.0, min(1.0, y_pred))  # Clamp to valid range
                predictions.append(y_pred)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in trend-based prediction: {str(e)}")
            return [np.mean(scores[-3:]) if len(scores) >= 3 else 0.5] * horizon
    
    def _analyze_trajectory_trend(self, predictions: List[float]) -> Dict:
        """
        Analyze trend characteristics of predicted trajectory
        
        Determines if trajectory is increasing, decreasing, or stable
        """
        try:
            if len(predictions) < 2:
                return {'direction': 'stable', 'strength': 0.0, 'consistency': 0.0}
            
            # Calculate changes
            changes = np.diff(predictions)
            
            # Overall direction
            total_change = predictions[-1] - predictions[0]
            avg_change = np.mean(changes)
            
            if total_change > 0.05:
                direction = 'increasing'
            elif total_change < -0.05:
                direction = 'decreasing'
            else:
                direction = 'stable'
            
            # Trend strength (how strong is the trend)
            strength = abs(total_change) / len(predictions)
            
            # Trend consistency (how consistent is the direction)
            if len(changes) > 0:
                positive_changes = sum(1 for c in changes if c > 0)
                negative_changes = sum(1 for c in changes if c < 0)
                consistency = max(positive_changes, negative_changes) / len(changes)
            else:
                consistency = 0.0
            
            return {
                'direction': direction,
                'strength': float(strength),
                'consistency': float(consistency),
                'total_change': float(total_change),
                'average_change_per_step': float(avg_change)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trajectory trend: {str(e)}")
            return {'direction': 'stable', 'strength': 0.0, 'consistency': 0.0}
    
    def _calculate_trajectory_confidence(self, ar_pred: List[float], exp_pred: List[float], 
                                       pattern_pred: List[float], trend_pred: List[float]) -> float:
        """
        Calculate confidence in trajectory prediction based on method agreement
        
        Higher agreement between methods = higher confidence
        """
        try:
            predictions = [ar_pred, exp_pred, pattern_pred, trend_pred]
            
            # Remove empty predictions
            valid_predictions = [pred for pred in predictions if len(pred) > 0]
            
            if len(valid_predictions) < 2:
                return 0.1  # Low confidence if too few methods
            
            # Calculate agreement between methods
            agreements = []
            horizon = min(len(pred) for pred in valid_predictions)
            
            for i in range(horizon):
                values = [pred[i] for pred in valid_predictions]
                if len(values) >= 2:
                    variance = np.var(values)
                    # Lower variance = higher agreement = higher confidence
                    agreement = 1.0 / (1.0 + variance * 5.0)
                    agreements.append(agreement)
            
            if agreements:
                avg_agreement = np.mean(agreements)
                # Scale to reasonable confidence range
                confidence = min(0.95, max(0.1, avg_agreement))
                return confidence
            else:
                return 0.1
                
        except Exception as e:
            logger.error(f"Error calculating trajectory confidence: {str(e)}")
            return 0.1
    
    def _calculate_pattern_similarity_simple(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """
        Calculate simple pattern similarity using correlation and distance
        
        Combines correlation and normalized distance for similarity measure
        """
        try:
            if len(pattern1) != len(pattern2) or len(pattern1) == 0:
                return 0.0
            
            # Normalize patterns to 0-1 range
            p1_norm = (pattern1 - np.min(pattern1)) / (np.max(pattern1) - np.min(pattern1) + 1e-8)
            p2_norm = (pattern2 - np.min(pattern2)) / (np.max(pattern2) - np.min(pattern2) + 1e-8)
            
            # Calculate correlation
            try:
                correlation = np.corrcoef(p1_norm, p2_norm)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
            
            # Calculate normalized distance similarity
            distance = np.mean(np.abs(p1_norm - p2_norm))
            distance_similarity = 1.0 - distance
            
            # Combine correlation and distance similarity
            similarity = (abs(correlation) * 0.6 + distance_similarity * 0.4)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {str(e)}")
            return 0.0
    
    def _backtest_trajectory_prediction(self, historical_data: np.ndarray) -> np.ndarray:
        """
        Simplified trajectory prediction for backtesting
        
        Uses simplified prediction method for accuracy evaluation
        """
        try:
            if len(historical_data) < 3:
                return np.array([np.mean(historical_data)] * 5 if len(historical_data) > 0 else [0.5] * 5)
            
            # Simple combination of trend and exponential smoothing
            trend_pred = self._trend_based_prediction(historical_data, 5)
            exp_pred = self._exponential_smoothing_prediction(historical_data, 5)
            
            # Average the two methods
            combined = [(t + e) / 2.0 for t, e in zip(trend_pred, exp_pred)]
            
            return np.array(combined)
            
        except Exception as e:
            logger.error(f"Error in backtest trajectory prediction: {str(e)}")
            return np.array([0.5] * 5)
    
    # Additional placeholder helper methods for complete functionality
    def _calculate_momentum_strength(self, momentum: np.ndarray) -> float:
        """Calculate strength of current momentum"""
        try:
            if len(momentum) == 0:
                return 0.0
            return float(np.mean(momentum))
        except:
            return 0.0
    
    def _analyze_historical_sustainability(self, scores: np.ndarray) -> float:
        """Analyze historical sustainability patterns"""
        try:
            if len(scores) < 5:
                return 0.5
            # Simple sustainability based on variance
            return 1.0 - min(1.0, np.var(scores))
        except:
            return 0.5
    
    def _assess_volatility_sustainability(self, momentum: np.ndarray) -> float:
        """Assess sustainability based on volatility"""
        try:
            if len(momentum) < 2:
                return 0.5
            volatility = np.std(momentum)
            return 1.0 - min(1.0, volatility)
        except:
            return 0.5
    
    def _analyze_trend_consistency(self, scores: np.ndarray) -> float:
        """Analyze trend consistency"""
        try:
            if len(scores) < 3:
                return 0.5
            changes = np.diff(scores)
            if len(changes) == 0:
                return 0.5
            # Count direction changes
            direction_changes = sum(1 for i in range(len(changes)-1) if changes[i] * changes[i+1] < 0)
            consistency = 1.0 - (direction_changes / max(1, len(changes)-1))
            return consistency
        except:
            return 0.5
    
    def _assess_peak_valley_sustainability(self, scores: np.ndarray) -> float:
        """Assess peak/valley sustainability"""
        try:
            if len(scores) < 5:
                return 0.5
            # Find peaks and valleys
            peaks = []
            valleys = []
            for i in range(1, len(scores)-1):
                if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                    peaks.append(scores[i])
                elif scores[i] < scores[i-1] and scores[i] < scores[i+1]:
                    valleys.append(scores[i])
            
            # Balance between peaks and valleys indicates sustainability
            if len(peaks) + len(valleys) == 0:
                return 0.5
            
            peak_avg = np.mean(peaks) if peaks else 0.5
            valley_avg = np.mean(valleys) if valleys else 0.5
            balance = 1.0 - abs(peak_avg - valley_avg)
            return max(0.0, min(1.0, balance))
        except:
            return 0.5