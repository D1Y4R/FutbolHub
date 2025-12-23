"""
Meta-Learning Layer Implementation
Advanced meta-learning system that learns which models succeed under which conditions
and optimizes prediction quality through intelligent model selection and adaptation.

Features:
- Model Performance Profiling: Track algorithm success patterns per context
- Intelligent Model Selection: Dynamic algorithm weighting based on context
- Learning from Errors: Pattern analysis and systematic bias correction
- Adaptive Intelligence: Self-improving prediction strategies with concept drift detection
- Real-time Learning: Continuous improvement through feedback loops
"""

import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Meta-learning operation modes"""
    EXPLORATION = "exploration"  # Learn from all data
    EXPLOITATION = "exploitation"  # Use learned patterns
    ADAPTATION = "adaptation"  # Adapt to concept drift
    VALIDATION = "validation"  # Validate model performance

class ContextType(Enum):
    """Context types for model performance analysis"""
    LEAGUE = "league"
    TEAM_STRENGTH = "team_strength"
    MATCH_IMPORTANCE = "match_importance"
    RECENT_FORM = "recent_form"
    HEAD_TO_HEAD = "head_to_head"
    SEASONAL_PERIOD = "seasonal_period"
    VENUE_TYPE = "venue_type"
    WEATHER_CONDITIONS = "weather_conditions"

@dataclass
class ModelPerformanceProfile:
    """Comprehensive model performance profile for specific contexts"""
    model_name: str
    context_type: ContextType
    context_value: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_correlation: float
    prediction_count: int
    error_patterns: Dict[str, float]
    success_conditions: List[str]
    failure_indicators: List[str]
    optimal_parameters: Dict[str, Any]
    last_updated: str
    performance_trend: List[float]  # Last 10 accuracy scores
    stability_score: float
    reliability_index: float

@dataclass
class MetaFeatures:
    """Meta-features extracted for model selection"""
    league_difficulty: float
    team_predictability: float
    match_volatility: float
    historical_accuracy: float
    form_stability: float
    head_to_head_clarity: float
    seasonal_factor: float
    data_quality_score: float
    context_similarity: float
    uncertainty_level: float

@dataclass
class LearningSession:
    """Learning session information"""
    session_id: str
    start_time: str
    end_time: str
    predictions_analyzed: int
    patterns_discovered: int
    models_updated: int
    improvement_score: float
    concept_drift_detected: bool
    adaptation_actions: List[str]

class ConceptDriftDetector:
    """Detects concept drift in model performance"""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 0.05):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.performance_windows = defaultdict(deque)
        self.baseline_distributions = {}
        
    def add_performance_sample(self, model_name: str, accuracy: float, context: str):
        """Add a new performance sample"""
        key = f"{model_name}_{context}"
        window = self.performance_windows[key]
        
        window.append(accuracy)
        if len(window) > self.window_size:
            window.popleft()
            
        # Update baseline if we have enough samples
        if len(window) >= self.window_size // 2:
            self.baseline_distributions[key] = {
                'mean': np.mean(window),
                'std': np.std(window),
                'samples': list(window)
            }
    
    def detect_drift(self, model_name: str, context: str, recent_window: int = 20) -> Tuple[bool, float]:
        """Detect concept drift using statistical tests"""
        key = f"{model_name}_{context}"
        
        if key not in self.baseline_distributions:
            return False, 0.0
            
        window = self.performance_windows[key]
        if len(window) < recent_window * 2:
            return False, 0.0
            
        # Get recent and historical performance
        recent_performance = list(window)[-recent_window:]
        historical_performance = self.baseline_distributions[key]['samples']
        
        # Kolmogorov-Smirnov test for distribution change
        try:
            statistic, p_value = stats.ks_2samp(historical_performance, recent_performance)
            drift_detected = p_value < self.sensitivity
            drift_magnitude = statistic
            
            return drift_detected, drift_magnitude
        except:
            return False, 0.0

class ErrorPatternAnalyzer:
    """Analyzes error patterns and systematic biases"""
    
    def __init__(self):
        self.error_database = defaultdict(list)
        self.bias_patterns = defaultdict(dict)
        self.correction_strategies = {}
        
    def record_error(self, model_name: str, predicted: Dict, actual: Dict, context: Dict):
        """Record a prediction error for analysis"""
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'predicted': predicted,
            'actual': actual,
            'context': context,
            'error_magnitude': self._calculate_error_magnitude(predicted, actual),
            'error_type': self._classify_error_type(predicted, actual)
        }
        
        self.error_database[model_name].append(error_record)
        
        # Keep only recent errors (last 1000)
        if len(self.error_database[model_name]) > 1000:
            self.error_database[model_name] = self.error_database[model_name][-1000:]
    
    def _calculate_error_magnitude(self, predicted: Dict, actual: Dict) -> float:
        """Calculate the magnitude of prediction error"""
        try:
            # For match result prediction
            if 'home_win_probability' in predicted and 'result' in actual:
                pred_probs = [
                    predicted.get('home_win_probability', 0),
                    predicted.get('draw_probability', 0),
                    predicted.get('away_win_probability', 0)
                ]
                
                # Convert actual result to probability vector
                actual_vector = [0, 0, 0]
                if actual['result'] == 'H':
                    actual_vector[0] = 1
                elif actual['result'] == 'D':
                    actual_vector[1] = 1
                elif actual['result'] == 'A':
                    actual_vector[2] = 1
                
                # Calculate cross-entropy loss
                epsilon = 1e-15
                pred_probs = np.clip(pred_probs, epsilon, 1 - epsilon)
                return -np.sum(actual_vector * np.log(pred_probs))
            
            return 0.5  # Default moderate error
        except:
            return 0.5
    
    def _classify_error_type(self, predicted: Dict, actual: Dict) -> str:
        """Classify the type of prediction error"""
        try:
            if 'result' in actual:
                pred_result = self._get_predicted_result(predicted)
                actual_result = actual['result']
                
                if pred_result == 'H' and actual_result == 'A':
                    return 'home_overconfidence'
                elif pred_result == 'A' and actual_result == 'H':
                    return 'away_overconfidence'
                elif pred_result in ['H', 'A'] and actual_result == 'D':
                    return 'draw_underestimation'
                elif pred_result == 'D' and actual_result in ['H', 'A']:
                    return 'draw_overestimation'
                    
            return 'unknown_error'
        except:
            return 'unknown_error'
    
    def _get_predicted_result(self, predicted: Dict) -> str:
        """Get the most likely predicted result"""
        probs = {
            'H': predicted.get('home_win_probability', 0),
            'D': predicted.get('draw_probability', 0),
            'A': predicted.get('away_win_probability', 0)
        }
        return max(probs, key=probs.get)
    
    def analyze_bias_patterns(self, model_name: str) -> Dict[str, Any]:
        """Analyze systematic bias patterns for a model"""
        if model_name not in self.error_database:
            return {}
            
        errors = self.error_database[model_name]
        
        # Analyze error types frequency
        error_types = defaultdict(int)
        for error in errors:
            error_types[error['error_type']] += 1
        
        # Analyze context-specific biases
        context_biases = defaultdict(list)
        for error in errors:
            context = error['context']
            for key, value in context.items():
                context_biases[f"{key}_{value}"].append(error['error_magnitude'])
        
        # Calculate bias scores
        bias_analysis = {
            'error_type_distribution': dict(error_types),
            'context_specific_biases': {},
            'overall_bias_score': np.mean([e['error_magnitude'] for e in errors]) if errors else 0,
            'systematic_patterns': []
        }
        
        for context_key, magnitudes in context_biases.items():
            if len(magnitudes) >= 5:  # Minimum samples for reliable analysis
                bias_analysis['context_specific_biases'][context_key] = {
                    'mean_error': np.mean(magnitudes),
                    'std_error': np.std(magnitudes),
                    'sample_count': len(magnitudes)
                }
        
        return bias_analysis
    
    def suggest_corrections(self, model_name: str) -> List[str]:
        """Suggest corrections based on identified patterns"""
        bias_analysis = self.analyze_bias_patterns(model_name)
        suggestions = []
        
        # Check for systematic overconfidence in home/away predictions
        error_types = bias_analysis.get('error_type_distribution', {})
        
        if error_types.get('home_overconfidence', 0) > error_types.get('away_overconfidence', 0) * 1.5:
            suggestions.append("Reduce home team advantage weighting")
            
        if error_types.get('draw_underestimation', 0) > 10:
            suggestions.append("Increase draw probability baseline")
            
        if error_types.get('draw_overestimation', 0) > 10:
            suggestions.append("Reduce draw probability in decisive contexts")
        
        # Context-specific suggestions
        context_biases = bias_analysis.get('context_specific_biases', {})
        for context, bias_info in context_biases.items():
            if bias_info['mean_error'] > 0.7:  # High error threshold
                suggestions.append(f"Review model parameters for context: {context}")
        
        return suggestions

class MetaLearningLayer:
    """
    Advanced Meta-Learning Layer for Football Prediction System
    
    This class implements a comprehensive meta-learning system that:
    1. Profiles model performance across different contexts
    2. Intelligently selects optimal models for specific situations
    3. Learns from prediction errors and adapts
    4. Implements adaptive intelligence with concept drift detection
    """
    
    def __init__(self, save_interval: int = 300):  # Save every 5 minutes
        """Initialize the Meta-Learning Layer"""
        self.save_interval = save_interval
        self.last_save_time = time.time()
        
        # Core components
        self.concept_drift_detector = ConceptDriftDetector()
        self.error_analyzer = ErrorPatternAnalyzer()
        
        # Performance database
        self.performance_profiles: Dict[str, ModelPerformanceProfile] = {}
        self.meta_features_cache: Dict[str, MetaFeatures] = {}
        self.learning_sessions: List[LearningSession] = []
        
        # Model selection intelligence
        self.model_rankings: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.context_model_map: Dict[str, str] = {}
        self.confidence_thresholds: Dict[str, float] = {}
        
        # Adaptive learning parameters
        self.learning_rate = 0.01
        self.adaptation_threshold = 0.1
        self.performance_window_size = 50
        self.learning_mode = LearningMode.EXPLORATION
        
        # Real-time learning components
        self.prediction_feedback_queue = deque(maxlen=1000)
        self.continuous_learning_enabled = True
        self.learning_thread = None
        self._start_continuous_learning()
        
        # File paths for persistence
        self.profiles_file = "algorithms/meta_learning_profiles.json"
        self.sessions_file = "algorithms/meta_learning_sessions.json"
        self.rankings_file = "algorithms/meta_learning_rankings.json"
        
        # Load existing data
        self._load_persistent_data()
        
        # Performance tracking integration
        self.performance_tracker = None
        self._initialize_performance_tracker()
        
        logger.info("🧠 Meta-Learning Layer initialized with advanced capabilities")
    
    def _initialize_performance_tracker(self):
        """Initialize integration with existing performance tracker"""
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from model_performance_tracker import ModelPerformanceTracker
            self.performance_tracker = ModelPerformanceTracker()
            logger.info("Integrated with ModelPerformanceTracker")
        except Exception as e:
            logger.warning(f"Could not integrate with ModelPerformanceTracker: {e}")
    
    def _start_continuous_learning(self):
        """Start continuous learning thread"""
        if self.learning_thread is None or not self.learning_thread.is_alive():
            self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
            self.learning_thread.start()
            logger.info("Continuous learning thread started")
    
    def _continuous_learning_loop(self):
        """Continuous learning background process"""
        while self.continuous_learning_enabled:
            try:
                # Process feedback queue
                if self.prediction_feedback_queue:
                    self._process_feedback_batch()
                
                # Periodic model ranking updates
                self._update_model_rankings()
                
                # Check for concept drift
                self._check_concept_drift()
                
                # Save data periodically
                current_time = time.time()
                if current_time - self.last_save_time > self.save_interval:
                    self._save_persistent_data()
                    self.last_save_time = current_time
                
                time.sleep(30)  # Sleep for 30 seconds
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def extract_meta_features(self, match_context: Dict) -> MetaFeatures:
        """Extract meta-features for intelligent model selection"""
        try:
            # League difficulty analysis
            league = match_context.get('league', '')
            league_difficulty = self._calculate_league_difficulty(league)
            
            # Team predictability analysis
            home_stats = match_context.get('home_stats', {})
            away_stats = match_context.get('away_stats', {})
            team_predictability = self._calculate_team_predictability(home_stats, away_stats)
            
            # Match volatility assessment
            match_volatility = self._calculate_match_volatility(match_context)
            
            # Historical accuracy for similar contexts
            historical_accuracy = self._get_historical_accuracy(match_context)
            
            # Form stability analysis
            form_stability = self._calculate_form_stability(home_stats, away_stats)
            
            # Head-to-head clarity
            h2h_data = match_context.get('head_to_head', {})
            head_to_head_clarity = self._calculate_h2h_clarity(h2h_data)
            
            # Seasonal factor
            seasonal_factor = self._calculate_seasonal_factor(match_context.get('date', ''))
            
            # Data quality assessment
            data_quality_score = self._assess_data_quality(match_context)
            
            # Context similarity to known patterns
            context_similarity = self._calculate_context_similarity(match_context)
            
            # Uncertainty level assessment
            uncertainty_level = self._calculate_uncertainty_level(match_context)
            
            meta_features = MetaFeatures(
                league_difficulty=league_difficulty,
                team_predictability=team_predictability,
                match_volatility=match_volatility,
                historical_accuracy=historical_accuracy,
                form_stability=form_stability,
                head_to_head_clarity=head_to_head_clarity,
                seasonal_factor=seasonal_factor,
                data_quality_score=data_quality_score,
                context_similarity=context_similarity,
                uncertainty_level=uncertainty_level
            )
            
            return meta_features
            
        except Exception as e:
            logger.error(f"Error extracting meta-features: {e}")
            # Return default meta-features
            return MetaFeatures(
                league_difficulty=0.5, team_predictability=0.5, match_volatility=0.5,
                historical_accuracy=0.5, form_stability=0.5, head_to_head_clarity=0.5,
                seasonal_factor=0.5, data_quality_score=0.5, context_similarity=0.5,
                uncertainty_level=0.5
            )
    
    def _calculate_league_difficulty(self, league: str) -> float:
        """Calculate league difficulty based on historical prediction accuracy"""
        # Check if we have performance data for this league
        league_accuracies = []
        
        for profile in self.performance_profiles.values():
            if profile.context_type == ContextType.LEAGUE and profile.context_value == league:
                league_accuracies.append(profile.accuracy)
        
        if league_accuracies:
            # Higher difficulty = lower accuracy = higher difficulty score
            avg_accuracy = np.mean(league_accuracies)
            difficulty = 1.0 - avg_accuracy  # Invert accuracy to get difficulty
        else:
            # Default difficulty for unknown leagues
            difficulty = 0.5
        
        return np.clip(difficulty, 0.0, 1.0)
    
    def _calculate_team_predictability(self, home_stats: Dict, away_stats: Dict) -> float:
        """Calculate how predictable the teams are based on their statistics"""
        try:
            predictability_factors = []
            
            # Form consistency
            for stats in [home_stats, away_stats]:
                if 'recent_results' in stats:
                    results = stats['recent_results']
                    if len(results) >= 3:
                        # Calculate consistency in results
                        win_rate = sum(1 for r in results if r == 'W') / len(results)
                        consistency = 1.0 - abs(win_rate - 0.5) * 2  # Higher for teams close to 50% or very high/low
                        predictability_factors.append(consistency)
                
                # Goal scoring consistency
                if 'goals_for_avg' in stats and 'goals_for_std' in stats:
                    avg_goals = stats['goals_for_avg']
                    std_goals = stats['goals_for_std']
                    if avg_goals > 0:
                        coefficient_of_variation = std_goals / avg_goals
                        goal_predictability = 1.0 / (1.0 + coefficient_of_variation)
                        predictability_factors.append(goal_predictability)
            
            return np.mean(predictability_factors) if predictability_factors else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating team predictability: {e}")
            return 0.5
    
    def _calculate_match_volatility(self, match_context: Dict) -> float:
        """Calculate match volatility based on various factors"""
        try:
            volatility_factors = []
            
            # ELO difference (smaller diff = higher volatility)
            elo_diff = abs(match_context.get('elo_diff', 0))
            elo_volatility = 1.0 / (1.0 + elo_diff / 100.0)  # Normalize and invert
            volatility_factors.append(elo_volatility)
            
            # Position difference in league table
            home_pos = match_context.get('home_position', 10)
            away_pos = match_context.get('away_position', 10)
            pos_diff = abs(home_pos - away_pos)
            pos_volatility = 1.0 / (1.0 + pos_diff / 5.0)
            volatility_factors.append(pos_volatility)
            
            # Recent form volatility
            home_form = match_context.get('home_stats', {}).get('form_score', 0.5)
            away_form = match_context.get('away_stats', {}).get('form_score', 0.5)
            form_diff = abs(home_form - away_form)
            form_volatility = 1.0 - form_diff  # Closer form = higher volatility
            volatility_factors.append(form_volatility)
            
            return np.mean(volatility_factors) if volatility_factors else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating match volatility: {e}")
            return 0.5
    
    def _get_historical_accuracy(self, match_context: Dict) -> float:
        """Get historical accuracy for similar contexts"""
        try:
            similar_contexts = []
            
            league = match_context.get('league', '')
            team_strength_category = self._categorize_team_strength(match_context)
            
            # Find profiles for similar contexts
            for profile in self.performance_profiles.values():
                if (profile.context_type == ContextType.LEAGUE and 
                    profile.context_value == league):
                    similar_contexts.append(profile.accuracy)
                elif (profile.context_type == ContextType.TEAM_STRENGTH and 
                      profile.context_value == team_strength_category):
                    similar_contexts.append(profile.accuracy)
            
            return np.mean(similar_contexts) if similar_contexts else 0.5
            
        except Exception as e:
            logger.warning(f"Error getting historical accuracy: {e}")
            return 0.5
    
    def _categorize_team_strength(self, match_context: Dict) -> str:
        """Categorize team strength based on various metrics"""
        try:
            home_elo = match_context.get('home_stats', {}).get('elo_rating', 1500)
            away_elo = match_context.get('away_stats', {}).get('elo_rating', 1500)
            avg_elo = (home_elo + away_elo) / 2
            
            if avg_elo > 1700:
                return "high_strength"
            elif avg_elo > 1300:
                return "medium_strength"
            else:
                return "low_strength"
                
        except:
            return "medium_strength"
    
    def _calculate_form_stability(self, home_stats: Dict, away_stats: Dict) -> float:
        """Calculate form stability for both teams"""
        try:
            stability_scores = []
            
            for stats in [home_stats, away_stats]:
                if 'form_analysis' in stats:
                    form_data = stats['form_analysis']
                    momentum = form_data.get('momentum_score', 0)
                    consistency = form_data.get('consistency_score', 0)
                    stability = (momentum + consistency) / 2
                    stability_scores.append(stability)
                elif 'recent_results' in stats:
                    # Calculate stability from recent results
                    results = stats['recent_results']
                    if len(results) >= 3:
                        # Count result type consistency
                        wins = results.count('W')
                        draws = results.count('D')
                        losses = results.count('L')
                        
                        # Higher stability for consistent patterns
                        max_type = max(wins, draws, losses)
                        stability = max_type / len(results)
                        stability_scores.append(stability)
            
            return np.mean(stability_scores) if stability_scores else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating form stability: {e}")
            return 0.5
    
    def _calculate_h2h_clarity(self, h2h_data: Dict) -> float:
        """Calculate head-to-head pattern clarity"""
        try:
            if not h2h_data or 'matches' not in h2h_data:
                return 0.3  # Low clarity for no H2H data
            
            matches = h2h_data['matches']
            if len(matches) < 3:
                return 0.4  # Low clarity for few matches
            
            # Analyze result patterns
            results = [match.get('result', '') for match in matches]
            home_wins = results.count('H')
            draws = results.count('D')
            away_wins = results.count('A')
            
            total = len(results)
            if total == 0:
                return 0.3
            
            # Calculate dominance clarity
            max_result_type = max(home_wins, draws, away_wins)
            dominance_clarity = max_result_type / total
            
            # Calculate goal pattern clarity
            goal_patterns = []
            for match in matches:
                home_goals = match.get('home_goals', 0)
                away_goals = match.get('away_goals', 0)
                goal_patterns.append(abs(home_goals - away_goals))
            
            goal_clarity = 1.0 - (np.std(goal_patterns) / (np.mean(goal_patterns) + 1))
            
            # Combine clarities
            overall_clarity = (dominance_clarity + goal_clarity) / 2
            return np.clip(overall_clarity, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating H2H clarity: {e}")
            return 0.3
    
    def _calculate_seasonal_factor(self, date_str: str) -> float:
        """Calculate seasonal factor based on date"""
        try:
            if not date_str:
                return 0.5
            
            # Parse date
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            month = date_obj.month
            
            # Define seasonal periods
            if month in [8, 9, 10]:  # Early season
                return 0.7  # Less predictable
            elif month in [11, 12, 1, 2]:  # Mid season
                return 0.9  # More predictable
            elif month in [3, 4, 5]:  # Late season
                return 0.8  # Moderately predictable
            else:  # Summer break / pre-season
                return 0.4  # Highly unpredictable
                
        except Exception as e:
            logger.warning(f"Error calculating seasonal factor: {e}")
            return 0.5
    
    def _assess_data_quality(self, match_context: Dict) -> float:
        """Assess the quality of available data for prediction"""
        try:
            quality_factors = []
            
            # Check data completeness
            required_fields = ['home_stats', 'away_stats', 'league', 'date']
            completeness = sum(1 for field in required_fields if field in match_context) / len(required_fields)
            quality_factors.append(completeness)
            
            # Check stats richness
            for team_key in ['home_stats', 'away_stats']:
                if team_key in match_context:
                    stats = match_context[team_key]
                    stats_fields = ['recent_matches', 'goals_for_avg', 'goals_against_avg', 'elo_rating']
                    stats_completeness = sum(1 for field in stats_fields if field in stats) / len(stats_fields)
                    quality_factors.append(stats_completeness)
            
            # Check temporal freshness
            if 'date' in match_context:
                try:
                    match_date = datetime.fromisoformat(match_context['date'].replace('Z', '+00:00'))
                    days_old = (datetime.now() - match_date).days
                    freshness = max(0, 1.0 - days_old / 30.0)  # Fresher data = higher quality
                    quality_factors.append(freshness)
                except:
                    quality_factors.append(0.5)
            
            return np.mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            logger.warning(f"Error assessing data quality: {e}")
            return 0.5
    
    def _calculate_context_similarity(self, match_context: Dict) -> float:
        """Calculate similarity to known successful prediction contexts"""
        try:
            if not self.performance_profiles:
                return 0.5  # No known contexts yet
            
            current_features = self.extract_meta_features(match_context)
            similarities = []
            
            # Compare with stored high-performance contexts
            for profile in self.performance_profiles.values():
                if profile.accuracy > 0.7:  # Only compare with successful contexts
                    # Simple feature similarity calculation
                    feature_diffs = []
                    # This would need the stored context features, simplified here
                    similarity = 0.6  # Placeholder
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating context similarity: {e}")
            return 0.5
    
    def _calculate_uncertainty_level(self, match_context: Dict) -> float:
        """Calculate overall uncertainty level for the match"""
        try:
            uncertainty_factors = []
            
            # ELO difference uncertainty (closer teams = higher uncertainty)
            elo_diff = abs(match_context.get('elo_diff', 0))
            elo_uncertainty = 1.0 / (1.0 + elo_diff / 50.0)
            uncertainty_factors.append(elo_uncertainty)
            
            # Form difference uncertainty
            home_form = match_context.get('home_stats', {}).get('form_score', 0.5)
            away_form = match_context.get('away_stats', {}).get('form_score', 0.5)
            form_uncertainty = 1.0 - abs(home_form - away_form)
            uncertainty_factors.append(form_uncertainty)
            
            # League predictability
            league_difficulty = self._calculate_league_difficulty(match_context.get('league', ''))
            uncertainty_factors.append(league_difficulty)
            
            # Data quality uncertainty (poor data = higher uncertainty)
            data_quality = self._assess_data_quality(match_context)
            data_uncertainty = 1.0 - data_quality
            uncertainty_factors.append(data_uncertainty)
            
            return np.mean(uncertainty_factors) if uncertainty_factors else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating uncertainty level: {e}")
            return 0.5
    
    def select_optimal_models(self, match_context: Dict, available_models: List[str]) -> List[Tuple[str, float]]:
        """
        Intelligently select optimal models and weights for given context
        
        Returns:
            List of (model_name, weight) tuples sorted by expected performance
        """
        logger.info("🧠 Selecting optimal models using meta-learning intelligence")
        
        # Extract meta-features for this context
        meta_features = self.extract_meta_features(match_context)
        
        # Get model performance scores for this context
        model_scores = {}
        
        for model_name in available_models:
            score = self._calculate_model_score(model_name, meta_features, match_context)
            model_scores[model_name] = score
        
        # Sort models by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert scores to weights (softmax-like normalization)
        scores = [score for _, score in sorted_models]
        if sum(scores) > 0:
            weights = self._softmax_normalize(scores)
        else:
            # Equal weights if no performance data
            weights = [1.0 / len(sorted_models)] * len(sorted_models)
        
        # Create final model-weight pairs
        optimal_models = [(model, weight) for (model, _), weight in zip(sorted_models, weights)]
        
        # Log selection reasoning
        top_3 = optimal_models[:3]
        logger.info(f"🎯 Top model selections: {[(m, f'{w:.3f}') for m, w in top_3]}")
        
        return optimal_models
    
    def _calculate_model_score(self, model_name: str, meta_features: MetaFeatures, match_context: Dict) -> float:
        """Calculate expected performance score for a model in given context"""
        base_score = 0.5  # Default score
        
        # Get historical performance for this model
        model_profiles = [p for p in self.performance_profiles.values() if p.model_name == model_name]
        
        if model_profiles:
            # Weight by context relevance
            weighted_scores = []
            
            for profile in model_profiles:
                relevance_weight = self._calculate_context_relevance(profile, match_context, meta_features)
                weighted_score = profile.accuracy * relevance_weight
                weighted_scores.append(weighted_score)
            
            if weighted_scores:
                base_score = np.mean(weighted_scores)
        
        # Apply meta-feature adjustments
        adjusted_score = self._apply_meta_feature_adjustments(model_name, base_score, meta_features)
        
        # Apply concept drift adjustments
        drift_adjustment = self._get_concept_drift_adjustment(model_name, match_context)
        final_score = adjusted_score * drift_adjustment
        
        return np.clip(final_score, 0.0, 1.0)
    
    def _calculate_context_relevance(self, profile: ModelPerformanceProfile, 
                                   match_context: Dict, meta_features: MetaFeatures) -> float:
        """Calculate how relevant a performance profile is to current context"""
        relevance_factors = []
        
        # League relevance
        if profile.context_type == ContextType.LEAGUE:
            if profile.context_value == match_context.get('league', ''):
                relevance_factors.append(1.0)
            else:
                relevance_factors.append(0.3)  # Different league
        
        # Team strength relevance
        elif profile.context_type == ContextType.TEAM_STRENGTH:
            current_strength = self._categorize_team_strength(match_context)
            if profile.context_value == current_strength:
                relevance_factors.append(0.8)
            else:
                relevance_factors.append(0.4)
        
        # Recent form relevance
        elif profile.context_type == ContextType.RECENT_FORM:
            form_similarity = 1.0 - abs(meta_features.form_stability - 0.5)  # Simplified
            relevance_factors.append(form_similarity)
        
        # Default relevance
        if not relevance_factors:
            relevance_factors.append(0.5)
        
        # Consider profile age (newer is more relevant)
        try:
            profile_date = datetime.fromisoformat(profile.last_updated)
            days_old = (datetime.now() - profile_date).days
            age_factor = max(0.3, 1.0 - days_old / 90.0)  # Decay over 90 days
            relevance_factors.append(age_factor)
        except:
            relevance_factors.append(0.5)
        
        return np.mean(relevance_factors)
    
    def _apply_meta_feature_adjustments(self, model_name: str, base_score: float, 
                                      meta_features: MetaFeatures) -> float:
        """Apply meta-feature based adjustments to model score"""
        
        # Model-specific adjustments based on characteristics
        adjustments = {
            'poisson': {
                'low_volatility_bonus': 0.1 if meta_features.match_volatility < 0.3 else 0,
                'high_predictability_bonus': 0.1 if meta_features.team_predictability > 0.7 else 0,
                'stable_form_bonus': 0.05 if meta_features.form_stability > 0.6 else 0
            },
            'xgboost': {
                'high_data_quality_bonus': 0.15 if meta_features.data_quality_score > 0.8 else 0,
                'complex_context_bonus': 0.1 if meta_features.uncertainty_level > 0.6 else 0,
                'league_difficulty_bonus': 0.05 if meta_features.league_difficulty > 0.5 else 0
            },
            'monte_carlo': {
                'high_uncertainty_bonus': 0.15 if meta_features.uncertainty_level > 0.7 else 0,
                'volatile_match_bonus': 0.1 if meta_features.match_volatility > 0.6 else 0,
                'low_predictability_bonus': 0.05 if meta_features.team_predictability < 0.4 else 0
            },
            'neural_network': {
                'high_data_quality_bonus': 0.12 if meta_features.data_quality_score > 0.7 else 0,
                'pattern_similarity_bonus': 0.08 if meta_features.context_similarity > 0.6 else 0,
                'complex_league_bonus': 0.05 if meta_features.league_difficulty > 0.6 else 0
            },
            'dixon_coles': {
                'low_scoring_bonus': 0.1,  # Generally good for low-scoring contexts
                'stable_form_bonus': 0.08 if meta_features.form_stability > 0.5 else 0,
                'clear_h2h_bonus': 0.05 if meta_features.head_to_head_clarity > 0.6 else 0
            },
            'crf': {
                'pattern_recognition_bonus': 0.1 if meta_features.context_similarity > 0.5 else 0,
                'moderate_uncertainty_bonus': 0.08 if 0.3 < meta_features.uncertainty_level < 0.7 else 0,
                'seasonal_bonus': 0.05 if meta_features.seasonal_factor > 0.6 else 0
            }
        }
        
        model_adjustments = adjustments.get(model_name, {})
        total_adjustment = sum(model_adjustments.values())
        
        return base_score + total_adjustment
    
    def _get_concept_drift_adjustment(self, model_name: str, match_context: Dict) -> float:
        """Get adjustment factor based on concept drift detection"""
        try:
            league = match_context.get('league', '')
            drift_detected, drift_magnitude = self.concept_drift_detector.detect_drift(model_name, league)
            
            if drift_detected:
                # Reduce confidence in model if drift detected
                adjustment = 1.0 - (drift_magnitude * 0.3)  # Max 30% reduction
                logger.warning(f"🌊 Concept drift detected for {model_name} in {league}, adjustment: {adjustment:.3f}")
                return max(0.5, adjustment)  # Minimum 50% confidence
            
            return 1.0  # No adjustment if no drift
            
        except Exception as e:
            logger.warning(f"Error calculating concept drift adjustment: {e}")
            return 1.0
    
    def _softmax_normalize(self, scores: List[float], temperature: float = 1.0) -> List[float]:
        """Apply softmax normalization to convert scores to weights"""
        if not scores:
            return []
        
        # Apply temperature scaling
        scaled_scores = [s / temperature for s in scores]
        
        # Softmax calculation
        exp_scores = [np.exp(s - max(scaled_scores)) for s in scaled_scores]  # Subtract max for stability
        sum_exp = sum(exp_scores)
        
        if sum_exp == 0:
            return [1.0 / len(scores)] * len(scores)
        
        return [exp_s / sum_exp for exp_s in exp_scores]
    
    def record_prediction_feedback(self, model_predictions: Dict, actual_result: Dict, 
                                 match_context: Dict, ensemble_result: Dict):
        """Record prediction feedback for continuous learning"""
        try:
            feedback_record = {
                'timestamp': datetime.now().isoformat(),
                'model_predictions': model_predictions,
                'actual_result': actual_result,
                'match_context': match_context,
                'ensemble_result': ensemble_result,
                'meta_features': asdict(self.extract_meta_features(match_context))
            }
            
            self.prediction_feedback_queue.append(feedback_record)
            
            # Immediate learning for critical feedback
            if self._is_critical_feedback(feedback_record):
                self._process_critical_feedback(feedback_record)
            
            logger.info(f"📝 Prediction feedback recorded, queue size: {len(self.prediction_feedback_queue)}")
            
        except Exception as e:
            logger.error(f"Error recording prediction feedback: {e}")
    
    def _is_critical_feedback(self, feedback_record: Dict) -> bool:
        """Determine if feedback requires immediate processing"""
        try:
            # Check for significant prediction errors
            ensemble_result = feedback_record['ensemble_result']
            actual_result = feedback_record['actual_result']
            
            # Calculate prediction error magnitude
            error_magnitude = self.error_analyzer._calculate_error_magnitude(
                ensemble_result, actual_result
            )
            
            return error_magnitude > 1.0  # High error threshold
            
        except:
            return False
    
    def _process_critical_feedback(self, feedback_record: Dict):
        """Process critical feedback immediately"""
        try:
            model_predictions = feedback_record['model_predictions']
            actual_result = feedback_record['actual_result']
            match_context = feedback_record['match_context']
            
            # Record errors for each model
            for model_name, prediction in model_predictions.items():
                self.error_analyzer.record_error(model_name, prediction, actual_result, match_context)
            
            # Update performance profiles
            self._update_performance_profiles(feedback_record)
            
            logger.info("🚨 Critical feedback processed immediately")
            
        except Exception as e:
            logger.error(f"Error processing critical feedback: {e}")
    
    def _process_feedback_batch(self):
        """Process a batch of feedback records"""
        try:
            if not self.prediction_feedback_queue:
                return
            
            # Process up to 10 records at a time
            batch_size = min(10, len(self.prediction_feedback_queue))
            batch = []
            
            for _ in range(batch_size):
                if self.prediction_feedback_queue:
                    batch.append(self.prediction_feedback_queue.popleft())
            
            # Process each feedback record
            for feedback_record in batch:
                self._update_performance_profiles(feedback_record)
                
                # Record errors for analysis
                model_predictions = feedback_record['model_predictions']
                actual_result = feedback_record['actual_result']
                match_context = feedback_record['match_context']
                
                for model_name, prediction in model_predictions.items():
                    self.error_analyzer.record_error(model_name, prediction, actual_result, match_context)
            
            logger.info(f"📊 Processed feedback batch of {len(batch)} records")
            
        except Exception as e:
            logger.error(f"Error processing feedback batch: {e}")
    
    def _update_performance_profiles(self, feedback_record: Dict):
        """Update performance profiles based on feedback"""
        try:
            model_predictions = feedback_record['model_predictions']
            actual_result = feedback_record['actual_result']
            match_context = feedback_record['match_context']
            meta_features = feedback_record['meta_features']
            
            # Extract context information
            league = match_context.get('league', '')
            team_strength = self._categorize_team_strength(match_context)
            
            # Update profiles for each model
            for model_name, prediction in model_predictions.items():
                # Calculate accuracy for this prediction
                accuracy = self._calculate_prediction_accuracy(prediction, actual_result)
                
                # Update league-specific profile
                league_profile_key = f"{model_name}_league_{league}"
                self._update_single_profile(
                    league_profile_key, model_name, ContextType.LEAGUE, league, 
                    accuracy, match_context, meta_features
                )
                
                # Update team strength profile
                strength_profile_key = f"{model_name}_strength_{team_strength}"
                self._update_single_profile(
                    strength_profile_key, model_name, ContextType.TEAM_STRENGTH, team_strength,
                    accuracy, match_context, meta_features
                )
                
                # Update concept drift detector
                self.concept_drift_detector.add_performance_sample(model_name, accuracy, league)
            
        except Exception as e:
            logger.error(f"Error updating performance profiles: {e}")
    
    def _calculate_prediction_accuracy(self, prediction: Dict, actual_result: Dict) -> float:
        """Calculate accuracy score for a single prediction"""
        try:
            # For match result prediction
            if 'result' in actual_result and 'home_win_probability' in prediction:
                predicted_probs = [
                    prediction.get('home_win_probability', 0),
                    prediction.get('draw_probability', 0),
                    prediction.get('away_win_probability', 0)
                ]
                
                # Get actual result index
                actual_result_map = {'H': 0, 'D': 1, 'A': 2}
                actual_idx = actual_result_map.get(actual_result['result'], 1)
                
                # Calculate accuracy as the probability assigned to correct outcome
                accuracy = predicted_probs[actual_idx]
                return np.clip(accuracy, 0.0, 1.0)
            
            return 0.5  # Default moderate accuracy
            
        except Exception as e:
            logger.warning(f"Error calculating prediction accuracy: {e}")
            return 0.5
    
    def _update_single_profile(self, profile_key: str, model_name: str, context_type: ContextType,
                             context_value: str, accuracy: float, match_context: Dict, meta_features: Dict):
        """Update a single performance profile"""
        try:
            if profile_key in self.performance_profiles:
                # Update existing profile
                profile = self.performance_profiles[profile_key]
                profile.prediction_count += 1
                
                # Update accuracy with learning rate
                old_accuracy = profile.accuracy
                profile.accuracy = old_accuracy + self.learning_rate * (accuracy - old_accuracy)
                
                # Update performance trend
                profile.performance_trend.append(accuracy)
                if len(profile.performance_trend) > 10:
                    profile.performance_trend = profile.performance_trend[-10:]
                
                # Update stability score
                if len(profile.performance_trend) > 3:
                    profile.stability_score = 1.0 - np.std(profile.performance_trend)
                
                # Update reliability index
                profile.reliability_index = min(1.0, profile.prediction_count / 50.0) * profile.stability_score
                
                profile.last_updated = datetime.now().isoformat()
                
            else:
                # Create new profile
                self.performance_profiles[profile_key] = ModelPerformanceProfile(
                    model_name=model_name,
                    context_type=context_type,
                    context_value=context_value,
                    accuracy=accuracy,
                    precision=accuracy,  # Simplified
                    recall=accuracy,     # Simplified
                    f1_score=accuracy,   # Simplified
                    confidence_correlation=0.5,
                    prediction_count=1,
                    error_patterns={},
                    success_conditions=[],
                    failure_indicators=[],
                    optimal_parameters={},
                    last_updated=datetime.now().isoformat(),
                    performance_trend=[accuracy],
                    stability_score=1.0,
                    reliability_index=0.1  # Low initially
                )
                
        except Exception as e:
            logger.error(f"Error updating single profile: {e}")
    
    def _update_model_rankings(self):
        """Update model rankings based on current performance profiles"""
        try:
            # Group profiles by context
            context_rankings = defaultdict(list)
            
            for profile in self.performance_profiles.values():
                context_key = f"{profile.context_type.value}_{profile.context_value}"
                context_rankings[context_key].append((
                    profile.model_name,
                    profile.accuracy * profile.reliability_index  # Weight by reliability
                ))
            
            # Sort rankings for each context
            for context, model_scores in context_rankings.items():
                sorted_models = sorted(model_scores, key=lambda x: x[1], reverse=True)
                self.model_rankings[context] = sorted_models
                
        except Exception as e:
            logger.error(f"Error updating model rankings: {e}")
    
    def _check_concept_drift(self):
        """Check for concept drift across all models and contexts"""
        try:
            drift_detections = []
            
            # Check each model-context combination
            for profile in self.performance_profiles.values():
                if profile.prediction_count > 20:  # Minimum samples for drift detection
                    drift_detected, drift_magnitude = self.concept_drift_detector.detect_drift(
                        profile.model_name, profile.context_value
                    )
                    
                    if drift_detected:
                        drift_detections.append({
                            'model': profile.model_name,
                            'context': f"{profile.context_type.value}_{profile.context_value}",
                            'magnitude': drift_magnitude,
                            'timestamp': datetime.now().isoformat()
                        })
            
            if drift_detections:
                logger.warning(f"🌊 Concept drift detected in {len(drift_detections)} contexts")
                self._handle_concept_drift(drift_detections)
                
        except Exception as e:
            logger.error(f"Error checking concept drift: {e}")
    
    def _handle_concept_drift(self, drift_detections: List[Dict]):
        """Handle detected concept drift"""
        try:
            for detection in drift_detections:
                model_name = detection['model']
                context = detection['context']
                magnitude = detection['magnitude']
                
                # Reduce reliability of affected profiles
                affected_profiles = [
                    p for p in self.performance_profiles.values()
                    if p.model_name == model_name and f"{p.context_type.value}_{p.context_value}" == context
                ]
                
                for profile in affected_profiles:
                    # Reduce reliability based on drift magnitude
                    drift_penalty = magnitude * 0.3
                    profile.reliability_index = max(0.2, profile.reliability_index - drift_penalty)
                    
                    # Reset some performance history to adapt faster
                    if len(profile.performance_trend) > 5:
                        profile.performance_trend = profile.performance_trend[-5:]
                
                logger.info(f"🔄 Adapted to concept drift in {model_name} for {context}")
                
        except Exception as e:
            logger.error(f"Error handling concept drift: {e}")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights and recommendations"""
        try:
            insights = {
                'performance_summary': self._get_performance_summary(),
                'model_rankings': dict(self.model_rankings),
                'error_patterns': self._get_error_insights(),
                'concept_drift_status': self._get_drift_status(),
                'learning_recommendations': self._generate_learning_recommendations(),
                'meta_learning_stats': {
                    'total_profiles': len(self.performance_profiles),
                    'learning_sessions': len(self.learning_sessions),
                    'feedback_queue_size': len(self.prediction_feedback_queue),
                    'learning_mode': self.learning_mode.value,
                    'last_update': datetime.now().isoformat()
                }
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating learning insights: {e}")
            return {'error': str(e)}
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all models and contexts"""
        model_performance = defaultdict(list)
        
        for profile in self.performance_profiles.values():
            model_performance[profile.model_name].append({
                'context': f"{profile.context_type.value}_{profile.context_value}",
                'accuracy': profile.accuracy,
                'reliability': profile.reliability_index,
                'predictions': profile.prediction_count
            })
        
        # Calculate aggregated stats
        summary = {}
        for model, performances in model_performance.items():
            accuracies = [p['accuracy'] for p in performances]
            reliabilities = [p['reliability'] for p in performances]
            total_predictions = sum(p['predictions'] for p in performances)
            
            summary[model] = {
                'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                'max_accuracy': max(accuracies) if accuracies else 0,
                'min_accuracy': min(accuracies) if accuracies else 0,
                'avg_reliability': np.mean(reliabilities) if reliabilities else 0,
                'total_predictions': total_predictions,
                'context_count': len(performances)
            }
        
        return summary
    
    def _get_error_insights(self) -> Dict[str, Any]:
        """Get insights from error pattern analysis"""
        error_insights = {}
        
        for model_name in ['poisson', 'dixon_coles', 'xgboost', 'monte_carlo', 'crf', 'neural_network']:
            bias_analysis = self.error_analyzer.analyze_bias_patterns(model_name)
            suggestions = self.error_analyzer.suggest_corrections(model_name)
            
            error_insights[model_name] = {
                'bias_analysis': bias_analysis,
                'correction_suggestions': suggestions
            }
        
        return error_insights
    
    def _get_drift_status(self) -> Dict[str, Any]:
        """Get concept drift status summary"""
        drift_summary = {
            'models_with_drift': [],
            'stable_models': [],
            'adaptation_needed': []
        }
        
        # Check drift status for each model
        for profile in self.performance_profiles.values():
            if profile.prediction_count > 10:
                drift_detected, magnitude = self.concept_drift_detector.detect_drift(
                    profile.model_name, profile.context_value
                )
                
                model_context = f"{profile.model_name}_{profile.context_value}"
                
                if drift_detected:
                    drift_summary['models_with_drift'].append({
                        'model': profile.model_name,
                        'context': profile.context_value,
                        'magnitude': magnitude,
                        'reliability': profile.reliability_index
                    })
                    
                    if magnitude > 0.3:  # High drift threshold
                        drift_summary['adaptation_needed'].append(model_context)
                else:
                    drift_summary['stable_models'].append(model_context)
        
        return drift_summary
    
    def _generate_learning_recommendations(self) -> List[str]:
        """Generate actionable learning recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        performance_summary = self._get_performance_summary()
        
        # Find best and worst performing models
        model_avg_accuracies = {
            model: stats['avg_accuracy'] 
            for model, stats in performance_summary.items()
        }
        
        if model_avg_accuracies:
            best_model = max(model_avg_accuracies, key=model_avg_accuracies.get)
            worst_model = min(model_avg_accuracies, key=model_avg_accuracies.get)
            
            recommendations.append(f"Best performing model: {best_model} ({model_avg_accuracies[best_model]:.3f} avg accuracy)")
            
            if model_avg_accuracies[worst_model] < 0.5:
                recommendations.append(f"Consider reviewing {worst_model} parameters (low accuracy: {model_avg_accuracies[worst_model]:.3f})")
        
        # Concept drift recommendations
        drift_status = self._get_drift_status()
        if drift_status['adaptation_needed']:
            recommendations.append(f"Immediate adaptation needed for: {', '.join(drift_status['adaptation_needed'])}")
        
        # Data collection recommendations
        low_data_models = [
            model for model, stats in performance_summary.items()
            if stats['total_predictions'] < 50
        ]
        
        if low_data_models:
            recommendations.append(f"Increase data collection for: {', '.join(low_data_models)}")
        
        # Error pattern recommendations
        error_insights = self._get_error_insights()
        for model, insights in error_insights.items():
            suggestions = insights.get('correction_suggestions', [])
            if suggestions:
                recommendations.extend([f"{model}: {suggestion}" for suggestion in suggestions[:2]])  # Top 2 suggestions
        
        return recommendations
    
    def _load_persistent_data(self):
        """Load persistent data from files"""
        try:
            # Load performance profiles
            if os.path.exists(self.profiles_file):
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    
                for key, profile_dict in profiles_data.items():
                    # Convert back to enum
                    profile_dict['context_type'] = ContextType(profile_dict['context_type'])
                    self.performance_profiles[key] = ModelPerformanceProfile(**profile_dict)
                
                logger.info(f"Loaded {len(self.performance_profiles)} performance profiles")
            
            # Load learning sessions
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    sessions_data = json.load(f)
                    self.learning_sessions = [LearningSession(**session) for session in sessions_data]
                
                logger.info(f"Loaded {len(self.learning_sessions)} learning sessions")
            
            # Load model rankings
            if os.path.exists(self.rankings_file):
                with open(self.rankings_file, 'r', encoding='utf-8') as f:
                    self.model_rankings = json.load(f)
                
                logger.info(f"Loaded model rankings for {len(self.model_rankings)} contexts")
                
        except Exception as e:
            logger.warning(f"Error loading persistent data: {e}")
    
    def _save_persistent_data(self):
        """Save persistent data to files"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.profiles_file), exist_ok=True)
            
            # Save performance profiles
            profiles_data = {}
            for key, profile in self.performance_profiles.items():
                profile_dict = asdict(profile)
                profile_dict['context_type'] = profile.context_type.value  # Convert enum to string
                profiles_data[key] = profile_dict
            
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, indent=2, ensure_ascii=False)
            
            # Save learning sessions
            sessions_data = [asdict(session) for session in self.learning_sessions]
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, indent=2, ensure_ascii=False)
            
            # Save model rankings
            with open(self.rankings_file, 'w', encoding='utf-8') as f:
                json.dump(dict(self.model_rankings), f, indent=2, ensure_ascii=False)
            
            logger.info("Meta-learning data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving persistent data: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.continuous_learning_enabled = False
            if self.learning_thread and self.learning_thread.is_alive():
                self.learning_thread.join(timeout=5)
            self._save_persistent_data()
        except:
            pass  # Ignore cleanup errors

# Factory function for easy instantiation
def create_meta_learning_layer() -> MetaLearningLayer:
    """Factory function to create MetaLearningLayer instance"""
    return MetaLearningLayer()

# Example usage
if __name__ == "__main__":
    # Initialize meta-learning layer
    meta_learner = create_meta_learning_layer()
    
    # Example context
    sample_context = {
        'league': 'Premier League',
        'home_team': 'Manchester City',
        'away_team': 'Liverpool',
        'elo_diff': 50,
        'home_stats': {'form_score': 0.8, 'recent_matches': []},
        'away_stats': {'form_score': 0.7, 'recent_matches': []},
        'date': '2025-01-15T15:00:00Z'
    }
    
    # Select optimal models
    available_models = ['poisson', 'dixon_coles', 'xgboost', 'monte_carlo', 'crf', 'neural_network']
    optimal_models = meta_learner.select_optimal_models(sample_context, available_models)
    
    print("🧠 Meta-Learning Layer Example:")
    print(f"Optimal models: {optimal_models[:3]}")
    
    # Get learning insights
    insights = meta_learner.get_learning_insights()
    print(f"Learning insights: {insights['meta_learning_stats']}")