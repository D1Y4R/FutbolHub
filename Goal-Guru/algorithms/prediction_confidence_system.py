"""
Advanced Prediction Confidence System Implementation
Comprehensive confidence scoring and uncertainty measurement system for football predictions.

This system provides sophisticated confidence assessment through:
- Multi-model agreement analysis
- Prediction variance quantification 
- Historical accuracy-based confidence
- Context-specific reliability scoring
- Bayesian uncertainty quantification
- Risk-adjusted prediction intervals
- Intelligent confidence communication

Features:
- Real-time confidence tracking
- Data quality impact assessment
- Model consensus scoring
- Prediction stability analysis
- Feature importance uncertainty
- Context familiarity assessment
- Conservative vs aggressive prediction modes
- Betting confidence optimization
- Alert systems for low confidence predictions
"""

import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import math
import scipy.stats as stats
from scipy.stats import entropy, beta, norm, t
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence level categories"""
    VERY_LOW = "very_low"      # 0-20
    LOW = "low"                # 20-40
    MODERATE = "moderate"      # 40-60
    HIGH = "high"              # 60-80
    VERY_HIGH = "very_high"    # 80-100

class RiskTolerance(Enum):
    """Risk tolerance modes for predictions"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

class PredictionType(Enum):
    """Types of predictions for context-specific confidence"""
    WIN_DRAW_LOSS = "1x2"
    EXACT_SCORE = "exact_score"
    OVER_UNDER = "over_under"
    BOTH_TEAMS_SCORE = "btts"
    HALFTIME_FULLTIME = "ht_ft"
    HANDICAP = "handicap"
    GOAL_RANGE = "goal_range"

@dataclass
class ConfidenceMetrics:
    """Comprehensive confidence metrics for a prediction"""
    overall_confidence: float  # 0-100 scale
    model_agreement: float     # How much models agree (0-1)
    prediction_variance: float # Variance in model outputs
    historical_accuracy: float # Historical performance in similar contexts
    data_quality_score: float  # Quality of input data (0-1)
    context_familiarity: float # How familiar is this context (0-1)
    stability_score: float     # Prediction stability (0-1)
    uncertainty_interval: Tuple[float, float]  # Confidence interval
    risk_adjusted_confidence: Dict[RiskTolerance, float]  # Risk-adjusted scores
    recommendation_strength: float  # Betting recommendation strength (0-100)
    explanation: str           # Human-readable confidence explanation
    alert_level: str          # Alert level for low confidence
    factors: Dict[str, float] # Individual confidence factors

@dataclass
class ModelPredictionInput:
    """Input from individual prediction model"""
    model_name: str
    prediction: Dict[str, float]  # Prediction probabilities
    confidence: float             # Model's own confidence
    historical_accuracy: float    # Model's historical accuracy
    context_performance: float    # Performance in this context
    data_quality: float          # Quality of data used
    features_used: List[str]     # Features that influenced prediction
    uncertainty: float           # Model's uncertainty estimate

@dataclass
class MatchContext:
    """Context information for confidence assessment"""
    league: str
    teams: Tuple[str, str]
    team_strengths: Tuple[float, float]
    recent_form: Tuple[float, float]
    head_to_head_history: int
    data_completeness: float
    match_importance: float
    seasonal_period: str
    venue_type: str
    weather_conditions: Optional[Dict]
    fixture_congestion: float

class PredictionConfidenceSystem:
    """
    Advanced Prediction Confidence System
    
    Provides comprehensive confidence assessment for football predictions
    using multi-model analysis, historical performance, and uncertainty quantification.
    """
    
    def __init__(self):
        """Initialize the confidence system"""
        logger.info("Initializing Prediction Confidence System...")
        
        # Configuration
        self.config = {
            'min_models_for_consensus': 3,
            'variance_threshold_high': 0.1,
            'variance_threshold_low': 0.05,
            'historical_window_days': 365,
            'min_historical_samples': 10,
            'data_quality_weights': {
                'completeness': 0.3,
                'freshness': 0.2,
                'reliability': 0.25,
                'consistency': 0.25
            },
            'confidence_factors_weights': {
                'model_agreement': 0.25,
                'historical_accuracy': 0.20,
                'data_quality': 0.15,
                'context_familiarity': 0.15,
                'prediction_stability': 0.15,
                'variance_penalty': 0.10
            }
        }
        
        # Historical performance tracking
        self.performance_history = defaultdict(lambda: defaultdict(list))
        self.context_performance = defaultdict(lambda: defaultdict(float))
        self.model_reliability = defaultdict(float)
        
        # Uncertainty quantification components
        self.bayesian_estimator = BayesianConfidenceEstimator()
        self.stability_analyzer = PredictionStabilityAnalyzer()
        self.consensus_analyzer = ModelConsensusAnalyzer()
        
        # Load historical data
        self._load_historical_performance()
        
        logger.info("Prediction Confidence System initialized successfully")
    
    def calculate_comprehensive_confidence(self, 
                                        model_predictions: List[ModelPredictionInput],
                                        match_context: MatchContext,
                                        prediction_type: PredictionType = PredictionType.WIN_DRAW_LOSS) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence metrics for a prediction
        
        Args:
            model_predictions: List of predictions from different models
            match_context: Context information about the match
            prediction_type: Type of prediction being made
            
        Returns:
            ConfidenceMetrics: Comprehensive confidence assessment
        """
        logger.info(f"Calculating confidence for {prediction_type.value} prediction")
        
        # 1. Multi-Model Agreement Assessment
        model_agreement = self._calculate_model_agreement(model_predictions)
        
        # 2. Prediction Variance Analysis
        prediction_variance = self._calculate_prediction_variance(model_predictions)
        
        # 3. Historical Accuracy-Based Confidence
        historical_accuracy = self._calculate_historical_accuracy(
            model_predictions, match_context, prediction_type
        )
        
        # 4. Data Quality Impact Assessment
        data_quality_score = self._assess_data_quality(model_predictions, match_context)
        
        # 5. Context Familiarity Assessment
        context_familiarity = self._assess_context_familiarity(match_context, prediction_type)
        
        # 6. Prediction Stability Analysis
        stability_score = self.stability_analyzer.analyze_stability(
            model_predictions, match_context
        )
        
        # 7. Overall Confidence Calculation
        overall_confidence = self._calculate_overall_confidence(
            model_agreement, prediction_variance, historical_accuracy,
            data_quality_score, context_familiarity, stability_score
        )
        
        # 8. Uncertainty Interval Estimation
        uncertainty_interval = self.bayesian_estimator.estimate_confidence_interval(
            model_predictions, overall_confidence
        )
        
        # 9. Risk-Adjusted Confidence
        risk_adjusted_confidence = self._calculate_risk_adjusted_confidence(
            overall_confidence, prediction_variance, match_context
        )
        
        # 10. Recommendation Strength
        recommendation_strength = self._calculate_recommendation_strength(
            overall_confidence, model_agreement, prediction_type
        )
        
        # 11. Generate Explanation
        explanation = self._generate_confidence_explanation(
            overall_confidence, model_agreement, historical_accuracy, 
            data_quality_score, prediction_type
        )
        
        # 12. Determine Alert Level
        alert_level = self._determine_alert_level(overall_confidence, prediction_variance)
        
        # 13. Compile Individual Factors
        factors = {
            'model_agreement': model_agreement,
            'prediction_variance': prediction_variance,
            'historical_accuracy': historical_accuracy,
            'data_quality': data_quality_score,
            'context_familiarity': context_familiarity,
            'stability_score': stability_score
        }
        
        # Create comprehensive confidence metrics
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=overall_confidence,
            model_agreement=model_agreement,
            prediction_variance=prediction_variance,
            historical_accuracy=historical_accuracy,
            data_quality_score=data_quality_score,
            context_familiarity=context_familiarity,
            stability_score=stability_score,
            uncertainty_interval=uncertainty_interval,
            risk_adjusted_confidence=risk_adjusted_confidence,
            recommendation_strength=recommendation_strength,
            explanation=explanation,
            alert_level=alert_level,
            factors=factors
        )
        
        # Update performance tracking
        self._update_performance_tracking(confidence_metrics, match_context, prediction_type)
        
        logger.info(f"Confidence calculated: {overall_confidence:.1f}% ({alert_level})")
        return confidence_metrics
    
    def _calculate_model_agreement(self, model_predictions: List[ModelPredictionInput]) -> float:
        """Calculate agreement between different models"""
        if len(model_predictions) < 2:
            return 1.0  # Single model = perfect agreement
        
        # Extract prediction vectors for each outcome
        outcome_predictions = defaultdict(list)
        for model_pred in model_predictions:
            for outcome, prob in model_pred.prediction.items():
                outcome_predictions[outcome].append(prob)
        
        # Calculate agreement for each outcome
        agreements = []
        for outcome, probs in outcome_predictions.items():
            if len(probs) < 2:
                continue
                
            # Calculate coefficient of variation (lower = more agreement)
            mean_prob = np.mean(probs)
            std_prob = np.std(probs)
            
            if mean_prob > 0:
                cv = std_prob / mean_prob
                # Convert to agreement score (0-1, higher = more agreement)
                agreement = max(0, 1 - cv)
                agreements.append(agreement)
        
        # Overall agreement
        if not agreements:
            return 0.5
        
        overall_agreement = np.mean(agreements)
        
        # Bonus for models with similar confidence levels
        confidences = [pred.confidence for pred in model_predictions]
        confidence_agreement = 1 - (np.std(confidences) / (np.mean(confidences) + 0.01))
        
        # Weighted combination
        final_agreement = 0.7 * overall_agreement + 0.3 * confidence_agreement
        return float(np.clip(final_agreement, 0, 1))
    
    def _calculate_prediction_variance(self, model_predictions: List[ModelPredictionInput]) -> float:
        """Calculate variance in model predictions"""
        if len(model_predictions) < 2:
            return 0.0
        
        # Calculate variance for main outcomes (home, draw, away)
        variances = []
        main_outcomes = ['home_win', 'draw', 'away_win']
        
        for outcome in main_outcomes:
            probs = []
            for model_pred in model_predictions:
                # Handle different outcome naming conventions
                if outcome in model_pred.prediction:
                    probs.append(model_pred.prediction[outcome])
                elif outcome == 'home_win' and 'home' in model_pred.prediction:
                    probs.append(model_pred.prediction['home'])
                elif outcome == 'away_win' and 'away' in model_pred.prediction:
                    probs.append(model_pred.prediction['away'])
            
            if len(probs) >= 2:
                variance = np.var(probs)
                variances.append(variance)
        
        if not variances:
            return 0.0
        
        # Average variance across outcomes
        avg_variance = np.mean(variances)
        
        # Normalize to 0-1 scale (higher variance = lower confidence)
        # Typical variance range: 0-0.25 for probabilities
        normalized_variance = min(avg_variance / 0.25, 1.0)
        
        return float(normalized_variance)
    
    def _calculate_historical_accuracy(self, 
                                     model_predictions: List[ModelPredictionInput],
                                     match_context: MatchContext,
                                     prediction_type: PredictionType) -> float:
        """Calculate confidence based on historical accuracy"""
        
        # Context key for historical lookup
        context_key = f"{match_context.league}_{prediction_type.value}"
        
        # Calculate weighted historical accuracy
        total_weight = 0
        weighted_accuracy = 0
        
        for model_pred in model_predictions:
            # Get model's historical accuracy in this context
            model_accuracy = self.context_performance[context_key].get(
                model_pred.model_name, model_pred.historical_accuracy
            )
            
            # Weight by model's current confidence
            weight = model_pred.confidence
            weighted_accuracy += model_accuracy * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5  # Default moderate confidence
        
        base_accuracy = weighted_accuracy / total_weight
        
        # Adjust based on context familiarity
        familiarity_factor = self._get_context_familiarity_factor(match_context)
        
        # Combine base accuracy with familiarity
        historical_confidence = base_accuracy * familiarity_factor
        
        return np.clip(historical_confidence, 0, 1)
    
    def _assess_data_quality(self, 
                           model_predictions: List[ModelPredictionInput],
                           match_context: MatchContext) -> float:
        """Assess the quality of data used for predictions"""
        
        quality_scores = []
        
        # 1. Data Completeness
        completeness = match_context.data_completeness
        quality_scores.append(('completeness', completeness))
        
        # 2. Data Freshness (based on how recent the data is)
        # Assume higher fixture congestion means more recent data
        freshness = min(1.0, 1 - match_context.fixture_congestion * 0.3)
        quality_scores.append(('freshness', freshness))
        
        # 3. Data Reliability (average of model data quality scores)
        model_data_qualities = [pred.data_quality for pred in model_predictions if pred.data_quality > 0]
        reliability = np.mean(model_data_qualities) if model_data_qualities else 0.7
        quality_scores.append(('reliability', reliability))
        
        # 4. Data Consistency (based on head-to-head history availability)
        consistency = min(1.0, match_context.head_to_head_history / 10.0)  # Normalize to 10 matches
        quality_scores.append(('consistency', consistency))
        
        # Calculate weighted average
        weights = self.config['data_quality_weights']
        total_score = sum(weights[factor] * score for factor, score in quality_scores)
        
        return np.clip(total_score, 0, 1)
    
    def _assess_context_familiarity(self, 
                                  match_context: MatchContext,
                                  prediction_type: PredictionType) -> float:
        """Assess how familiar this context is to our models"""
        
        familiarity_factors = []
        
        # 1. League Familiarity
        league_key = match_context.league
        league_predictions = len(self.performance_history.get(league_key, {}))
        league_familiarity = min(1.0, league_predictions / 100.0)  # Normalize to 100 predictions
        familiarity_factors.append(league_familiarity)
        
        # 2. Team Familiarity
        team_key = f"{match_context.teams[0]}_{match_context.teams[1]}"
        h2h_familiarity = min(1.0, match_context.head_to_head_history / 10.0)
        familiarity_factors.append(h2h_familiarity)
        
        # 3. Prediction Type Familiarity
        pred_type_key = f"{league_key}_{prediction_type.value}"
        type_predictions = len(self.performance_history.get(pred_type_key, {}))
        type_familiarity = min(1.0, type_predictions / 50.0)  # Normalize to 50 predictions
        familiarity_factors.append(type_familiarity)
        
        # 4. Match Context Familiarity (strength difference, importance)
        strength_diff = abs(match_context.team_strengths[0] - match_context.team_strengths[1])
        context_familiarity = 1 - min(0.5, strength_diff / 100.0)  # Penalty for extreme differences
        familiarity_factors.append(context_familiarity)
        
        # Average familiarity score
        overall_familiarity = np.mean(familiarity_factors)
        
        return float(np.clip(overall_familiarity, 0, 1))
    
    def _calculate_overall_confidence(self,
                                    model_agreement: float,
                                    prediction_variance: float,
                                    historical_accuracy: float,
                                    data_quality: float,
                                    context_familiarity: float,
                                    stability_score: float) -> float:
        """Calculate overall confidence score using weighted factors"""
        
        weights = self.config['confidence_factors_weights']
        
        # Positive factors (higher = better confidence)
        positive_factors = [
            ('model_agreement', model_agreement),
            ('historical_accuracy', historical_accuracy),
            ('data_quality', data_quality),
            ('context_familiarity', context_familiarity),
            ('prediction_stability', stability_score)
        ]
        
        # Calculate positive contribution
        positive_score = sum(weights[factor] * score for factor, score in positive_factors)
        
        # Negative factors (higher = worse confidence)
        variance_penalty = prediction_variance * weights['variance_penalty']
        
        # Overall confidence (0-1 scale)
        confidence = positive_score - variance_penalty
        
        # Apply non-linear transformation for better distribution
        confidence = self._apply_confidence_curve(confidence)
        
        # Convert to 0-100 scale
        return float(np.clip(confidence * 100, 0, 100))
    
    def _apply_confidence_curve(self, raw_confidence: float) -> float:
        """Apply non-linear curve to raw confidence for better distribution"""
        # Sigmoid-like transformation to spread confidence values
        # This helps avoid clustering around 50%
        
        # Shift to center around 0.5
        centered = raw_confidence - 0.5
        
        # Apply sigmoid transformation
        transformed = 1 / (1 + np.exp(-4 * centered))
        
        # Apply slight amplification for extreme values
        if transformed > 0.8:
            transformed = 0.8 + (transformed - 0.8) * 1.5
        elif transformed < 0.2:
            transformed = 0.2 - (0.2 - transformed) * 1.5
        
        return float(np.clip(transformed, 0, 1))
    
    def _calculate_risk_adjusted_confidence(self,
                                          overall_confidence: float,
                                          prediction_variance: float,
                                          match_context: MatchContext) -> Dict[RiskTolerance, float]:
        """Calculate risk-adjusted confidence for different risk tolerances"""
        
        risk_adjustments = {
            RiskTolerance.CONSERVATIVE: {
                'base_penalty': 15,    # Base penalty for conservative approach
                'variance_penalty': 25, # Higher penalty for variance
                'uncertainty_threshold': 70  # Lower threshold for recommendations
            },
            RiskTolerance.BALANCED: {
                'base_penalty': 5,     # Moderate penalty
                'variance_penalty': 15, # Moderate variance penalty
                'uncertainty_threshold': 60  # Balanced threshold
            },
            RiskTolerance.AGGRESSIVE: {
                'base_penalty': 0,     # No base penalty
                'variance_penalty': 5,  # Low variance penalty
                'uncertainty_threshold': 50  # Lower threshold for recommendations
            }
        }
        
        risk_adjusted = {}
        
        for risk_level, adjustments in risk_adjustments.items():
            # Start with overall confidence
            adjusted_confidence = overall_confidence
            
            # Apply base penalty
            adjusted_confidence -= adjustments['base_penalty']
            
            # Apply variance penalty
            variance_penalty = prediction_variance * adjustments['variance_penalty']
            adjusted_confidence -= variance_penalty
            
            # Apply match context adjustments
            if match_context.match_importance > 0.8:  # High importance match
                if risk_level == RiskTolerance.CONSERVATIVE:
                    adjusted_confidence -= 10  # Extra cautious for important matches
                elif risk_level == RiskTolerance.AGGRESSIVE:
                    adjusted_confidence += 5   # Slightly more confident for big matches
            
            # Ensure valid range
            adjusted_confidence = np.clip(adjusted_confidence, 0, 100)
            risk_adjusted[risk_level] = adjusted_confidence
        
        return risk_adjusted
    
    def _calculate_recommendation_strength(self,
                                         overall_confidence: float,
                                         model_agreement: float,
                                         prediction_type: PredictionType) -> float:
        """Calculate recommendation strength for betting optimization"""
        
        # Base strength from confidence
        base_strength = overall_confidence
        
        # Boost for high model agreement
        agreement_bonus = model_agreement * 15  # Up to 15 point bonus
        
        # Adjust by prediction type difficulty
        type_multipliers = {
            PredictionType.WIN_DRAW_LOSS: 1.0,      # Standard
            PredictionType.OVER_UNDER: 0.9,         # Slightly easier
            PredictionType.BOTH_TEAMS_SCORE: 0.85,  # Moderate difficulty
            PredictionType.EXACT_SCORE: 0.6,        # Very difficult
            PredictionType.HALFTIME_FULLTIME: 0.7,  # Difficult
            PredictionType.HANDICAP: 0.8,           # Moderate
            PredictionType.GOAL_RANGE: 0.75         # Moderate difficulty
        }
        
        type_multiplier = type_multipliers.get(prediction_type, 0.8)
        
        # Calculate final strength
        recommendation_strength = (base_strength + agreement_bonus) * type_multiplier
        
        # Apply threshold-based adjustments
        if recommendation_strength >= 85:
            recommendation_strength = min(100, recommendation_strength * 1.1)  # Boost high confidence
        elif recommendation_strength <= 35:
            recommendation_strength = max(0, recommendation_strength * 0.8)   # Reduce low confidence
        
        return np.clip(recommendation_strength, 0, 100)
    
    def _generate_confidence_explanation(self,
                                       overall_confidence: float,
                                       model_agreement: float,
                                       historical_accuracy: float,
                                       data_quality: float,
                                       prediction_type: PredictionType) -> str:
        """Generate human-readable explanation for confidence level"""
        
        confidence_level = self._get_confidence_level(overall_confidence)
        
        # Base explanation
        explanations = {
            ConfidenceLevel.VERY_HIGH: "Çok yüksek güven: Modeller yüksek oranda hemfikir ve geçmiş performans mükemmel.",
            ConfidenceLevel.HIGH: "Yüksek güven: Modeller büyük ölçüde hemfikir ve güçlü geçmiş performans.",
            ConfidenceLevel.MODERATE: "Orta güven: Modeller genel olarak hemfikir ancak bazı belirsizlikler mevcut.",
            ConfidenceLevel.LOW: "Düşük güven: Modeller arasında farklılıklar var veya veri kalitesi sınırlı.",
            ConfidenceLevel.VERY_LOW: "Çok düşük güven: Yüksek belirsizlik, modeller anlaşamıyor veya veri eksik."
        }
        
        base_explanation = explanations[confidence_level]
        
        # Add specific details
        details = []
        
        if model_agreement >= 0.8:
            details.append("modeller yüksek oranda hemfikir")
        elif model_agreement <= 0.5:
            details.append("modeller arasında anlaşmazlık var")
        
        if historical_accuracy >= 0.8:
            details.append("geçmiş performans mükemmel")
        elif historical_accuracy <= 0.5:
            details.append("geçmiş performans sınırlı")
        
        if data_quality >= 0.8:
            details.append("veri kalitesi yüksek")
        elif data_quality <= 0.5:
            details.append("veri kalitesi düşük")
        
        # Prediction type specific notes
        if prediction_type == PredictionType.EXACT_SCORE:
            details.append("kesin skor tahmini doğası gereği zor")
        elif prediction_type == PredictionType.WIN_DRAW_LOSS:
            details.append("1X2 tahmini için iyi model güvenilirliği")
        
        # Combine base explanation with details
        if details:
            full_explanation = f"{base_explanation} Detaylar: {', '.join(details)}."
        else:
            full_explanation = base_explanation
        
        return full_explanation
    
    def _determine_alert_level(self, overall_confidence: float, prediction_variance: float) -> str:
        """Determine alert level for low confidence predictions"""
        
        if overall_confidence <= 30 or prediction_variance >= 0.4:
            return "YÜKSEK_RİSK"
        elif overall_confidence <= 50 or prediction_variance >= 0.25:
            return "ORTA_RİSK"
        elif overall_confidence <= 70:
            return "DÜŞÜK_RİSK"
        else:
            return "GÜVENİLİR"
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        if confidence >= 80:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 60:
            return ConfidenceLevel.HIGH
        elif confidence >= 40:
            return ConfidenceLevel.MODERATE
        elif confidence >= 20:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _get_context_familiarity_factor(self, match_context: MatchContext) -> float:
        """Get familiarity factor for the match context"""
        # This would typically use historical data
        # For now, return a reasonable default based on available information
        
        base_familiarity = 0.7
        
        # Adjust based on head-to-head history
        if match_context.head_to_head_history >= 10:
            base_familiarity += 0.2
        elif match_context.head_to_head_history >= 5:
            base_familiarity += 0.1
        
        # Adjust based on data completeness
        base_familiarity *= match_context.data_completeness
        
        return np.clip(base_familiarity, 0, 1)
    
    def _update_performance_tracking(self,
                                   confidence_metrics: ConfidenceMetrics,
                                   match_context: MatchContext,
                                   prediction_type: PredictionType):
        """Update performance tracking for future confidence assessments"""
        
        # Create context keys
        league_key = match_context.league
        prediction_key = f"{league_key}_{prediction_type.value}"
        
        # Store confidence metrics for future reference
        timestamp = datetime.now().isoformat()
        performance_record = {
            'timestamp': timestamp,
            'confidence': confidence_metrics.overall_confidence,
            'model_agreement': confidence_metrics.model_agreement,
            'data_quality': confidence_metrics.data_quality_score,
            'context_familiarity': confidence_metrics.context_familiarity
        }
        
        # Add to performance history
        if prediction_key not in self.performance_history:
            self.performance_history[prediction_key] = defaultdict(list)
        
        self.performance_history[prediction_key]['confidence_history'].append(performance_record)
        
        # Maintain rolling window
        max_history = 1000
        if len(self.performance_history[prediction_key]['confidence_history']) > max_history:
            self.performance_history[prediction_key]['confidence_history'] = \
                self.performance_history[prediction_key]['confidence_history'][-max_history:]
    
    def _load_historical_performance(self):
        """Load historical performance data"""
        try:
            history_file = "algorithms/confidence_performance_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.performance_history = defaultdict(lambda: defaultdict(list), data.get('performance_history', {}))
                    self.context_performance = defaultdict(lambda: defaultdict(float), data.get('context_performance', {}))
                    self.model_reliability = defaultdict(float, data.get('model_reliability', {}))
                logger.info("Historical performance data loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load historical performance data: {e}")
            self.performance_history = defaultdict(lambda: defaultdict(list))
            self.context_performance = defaultdict(lambda: defaultdict(float))
            self.model_reliability = defaultdict(float)
    
    def save_performance_history(self):
        """Save historical performance data"""
        try:
            history_file = "algorithms/confidence_performance_history.json"
            data = {
                'performance_history': dict(self.performance_history),
                'context_performance': dict(self.context_performance),
                'model_reliability': dict(self.model_reliability),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info("Performance history saved successfully")
        except Exception as e:
            logger.error(f"Could not save performance history: {e}")


class BayesianConfidenceEstimator:
    """Bayesian approach to confidence interval estimation"""
    
    def estimate_confidence_interval(self, 
                                   model_predictions: List[ModelPredictionInput],
                                   overall_confidence: float,
                                   confidence_level: float = 0.95) -> Tuple[float, float]:
        """Estimate Bayesian confidence interval"""
        
        if len(model_predictions) < 2:
            # Single model - use confidence to estimate interval
            margin = (100 - overall_confidence) / 100 * 0.5
            return (max(0, overall_confidence - margin * 100), 
                   min(100, overall_confidence + margin * 100))
        
        # Extract main outcome probabilities
        home_probs = []
        for pred in model_predictions:
            if 'home_win' in pred.prediction:
                home_probs.append(pred.prediction['home_win'])
            elif 'home' in pred.prediction:
                home_probs.append(pred.prediction['home'])
        
        if not home_probs:
            return (overall_confidence - 10, overall_confidence + 10)
        
        # Bayesian Beta distribution estimation
        # Convert probabilities to successes and trials
        successes = np.mean(home_probs) * 100
        trials = 100
        
        # Beta distribution parameters
        alpha = successes + 1
        beta_param = trials - successes + 1
        
        # Calculate confidence interval using scipy.stats.beta
        from scipy.stats import beta as beta_dist
        lower = beta_dist.ppf((1 - confidence_level) / 2, alpha, beta_param) * 100
        upper = beta_dist.ppf(1 - (1 - confidence_level) / 2, alpha, beta_param) * 100
        
        return (float(np.clip(lower, 0, 100)), float(np.clip(upper, 0, 100)))


class PredictionStabilityAnalyzer:
    """Analyzes prediction stability across slight variations"""
    
    def analyze_stability(self, 
                         model_predictions: List[ModelPredictionInput],
                         match_context: MatchContext,
                         perturbation_level: float = 0.05) -> float:
        """Analyze how stable predictions are to small changes"""
        
        if len(model_predictions) < 2:
            return 0.8  # Default stability for single model
        
        # Calculate stability based on prediction variance
        outcome_stabilities = []
        
        main_outcomes = ['home_win', 'draw', 'away_win', 'home', 'away']
        
        for outcome in main_outcomes:
            probs = []
            for pred in model_predictions:
                if outcome in pred.prediction:
                    probs.append(pred.prediction[outcome])
            
            if len(probs) >= 2:
                # Calculate coefficient of variation
                mean_prob = np.mean(probs)
                std_prob = np.std(probs)
                
                if mean_prob > 0:
                    cv = std_prob / mean_prob
                    stability = max(0, 1 - cv * 2)  # Lower CV = higher stability
                    outcome_stabilities.append(stability)
        
        if not outcome_stabilities:
            return 0.5
        
        # Average stability across outcomes
        base_stability = np.mean(outcome_stabilities)
        
        # Adjust for match context factors
        if match_context.fixture_congestion > 0.7:
            base_stability *= 0.9  # High congestion reduces stability
        
        if match_context.data_completeness < 0.8:
            base_stability *= 0.95  # Incomplete data reduces stability
        
        return float(np.clip(base_stability, 0, 1))


class ModelConsensusAnalyzer:
    """Analyzes consensus between different prediction models"""
    
    def calculate_consensus_score(self, model_predictions: List[ModelPredictionInput]) -> float:
        """Calculate consensus score between models"""
        
        if len(model_predictions) < 2:
            return 1.0
        
        # Calculate pairwise agreement
        agreements = []
        
        for i in range(len(model_predictions)):
            for j in range(i + 1, len(model_predictions)):
                agreement = self._calculate_pairwise_agreement(
                    model_predictions[i], model_predictions[j]
                )
                agreements.append(agreement)
        
        if not agreements:
            return 0.5
        
        return float(np.mean(agreements))
    
    def _calculate_pairwise_agreement(self, 
                                    pred1: ModelPredictionInput,
                                    pred2: ModelPredictionInput) -> float:
        """Calculate agreement between two model predictions"""
        
        # Find common outcomes
        common_outcomes = set(pred1.prediction.keys()) & set(pred2.prediction.keys())
        
        if not common_outcomes:
            return 0.0
        
        # Calculate agreement for each common outcome
        agreements = []
        for outcome in common_outcomes:
            prob1 = pred1.prediction[outcome]
            prob2 = pred2.prediction[outcome]
            
            # Calculate agreement as inverse of absolute difference
            diff = abs(prob1 - prob2)
            agreement = max(0, 1 - diff)
            agreements.append(agreement)
        
        return float(np.mean(agreements))