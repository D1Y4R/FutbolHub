"""
Meta-Learning Integration Module
Provides integration utilities and real-time learning capabilities for the meta-learning layer.

Features:
- Real-time performance monitoring
- Automatic feedback collection
- Performance improvement tracking
- System health monitoring
- Integration with existing components
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import threading
import time
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric for tracking improvements"""
    timestamp: str
    context: str
    model_name: str
    accuracy: float
    confidence: float
    prediction_time: float
    optimization_method: str
    success: bool
    error_magnitude: float = 0.0

@dataclass
class LearningProgress:
    """Learning progress tracking"""
    start_time: str
    total_predictions: int
    successful_predictions: int
    accuracy_improvements: List[float]
    concept_drifts_detected: int
    adaptations_performed: int
    avg_optimization_time: float
    best_model_selections: Dict[str, int]
    error_reduction_rate: float

class RealTimeLearningMonitor:
    """Real-time learning and performance monitoring system"""
    
    def __init__(self, save_interval: int = 300):  # 5 minutes
        self.save_interval = save_interval
        self.last_save_time = time.time()
        
        # Performance tracking
        self.performance_metrics: deque = deque(maxlen=1000)
        self.accuracy_history: Dict[str, List[float]] = defaultdict(list)
        self.learning_progress = self._initialize_learning_progress()
        
        # Real-time monitoring
        self.monitoring_enabled = True
        self.monitor_thread = None
        self.performance_alerts: List[Dict] = []
        
        # Integration components
        self.meta_learning_layer = None
        self.ensemble_predictor = None
        self.performance_tracker = None
        
        # Files for persistence
        self.metrics_file = "algorithms/meta_learning_metrics.json"
        self.progress_file = "algorithms/meta_learning_progress.json"
        
        self._initialize_components()
        self._start_monitoring()
        
        logger.info("🔄 Real-time learning monitor initialized")
    
    def _initialize_learning_progress(self) -> LearningProgress:
        """Initialize learning progress tracking"""
        return LearningProgress(
            start_time=datetime.now().isoformat(),
            total_predictions=0,
            successful_predictions=0,
            accuracy_improvements=[],
            concept_drifts_detected=0,
            adaptations_performed=0,
            avg_optimization_time=0.0,
            best_model_selections=defaultdict(int),
            error_reduction_rate=0.0
        )
    
    def _initialize_components(self):
        """Initialize integration components"""
        try:
            # Import meta-learning layer
            from algorithms.meta_learning_layer import MetaLearningLayer
            self.meta_learning_layer = MetaLearningLayer()
            
            # Import ensemble predictor
            from algorithms.ensemble import EnsemblePredictor
            self.ensemble_predictor = EnsemblePredictor()
            
            # Import performance tracker
            from model_performance_tracker import ModelPerformanceTracker
            self.performance_tracker = ModelPerformanceTracker()
            
            logger.info("✅ Meta-learning integration components initialized")
            
        except Exception as e:
            logger.warning(f"Some integration components failed to initialize: {e}")
    
    def _start_monitoring(self):
        """Start real-time monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("📊 Real-time monitoring thread started")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                # Check system health
                self._check_system_health()
                
                # Update learning progress
                self._update_learning_progress()
                
                # Detect performance anomalies
                self._detect_performance_anomalies()
                
                # Save metrics periodically
                current_time = time.time()
                if current_time - self.last_save_time > self.save_interval:
                    self._save_metrics()
                    self.last_save_time = current_time
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def record_prediction_performance(self, 
                                    prediction_result: Dict, 
                                    actual_result: Optional[Dict] = None,
                                    match_context: Optional[Dict] = None,
                                    optimization_time: float = 0.0):
        """Record performance metrics for a prediction"""
        try:
            # Extract basic information
            optimization_method = prediction_result.get('algorithm', 'Unknown')
            confidence = prediction_result.get('confidence', 0.0)
            
            # Calculate accuracy if actual result is provided
            accuracy = 0.5  # Default
            success = False
            error_magnitude = 0.0
            
            if actual_result:
                accuracy = self._calculate_accuracy(prediction_result, actual_result)
                success = accuracy > 0.6  # Threshold for success
                error_magnitude = self._calculate_error_magnitude(prediction_result, actual_result)
            
            # Create performance metric
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                context=self._extract_context_key(match_context or {}),
                model_name=optimization_method,
                accuracy=accuracy,
                confidence=confidence,
                prediction_time=optimization_time,
                optimization_method=optimization_method,
                success=success,
                error_magnitude=error_magnitude
            )
            
            # Store metric
            self.performance_metrics.append(metric)
            
            # Update accuracy history
            self.accuracy_history[optimization_method].append(accuracy)
            if len(self.accuracy_history[optimization_method]) > 50:
                self.accuracy_history[optimization_method] = self.accuracy_history[optimization_method][-50:]
            
            # Update learning progress
            self.learning_progress.total_predictions += 1
            if success:
                self.learning_progress.successful_predictions += 1
            
            # Track best model selections
            if success:
                self.learning_progress.best_model_selections[optimization_method] += 1
            
            logger.debug(f"📈 Performance recorded: {optimization_method} - {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error recording prediction performance: {e}")
    
    def _calculate_accuracy(self, prediction: Dict, actual: Dict) -> float:
        """Calculate prediction accuracy"""
        try:
            # For match result prediction
            if 'result' in actual:
                pred_home = prediction.get('home_win_probability', 0)
                pred_draw = prediction.get('draw_probability', 0)
                pred_away = prediction.get('away_win_probability', 0)
                
                actual_result = actual['result']
                
                if actual_result == 'H':
                    return pred_home / 100.0 if pred_home > 0 else 0.0
                elif actual_result == 'D':
                    return pred_draw / 100.0 if pred_draw > 0 else 0.0
                elif actual_result == 'A':
                    return pred_away / 100.0 if pred_away > 0 else 0.0
            
            return 0.5  # Default moderate accuracy
            
        except Exception as e:
            logger.warning(f"Error calculating accuracy: {e}")
            return 0.5
    
    def _calculate_error_magnitude(self, prediction: Dict, actual: Dict) -> float:
        """Calculate prediction error magnitude"""
        try:
            accuracy = self._calculate_accuracy(prediction, actual)
            return 1.0 - accuracy  # Error magnitude is inverse of accuracy
        except:
            return 0.5
    
    def _extract_context_key(self, match_context: Dict) -> str:
        """Extract a context key from match context"""
        try:
            league = match_context.get('league', 'unknown')
            elo_diff = match_context.get('elo_diff', 0)
            
            # Categorize ELO difference
            if abs(elo_diff) > 300:
                strength_diff = 'high_diff'
            elif abs(elo_diff) > 100:
                strength_diff = 'medium_diff'
            else:
                strength_diff = 'low_diff'
            
            return f"{league}_{strength_diff}"
        except:
            return "unknown_context"
    
    def _check_system_health(self):
        """Check overall system health"""
        try:
            health_issues = []
            
            # Check recent performance
            if len(self.performance_metrics) > 10:
                recent_accuracy = [m.accuracy for m in list(self.performance_metrics)[-10:]]
                avg_accuracy = sum(recent_accuracy) / len(recent_accuracy)
                
                if avg_accuracy < 0.4:
                    health_issues.append("Low recent accuracy detected")
            
            # Check meta-learning layer health
            if self.meta_learning_layer:
                try:
                    insights = self.meta_learning_layer.get_learning_insights()
                    if 'error' in insights:
                        health_issues.append("Meta-learning layer error")
                except:
                    health_issues.append("Meta-learning layer unresponsive")
            
            # Check optimization times
            if len(self.performance_metrics) > 5:
                recent_times = [m.prediction_time for m in list(self.performance_metrics)[-5:]]
                avg_time = sum(recent_times) / len(recent_times)
                
                if avg_time > 5.0:  # More than 5 seconds
                    health_issues.append("High optimization times detected")
            
            # Log health issues
            if health_issues:
                logger.warning(f"🏥 System health issues: {health_issues}")
                
                # Create performance alert
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'health_check',
                    'issues': health_issues,
                    'severity': 'warning'
                }
                self.performance_alerts.append(alert)
                
                # Keep only recent alerts
                if len(self.performance_alerts) > 50:
                    self.performance_alerts = self.performance_alerts[-50:]
                    
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    def _update_learning_progress(self):
        """Update learning progress metrics"""
        try:
            if len(self.performance_metrics) < 2:
                return
            
            # Calculate accuracy improvements
            recent_metrics = list(self.performance_metrics)[-20:]  # Last 20 predictions
            older_metrics = list(self.performance_metrics)[-40:-20] if len(self.performance_metrics) >= 40 else []
            
            if older_metrics:
                recent_avg = sum(m.accuracy for m in recent_metrics) / len(recent_metrics)
                older_avg = sum(m.accuracy for m in older_metrics) / len(older_metrics)
                improvement = recent_avg - older_avg
                
                if improvement > 0.01:  # Significant improvement
                    self.learning_progress.accuracy_improvements.append(improvement)
                    if len(self.learning_progress.accuracy_improvements) > 20:
                        self.learning_progress.accuracy_improvements = self.learning_progress.accuracy_improvements[-20:]
            
            # Update average optimization time
            optimization_times = [m.prediction_time for m in recent_metrics if m.prediction_time > 0]
            if optimization_times:
                self.learning_progress.avg_optimization_time = sum(optimization_times) / len(optimization_times)
            
            # Calculate error reduction rate
            if len(self.performance_metrics) >= 10:
                recent_errors = [m.error_magnitude for m in list(self.performance_metrics)[-10:]]
                older_errors = [m.error_magnitude for m in list(self.performance_metrics)[-20:-10]]
                
                if older_errors:
                    recent_error_avg = sum(recent_errors) / len(recent_errors)
                    older_error_avg = sum(older_errors) / len(older_errors)
                    
                    if older_error_avg > 0:
                        reduction_rate = (older_error_avg - recent_error_avg) / older_error_avg
                        self.learning_progress.error_reduction_rate = max(0, reduction_rate)
                        
        except Exception as e:
            logger.error(f"Error updating learning progress: {e}")
    
    def _detect_performance_anomalies(self):
        """Detect performance anomalies and suggest adaptations"""
        try:
            if len(self.performance_metrics) < 20:
                return
            
            recent_metrics = list(self.performance_metrics)[-10:]
            
            # Detect accuracy drops
            accuracies = [m.accuracy for m in recent_metrics]
            avg_accuracy = sum(accuracies) / len(accuracies)
            
            if avg_accuracy < 0.4:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'accuracy_drop',
                    'description': f'Average accuracy dropped to {avg_accuracy:.3f}',
                    'severity': 'high',
                    'suggested_action': 'force_meta_learning_adaptation'
                }
                self.performance_alerts.append(alert)
                
                # Trigger automatic adaptation
                if self.ensemble_predictor and hasattr(self.ensemble_predictor, 'force_meta_learning_adaptation'):
                    try:
                        self.ensemble_predictor.force_meta_learning_adaptation("automatic_anomaly_detection")
                        self.learning_progress.adaptations_performed += 1
                        logger.info("🔄 Automatic adaptation triggered due to accuracy drop")
                    except Exception as e:
                        logger.error(f"Error triggering automatic adaptation: {e}")
            
            # Detect concept drift patterns
            optimization_methods = [m.optimization_method for m in recent_metrics]
            method_counts = {}
            for method in optimization_methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            
            # If meta-learning is not being used much, suggest investigation
            meta_learning_usage = method_counts.get('🧠 Meta-Learning Akıllı Seçim', 0)
            if meta_learning_usage < len(recent_metrics) * 0.5:  # Less than 50% usage
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'low_meta_learning_usage',
                    'description': f'Meta-learning used in only {meta_learning_usage}/{len(recent_metrics)} predictions',
                    'severity': 'medium',
                    'suggested_action': 'check_meta_learning_integration'
                }
                self.performance_alerts.append(alert)
                
        except Exception as e:
            logger.error(f"Error detecting performance anomalies: {e}")
    
    def _save_metrics(self):
        """Save performance metrics to file"""
        try:
            # Save performance metrics
            metrics_data = [asdict(metric) for metric in self.performance_metrics]
            
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            # Save learning progress
            progress_data = asdict(self.learning_progress)
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            
            logger.debug("💾 Meta-learning metrics saved")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            if not self.performance_metrics:
                return {'error': 'No performance data available'}
            
            recent_metrics = list(self.performance_metrics)[-20:]
            
            # Overall statistics
            total_predictions = len(self.performance_metrics)
            recent_accuracy = sum(m.accuracy for m in recent_metrics) / len(recent_metrics)
            success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
            avg_optimization_time = sum(m.prediction_time for m in recent_metrics) / len(recent_metrics)
            
            # Method performance
            method_performance = defaultdict(list)
            for metric in recent_metrics:
                method_performance[metric.optimization_method].append(metric.accuracy)
            
            method_averages = {
                method: sum(accuracies) / len(accuracies)
                for method, accuracies in method_performance.items()
            }
            
            # Learning progress summary
            progress_summary = asdict(self.learning_progress)
            
            report = {
                'summary': {
                    'total_predictions': total_predictions,
                    'recent_accuracy': recent_accuracy,
                    'success_rate': success_rate,
                    'avg_optimization_time': avg_optimization_time,
                    'monitoring_duration': self._calculate_monitoring_duration(),
                },
                'method_performance': method_averages,
                'learning_progress': progress_summary,
                'recent_alerts': self.performance_alerts[-10:],  # Last 10 alerts
                'accuracy_trends': {
                    method: accuracies[-10:] for method, accuracies in self.accuracy_history.items()
                },
                'recommendations': self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _calculate_monitoring_duration(self) -> str:
        """Calculate how long monitoring has been active"""
        try:
            start_time = datetime.fromisoformat(self.learning_progress.start_time)
            duration = datetime.now() - start_time
            
            hours = duration.total_seconds() / 3600
            if hours < 1:
                return f"{duration.total_seconds() / 60:.1f} minutes"
            elif hours < 24:
                return f"{hours:.1f} hours"
            else:
                return f"{hours / 24:.1f} days"
                
        except:
            return "unknown"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        try:
            if not self.performance_metrics:
                return recommendations
            
            recent_metrics = list(self.performance_metrics)[-20:]
            
            # Accuracy-based recommendations
            recent_accuracy = sum(m.accuracy for m in recent_metrics) / len(recent_metrics)
            
            if recent_accuracy < 0.5:
                recommendations.append("Consider reviewing model parameters - accuracy is below 50%")
            elif recent_accuracy < 0.6:
                recommendations.append("Accuracy could be improved - consider meta-learning adaptation")
            
            # Optimization time recommendations
            avg_time = sum(m.prediction_time for m in recent_metrics) / len(recent_metrics)
            if avg_time > 3.0:
                recommendations.append("High optimization times detected - consider performance optimization")
            
            # Method usage recommendations
            method_counts = defaultdict(int)
            for metric in recent_metrics:
                method_counts[metric.optimization_method] += 1
            
            meta_learning_usage = method_counts.get('🧠 Meta-Learning Akıllı Seçim', 0)
            total_predictions = len(recent_metrics)
            
            if meta_learning_usage < total_predictions * 0.3:
                recommendations.append("Low meta-learning usage - check integration and enable meta-learning")
            
            # Error pattern recommendations
            error_rates = [m.error_magnitude for m in recent_metrics]
            if error_rates and sum(error_rates) / len(error_rates) > 0.6:
                recommendations.append("High error rates detected - analyze error patterns and adjust models")
            
            # Learning progress recommendations
            if self.learning_progress.accuracy_improvements:
                avg_improvement = sum(self.learning_progress.accuracy_improvements) / len(self.learning_progress.accuracy_improvements)
                if avg_improvement < 0.01:
                    recommendations.append("Learning improvements are stagnating - consider concept drift detection")
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations - check system logs")
        
        return recommendations
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_enabled = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        # Save final metrics
        self._save_metrics()
        logger.info("🛑 Real-time monitoring stopped")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_monitoring()
        except:
            pass  # Ignore cleanup errors

# Global instance for easy access
real_time_monitor = None

def get_real_time_monitor() -> RealTimeLearningMonitor:
    """Get or create the global real-time monitor instance"""
    global real_time_monitor
    
    if real_time_monitor is None:
        real_time_monitor = RealTimeLearningMonitor()
    
    return real_time_monitor

def initialize_meta_learning_integration():
    """Initialize meta-learning integration components"""
    monitor = get_real_time_monitor()
    logger.info("🚀 Meta-learning integration initialized")
    return monitor

# Example usage and testing
if __name__ == "__main__":
    # Initialize monitoring
    monitor = initialize_meta_learning_integration()
    
    # Example prediction performance recording
    sample_prediction = {
        'home_win_probability': 45.5,
        'draw_probability': 30.2,
        'away_win_probability': 24.3,
        'confidence': 72.5,
        'algorithm': '🧠 Meta-Learning Akıllı Seçim'
    }
    
    sample_actual = {
        'result': 'H'  # Home win
    }
    
    sample_context = {
        'league': 'Premier League',
        'elo_diff': 150
    }
    
    # Record performance
    monitor.record_prediction_performance(
        prediction_result=sample_prediction,
        actual_result=sample_actual,
        match_context=sample_context,
        optimization_time=1.2
    )
    
    # Get performance report
    report = monitor.get_performance_report()
    print("📊 Performance Report:")
    print(f"Recent accuracy: {report['summary']['recent_accuracy']:.3f}")
    print(f"Success rate: {report['summary']['success_rate']:.3f}")
    print(f"Recommendations: {report['recommendations']}")