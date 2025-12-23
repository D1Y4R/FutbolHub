"""
Advanced Genetic Algorithm Engine for Ensemble Weight Optimization
Implements sophisticated genetic algorithm with multi-objective optimization for football prediction ensembles.

Features:
- Population-based weight optimization
- Multi-objective fitness function (accuracy + diversity + efficiency)
- Adaptive parameter tuning
- Elitism strategy
- Real-time performance adaptation
- Context-aware weight evolution
- Advanced meta-learning capabilities
"""

import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os
from concurrent.futures import ThreadPoolExecutor
import math
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Optimization objectives for multi-objective GA"""
    ACCURACY = "accuracy"
    DIVERSITY = "diversity"
    EFFICIENCY = "efficiency"
    STABILITY = "stability"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"

@dataclass
class Individual:
    """Individual in the GA population representing a weight configuration"""
    weights: Dict[str, float]
    fitness_scores: Dict[OptimizationObjective, float]
    overall_fitness: float
    age: int = 0
    performance_history: List[float] = None
    context_performance: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []
        if self.context_performance is None:
            self.context_performance = {}

@dataclass
class EvolutionConfig:
    """Configuration for genetic algorithm evolution"""
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    max_generations: int = 100
    convergence_threshold: float = 0.001
    diversity_threshold: float = 0.05
    tournament_size: int = 3
    adaptive_parameters: bool = True
    multi_objective_weights: Dict[OptimizationObjective, float] = None
    
    def __post_init__(self):
        if self.multi_objective_weights is None:
            self.multi_objective_weights = {
                OptimizationObjective.ACCURACY: 0.4,
                OptimizationObjective.DIVERSITY: 0.25,
                OptimizationObjective.EFFICIENCY: 0.15,
                OptimizationObjective.STABILITY: 0.15,
                OptimizationObjective.RISK_ADJUSTED_RETURN: 0.05
            }

class GeneticEnsembleOptimizer:
    """
    Advanced Genetic Algorithm Engine for Ensemble Weight Optimization
    
    This class implements a sophisticated genetic algorithm that optimizes model weights
    for ensemble predictions using multi-objective optimization principles.
    """
    
    def __init__(self, config: EvolutionConfig = None):
        """Initialize the genetic optimizer"""
        self.config = config or EvolutionConfig()
        
        # Available models for weight optimization
        self.model_names = [
            'poisson', 'dixon_coles', 'xgboost', 'monte_carlo', 
            'crf', 'neural_network'
        ]
        
        # Population management
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.evolution_history: List[Dict] = []
        
        # Performance tracking
        self.performance_tracker = None
        self._initialize_performance_tracker()
        
        # Context-aware optimization
        self.context_populations: Dict[str, List[Individual]] = {}
        self.context_performance: Dict[str, Dict] = defaultdict(dict)
        
        # Adaptive parameter management
        self.adaptive_state = {
            'mutation_rate': self.config.mutation_rate,
            'crossover_rate': self.config.crossover_rate,
            'stagnation_counter': 0,
            'diversity_history': deque(maxlen=10),
            'fitness_history': deque(maxlen=10)
        }
        
        # Meta-learning components
        self.meta_learner = MetaLearningEngine()
        self.pattern_recognizer = PatternRecognizer()
        
        # File paths for persistence
        self.save_path = "algorithms/genetic_optimizer_state.json"
        self.history_path = "algorithms/genetic_evolution_history.json"
        
        logger.info("GeneticEnsembleOptimizer initialized with advanced features")
    
    def _initialize_performance_tracker(self):
        """Initialize performance tracking system"""
        try:
            from model_performance_tracker import ModelPerformanceTracker
            self.performance_tracker = ModelPerformanceTracker()
        except ImportError:
            logger.warning("ModelPerformanceTracker not available, using mock tracker")
            self.performance_tracker = MockPerformanceTracker()
    
    def optimize_weights(self, 
                        match_contexts: List[Dict], 
                        performance_data: Dict = None,
                        target_context: str = None) -> Dict[str, float]:
        """
        Main optimization function using genetic algorithm
        
        Args:
            match_contexts: List of match contexts for evaluation
            performance_data: Historical performance data
            target_context: Specific context to optimize for (e.g., 'premier_league', 'derby_matches')
            
        Returns:
            Dict[str, float]: Optimized weights for models
        """
        logger.info(f"Starting genetic optimization for {len(match_contexts)} contexts")
        
        # Initialize population if needed
        if not self.population or target_context:
            self._initialize_population(target_context)
        
        # Evolution loop
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Evaluate fitness for entire population
            self._evaluate_population_fitness(match_contexts, performance_data)
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Converged at generation {generation}")
                break
            
            # Evolve population
            new_population = self._evolve_population()
            self.population = new_population
            
            # Adaptive parameter adjustment
            if self.config.adaptive_parameters:
                self._adapt_parameters()
            
            # Log progress
            if generation % 10 == 0:
                self._log_generation_progress(generation)
        
        # Extract best weights
        best_weights = self._extract_best_weights()
        
        # Save evolution state
        self._save_evolution_state()
        
        logger.info("Genetic optimization completed")
        return best_weights
    
    def _initialize_population(self, context: str = None):
        """Initialize the GA population with diverse weight configurations"""
        logger.info(f"Initializing population of size {self.config.population_size}")
        
        self.population = []
        
        # Strategy 1: Uniform random weights (20% of population)
        uniform_count = int(0.2 * self.config.population_size)
        for _ in range(uniform_count):
            weights = self._generate_random_weights()
            individual = Individual(
                weights=weights,
                fitness_scores={obj: 0.0 for obj in OptimizationObjective},
                overall_fitness=0.0
            )
            self.population.append(individual)
        
        # Strategy 2: Expert-guided initialization (30% of population)
        expert_count = int(0.3 * self.config.population_size)
        expert_strategies = self._get_expert_weight_strategies()
        for i in range(expert_count):
            strategy = expert_strategies[i % len(expert_strategies)]
            weights = self._apply_noise_to_weights(strategy, noise_level=0.1)
            individual = Individual(
                weights=weights,
                fitness_scores={obj: 0.0 for obj in OptimizationObjective},
                overall_fitness=0.0
            )
            self.population.append(individual)
        
        # Strategy 3: Context-specific initialization (30% of population)
        if context and context in self.context_populations:
            context_count = int(0.3 * self.config.population_size)
            context_individuals = self.context_populations[context][:context_count]
            for individual in context_individuals:
                # Create new individual based on context performance
                new_weights = self._mutate_weights(individual.weights, mutation_rate=0.05)
                new_individual = Individual(
                    weights=new_weights,
                    fitness_scores={obj: 0.0 for obj in OptimizationObjective},
                    overall_fitness=0.0
                )
                self.population.append(new_individual)
        
        # Strategy 4: Performance-guided initialization (20% of population)
        remaining_count = self.config.population_size - len(self.population)
        for _ in range(remaining_count):
            weights = self._generate_performance_guided_weights()
            individual = Individual(
                weights=weights,
                fitness_scores={obj: 0.0 for obj in OptimizationObjective},
                overall_fitness=0.0
            )
            self.population.append(individual)
        
        logger.info(f"Population initialized with {len(self.population)} individuals")
    
    def _generate_random_weights(self) -> Dict[str, float]:
        """Generate random weight configuration"""
        weights = {}
        raw_weights = [random.random() for _ in self.model_names]
        total = sum(raw_weights)
        
        for i, model in enumerate(self.model_names):
            weights[model] = raw_weights[i] / total
        
        return weights
    
    def _get_expert_weight_strategies(self) -> List[Dict[str, float]]:
        """Get expert-defined weight strategies"""
        strategies = []
        
        # Strategy 1: Balanced approach
        balanced = {model: 1.0/len(self.model_names) for model in self.model_names}
        strategies.append(balanced)
        
        # Strategy 2: Statistical model focused
        statistical_focused = {
            'poisson': 0.3, 'dixon_coles': 0.25, 'xgboost': 0.15,
            'monte_carlo': 0.15, 'crf': 0.075, 'neural_network': 0.075
        }
        strategies.append(statistical_focused)
        
        # Strategy 3: ML model focused
        ml_focused = {
            'poisson': 0.15, 'dixon_coles': 0.1, 'xgboost': 0.3,
            'monte_carlo': 0.15, 'crf': 0.15, 'neural_network': 0.15
        }
        strategies.append(ml_focused)
        
        # Strategy 4: Conservative approach (favor proven models)
        conservative = {
            'poisson': 0.35, 'dixon_coles': 0.3, 'xgboost': 0.1,
            'monte_carlo': 0.1, 'crf': 0.075, 'neural_network': 0.075
        }
        strategies.append(conservative)
        
        # Strategy 5: Aggressive approach (favor advanced models)
        aggressive = {
            'poisson': 0.1, 'dixon_coles': 0.1, 'xgboost': 0.25,
            'monte_carlo': 0.2, 'crf': 0.175, 'neural_network': 0.175
        }
        strategies.append(aggressive)
        
        return strategies
    
    def _generate_performance_guided_weights(self) -> Dict[str, float]:
        """Generate weights based on historical performance"""
        weights = {}
        
        if not self.performance_tracker:
            return self._generate_random_weights()
        
        # Get performance scores for each model
        performance_scores = {}
        for model in self.model_names:
            performance = self.performance_tracker.get_model_performance(model)
            if performance and performance.get('overall', {}).get('accuracy', 0) > 0:
                performance_scores[model] = performance['overall']['accuracy'] / 100.0
            else:
                performance_scores[model] = 0.5  # Default neutral score
        
        # Convert performance to weights with some randomization
        total_performance = sum(performance_scores.values())
        if total_performance > 0:
            for model in self.model_names:
                base_weight = performance_scores[model] / total_performance
                # Add randomization (±20%)
                noise = random.uniform(-0.2, 0.2) * base_weight
                weights[model] = max(0.01, base_weight + noise)
        else:
            return self._generate_random_weights()
        
        # Normalize
        total = sum(weights.values())
        for model in weights:
            weights[model] /= total
        
        return weights
    
    def _evaluate_population_fitness(self, match_contexts: List[Dict], performance_data: Dict = None):
        """Evaluate fitness for entire population using multi-objective approach"""
        logger.debug(f"Evaluating fitness for {len(self.population)} individuals")
        
        # Parallel fitness evaluation for efficiency
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for individual in self.population:
                future = executor.submit(
                    self._evaluate_individual_fitness, 
                    individual, 
                    match_contexts, 
                    performance_data
                )
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                fitness_scores = future.result()
                self.population[i].fitness_scores = fitness_scores
                self.population[i].overall_fitness = self._calculate_overall_fitness(fitness_scores)
        
        # Update best individual
        self._update_best_individual()
        
        # Update diversity metrics
        self._update_diversity_metrics()
    
    def _evaluate_individual_fitness(self, 
                                   individual: Individual, 
                                   match_contexts: List[Dict], 
                                   performance_data: Dict = None) -> Dict[OptimizationObjective, float]:
        """Evaluate fitness for a single individual"""
        fitness_scores = {}
        
        # Accuracy objective
        fitness_scores[OptimizationObjective.ACCURACY] = self._evaluate_accuracy_fitness(
            individual.weights, match_contexts, performance_data
        )
        
        # Diversity objective
        fitness_scores[OptimizationObjective.DIVERSITY] = self._evaluate_diversity_fitness(
            individual.weights
        )
        
        # Efficiency objective
        fitness_scores[OptimizationObjective.EFFICIENCY] = self._evaluate_efficiency_fitness(
            individual.weights
        )
        
        # Stability objective
        fitness_scores[OptimizationObjective.STABILITY] = self._evaluate_stability_fitness(
            individual, match_contexts
        )
        
        # Risk-adjusted return objective
        fitness_scores[OptimizationObjective.RISK_ADJUSTED_RETURN] = self._evaluate_risk_adjusted_fitness(
            individual.weights, match_contexts, performance_data
        )
        
        return fitness_scores
    
    def _evaluate_accuracy_fitness(self, 
                                 weights: Dict[str, float], 
                                 match_contexts: List[Dict], 
                                 performance_data: Dict = None) -> float:
        """Evaluate accuracy-based fitness"""
        if not self.performance_tracker:
            return 0.5
        
        total_accuracy = 0.0
        valid_contexts = 0
        
        for context in match_contexts:
            context_accuracy = 0.0
            
            # Calculate weighted accuracy based on individual model performance
            for model_name, weight in weights.items():
                model_performance = self.performance_tracker.get_model_performance(model_name)
                if model_performance:
                    model_accuracy = model_performance.get('overall', {}).get('accuracy', 50.0)
                    context_accuracy += weight * (model_accuracy / 100.0)
            
            # Context-specific adjustments
            league = context.get('league', '')
            match_type = context.get('match_type', 'balanced')
            
            # Apply context-specific performance modifications
            context_modifier = self._get_context_accuracy_modifier(league, match_type, weights)
            context_accuracy *= context_modifier
            
            total_accuracy += context_accuracy
            valid_contexts += 1
        
        if valid_contexts == 0:
            return 0.5
        
        average_accuracy = total_accuracy / valid_contexts
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, average_accuracy))
    
    def _evaluate_diversity_fitness(self, weights: Dict[str, float]) -> float:
        """Evaluate diversity fitness (entropy-based)"""
        # Calculate entropy of weight distribution
        entropy = 0.0
        for weight in weights.values():
            if weight > 0:
                entropy -= weight * math.log(weight)
        
        # Normalize entropy (max entropy occurs when all weights are equal)
        max_entropy = math.log(len(weights))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Penalize extreme weight concentrations
        max_weight = max(weights.values())
        concentration_penalty = max_weight ** 2  # Quadratic penalty for concentration
        
        diversity_score = normalized_entropy * (1 - concentration_penalty)
        return max(0.0, min(1.0, diversity_score))
    
    def _evaluate_efficiency_fitness(self, weights: Dict[str, float]) -> float:
        """Evaluate computational efficiency fitness"""
        # Model complexity weights (relative computational cost)
        complexity_weights = {
            'poisson': 0.1,
            'dixon_coles': 0.15,
            'xgboost': 0.8,
            'monte_carlo': 1.0,
            'crf': 0.6,
            'neural_network': 0.9
        }
        
        # Calculate weighted complexity
        total_complexity = sum(weights[model] * complexity_weights.get(model, 0.5) 
                              for model in weights)
        
        # Efficiency is inverse of complexity (lower complexity = higher efficiency)
        max_complexity = max(complexity_weights.values())
        efficiency_score = 1.0 - (total_complexity / max_complexity)
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _evaluate_stability_fitness(self, 
                                  individual: Individual, 
                                  match_contexts: List[Dict]) -> float:
        """Evaluate stability fitness based on performance consistency"""
        if len(individual.performance_history) < 3:
            return 0.5  # Neutral score for new individuals
        
        # Calculate coefficient of variation (CV) for stability
        performance_array = np.array(individual.performance_history[-10:])  # Last 10 performances
        
        if len(performance_array) == 0 or np.std(performance_array) == 0:
            return 0.5
        
        cv = np.std(performance_array) / np.mean(performance_array)
        
        # Lower CV means higher stability
        stability_score = 1.0 / (1.0 + cv)
        
        return max(0.0, min(1.0, stability_score))
    
    def _evaluate_risk_adjusted_fitness(self, 
                                      weights: Dict[str, float], 
                                      match_contexts: List[Dict], 
                                      performance_data: Dict = None) -> float:
        """Evaluate risk-adjusted return fitness"""
        if not performance_data:
            return 0.5
        
        # Calculate Sharpe ratio-like metric for ensemble
        returns = []
        
        for context in match_contexts:
            # Simulate return based on prediction accuracy
            context_return = 0.0
            
            for model_name, weight in weights.items():
                model_performance = performance_data.get(model_name, {})
                accuracy = model_performance.get('accuracy', 0.5)
                
                # Convert accuracy to return (simplified)
                model_return = (accuracy - 0.5) * 2  # Maps 0.5-1.0 accuracy to 0-1 return
                context_return += weight * model_return
            
            returns.append(context_return)
        
        if len(returns) < 2:
            return 0.5
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.5
        
        # Sharpe-like ratio
        risk_adjusted_return = mean_return / std_return
        
        # Normalize to 0-1 range
        normalized_score = 1.0 / (1.0 + np.exp(-risk_adjusted_return))
        
        return max(0.0, min(1.0, normalized_score))
    
    def _calculate_overall_fitness(self, fitness_scores: Dict[OptimizationObjective, float]) -> float:
        """Calculate overall fitness using weighted combination of objectives"""
        overall_fitness = 0.0
        
        for objective, score in fitness_scores.items():
            weight = self.config.multi_objective_weights.get(objective, 0.0)
            overall_fitness += weight * score
        
        return overall_fitness
    
    def _get_context_accuracy_modifier(self, 
                                     league: str, 
                                     match_type: str, 
                                     weights: Dict[str, float]) -> float:
        """Get context-specific accuracy modifier"""
        modifier = 1.0
        
        # League-specific adjustments
        if 'premier_league' in league.lower():
            # Premier League: favor ML models
            modifier += 0.1 * (weights.get('xgboost', 0) + weights.get('neural_network', 0))
        elif 'serie_a' in league.lower():
            # Serie A: favor defensive models
            modifier += 0.1 * weights.get('dixon_coles', 0)
        
        # Match type adjustments
        if match_type == 'derby':
            # Derby matches: favor uncertainty models
            modifier += 0.1 * weights.get('monte_carlo', 0)
        elif match_type == 'heavy_favorite':
            # Clear favorites: favor statistical models
            modifier += 0.1 * (weights.get('poisson', 0) + weights.get('dixon_coles', 0))
        
        return modifier
    
    def _evolve_population(self) -> List[Individual]:
        """Evolve the population using genetic operators"""
        new_population = []
        
        # Elitism: Keep best individuals
        elite_individuals = self._select_elite()
        new_population.extend(elite_individuals)
        
        # Generate offspring to fill remaining population
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.adaptive_state['crossover_rate']:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            if random.random() < self.adaptive_state['mutation_rate']:
                child1 = self._mutate(child1)
            if random.random() < self.adaptive_state['mutation_rate']:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.config.population_size]
    
    def _select_elite(self) -> List[Individual]:
        """Select elite individuals for next generation"""
        # Sort by fitness and select top individuals
        sorted_population = sorted(
            self.population, 
            key=lambda x: x.overall_fitness, 
            reverse=True
        )
        
        elite = sorted_population[:self.config.elite_size]
        
        # Age the elite individuals
        for individual in elite:
            individual.age += 1
        
        return elite
    
    def _tournament_selection(self) -> Individual:
        """Tournament selection for parent selection"""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda x: x.overall_fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Arithmetic crossover for weight combinations"""
        alpha = random.random()  # Crossover ratio
        
        child1_weights = {}
        child2_weights = {}
        
        for model in self.model_names:
            # Arithmetic crossover
            weight1 = parent1.weights[model]
            weight2 = parent2.weights[model]
            
            child1_weights[model] = alpha * weight1 + (1 - alpha) * weight2
            child2_weights[model] = (1 - alpha) * weight1 + alpha * weight2
        
        # Normalize weights
        child1_weights = self._normalize_weights(child1_weights)
        child2_weights = self._normalize_weights(child2_weights)
        
        # Create new individuals
        child1 = Individual(
            weights=child1_weights,
            fitness_scores={obj: 0.0 for obj in OptimizationObjective},
            overall_fitness=0.0
        )
        
        child2 = Individual(
            weights=child2_weights,
            fitness_scores={obj: 0.0 for obj in OptimizationObjective},
            overall_fitness=0.0
        )
        
        return child1, child2
    
    def _mutate(self, individual: Individual) -> Individual:
        """Gaussian mutation for weight adjustment"""
        mutated_weights = {}
        
        # Standard deviation for mutation (adaptive)
        std_dev = 0.05 * (1.0 + self.adaptive_state['stagnation_counter'] / 10.0)
        
        for model in self.model_names:
            original_weight = individual.weights[model]
            
            # Gaussian mutation
            mutation = np.random.normal(0, std_dev)
            mutated_weight = original_weight + mutation
            
            # Ensure positive weights
            mutated_weights[model] = max(0.001, mutated_weight)
        
        # Normalize weights
        mutated_weights = self._normalize_weights(mutated_weights)
        
        # Create new individual
        mutated_individual = Individual(
            weights=mutated_weights,
            fitness_scores={obj: 0.0 for obj in OptimizationObjective},
            overall_fitness=0.0
        )
        
        return mutated_individual
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0"""
        total = sum(weights.values())
        if total > 0:
            return {model: weight / total for model, weight in weights.items()}
        else:
            # Equal weights if all are zero
            return {model: 1.0 / len(weights) for model in weights}
    
    def _mutate_weights(self, weights: Dict[str, float], mutation_rate: float = 0.1) -> Dict[str, float]:
        """Apply mutation to weights"""
        mutated = weights.copy()
        
        for model in mutated:
            if random.random() < mutation_rate:
                # Gaussian noise
                noise = np.random.normal(0, 0.05)
                mutated[model] = max(0.001, mutated[model] + noise)
        
        return self._normalize_weights(mutated)
    
    def _apply_noise_to_weights(self, weights: Dict[str, float], noise_level: float = 0.1) -> Dict[str, float]:
        """Apply noise to weight configuration"""
        noisy_weights = {}
        
        for model, weight in weights.items():
            noise = np.random.normal(0, noise_level * weight)
            noisy_weights[model] = max(0.001, weight + noise)
        
        return self._normalize_weights(noisy_weights)
    
    def _check_convergence(self) -> bool:
        """Check if the algorithm has converged"""
        if len(self.evolution_history) < 5:
            return False
        
        # Check fitness improvement over last 5 generations
        recent_fitness = [gen['best_fitness'] for gen in self.evolution_history[-5:]]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        if fitness_improvement < self.config.convergence_threshold:
            self.adaptive_state['stagnation_counter'] += 1
            if self.adaptive_state['stagnation_counter'] > 5:
                return True
        else:
            self.adaptive_state['stagnation_counter'] = 0
        
        return False
    
    def _adapt_parameters(self):
        """Adapt genetic algorithm parameters based on evolution progress"""
        # Adapt mutation rate based on diversity
        current_diversity = self._calculate_population_diversity()
        self.adaptive_state['diversity_history'].append(current_diversity)
        
        if len(self.adaptive_state['diversity_history']) > 3:
            avg_diversity = np.mean(list(self.adaptive_state['diversity_history']))
            
            if avg_diversity < self.config.diversity_threshold:
                # Low diversity: increase mutation
                self.adaptive_state['mutation_rate'] = min(0.3, 
                    self.adaptive_state['mutation_rate'] * 1.1)
            else:
                # High diversity: decrease mutation
                self.adaptive_state['mutation_rate'] = max(0.01, 
                    self.adaptive_state['mutation_rate'] * 0.95)
        
        # Adapt crossover rate based on fitness progress
        if self.adaptive_state['stagnation_counter'] > 3:
            # Stagnation: increase crossover to explore more
            self.adaptive_state['crossover_rate'] = min(0.95, 
                self.adaptive_state['crossover_rate'] * 1.05)
        
        logger.debug(f"Adapted parameters: mutation_rate={self.adaptive_state['mutation_rate']:.3f}, "
                    f"crossover_rate={self.adaptive_state['crossover_rate']:.3f}")
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity metric"""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate pairwise distances between individuals
        distances = []
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._calculate_individual_distance(
                    self.population[i], 
                    self.population[j]
                )
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_individual_distance(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate distance between two individuals"""
        # Euclidean distance in weight space
        distance = 0.0
        
        for model in self.model_names:
            diff = ind1.weights[model] - ind2.weights[model]
            distance += diff ** 2
        
        return math.sqrt(distance)
    
    def _update_best_individual(self):
        """Update the best individual in the population"""
        current_best = max(self.population, key=lambda x: x.overall_fitness)
        
        if self.best_individual is None or current_best.overall_fitness > self.best_individual.overall_fitness:
            self.best_individual = current_best
            logger.debug(f"New best individual found with fitness: {current_best.overall_fitness:.4f}")
    
    def _update_diversity_metrics(self):
        """Update diversity tracking metrics"""
        diversity = self._calculate_population_diversity()
        self.adaptive_state['diversity_history'].append(diversity)
        
        # Track fitness history for best individual
        if self.best_individual:
            self.adaptive_state['fitness_history'].append(self.best_individual.overall_fitness)
    
    def _extract_best_weights(self) -> Dict[str, float]:
        """Extract the best weight configuration"""
        if self.best_individual:
            return self.best_individual.weights.copy()
        else:
            # Fallback to equal weights
            return {model: 1.0 / len(self.model_names) for model in self.model_names}
    
    def _log_generation_progress(self, generation: int):
        """Log progress of current generation"""
        if not self.population:
            return
        
        best_fitness = max(ind.overall_fitness for ind in self.population)
        avg_fitness = np.mean([ind.overall_fitness for ind in self.population])
        diversity = self._calculate_population_diversity()
        
        logger.info(f"Generation {generation}: Best={best_fitness:.4f}, "
                   f"Avg={avg_fitness:.4f}, Diversity={diversity:.4f}")
        
        # Store evolution history
        generation_stats = {
            'generation': generation,
            'best_fitness': best_fitness,
            'average_fitness': avg_fitness,
            'diversity': diversity,
            'mutation_rate': self.adaptive_state['mutation_rate'],
            'crossover_rate': self.adaptive_state['crossover_rate'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.evolution_history.append(generation_stats)
    
    def _save_evolution_state(self):
        """Save the current evolution state"""
        try:
            state_data = {
                'generation': self.generation,
                'best_individual': {
                    'weights': self.best_individual.weights if self.best_individual else None,
                    'fitness': self.best_individual.overall_fitness if self.best_individual else 0.0
                },
                'adaptive_state': {
                    'mutation_rate': self.adaptive_state['mutation_rate'],
                    'crossover_rate': self.adaptive_state['crossover_rate'],
                    'stagnation_counter': self.adaptive_state['stagnation_counter']
                },
                'config': {
                    'population_size': self.config.population_size,
                    'elite_size': self.config.elite_size,
                    'max_generations': self.config.max_generations
                },
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Save evolution history
            with open(self.history_path, 'w') as f:
                json.dump(self.evolution_history, f, indent=2)
                
            logger.info("Evolution state saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save evolution state: {e}")
    
    def load_evolution_state(self) -> bool:
        """Load previous evolution state"""
        try:
            if not os.path.exists(self.save_path):
                return False
            
            with open(self.save_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore state
            self.generation = state_data.get('generation', 0)
            
            best_data = state_data.get('best_individual', {})
            if best_data.get('weights'):
                self.best_individual = Individual(
                    weights=best_data['weights'],
                    fitness_scores={obj: 0.0 for obj in OptimizationObjective},
                    overall_fitness=best_data.get('fitness', 0.0)
                )
            
            adaptive_data = state_data.get('adaptive_state', {})
            self.adaptive_state.update(adaptive_data)
            
            # Load evolution history
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    self.evolution_history = json.load(f)
            
            logger.info("Evolution state loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load evolution state: {e}")
            return False
    
    def get_evolution_analysis(self) -> Dict:
        """Get comprehensive analysis of evolution process"""
        if not self.evolution_history:
            return {}
        
        analysis = {
            'total_generations': len(self.evolution_history),
            'final_fitness': self.evolution_history[-1]['best_fitness'],
            'fitness_improvement': (
                self.evolution_history[-1]['best_fitness'] - 
                self.evolution_history[0]['best_fitness']
            ),
            'convergence_generation': self._find_convergence_generation(),
            'diversity_trend': self._analyze_diversity_trend(),
            'parameter_adaptation': self._analyze_parameter_adaptation(),
            'best_weights': self.best_individual.weights if self.best_individual else None,
            'fitness_breakdown': (
                self.best_individual.fitness_scores if self.best_individual else None
            )
        }
        
        return analysis
    
    def _find_convergence_generation(self) -> Optional[int]:
        """Find the generation where convergence occurred"""
        if len(self.evolution_history) < 10:
            return None
        
        for i in range(5, len(self.evolution_history)):
            recent_fitness = [
                gen['best_fitness'] 
                for gen in self.evolution_history[i-5:i]
            ]
            
            if max(recent_fitness) - min(recent_fitness) < self.config.convergence_threshold:
                return i
        
        return None
    
    def _analyze_diversity_trend(self) -> Dict:
        """Analyze diversity trends during evolution"""
        diversities = [gen.get('diversity', 0) for gen in self.evolution_history]
        
        if not diversities:
            return {}
        
        return {
            'initial_diversity': diversities[0],
            'final_diversity': diversities[-1],
            'max_diversity': max(diversities),
            'min_diversity': min(diversities),
            'diversity_trend': 'increasing' if diversities[-1] > diversities[0] else 'decreasing'
        }
    
    def _analyze_parameter_adaptation(self) -> Dict:
        """Analyze how parameters adapted during evolution"""
        mutation_rates = [gen.get('mutation_rate', 0) for gen in self.evolution_history]
        crossover_rates = [gen.get('crossover_rate', 0) for gen in self.evolution_history]
        
        return {
            'mutation_rate_change': (
                mutation_rates[-1] - mutation_rates[0] if mutation_rates else 0
            ),
            'crossover_rate_change': (
                crossover_rates[-1] - crossover_rates[0] if crossover_rates else 0
            ),
            'parameter_stability': np.std(mutation_rates) if mutation_rates else 0
        }


class MetaLearningEngine:
    """Advanced meta-learning engine for pattern recognition in ensemble optimization"""
    
    def __init__(self):
        self.pattern_db = defaultdict(list)
        self.context_patterns = defaultdict(dict)
        
    def learn_from_optimization(self, 
                              context: Dict, 
                              optimization_result: Dict, 
                              performance: float):
        """Learn patterns from optimization results"""
        pattern_key = self._extract_pattern_key(context)
        
        pattern_entry = {
            'context': context,
            'weights': optimization_result,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        }
        
        self.pattern_db[pattern_key].append(pattern_entry)
        self._update_context_patterns(pattern_key, pattern_entry)
    
    def _extract_pattern_key(self, context: Dict) -> str:
        """Extract pattern key from context"""
        league = context.get('league', 'unknown')
        match_type = context.get('match_type', 'balanced')
        return f"{league}_{match_type}"
    
    def _update_context_patterns(self, pattern_key: str, pattern_entry: Dict):
        """Update context-specific patterns"""
        if pattern_key not in self.context_patterns:
            self.context_patterns[pattern_key] = {
                'count': 0,
                'avg_performance': 0.0,
                'best_weights': None,
                'best_performance': 0.0
            }
        
        patterns = self.context_patterns[pattern_key]
        patterns['count'] += 1
        
        # Update average performance
        current_avg = patterns['avg_performance']
        new_performance = pattern_entry['performance']
        patterns['avg_performance'] = (
            (current_avg * (patterns['count'] - 1) + new_performance) / patterns['count']
        )
        
        # Update best performance
        if new_performance > patterns['best_performance']:
            patterns['best_performance'] = new_performance
            patterns['best_weights'] = pattern_entry['weights']


class PatternRecognizer:
    """Pattern recognition system for ensemble optimization"""
    
    def __init__(self):
        self.failure_patterns = defaultdict(list)
        self.success_patterns = defaultdict(list)
        
    def analyze_failure(self, 
                       context: Dict, 
                       weights: Dict[str, float], 
                       expected_performance: float, 
                       actual_performance: float):
        """Analyze failure patterns"""
        if actual_performance < expected_performance * 0.8:  # 20% worse than expected
            failure_pattern = {
                'context': context,
                'weights': weights,
                'performance_gap': expected_performance - actual_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            pattern_key = self._get_failure_pattern_key(context, weights)
            self.failure_patterns[pattern_key].append(failure_pattern)
    
    def _get_failure_pattern_key(self, context: Dict, weights: Dict[str, float]) -> str:
        """Generate failure pattern key"""
        # Identify dominant model
        dominant_model = max(weights, key=weights.get)
        league_type = context.get('league_category', 'unknown')
        
        return f"failure_{dominant_model}_{league_type}"


class MockPerformanceTracker:
    """Mock performance tracker for testing purposes"""
    
    def get_model_performance(self, model_name: str) -> Dict:
        """Return mock performance data"""
        return {
            'overall': {
                'accuracy': random.uniform(45, 75),
                'predictions': random.randint(50, 200)
            }
        }


# Context-aware optimization interface
class ContextAwareOptimizer:
    """Context-aware optimization interface for the genetic algorithm"""
    
    def __init__(self, genetic_optimizer: GeneticEnsembleOptimizer):
        self.genetic_optimizer = genetic_optimizer
        self.context_cache = {}
        
    def optimize_for_context(self, 
                           context_type: str, 
                           match_contexts: List[Dict], 
                           cache_timeout: int = 3600) -> Dict[str, float]:
        """Optimize weights for specific context with caching"""
        cache_key = f"{context_type}_{hash(str(match_contexts))}"
        
        # Check cache
        if cache_key in self.context_cache:
            cached_result, timestamp = self.context_cache[cache_key]
            if (datetime.now() - timestamp).seconds < cache_timeout:
                return cached_result
        
        # Perform optimization
        optimized_weights = self.genetic_optimizer.optimize_weights(
            match_contexts=match_contexts,
            target_context=context_type
        )
        
        # Cache result
        self.context_cache[cache_key] = (optimized_weights, datetime.now())
        
        return optimized_weights
    
    def get_context_specific_weights(self, context: Dict) -> Dict[str, float]:
        """Get context-specific weights using stored patterns"""
        context_type = f"{context.get('league', 'unknown')}_{context.get('match_type', 'balanced')}"
        
        # Try to use cached optimization result
        for cache_key, (weights, timestamp) in self.context_cache.items():
            if context_type in cache_key:
                return weights
        
        # Fallback to genetic optimization with single context
        return self.genetic_optimizer.optimize_weights([context])