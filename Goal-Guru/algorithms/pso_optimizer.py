"""
Particle Swarm Optimization (PSO) for Parameter Tuning
Soccer Prediction projesinden esinlenilerek geliştirildi
Model parametrelerini optimize etmek için kullanılır
"""

import numpy as np
from typing import Callable, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class PSOOptimizer:
    """
    Particle Swarm Optimization algoritması
    Model parametrelerini optimize etmek için kullanılır
    """
    
    def __init__(self, 
                 n_particles: int = 80,
                 n_dimensions: int = 11,
                 max_iterations: int = 100,
                 w: float = 0.729,      # İnertia weight
                 c1: float = 1.49445,   # Cognitive parameter
                 c2: float = 1.49445):  # Social parameter
        
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Parametre sınırları (Soccer Prediction'dan)
        self.bounds = {
            'beta_h': (0.01, 0.1),
            'beta_a': (0.01, 0.1),
            'gamma_h': (-2.0, 0.0),
            'gamma_a': (-2.0, 0.0),
            'omega_hatt': (0.5, 3.0),
            'omega_hdef': (0.5, 3.0),
            'omega_aatt': (0.5, 3.0),
            'omega_adef': (0.5, 3.0),
            'alpha_h': (3.0, 5.0),
            'alpha_a': (3.0, 5.0),
            'rho': (0.5, 1.0)
        }
        
        logger.info(f"PSO Optimizer başlatıldı - {n_particles} parçacık, {n_dimensions} boyut")
    
    def initialize_swarm(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parçacık sürüsünü başlat"""
        positions = np.zeros((self.n_particles, self.n_dimensions))
        velocities = np.zeros((self.n_particles, self.n_dimensions))
        
        # Her boyut için rastgele pozisyonlar oluştur
        for i, (param, (low, high)) in enumerate(self.bounds.items()):
            positions[:, i] = np.random.uniform(low, high, self.n_particles)
            velocities[:, i] = np.random.uniform(-0.1 * (high - low), 
                                               0.1 * (high - low), 
                                               self.n_particles)
        
        return positions, velocities
    
    def evaluate_fitness(self, position: np.ndarray, 
                        fitness_function: Callable) -> float:
        """
        Bir pozisyonun fitness değerini hesapla
        fitness_function: parametre dizisini alıp hata değeri döndüren fonksiyon
        """
        # Parametreleri dictionary'e çevir
        params = {}
        for i, param_name in enumerate(self.bounds.keys()):
            params[param_name] = position[i]
        
        # Fitness fonksiyonunu çağır (minimize ediyoruz)
        return fitness_function(params)
    
    def optimize(self, fitness_function: Callable, 
                early_stopping_tol: float = 1e-6) -> Dict[str, float]:
        """
        PSO algoritmasını çalıştır
        Returns: Optimal parametreler
        """
        # Sürüyü başlat
        positions, velocities = self.initialize_swarm()
        
        # En iyi pozisyonları takip et
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.n_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        # İlk değerlendirme
        for i in range(self.n_particles):
            score = self.evaluate_fitness(positions[i], fitness_function)
            personal_best_scores[i] = score
            
            if score < global_best_score:
                global_best_score = score
                global_best_position = positions[i].copy()
        
        # Ana optimizasyon döngüsü
        best_scores_history = []
        
        for iteration in range(self.max_iterations):
            # Her parçacık için
            for i in range(self.n_particles):
                # Hızı güncelle
                r1 = np.random.rand(self.n_dimensions)
                r2 = np.random.rand(self.n_dimensions)
                
                cognitive = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2 * (global_best_position - positions[i])
                
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # Pozisyonu güncelle
                positions[i] += velocities[i]
                
                # Sınırları kontrol et
                for j, (param, (low, high)) in enumerate(self.bounds.items()):
                    positions[i, j] = np.clip(positions[i, j], low, high)
                
                # Fitness değerini hesapla
                score = self.evaluate_fitness(positions[i], fitness_function)
                
                # Personal best güncelle
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                
                # Global best güncelle
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()
            
            best_scores_history.append(global_best_score)
            
            # İlerleme logu
            if iteration % 10 == 0:
                logger.info(f"PSO İterasyon {iteration}: En iyi skor = {global_best_score:.6f}")
            
            # Early stopping kontrolü
            if len(best_scores_history) > 10:
                recent_improvement = abs(best_scores_history[-10] - best_scores_history[-1])
                if recent_improvement < early_stopping_tol:
                    logger.info(f"Early stopping - İterasyon {iteration}")
                    break
        
        # Optimal parametreleri döndür
        if global_best_position is None:
            logger.warning("PSO optimizasyonu başarısız - hiçbir geçerli çözüm bulunamadı")
            # Varsayılan parametreleri döndür
            return {
                'beta_h': 0.02539,
                'beta_a': 0.03,
                'gamma_h': -0.6711,
                'gamma_a': -0.7728,
                'omega_hatt': 2.1694,
                'omega_hdef': 1.7701,
                'omega_aatt': 1.3964,
                'omega_adef': 2.4794,
                'alpha_h': 4.2,
                'alpha_a': 4.0758,
                'rho': 0.876
            }
        
        optimal_params = {}
        for i, param_name in enumerate(self.bounds.keys()):
            optimal_params[param_name] = global_best_position[i]
        
        logger.info(f"PSO tamamlandı - Final skor: {global_best_score:.6f}")
        logger.info(f"Optimal parametreler: {optimal_params}")
        
        return optimal_params
    
    def optimize_xg_rating_params(self, training_data: List[Dict]) -> Dict[str, float]:
        """
        XG Rating System parametrelerini optimize et
        training_data: Maç verileri listesi
        """
        from .xg_rating_system import XGRatingSystem
        
        def fitness_function(params: Dict[str, float]) -> float:
            """Parametre setinin fitness değerini hesapla"""
            # XG Rating System'i verilen parametrelerle oluştur
            xg_system = XGRatingSystem()
            
            # Parametreleri güncelle
            xg_system.beta_h = params['beta_h']
            xg_system.beta_a = params['beta_a']
            xg_system.gamma_h = params['gamma_h']
            xg_system.gamma_a = params['gamma_a']
            xg_system.omega_hatt = params['omega_hatt']
            xg_system.omega_hdef = params['omega_hdef']
            xg_system.omega_aatt = params['omega_aatt']
            xg_system.omega_adef = params['omega_adef']
            xg_system.alpha_h = params['alpha_h']
            xg_system.alpha_a = params['alpha_a']
            xg_system.rho = params['rho']
            
            # Toplam hatayı hesapla
            total_error = 0.0
            
            for match in training_data:
                # Tahmin yap
                pred_home, pred_away = xg_system.predict_goals(
                    match['home_team_id'], 
                    match['away_team_id']
                )
                
                # Hatayı hesapla
                error = xg_system.calculate_goal_prediction_error(
                    match['home_goals'],
                    match['away_goals'],
                    pred_home,
                    pred_away,
                    match.get('home_xg'),
                    match.get('away_xg')
                )
                
                total_error += error
                
                # Ratingleri güncelle
                xg_system.update_ratings(
                    match['home_team_id'],
                    match['away_team_id'],
                    match['home_goals'],
                    match['away_goals'],
                    match.get('home_xg'),
                    match.get('away_xg')
                )
            
            return total_error / len(training_data)
        
        # PSO ile optimize et
        return self.optimize(fitness_function)