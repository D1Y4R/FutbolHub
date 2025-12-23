"""
Expected Goals (xG) Rating System
Soccer Prediction projesinden esinlenilerek geliştirildi
xG verilerini kullanarak takım güçlerini dinamik olarak günceller
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class XGRatingSystem:
    """
    xG tabanlı takım güç değerlendirme sistemi
    Soccer Prediction projesindeki yaklaşımı uygular
    """
    
    def __init__(self):
        # Soccer Prediction'dan alınan optimal parametreler
        self.beta_h = 0.02539  # Ev sahibi sigmoid eğimi
        self.beta_a = 0.03     # Deplasman sigmoid eğimi
        self.gamma_h = -0.6711 # Ev sahibi eşik değeri
        self.gamma_a = -0.7728 # Deplasman eşik değeri
        self.alpha_h = 4.2     # Ev sahibi maksimum gol
        self.alpha_a = 4.0758  # Deplasman maksimum gol
        self.rho = 0.876       # xG ağırlığı (%87.6)
        
        # Güncelleme ağırlıkları
        self.omega_hatt = 2.1694  # Ev sahibi hücum güncelleme
        self.omega_hdef = 1.7701  # Ev sahibi savunma güncelleme
        self.omega_aatt = 1.3964  # Deplasman hücum güncelleme
        self.omega_adef = 2.4794  # Deplasman savunma güncelleme
        
        # Takım güç değerlendirmeleri
        self.team_ratings = {}
        
        logger.info("XG Rating System başlatıldı - Soccer Prediction parametreleri yüklendi")
    
    def get_team_rating(self, team_id: str) -> Dict[str, float]:
        """Takım güç değerlendirmelerini getir"""
        if team_id not in self.team_ratings:
            # Yeni takım için başlangıç değerleri
            self.team_ratings[team_id] = {
                'h_att': 0.0,  # Ev sahibi hücum gücü
                'h_def': 0.0,  # Ev sahibi savunma zayıflığı
                'a_att': 0.0,  # Deplasman hücum gücü
                'a_def': 0.0   # Deplasman savunma zayıflığı
            }
        return self.team_ratings[team_id]
    
    def predict_goals(self, home_team_id: str, away_team_id: str) -> Tuple[float, float]:
        """
        Sigmoid fonksiyonlar kullanarak gol tahmini
        Denklem 1 ve 2'yi uygular
        """
        home_ratings = self.get_team_rating(home_team_id)
        away_ratings = self.get_team_rating(away_team_id)
        
        # Ev sahibi gol tahmini (Denklem 1)
        h_att = home_ratings['h_att']
        a_def = away_ratings['a_def']
        pred_home_goals = self.alpha_h / (1 + np.exp(-self.beta_h * (h_att + a_def) - self.gamma_h))
        
        # Deplasman gol tahmini (Denklem 2)
        a_att = away_ratings['a_att']
        h_def = home_ratings['h_def']
        pred_away_goals = self.alpha_a / (1 + np.exp(-self.beta_a * (a_att + h_def) - self.gamma_a))
        
        return pred_home_goals, pred_away_goals
    
    def calculate_combined_goals(self, actual_goals: int, xg: float) -> float:
        """
        xG ve gerçek golleri birleştir (Denklem 8 ve 9)
        g = (xG × ρ) + (goals × (1-ρ))
        """
        return (xg * self.rho) + (actual_goals * (1 - self.rho))
    
    def update_ratings(self, home_team_id: str, away_team_id: str, 
                      home_goals: int, away_goals: int,
                      home_xg: Optional[float] = None, 
                      away_xg: Optional[float] = None):
        """
        Maç sonrası takım güçlerini güncelle (Denklem 3-6)
        """
        # Tahmin edilen goller
        pred_home, pred_away = self.predict_goals(home_team_id, away_team_id)
        
        # xG varsa birleştirilmiş gol değerlerini kullan
        if home_xg is not None and away_xg is not None:
            g_h = self.calculate_combined_goals(home_goals, home_xg)
            g_a = self.calculate_combined_goals(away_goals, away_xg)
        else:
            # xG yoksa sadece gerçek golleri kullan
            g_h = float(home_goals)
            g_a = float(away_goals)
        
        # Ev sahibi takım güncellemeleri
        home_ratings = self.get_team_rating(home_team_id)
        home_ratings['h_att'] += self.omega_hatt * (g_h - pred_home)  # Denklem 3
        home_ratings['h_def'] += self.omega_hdef * (g_a - pred_away)  # Denklem 4
        
        # Deplasman takım güncellemeleri
        away_ratings = self.get_team_rating(away_team_id)
        away_ratings['a_att'] += self.omega_aatt * (g_a - pred_away)  # Denklem 5
        away_ratings['a_def'] += self.omega_adef * (g_h - pred_home)  # Denklem 6
        
        logger.debug(f"Ratings güncellendi - {home_team_id}: {home_ratings}, {away_team_id}: {away_ratings}")
    
    def get_team_strength_scores(self, team_id: str) -> Dict[str, float]:
        """
        Takım güç puanlarını hesapla
        Hybrid ML sistemine entegrasyon için
        """
        ratings = self.get_team_rating(team_id)
        
        # Hücum ve savunma güçlerini normalize et
        attack_strength = (ratings['h_att'] + ratings['a_att']) / 2
        defense_strength = -(ratings['h_def'] + ratings['a_def']) / 2  # Negatif çünkü zayıflık
        
        # 0-100 arasına normalize et
        # Sigmoid fonksiyon kullanarak
        normalized_attack = 100 / (1 + np.exp(-0.1 * attack_strength))
        normalized_defense = 100 / (1 + np.exp(-0.1 * defense_strength))
        
        return {
            'attack_rating': normalized_attack,
            'defense_rating': normalized_defense,
            'overall_rating': (normalized_attack + normalized_defense) / 2
        }
    
    def calculate_goal_prediction_error(self, home_goals: int, away_goals: int,
                                      pred_home: float, pred_away: float,
                                      home_xg: Optional[float] = None,
                                      away_xg: Optional[float] = None) -> float:
        """
        Gol tahmin hatasını hesapla (Denklem 7)
        PSO optimizasyonu için kullanılır
        """
        # xG varsa birleştirilmiş değerleri kullan
        if home_xg is not None and away_xg is not None:
            g_h = self.calculate_combined_goals(home_goals, home_xg)
            g_a = self.calculate_combined_goals(away_goals, away_xg)
        else:
            g_h = float(home_goals)
            g_a = float(away_goals)
        
        # Hata hesaplama
        error = 0.5 * ((g_h - pred_home)**2 + (g_a - pred_away)**2)
        return error