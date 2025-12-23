"""
TrueSkill Adapter
Microsoft TrueSkill sistemini futbol tahminlerine uyarlama
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from trueskill import Rating as TrueSkillRating, rate_1vs1, TrueSkill
import math

logger = logging.getLogger(__name__)

class TrueSkillAdapter:
    """
    Futbol için TrueSkill adaptasyonu
    Takım ve oyuncu bazlı analizler
    """
    
    def __init__(self, mu=25.0, sigma=8.333, beta=4.166, tau=0.083):
        # TrueSkill ortamı oluştur
        self.env = TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=0.25)
        self.team_ratings = {}  # Takım ID -> TrueSkillRating
        self.player_ratings = {}  # Oyuncu ID -> TrueSkillRating  
        self.team_chemistry = {}  # Takım ID -> kimya faktörü (0-1)
        self.rating_history = {}  # Takım ID -> [(date, mu, sigma)]
        
    def get_team_rating(self, team_id):
        """Takımın TrueSkill rating'ini getir"""
        if team_id not in self.team_ratings:
            self.team_ratings[team_id] = self.env.create_rating()
        return self.team_ratings[team_id]
        
    def get_team_skill_values(self, team_id):
        """Takımın skill değerlerini getir"""
        rating = self.get_team_rating(team_id)
        conservative_skill = rating.mu - 3 * rating.sigma  # %99.7 güven aralığı
        
        return {
            'skill': rating.mu,
            'uncertainty': rating.sigma,
            'conservative_skill': conservative_skill,
            'confidence': self._calculate_confidence(rating.sigma),
            'chemistry': self.team_chemistry.get(team_id, 0.7)  # Default %70
        }
        
    def _calculate_confidence(self, sigma):
        """Sigma'ya göre güven seviyesi (0-100)"""
        # Düşük sigma = yüksek güven
        # Sigma 1-10 aralığında, güven 100-20 aralığında
        confidence = 100 - ((sigma - 1) / 9 * 80)
        return max(20, min(100, confidence))
        
    def update_ratings_from_match(self, home_id, away_id, home_goals, away_goals):
        """
        Maç sonucuna göre rating güncelle
        
        Returns:
            dict: Güncelleme detayları
        """
        home_rating = self.get_team_rating(home_id)
        away_rating = self.get_team_rating(away_id)
        
        # TrueSkill güncellemesi
        if home_goals > away_goals:
            new_home, new_away = rate_1vs1(home_rating, away_rating, env=self.env)
        elif home_goals < away_goals:
            new_away, new_home = rate_1vs1(away_rating, home_rating, env=self.env)
        else:  # Beraberlik
            new_home, new_away = rate_1vs1(home_rating, away_rating, drawn=True, env=self.env)
            
        # Yeni rating'leri kaydet
        self.team_ratings[home_id] = new_home
        self.team_ratings[away_id] = new_away
        
        # Takım kimyası güncelleme (kazananın kimyası artar)
        if home_goals > away_goals:
            self._update_team_chemistry(home_id, 0.02)  # %2 artış
            self._update_team_chemistry(away_id, -0.01)  # %1 azalış
        elif away_goals > home_goals:
            self._update_team_chemistry(away_id, 0.02)
            self._update_team_chemistry(home_id, -0.01)
        else:  # Beraberlik
            self._update_team_chemistry(home_id, 0.005)  # %0.5 artış
            self._update_team_chemistry(away_id, 0.005)
            
        # Geçmişe ekle
        now = datetime.now()
        self._add_to_history(home_id, now, new_home)
        self._add_to_history(away_id, now, new_away)
        
        logger.info(f"TrueSkill güncellendi - {home_id}: μ={home_rating.mu:.1f}→{new_home.mu:.1f}, "
                   f"σ={home_rating.sigma:.2f}→{new_home.sigma:.2f}")
        logger.info(f"TrueSkill güncellendi - {away_id}: μ={away_rating.mu:.1f}→{new_away.mu:.1f}, "
                   f"σ={away_rating.sigma:.2f}→{new_away.sigma:.2f}")
        
        return {
            'home': {
                'old_mu': home_rating.mu,
                'new_mu': new_home.mu,
                'old_sigma': home_rating.sigma,
                'new_sigma': new_home.sigma,
                'chemistry': self.team_chemistry.get(home_id, 0.7)
            },
            'away': {
                'old_mu': away_rating.mu,
                'new_mu': new_away.mu,
                'old_sigma': away_rating.sigma,
                'new_sigma': new_away.sigma,
                'chemistry': self.team_chemistry.get(away_id, 0.7)
            }
        }
        
    def _update_team_chemistry(self, team_id, change):
        """Takım kimyasını güncelle (0-1 aralığında)"""
        current = self.team_chemistry.get(team_id, 0.7)
        new_chemistry = max(0.1, min(1.0, current + change))
        self.team_chemistry[team_id] = new_chemistry
        
    def _add_to_history(self, team_id, date, rating):
        """Rating geçmişine ekle"""
        if team_id not in self.rating_history:
            self.rating_history[team_id] = []
        self.rating_history[team_id].append((date, rating.mu, rating.sigma))
        
    def calculate_match_quality(self, home_id, away_id):
        """
        Maç kalitesini hesapla (0-1)
        Yakın skill seviyesi = yüksek kalite
        """
        home_rating = self.get_team_rating(home_id)
        away_rating = self.get_team_rating(away_id)
        
        # Manuel match quality hesaplama
        delta_mu = abs(home_rating.mu - away_rating.mu)
        sum_sigma = home_rating.sigma + away_rating.sigma
        
        # Yakın skill = yüksek kalite
        quality = math.exp(-delta_mu / (2 * sum_sigma))
        return quality
        
    def get_win_probability(self, home_id, away_id):
        """
        Kazanma olasılıklarını hesapla
        
        Returns:
            dict: home_win, draw, away_win olasılıkları
        """
        home_rating = self.get_team_rating(home_id)
        away_rating = self.get_team_rating(away_id)
        
        # Skill farkı
        delta_mu = home_rating.mu - away_rating.mu
        sum_sigma = math.sqrt(home_rating.sigma**2 + away_rating.sigma**2)
        
        # TrueSkill formülü
        denom = math.sqrt(2 * (self.env.beta * self.env.beta) + sum_sigma**2)
        
        # CDF hesaplama için z-score
        from scipy.stats import norm
        
        # Ev sahibi kazanma olasılığı
        home_z = delta_mu / denom
        home_win_prob = norm.cdf(home_z)
        
        # Deplasman kazanma olasılığı  
        away_z = -delta_mu / denom
        away_win_prob = norm.cdf(away_z)
        
        # Beraberlik (draw margin dahilinde)
        # Epsilon yerine sabit draw margin kullan
        draw_margin = 0.74 / denom  # TrueSkill default epsilon değeri
        draw_prob = norm.cdf(draw_margin) - norm.cdf(-draw_margin)
        
        # Normalize et
        total = home_win_prob + draw_prob + away_win_prob
        
        return {
            'home_win': (home_win_prob / total) * 100,
            'draw': (draw_prob / total) * 100,
            'away_win': (away_win_prob / total) * 100,
            'match_quality': self.calculate_match_quality(home_id, away_id) * 100
        }
        
    def analyze_team_dynamics(self, team_id, recent_matches):
        """
        Takım dinamiklerini analiz et
        
        Args:
            team_id: Takım ID
            recent_matches: Son maçlar
            
        Returns:
            dict: Takım dinamik analizi
        """
        if not recent_matches:
            return {
                'form_trend': 'unknown',
                'consistency': 0,
                'momentum': 50,
                'chemistry_trend': 'stable'
            }
            
        # Son 5 maç
        last_5 = recent_matches[:5]
        
        # Form trendi
        wins = sum(1 for m in last_5 if m.get('goals_scored', 0) > m.get('goals_conceded', 0))
        draws = sum(1 for m in last_5 if m.get('goals_scored', 0) == m.get('goals_conceded', 0))
        
        if wins >= 4:
            form_trend = 'excellent'
            momentum = 90
        elif wins >= 3:
            form_trend = 'good'
            momentum = 75
        elif wins >= 2:
            form_trend = 'average'
            momentum = 50
        elif wins >= 1:
            form_trend = 'poor'
            momentum = 25
        else:
            form_trend = 'terrible'
            momentum = 10
            
        # Tutarlılık (gol farkı standart sapması)
        goal_diffs = [m.get('goals_scored', 0) - m.get('goals_conceded', 0) for m in last_5]
        consistency = 100 - min(100, np.std(goal_diffs) * 20)
        
        # Kimya trendi
        chemistry = self.team_chemistry.get(team_id, 0.7)
        if chemistry > 0.85:
            chemistry_trend = 'excellent'
        elif chemistry > 0.75:
            chemistry_trend = 'good'
        elif chemistry > 0.65:
            chemistry_trend = 'stable'
        else:
            chemistry_trend = 'poor'
            
        return {
            'form_trend': form_trend,
            'consistency': consistency,
            'momentum': momentum,
            'chemistry_trend': chemistry_trend,
            'chemistry_value': chemistry * 100,
            'recent_performance': f"{wins}W-{draws}D-{5-wins-draws}L"
        }
        
    def calculate_team_trueskill(self, team_id, matches):
        """
        Takımın son maçlarına göre TrueSkill hesapla
        
        Args:
            team_id: Takım ID
            matches: Maç listesi
            
        Returns:
            dict: Skill detayları
        """
        if not matches:
            return self.get_team_skill_values(team_id)
            
        # Son 120 gündeki maçları filtrele
        today = datetime.now()
        cutoff = today - timedelta(days=120)
        filtered_matches = []
        
        for match in matches:
            try:
                match_date = datetime.strptime(match.get('date', ''), '%Y-%m-%d')
                if match_date >= cutoff:
                    filtered_matches.append(match)
            except:
                filtered_matches.append(match)
                
        if not filtered_matches:
            logger.warning(f"Takım {team_id} için son 120 günde maç bulunamadı")
            return self.get_team_skill_values(team_id)
            
        # Başlangıç rating
        self.team_ratings[team_id] = self.env.create_rating()
        self.team_chemistry[team_id] = 0.7  # Başlangıç kimyası
        
        # Maçları işle (eskiden yeniye)
        for match in reversed(filtered_matches):
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            # Tahmini rakip skill'i
            goal_diff = goals_for - goals_against
            if abs(goal_diff) >= 3:
                opponent_mu = 25 + (-8 if goal_diff > 0 else 8)
            elif abs(goal_diff) == 2:
                opponent_mu = 25 + (-4 if goal_diff > 0 else 4)
            elif abs(goal_diff) == 1:
                opponent_mu = 25 + (-2 if goal_diff > 0 else 2)
            else:
                opponent_mu = 25
                
            # Tahmini rakip rating'i
            opponent_rating = self.env.create_rating(mu=opponent_mu)
            
            # TrueSkill güncellemesi
            if goals_for > goals_against:
                new_rating, _ = rate_1vs1(self.team_ratings[team_id], opponent_rating, env=self.env)
                self._update_team_chemistry(team_id, 0.01)
            elif goals_for < goals_against:
                _, new_rating = rate_1vs1(opponent_rating, self.team_ratings[team_id], env=self.env)
                self._update_team_chemistry(team_id, -0.005)
            else:  # Beraberlik
                new_rating, _ = rate_1vs1(self.team_ratings[team_id], opponent_rating, drawn=True, env=self.env)
                self._update_team_chemistry(team_id, 0.002)
                
            self.team_ratings[team_id] = new_rating
            
        logger.info(f"Takım {team_id} için TrueSkill hesaplandı: μ={self.team_ratings[team_id].mu:.1f}, "
                   f"σ={self.team_ratings[team_id].sigma:.2f}, kimya={self.team_chemistry[team_id]:.2f} "
                   f"({len(filtered_matches)} maç)")
        
        return self.get_team_skill_values(team_id)