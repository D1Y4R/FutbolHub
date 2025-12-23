"""
Glicko-2 Rating Sistemi
Belirsizlik ve volatilite ile gelişmiş rating hesaplama
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from glicko2 import Player
import math

logger = logging.getLogger(__name__)

class Glicko2System:
    """
    Futbol için özelleştirilmiş Glicko-2 rating sistemi
    """
    
    def __init__(self, initial_rating=1500, initial_rd=350, initial_volatility=0.06):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd  # Rating Deviation (belirsizlik)
        self.initial_volatility = initial_volatility  # Volatilite (tutarlılık)
        self.system_constant = 0.2  # Tau - sistem sabiti
        self.ratings = {}  # Takım ID -> Rating objesi
        self.rating_history = {}  # Takım ID -> [(date, rating, rd, vol)]
        
    def get_rating(self, team_id):
        """Takımın mevcut rating objesini getir"""
        if team_id not in self.ratings:
            player = Player(rating=self.initial_rating, rd=self.initial_rd)
            player.vol = self.initial_volatility
            self.ratings[team_id] = player
        return self.ratings[team_id]
        
    def get_rating_values(self, team_id):
        """Takımın rating değerlerini getir"""
        rating = self.get_rating(team_id)
        return {
            'rating': rating.getRating(),
            'rd': rating.getRd(),  
            'volatility': rating.vol,
            'confidence': self._calculate_confidence(rating.getRd())
        }
        
    def _calculate_confidence(self, rd):
        """RD'ye göre güven seviyesi hesapla (0-100)"""
        # Düşük RD = yüksek güven
        # RD 30-350 aralığında, güven 100-20 aralığında
        confidence = 100 - ((rd - 30) / 320 * 80)
        return max(20, min(100, confidence))
        
    def update_ratings_from_match(self, home_id, away_id, home_goals, away_goals):
        """
        Maç sonucuna göre her iki takımın rating'ini güncelle
        
        Returns:
            dict: Güncelleme detayları
        """
        home_rating = self.get_rating(home_id)
        away_rating = self.get_rating(away_id)
        
        # Maç sonucu skoru (1=galibiyet, 0.5=beraberlik, 0=mağlubiyet)
        if home_goals > away_goals:
            home_score = 1.0
            away_score = 0.0
        elif home_goals == away_goals:
            home_score = 0.5
            away_score = 0.5
        else:
            home_score = 0.0
            away_score = 1.0
            
        # Glicko-2 güncellemesi
        new_home_rating = self._glicko2_update(home_rating, [(away_rating, home_score)])
        new_away_rating = self._glicko2_update(away_rating, [(home_rating, away_score)])
        
        # Yeni rating'leri kaydet
        self.ratings[home_id] = new_home_rating
        self.ratings[away_id] = new_away_rating
        
        # Geçmişe ekle
        now = datetime.now()
        self._add_to_history(home_id, now, new_home_rating)
        self._add_to_history(away_id, now, new_away_rating)
        
        logger.info(f"Glicko-2 güncellendi - {home_id}: {home_rating.getRating():.0f}→{new_home_rating.getRating():.0f} "
                   f"(RD: {home_rating.getRd():.0f}→{new_home_rating.getRd():.0f})")
        logger.info(f"Glicko-2 güncellendi - {away_id}: {away_rating.getRating():.0f}→{new_away_rating.getRating():.0f} "
                   f"(RD: {away_rating.getRd():.0f}→{new_away_rating.getRd():.0f})")
        
        return {
            'home': {
                'old_rating': home_rating.getRating(),
                'new_rating': new_home_rating.getRating(),
                'old_rd': home_rating.getRd(),
                'new_rd': new_home_rating.getRd(),
                'volatility': new_home_rating.vol
            },
            'away': {
                'old_rating': away_rating.getRating(),
                'new_rating': new_away_rating.getRating(),
                'old_rd': away_rating.getRd(),
                'new_rd': new_away_rating.getRd(),
                'volatility': new_away_rating.vol
            }
        }
        
    def _glicko2_update(self, player_rating, matches):
        """
        Glicko-2 algoritması ile rating güncelleme
        
        Args:
            player_rating: Güncellenen oyuncunun rating'i
            matches: [(opponent_rating, score)] listesi
            
        Returns:
            Player: Güncellenmiş rating objesi
        """
        # Glicko-2 kütüphanesinin update_player metodunu kullan
        # matches listesini uygun formata çevir
        rating_list = []
        rd_list = []
        outcome_list = []
        
        for opp_rating, score in matches:
            rating_list.append(opp_rating.getRating())
            rd_list.append(opp_rating.getRd())
            outcome_list.append(score)  # 1.0 win, 0.5 draw, 0.0 loss
        
        # Yeni player objesi oluştur ve güncelle
        new_player = Player(rating=player_rating.getRating(), rd=player_rating.getRd())
        new_player.vol = player_rating.vol
        
        if rating_list:  # Eğer maç varsa güncelle
            new_player.update_player(rating_list, rd_list, outcome_list)
        
        return new_player

        
    def _g(self, phi):
        """g(φ) fonksiyonu"""
        return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)
        
    def _E(self, mu, mu_j, phi_j):
        """E(μ, μⱼ, φⱼ) beklenen skor"""
        return 1 / (1 + math.exp(-self._g(phi_j) * (mu - mu_j)))
        
    def _compute_variance(self, mu, m, phi_j):
        """Variance hesapla"""
        return 1 / sum(self._g(phi_j[j])**2 * self._E(mu, m[j], phi_j[j]) * 
                      (1 - self._E(mu, m[j], phi_j[j])) for j in range(len(m)))
        
    def _compute_volatility(self, sigma, delta, phi, v):
        """Yeni volatilite hesapla (iteratif)"""
        a = math.log(sigma**2)
        tau = self.system_constant
        
        def f(x):
            ex = math.exp(x)
            d2 = delta**2
            p2 = phi**2
            return (ex * (d2 - p2 - v - ex)) / (2 * (p2 + v + ex)**2) - (x - a) / tau**2
            
        # Illinois algoritması
        A = a
        if delta**2 > phi**2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau
            
        # Iterasyon
        f_A = f(A)
        f_B = f(B)
        
        while abs(B - A) > 0.000001:
            C = A + (A - B) * f_A / (f_B - f_A)
            f_C = f(C)
            
            if f_C * f_B < 0:
                A = B
                f_A = f_B
            else:
                f_A = f_A / 2
                
            B = C
            f_B = f_C
            
        return math.exp(B / 2)
        
    def _add_to_history(self, team_id, date, rating):
        """Rating geçmişine ekle"""
        if team_id not in self.rating_history:
            self.rating_history[team_id] = []
        self.rating_history[team_id].append((date, rating.getRating(), rating.getRd(), rating.vol))
        
    def time_decay_rd(self, team_id, days_inactive):
        """
        Oynamayan takımlar için RD artışı
        
        Args:
            team_id: Takım ID
            days_inactive: Oynamadığı gün sayısı
        """
        rating = self.get_rating(team_id)
        
        # Her 30 gün için RD %5 artar (max 350'ye kadar)
        periods = days_inactive / 30.0
        current_rd = rating.getRd()
        new_rd = min(350, current_rd * (1.05 ** periods))
        
        # Yeni player objesi oluştur
        new_player = Player(rating=rating.getRating(), rd=new_rd)
        new_player.vol = rating.vol
        self.ratings[team_id] = new_player
        
        logger.debug(f"RD time decay - Takım: {team_id}, {days_inactive} gün, "
                    f"RD: {current_rd:.0f} → {new_rd:.0f}")
        
    def get_match_prediction(self, home_id, away_id):
        """
        İki takım arasındaki maç için tahmin üret
        
        Returns:
            dict: Tahmin detayları
        """
        home_rating = self.get_rating(home_id)
        away_rating = self.get_rating(away_id)
        
        # Rating farkı
        rating_diff = home_rating.getRating() - away_rating.getRating()
        
        # Belirsizlik kombinasyonu
        combined_rd = math.sqrt(home_rating.getRd()**2 + away_rating.getRd()**2)
        
        # Kazanma olasılıkları (belirsizlik dahil)
        home_win_prob = self._win_probability(home_rating, away_rating)
        away_win_prob = self._win_probability(away_rating, home_rating)
        draw_prob = 1 - home_win_prob - away_win_prob
        
        # Güven seviyesi
        confidence = self._calculate_confidence(combined_rd)
        
        return {
            'rating_diff': rating_diff,
            'home_rating': home_rating.getRating(),
            'away_rating': away_rating.getRating(),
            'home_rd': home_rating.getRd(),
            'away_rd': away_rating.getRd(),
            'combined_uncertainty': combined_rd,
            'home_win_prob': home_win_prob * 100,
            'draw_prob': draw_prob * 100,
            'away_win_prob': away_win_prob * 100,
            'confidence': confidence,
            'volatility_factor': (home_rating.vol + away_rating.vol) / 2
        }
        
    def _win_probability(self, rating_a, rating_b):
        """A'nın B'ye karşı kazanma olasılığı"""
        g_phi = self._g(math.sqrt(rating_a.getRd()**2 + rating_b.getRd()**2) / 173.7178)
        return 1 / (1 + 10**(-g_phi * (rating_a.getRating() - rating_b.getRating()) / 400))
        
    def calculate_team_glicko2(self, team_id, matches):
        """
        Takımın son maçlarına göre Glicko-2 hesapla
        
        Args:
            team_id: Takım ID
            matches: Maç listesi
            
        Returns:
            dict: Rating detayları
        """
        if not matches:
            return self.get_rating_values(team_id)
            
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
            return self.get_rating_values(team_id)
            
        # Başlangıç rating
        player = Player(rating=self.initial_rating, rd=self.initial_rd)
        player.vol = self.initial_volatility
        self.ratings[team_id] = player
        
        # Maçları işle (eskiden yeniye)
        for match in reversed(filtered_matches):
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            # Tahmini rakip rating'i (gol farkına göre)
            goal_diff = goals_for - goals_against
            if abs(goal_diff) >= 3:
                opponent_mu = self.initial_rating + (-200 if goal_diff > 0 else 200)
            elif abs(goal_diff) == 2:
                opponent_mu = self.initial_rating + (-100 if goal_diff > 0 else 100)
            elif abs(goal_diff) == 1:
                opponent_mu = self.initial_rating + (-50 if goal_diff > 0 else 50)
            else:
                opponent_mu = self.initial_rating
                
            # Tahmini rakip objesi
            opponent_rating = Player(rating=opponent_mu, rd=200)
            opponent_rating.vol = 0.06
            
            # Skor
            if goals_for > goals_against:
                score = 1.0
            elif goals_for == goals_against:
                score = 0.5
            else:
                score = 0.0
                
            # Güncelle
            new_rating = self._glicko2_update(self.ratings[team_id], [(opponent_rating, score)])
            self.ratings[team_id] = new_rating
            
        logger.info(f"Takım {team_id} için Glicko-2 hesaplandı: μ={self.ratings[team_id].getRating():.0f}, "
                   f"RD={self.ratings[team_id].getRd():.0f}, σ={self.ratings[team_id].vol:.3f} "
                   f"({len(filtered_matches)} maç)")
        
        return self.get_rating_values(team_id)