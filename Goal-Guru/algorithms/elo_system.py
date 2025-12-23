"""
Elo Rating Sistemi - FIFA/FIDE Standardına Uygun
Takım güçlerini dinamik olarak hesaplar
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class EloSystem:
    """
    FIFA standardına uygun Elo rating sistemi
    
    Özellikler:
    - Dinamik K-faktörü (maç tipine göre)
    - Gol farkı çarpanı (FIFA formülü)
    - Ev sahası avantajı (+100 geçici)
    - Zaman bazlı decay
    """
    
    def __init__(self, initial_rating=1500):
        self.initial_rating = initial_rating
        self.ratings = {}  # Takım ID -> Elo rating
        self.match_counts = {}  # Takım ID -> Maç sayısı
        
        # FIFA standardı K-faktörleri
        self.k_factors = {
            'world_cup': 60,
            'continental_cup': 50,  # UEFA Champions League, Europa League vb.
            'qualifier': 40,
            'friendly': 20,
            'league': 35,  # Lig maçları
            'cup': 40,  # Ulusal kupalar
            'default': 30
        }
        
        # Ev sahası avantajı (geçici Elo bonusu)
        self.home_advantage = 100
        
    def get_k_factor(self, match_type: str = 'league', team_experience: int = 0) -> float:
        """
        Maç tipine ve takım deneyimine göre K-faktörü hesapla
        
        Args:
            match_type: Maç tipi ('league', 'cup', 'continental_cup' vb.)
            team_experience: Takımın toplam maç sayısı
            
        Returns:
            float: K-faktörü
        """
        # Temel K-faktörü
        base_k = self.k_factors.get(match_type, self.k_factors['default'])
        
        # Deneyim faktörü (yeni takımlar için K daha yüksek)
        if team_experience < 10:
            experience_multiplier = 1.3  # Yeni takımlar hızlı değişir
        elif team_experience < 30:
            experience_multiplier = 1.1
        else:
            experience_multiplier = 1.0  # Deneyimli takımlar daha stabil
            
        return base_k * experience_multiplier
        
    def get_goal_difference_multiplier(self, goal_diff: int) -> float:
        """
        FIFA standardı gol farkı çarpanı
        
        Args:
            goal_diff: Gol farkı (mutlak değer)
            
        Returns:
            float: Çarpan değeri
        """
        if goal_diff <= 1:
            return 1.0
        elif goal_diff == 2:
            return 1.5
        else:  # 3+
            return 1.75 + (goal_diff - 3) * 0.5
    
    def get_expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Standart Elo beklenen skor formülü
        
        Args:
            rating_a: Takım A'nın Elo puanı
            rating_b: Takım B'nin Elo puanı
            
        Returns:
            float: Beklenen skor (0-1 arası)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
        
    def update_rating(self, 
                     team_id: int,
                     team_rating: float,
                     opponent_rating: float, 
                     actual_score: float,
                     goal_diff: int = 1,
                     match_type: str = 'league',
                     is_home: bool = False,
                     match_age_days: int = 0) -> float:
        """
        FIFA standardına uygun Elo rating güncelleme
        
        Args:
            team_id: Takım ID
            team_rating: Takımın mevcut Elo'su
            opponent_rating: Rakibin Elo'su
            actual_score: Gerçek sonuç (1=galibiyet, 0.5=beraberlik, 0=mağlubiyet)
            goal_diff: Gol farkı (mutlak değer)
            match_type: Maç tipi
            is_home: Ev sahibi mi?
            match_age_days: Maçın kaç gün önce olduğu
            
        Returns:
            float: Yeni Elo rating
        """
        # Ev sahası avantajı ekle (sadece hesaplama için, kalıcı değil)
        adjusted_team_rating = team_rating + (self.home_advantage if is_home else 0)
        adjusted_opponent_rating = opponent_rating + (0 if is_home else self.home_advantage)
        
        # Beklenen skoru hesapla
        expected_score = self.get_expected_score(adjusted_team_rating, adjusted_opponent_rating)
        
        # K-faktörü (takım deneyimi ile)
        team_experience = self.match_counts.get(team_id, 0)
        k = self.get_k_factor(match_type, team_experience)
        
        # Gol farkı çarpanı (FIFA standardı)
        goal_multiplier = self.get_goal_difference_multiplier(goal_diff)
        
        # Zaman bazlı ağırlık (eski maçlar daha az etkili)
        time_weight = 0.95 ** (match_age_days / 30) if match_age_days > 0 else 1.0
        
        # Elo güncelleme formülü (FIFA)
        rating_change = k * goal_multiplier * (actual_score - expected_score) * time_weight
        new_rating = team_rating + rating_change
        
        # Rating'i kaydet
        self.ratings[team_id] = new_rating
        self.match_counts[team_id] = team_experience + 1
        
        logger.debug(
            f"Elo güncellendi - Takım: {team_id}, "
            f"Eski: {team_rating:.0f}, Yeni: {new_rating:.0f}, "
            f"Değişim: {rating_change:+.1f} "
            f"(K={k:.1f}, GD={goal_diff}, Mult={goal_multiplier:.2f})"
        )
        
        return new_rating
        
    def calculate_team_elo(self, 
                          team_id: int,
                          matches: list,
                          opponent_elos: Optional[Dict[int, float]] = None) -> float:
        """
        Takımın son maçlarına göre Elo hesapla
        
        Args:
            team_id: Takım ID
            matches: Maç listesi (en yeni önce)
            opponent_elos: Rakip takımların Elo değerleri {opponent_id: elo}
            
        Returns:
            float: Güncel Elo rating
        """
        if not matches:
            return self.initial_rating
            
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
            return self.initial_rating
            
        # Başlangıç Elo
        current_elo = self.initial_rating
        self.ratings[team_id] = current_elo
        self.match_counts[team_id] = 0
        
        # Maçları tersten işle (eskiden yeniye)
        for match in reversed(filtered_matches):
            # Maç sonucu
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            goal_diff = abs(goals_for - goals_against)
            
            if goals_for > goals_against:
                actual_score = 1.0
            elif goals_for == goals_against:
                actual_score = 0.5
            else:
                actual_score = 0.0
                
            # Rakip Elo'sunu belirle
            opponent_id = match.get('opponent_id')
            if opponent_elos and opponent_id and opponent_id in opponent_elos:
                # GERÇEK rakip Elo'su kullan
                opponent_elo = opponent_elos[opponent_id]
            else:
                # Eğer rakip Elo'su yoksa, performansa göre tahmin et
                # Bu ideal değil ama fallback olarak gerekli
                if goal_diff >= 3:
                    opponent_elo = self.initial_rating + (150 if actual_score == 0 else -150)
                elif goal_diff == 2:
                    opponent_elo = self.initial_rating + (100 if actual_score == 0 else -100)
                elif goal_diff == 1:
                    opponent_elo = self.initial_rating + (50 if actual_score == 0 else -50)
                else:
                    opponent_elo = self.initial_rating
            
            # Maç tipi
            match_type = match.get('competition_type', 'league')
            is_home = match.get('is_home', False)
            
            # Maç yaşı
            match_age_days = 0
            if 'date' in match:
                try:
                    match_date = datetime.strptime(match['date'], '%Y-%m-%d')
                    match_age_days = (today - match_date).days
                except:
                    pass
                    
            # Elo güncelle
            current_elo = self.update_rating(
                team_id=team_id,
                team_rating=current_elo,
                opponent_rating=opponent_elo,
                actual_score=actual_score,
                goal_diff=goal_diff,
                match_type=match_type,
                is_home=is_home,
                match_age_days=match_age_days
            )
            
        logger.info(
            f"Takım {team_id} için Elo hesaplandı: {current_elo:.0f} "
            f"({len(filtered_matches)} maç, son 120 gün)"
        )
        return current_elo
        
    def get_rating(self, team_id: int) -> float:
        """Takımın mevcut Elo rating'ini getir"""
        return self.ratings.get(team_id, self.initial_rating)
        
    def get_elo_difference(self, home_id: int, away_id: int) -> float:
        """
        İki takım arasındaki Elo farkını hesapla
        
        Returns:
            float: home_elo - away_elo
        """
        home_elo = self.get_rating(home_id)
        away_elo = self.get_rating(away_id)
        return home_elo - away_elo
        
    def predict_match(self, 
                     home_id: int,
                     away_id: int,
                     home_elo: Optional[float] = None,
                     away_elo: Optional[float] = None) -> Dict[str, float]:
        """
        İki takım arasındaki maç tahminini yap
        
        Args:
            home_id: Ev sahibi takım ID
            away_id: Deplasman takım ID
            home_elo: Ev sahibi Elo (None ise kayıtlıdan al)
            away_elo: Deplasman Elo (None ise kayıtlıdan al)
            
        Returns:
            dict: {
                'home_win': float,
                'draw': float,
                'away_win': float,
                'home_elo': float,
                'away_elo': float
            }
        """
        if home_elo is None:
            home_elo = self.get_rating(home_id)
        if away_elo is None:
            away_elo = self.get_rating(away_id)
            
        # Ev sahası avantajı ekle
        adjusted_home_elo = home_elo + self.home_advantage
        
        # Beklenen skorlar
        home_expected = self.get_expected_score(adjusted_home_elo, away_elo)
        
        # GELİŞTİRİLMİŞ BERABERLİK HESAPLAMASI
        # Futbol istatistiklerine göre: Ortalama lig beraberlik oranı %25-28
        # Elo farkına göre ölçeklendirilmiş beraberlik
        elo_diff = abs(adjusted_home_elo - away_elo)
        
        # Temel beraberlik olasılığı (lig ortalaması bazlı)
        # Yakın takımlar için daha yüksek, uzak takımlar için daha düşük
        if elo_diff < 50:
            # Çok yakın takımlar - yüksek beraberlik şansı
            draw_prob = 0.32  # %32
        elif elo_diff < 100:
            draw_prob = 0.28  # %28
        elif elo_diff < 150:
            draw_prob = 0.25  # %25
        elif elo_diff < 200:
            draw_prob = 0.22  # %22
        elif elo_diff < 300:
            draw_prob = 0.18  # %18
        else:
            # Çok farklı takımlar - ama yine de minimum beraberlik şansı
            draw_prob = max(0.15, 0.25 - (elo_diff / 1500))  # Minimum %15
        
        # Galibiyet olasılıkları (beraberliği düşerek)
        home_win = home_expected * (1 - draw_prob)
        away_win = (1 - home_expected) * (1 - draw_prob)
        
        # Normalize et
        total = home_win + draw_prob + away_win
        
        return {
            'home_win': home_win / total,
            'draw': draw_prob / total,
            'away_win': away_win / total,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_difference': home_elo - away_elo
        }
