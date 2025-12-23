"""
League Context Analyzer
Lig bağlamsal analizleri ve lig ortalama gollerini hesaplayan modül
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class LeagueContextAnalyzer:
    """
    Lig seviyesinde bağlamsal analiz ve istatistikler
    """
    
    def __init__(self):
        # Bilinen liglerin karakteristikleri (cache için)
        self.league_profiles = {
            'Premier League': {'avg_goals': 2.8, 'type': 'high_scoring', 'quality': 'elite'},
            'La Liga': {'avg_goals': 2.5, 'type': 'medium_scoring', 'quality': 'elite'},
            'Serie A': {'avg_goals': 2.6, 'type': 'medium_scoring', 'quality': 'elite'},
            'Bundesliga': {'avg_goals': 3.1, 'type': 'high_scoring', 'quality': 'elite'},
            'Ligue 1': {'avg_goals': 2.4, 'type': 'medium_scoring', 'quality': 'elite'},
            'Eredivisie': {'avg_goals': 3.2, 'type': 'high_scoring', 'quality': 'high'},
            'Super Lig': {'avg_goals': 2.7, 'type': 'medium_scoring', 'quality': 'high'},
            'Championship': {'avg_goals': 2.5, 'type': 'medium_scoring', 'quality': 'high'},
            'MLS': {'avg_goals': 2.9, 'type': 'high_scoring', 'quality': 'medium'},
            'Brasileirao': {'avg_goals': 2.3, 'type': 'low_scoring', 'quality': 'high'}
        }
        
        # Varsayılan lig ortalama gol
        self.default_league_avg = 2.5
        
    def analyze_league_context(self, league_name: str, recent_matches: List[Dict] = None) -> Dict:
        """
        Lig bağlamını analiz et
        
        Args:
            league_name: Lig adı
            recent_matches: Son maçlar (opsiyonel, dinamik hesaplama için)
            
        Returns:
            Lig bağlam analizi
        """
        try:
            # Önce cache'den kontrol et
            if league_name in self.league_profiles:
                profile = self.league_profiles[league_name]
                league_avg_goals = profile['avg_goals']
                league_type = profile['type']
                league_quality = profile['quality']
                logger.info(f"Lig profili cache'den alındı: {league_name} - {league_avg_goals:.2f} gol/maç")
            else:
                # Cache'de yoksa dinamik hesapla
                if recent_matches:
                    league_avg_goals = self._calculate_league_average(recent_matches)
                    league_type = self._determine_league_type(league_avg_goals)
                    league_quality = 'unknown'
                    logger.info(f"Lig ortalaması dinamik hesaplandı: {league_name} - {league_avg_goals:.2f} gol/maç")
                else:
                    # Veri yoksa varsayılan kullan
                    league_avg_goals = self.default_league_avg
                    league_type = 'medium_scoring'
                    league_quality = 'unknown'
                    logger.info(f"Varsayılan lig ortalaması kullanıldı: {league_avg_goals:.2f} gol/maç")
            
            # Lig karakteristikleri
            characteristics = self._analyze_league_characteristics(
                league_avg_goals, league_type, recent_matches
            )
            
            # Lambda faktörü hesapla (lig ortalamasına göre)
            lambda_factor = self._calculate_lambda_factor(league_avg_goals)
            
            return {
                'league_name': league_name,
                'avg_goals_per_match': league_avg_goals,
                'league_type': league_type,
                'league_quality': league_quality,
                'lambda_factor': lambda_factor,
                'characteristics': characteristics,
                'scoring_tendency': self._calculate_scoring_tendency(league_avg_goals),
                'defensive_tendency': self._calculate_defensive_tendency(league_avg_goals),
                'predictability': self._calculate_league_predictability(recent_matches),
                'home_advantage_factor': self._calculate_home_advantage(recent_matches)
            }
            
        except Exception as e:
            logger.error(f"Lig bağlam analizi hatası: {e}")
            return self._get_default_context()
    
    def _calculate_league_average(self, matches: List[Dict]) -> float:
        """
        Maçlardan lig ortalama golünü hesapla
        """
        if not matches:
            return self.default_league_avg
        
        total_goals = []
        for match in matches:
            home_goals = match.get('home_score', 0) or 0
            away_goals = match.get('away_score', 0) or 0
            # Sadece oynandı maçları dahil et
            if match.get('status') in ['FINISHED', 'FT', 'AET', 'PEN']:
                total_goals.append(home_goals + away_goals)
        
        if total_goals:
            return np.mean(total_goals)
        return self.default_league_avg
    
    def _determine_league_type(self, avg_goals: float) -> str:
        """
        Lig tipini belirle (gol ortalamasına göre)
        """
        if avg_goals >= 3.0:
            return 'high_scoring'
        elif avg_goals >= 2.3:
            return 'medium_scoring'
        else:
            return 'low_scoring'
    
    def _analyze_league_characteristics(self, avg_goals: float, league_type: str, 
                                       matches: List[Dict] = None) -> Dict:
        """
        Lig karakteristiklerini analiz et
        """
        characteristics = {
            'goal_distribution': 'normal',
            'home_dominance': 'moderate',
            'draw_frequency': 'normal',
            'upset_frequency': 'normal'
        }
        
        if matches and len(matches) >= 20:
            # Ev sahibi dominansı
            home_wins = sum(1 for m in matches if (m.get('home_score', 0) or 0) > (m.get('away_score', 0) or 0))
            away_wins = sum(1 for m in matches if (m.get('away_score', 0) or 0) > (m.get('home_score', 0) or 0))
            draws = sum(1 for m in matches if (m.get('home_score', 0) or 0) == (m.get('away_score', 0) or 0))
            
            total = len(matches)
            home_win_rate = home_wins / total if total > 0 else 0.45
            draw_rate = draws / total if total > 0 else 0.25
            
            # Ev dominansı
            if home_win_rate > 0.50:
                characteristics['home_dominance'] = 'strong'
            elif home_win_rate < 0.40:
                characteristics['home_dominance'] = 'weak'
            
            # Beraberlik sıklığı
            if draw_rate > 0.30:
                characteristics['draw_frequency'] = 'high'
            elif draw_rate < 0.20:
                characteristics['draw_frequency'] = 'low'
            
            # Gol dağılımı
            goals_per_match = [
                (m.get('home_score', 0) or 0) + (m.get('away_score', 0) or 0) 
                for m in matches
            ]
            std_goals = np.std(goals_per_match) if goals_per_match else 1.0
            
            if std_goals > 2.0:
                characteristics['goal_distribution'] = 'volatile'
            elif std_goals < 1.0:
                characteristics['goal_distribution'] = 'consistent'
        
        return characteristics
    
    def _calculate_lambda_factor(self, league_avg_goals: float) -> float:
        """
        Lambda hesaplaması için lig faktörü
        Dixon-Coles modelindeki gibi lig ortalamasını normalize et
        """
        # 2.5 gol referans alınarak normalize edilir
        # Lig ortalaması yüksekse lambda faktörü > 1.0
        # Lig ortalaması düşükse lambda faktörü < 1.0
        reference_goals = 2.5
        lambda_factor = league_avg_goals / reference_goals
        
        # Ekstrem değerleri sınırla
        lambda_factor = max(0.7, min(1.3, lambda_factor))
        
        logger.info(f"Lig lambda faktörü: {lambda_factor:.3f} (Lig ort: {league_avg_goals:.2f})")
        return lambda_factor
    
    def _calculate_scoring_tendency(self, avg_goals: float) -> float:
        """
        Ligin gol atma eğilimini hesapla (0-1 arası)
        """
        # 4 gol üstü = 1.0, 1 gol altı = 0.0
        tendency = (avg_goals - 1.0) / 3.0
        return max(0.0, min(1.0, tendency))
    
    def _calculate_defensive_tendency(self, avg_goals: float) -> float:
        """
        Ligin savunma eğilimini hesapla (0-1 arası)
        Düşük gol = yüksek savunma eğilimi
        """
        # 1.5 gol altı = 1.0 (çok savunmacı), 3.5 gol üstü = 0.0 (açık oyun)
        tendency = (3.5 - avg_goals) / 2.0
        return max(0.0, min(1.0, tendency))
    
    def _calculate_league_predictability(self, matches: List[Dict] = None) -> float:
        """
        Ligin tahmin edilebilirliğini hesapla
        """
        if not matches or len(matches) < 20:
            return 0.5  # Orta düzey tahmin edilebilirlik
        
        # Favori kazanma oranını kontrol et
        predictable_results = 0
        total_matches = 0
        
        for match in matches:
            home_score = match.get('home_score', 0) or 0
            away_score = match.get('away_score', 0) or 0
            
            if match.get('status') in ['FINISHED', 'FT']:
                total_matches += 1
                # Basit bir favori belirleme (ev sahibi genelde favori)
                if home_score >= away_score:  # Ev sahibi kazandı veya berabere
                    predictable_results += 1
        
        if total_matches > 0:
            predictability = predictable_results / total_matches
            # 0.3-0.7 arasına normalize et (çok tahmin edilebilir veya edilemez ligller yok)
            predictability = 0.3 + (predictability * 0.4)
            return predictability
        
        return 0.5
    
    def _calculate_home_advantage(self, matches: List[Dict] = None) -> float:
        """
        Ev sahibi avantaj faktörünü hesapla
        """
        if not matches or len(matches) < 10:
            return 1.1  # Varsayılan ev avantajı
        
        home_points = 0
        away_points = 0
        total_matches = 0
        
        for match in matches:
            if match.get('status') in ['FINISHED', 'FT']:
                home_score = match.get('home_score', 0) or 0
                away_score = match.get('away_score', 0) or 0
                
                if home_score > away_score:
                    home_points += 3
                elif home_score == away_score:
                    home_points += 1
                    away_points += 1
                else:
                    away_points += 3
                    
                total_matches += 1
        
        if total_matches > 0:
            avg_home_points = home_points / total_matches
            avg_away_points = away_points / total_matches
            
            if avg_home_points > 0:
                # Ev sahibi avantajı = ev puanları / deplasman puanları oranı
                advantage = 1.0 + (avg_home_points - avg_away_points) / 6.0
                # 0.95 - 1.15 arasında sınırla
                return max(0.95, min(1.15, advantage))
        
        return 1.1
    
    def get_cross_league_factor(self, home_league: str, away_league: str) -> float:
        """
        Farklı liglerden takımlar için düzeltme faktörü
        (Uluslararası maçlar için)
        """
        # Lig kalite skorları
        quality_scores = {
            'elite': 1.0,
            'high': 0.9,
            'medium': 0.8,
            'low': 0.7,
            'unknown': 0.85
        }
        
        home_quality = self.league_profiles.get(home_league, {}).get('quality', 'unknown')
        away_quality = self.league_profiles.get(away_league, {}).get('quality', 'unknown')
        
        home_score = quality_scores[home_quality]
        away_score = quality_scores[away_quality]
        
        # Kalite farkı faktörü
        quality_diff = home_score - away_score
        
        if abs(quality_diff) > 0.2:
            # Büyük kalite farkı varsa düzeltme uygula
            return 1.0 + (quality_diff * 0.2)
        
        return 1.0
    
    def _get_default_context(self) -> Dict:
        """
        Varsayılan lig bağlamı döndür
        """
        return {
            'league_name': 'Unknown',
            'avg_goals_per_match': self.default_league_avg,
            'league_type': 'medium_scoring',
            'league_quality': 'unknown',
            'lambda_factor': 1.0,
            'characteristics': {
                'goal_distribution': 'normal',
                'home_dominance': 'moderate',
                'draw_frequency': 'normal',
                'upset_frequency': 'normal'
            },
            'scoring_tendency': 0.5,
            'defensive_tendency': 0.5,
            'predictability': 0.5,
            'home_advantage_factor': 1.1
        }
    
    def calculate_adjusted_lambda(self, base_lambda: float, league_factor: float, 
                                 team_factor: float = 1.0) -> float:
        """
        Lig faktörü ile düzeltilmiş lambda hesapla
        
        Args:
            base_lambda: Temel lambda değeri (xG cross çarpımından)
            league_factor: Lig ortalama gol faktörü
            team_factor: Takım karakteristik faktörü
            
        Returns:
            Düzeltilmiş lambda değeri
        """
        # Dixon-Coles benzeri formül:
        # Lambda = base_lambda × league_factor × team_factor
        adjusted_lambda = base_lambda * league_factor * team_factor
        
        # Mantıklı sınırlar içinde tut
        adjusted_lambda = max(0.3, min(5.0, adjusted_lambda))
        
        logger.info(f"Lambda düzeltmesi: {base_lambda:.2f} × {league_factor:.3f} × {team_factor:.3f} = {adjusted_lambda:.2f}")
        
        return adjusted_lambda