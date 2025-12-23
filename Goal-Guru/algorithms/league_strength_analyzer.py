"""
League Strength Analyzer
Farklı liglerin güç seviyelerini analiz eder ve takımlar arası gerçek güç farkını hesaplar.
UEFA katsayıları ve lig seviyeleri kullanılarak dinamik ayarlama yapar.
"""

import logging
from typing import Dict, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

class LeagueStrengthAnalyzer:
    def __init__(self):
        # GERÇEK UEFA 5-YILLIK LİG KATSAYILARI (2024-2025 Sezon)
        # Kaynak: Official UEFA Club Coefficients Rankings
        self.league_coefficients = {
            # Top 5 Elite Leagues
            "Premier League": 94.157,
            "La Liga": 91.216,
            "Serie A": 85.962,
            "Bundesliga": 83.740,
            "Ligue 1": 74.165,
            
            # Tier 1 Strong Leagues (60-70)
            "Primeira Liga": 67.632,
            "Eredivisie": 63.150,
            "Jupiler Pro League": 58.300,
            "First Division A": 58.300,  # Belgium alias
            
            # Tier 2 Medium Leagues (50-60)
            "Scottish Premiership": 55.625,
            "Austrian Bundesliga": 54.050,
            "Super Lig": 52.500,
            "Süper Lig": 52.500,  # Turkish name
            "Czech Liga": 50.850,
            
            # Tier 3 Developing Leagues (40-50)
            "Eliteserien": 47.900,
            "Super League": 47.050,  # Greece
            "Danish Superliga": 46.825,
            "Swiss Super League": 45.975,
            "Croatian First League": 44.625,
            "Ukrainian Premier League": 43.600,
            "Serbian SuperLiga": 42.875,
            "Ekstraklasa": 41.500,
            "Allsvenskan": 40.850,
            
            # Tier 4 Secondary Leagues (30-40)
            "Championship": 38.500,  # England 2nd tier
            "Romanian Liga I": 37.950,
            "Israeli Premier League": 36.750,
            "Bulgarian First League": 35.625,
            "MLS": 35.000,
            "Liga MX": 34.500,
            "Brasileirão": 33.800,
            "Argentine Primera": 32.500,
            "J1 League": 31.200,
            
            # Unknown/Default
            "Unknown": 25.000,
        }
        
        # Premier League'i referans al (100 puan)
        self.reference_coefficient = 94.157
        
        # Lig strength multipliers (coefficient / reference)
        self.league_strength_multipliers = {}
        for league, coef in self.league_coefficients.items():
            self.league_strength_multipliers[league] = coef / self.reference_coefficient
        
        # Ülke futbol güç sıralaması (eski sistem - backward compatibility)
        self.country_strength = {
            # Elite Ülkeler (90-100)
            "England": 100, "Spain": 97, "Germany": 89, "Italy": 91, "France": 79,
            
            # Güçlü Ülkeler (60-79)
            "Netherlands": 67, "Portugal": 72, "Belgium": 62, "Turkey": 56, "Scotland": 59,
            "Austria": 57, "Switzerland": 49, "Denmark": 50,
            
            # Orta Seviye Ülkeler (40-59)
            "Russia": 46, "Ukraine": 46, "Czech Republic": 54, "Greece": 50, "Croatia": 47,
            "Serbia": 46, "Poland": 44, "Sweden": 43, "Norway": 51, "Romania": 40,
            "Israel": 39, "Bulgaria": 38,
            
            # Alt Seviye Ülkeler (20-39)
            "Hungary": 36, "Slovakia": 33, "Slovenia": 30, "Cyprus": 28, "Belarus": 25,
            "Azerbaijan": 24, "Kazakhstan": 23, "Bosnia and Herzegovina": 27, "Albania": 26,
            "North Macedonia": 25, "Ireland": 29, "Northern Ireland": 24, "Finland": 28,
            
            # Düşük Seviye Ülkeler (10-19)
            "Iceland": 22, "Luxembourg": 18, "Armenia": 20, "Georgia": 19, "Moldova": 16,
            "Estonia": 14, "Latvia": 13, "Lithuania": 12, "Malta": 11,
            
            # Çok Düşük Seviye (5-10)
            "Faroe Islands": 10, "Andorra": 7, "San Marino": 5, "Gibraltar": 6
        }
        
        # Lig seviye çarpanları (1. lig = 1.0, 2. lig = 0.8, vb.)
        self.league_level_multipliers = {
            1: 1.0,    # Birinci lig
            2: 0.8,    # İkinci lig
            3: 0.6,    # Üçüncü lig
            4: 0.4,    # Dördüncü lig
            5: 0.25,   # Beşinci lig
            6: 0.15,   # Altıncı lig ve altı (amatör)
            "cup": 1.0, # Kupa maçları için birinci lig seviyesi
            "youth": 0.3, # Genç ligler
            "amateur": 0.1  # Amatör ligler (BAL ligi vb.)
        }
        
        # Kupa maçları için özel çarpanlar
        self.cup_competition_factors = {
            "UEFA Champions League": 1.2,      # En yüksek seviye
            "Champions League": 1.2,
            "UEFA Europa League": 1.15,
            "Europa League": 1.15,
            "UEFA Conference League": 1.1,
            "Conference League": 1.1,
            "FA Cup": 1.05,                    # Ulusal kupalar
            "Copa del Rey": 1.05,
            "DFB Pokal": 1.05,
            "Coppa Italia": 1.05,
            "Coupe de France": 1.05,
            "Türkiye Kupası": 1.05,
            "EFL Cup": 1.0,                    # Lig kupaları
            "Carabao Cup": 1.0,
        }
        
        # Lig seviye kategorileri
        self.league_tiers = {
            "elite": (90, 100),
            "strong": (75, 89),
            "medium": (60, 74),
            "lower": (40, 59),
            "amateur": (0, 39)
        }
        
        logger.info("LeagueStrengthAnalyzer başlatıldı")
    
    def detect_league_level(self, league_name: str) -> Union[int, str]:
        """Lig isminden seviyeyi tespit eder"""
        if not league_name:
            return 1
            
        league_lower = league_name.lower()
        
        # Amatör ligler
        if any(x in league_lower for x in ["bal", "amateur", "regional", "bölgesel", "yerel", "il ligi"]):
            return 6  # Amatör
            
        # Sayısal seviye tespiti
        if "2" in league_name or any(x in league_lower for x in ["championship", "segunda", "serie b", "ligue 2", "2. bundesliga", "1. lig"]):
            return 2
        elif "3" in league_name or any(x in league_lower for x in ["league one", "tercera", "serie c", "3. liga", "2. lig"]):
            return 3
        elif "4" in league_name or any(x in league_lower for x in ["league two", "regionalliga", "3. lig"]):
            return 4
        elif "5" in league_name or any(x in league_lower for x in ["national league", "oberliga"]):
            return 5
            
        # Kupa maçları
        if any(x in league_lower for x in ["cup", "kupa", "copa", "coupe", "pokal"]):
            return "cup"
            
        # Genç ligler
        if any(x in league_lower for x in ["u19", "u21", "youth", "genç", "junior"]):
            return "youth"
            
        # Varsayılan olarak birinci lig
        return 1
    
    def get_country_strength(self, country_name: str) -> int:
        """Ülke futbol gücünü döndürür"""
        if not country_name:
            return 50  # Bilinmeyen ülke için ortalama
            
        # Tam eşleşme
        if country_name in self.country_strength:
            return self.country_strength[country_name]
            
        # Kısmi eşleşme
        country_lower = country_name.lower()
        for country, strength in self.country_strength.items():
            if country.lower() in country_lower or country_lower in country.lower():
                return strength
                
        # Güney Amerika ülkeleri için özel kontrol
        south_america_countries = {
            "Brazil": 85, "Argentina": 83, "Uruguay": 75, "Colombia": 73,
            "Chile": 70, "Paraguay": 65, "Ecuador": 63, "Peru": 60,
            "Venezuela": 55, "Bolivia": 50
        }
        
        for country, strength in south_america_countries.items():
            if country.lower() in country_lower:
                return strength
                
        # Bilinmeyen ülke için varsayılan
        return 45
    
    def calculate_dynamic_strength(self, country: str, league_name: str) -> int:
        """Ülke ve lig seviyesine göre dinamik güç hesaplar"""
        country_strength = self.get_country_strength(country)
        league_level = self.detect_league_level(league_name)
        
        # Lig seviyesi çarpanını al
        if isinstance(league_level, int):
            multiplier = self.league_level_multipliers.get(league_level, 0.5)
        else:
            multiplier = self.league_level_multipliers.get(league_level, 1.0)
            
        # Dinamik güç hesaplama
        dynamic_strength = int(country_strength * multiplier)
        
        # Min-max sınırları
        return max(10, min(100, dynamic_strength))
    
    def get_league_strength(self, league_name: str, country: Optional[str] = None) -> int:
        """Lig güç seviyesini döndürür - gerçek UEFA katsayıları kullanır"""
        # Önce özel durumları kontrol et (UEFA Kupaları)
        if league_name and "champions league" in league_name.lower():
            return 95
        elif league_name and "europa league" in league_name.lower():
            return 90
        elif league_name and "conference league" in league_name.lower():
            return 85
        
        # Gerçek UEFA katsayısını kullan
        league_multiplier = self.get_league_multiplier(league_name)
        
        # 0-100 skalasına çevir
        strength = int(league_multiplier * 100)
        
        # Eğer bulunamazsa eski sistemi kullan
        if strength < 25:
            return self.calculate_dynamic_strength(country if country else "", league_name if league_name else "")
            
        return strength
    
    def get_league_multiplier(self, league_name: str) -> float:
        """
        Lig strength multiplier'ı döndürür (0-1 arası)
        Premier League = 1.0, Süper Lig = 0.56 vb.
        """
        if not league_name:
            return 0.27  # Unknown league default
        
        # Direct match
        if league_name in self.league_strength_multipliers:
            return self.league_strength_multipliers[league_name]
        
        # Case-insensitive match
        league_lower = league_name.lower()
        for key, multiplier in self.league_strength_multipliers.items():
            if key.lower() == league_lower:
                return multiplier
        
        # Partial match
        for key, multiplier in self.league_strength_multipliers.items():
            if key.lower() in league_lower or league_lower in key.lower():
                return multiplier
        
        # Default for unknown
        logger.warning(f"Unknown league: {league_name}, using default multiplier")
        return 0.27
    
    def get_league_tier(self, strength: int) -> str:
        """Lig gücüne göre seviye kategorisi döndürür"""
        for tier, (min_str, max_str) in self.league_tiers.items():
            if min_str <= strength <= max_str:
                return tier
        return "medium"
    
    def calculate_strength_difference(self, home_league: str, away_league: str, 
                                     competition_name: Optional[str] = None,
                                     home_country: Optional[str] = None,
                                     away_country: Optional[str] = None) -> Dict[str, Any]:
        """
        İki takım arasındaki lig güç farkını hesaplar
        
        Returns:
            Dict: {
                'home_strength': int,
                'away_strength': int,
                'strength_ratio': float,
                'tier_difference': int,
                'adjustment_factor': float,
                'is_cross_tier': bool,
                'analysis': str
            }
        """
        home_strength = self.get_league_strength(home_league, home_country)
        away_strength = self.get_league_strength(away_league, away_country)
        
        # Kupa maçı faktörü
        cup_factor = 1.0
        if competition_name:
            for cup_name, factor in self.cup_competition_factors.items():
                if cup_name.lower() in competition_name.lower():
                    cup_factor = factor
                    break
        
        # Güç oranı hesapla
        strength_ratio = (home_strength / away_strength) if away_strength > 0 else 2.0
        
        # Seviye farkı
        home_tier = self.get_league_tier(home_strength)
        away_tier = self.get_league_tier(away_strength)
        
        tier_values = {"elite": 5, "strong": 4, "medium": 3, "lower": 2, "amateur": 1}
        tier_difference = abs(tier_values.get(home_tier, 3) - tier_values.get(away_tier, 3))
        
        # Ayarlama faktörü hesapla
        if tier_difference >= 3:  # Çok büyük fark (örn: elite vs amateur)
            adjustment_factor = 0.3 * cup_factor
        elif tier_difference == 2:  # Büyük fark (örn: elite vs medium)
            adjustment_factor = 0.5 * cup_factor
        elif tier_difference == 1:  # Orta fark (örn: strong vs medium)
            adjustment_factor = 0.7 * cup_factor
        else:  # Aynı seviye
            adjustment_factor = 0.9 * cup_factor
        
        # Analiz metni oluştur
        if tier_difference >= 2:
            analysis = f"Büyük lig farkı var! {home_league} ({home_tier}) vs {away_league} ({away_tier})"
        elif tier_difference == 1:
            analysis = f"Orta seviye lig farkı: {home_league} vs {away_league}"
        else:
            analysis = f"Benzer seviye ligler: {home_league} vs {away_league}"
        
        return {
            'home_strength': home_strength,
            'away_strength': away_strength,
            'strength_ratio': strength_ratio,
            'tier_difference': tier_difference,
            'adjustment_factor': adjustment_factor,
            'is_cross_tier': tier_difference >= 2,
            'home_tier': home_tier,
            'away_tier': away_tier,
            'analysis': analysis
        }
    
    def adjust_team_strength(self, team_xg: float, opponent_xg: float, 
                           team_league: str, opponent_league: str,
                           competition_name: Optional[str] = None,
                           team_country: Optional[str] = None,
                           opponent_country: Optional[str] = None) -> Tuple[float, float]:
        """
        Lig güç farkına göre takım xG değerlerini ayarlar
        
        Returns:
            Tuple[float, float]: (adjusted_team_xg, adjusted_opponent_xg)
        """
        # Güç farkı analizi
        strength_analysis = self.calculate_strength_difference(
            team_league, opponent_league, competition_name, team_country, opponent_country
        )
        
        # Büyük lig farkı varsa ayarlama yap
        if strength_analysis['is_cross_tier']:
            team_strength = strength_analysis['home_strength']
            opp_strength = strength_analysis['away_strength']
            adjustment = strength_analysis['adjustment_factor']
            
            if team_strength > opp_strength:
                # Güçlü takım lehine ayarlama
                strength_boost = 1 + (0.3 * (1 - adjustment))  # Max %30 artış
                weakness_penalty = adjustment  # Zayıf takım cezası
                
                adjusted_team_xg = team_xg * strength_boost
                adjusted_opponent_xg = opponent_xg * weakness_penalty
                
                logger.info(f"Lig farkı ayarlaması: {team_league} lehine - "
                          f"xG: {team_xg:.2f} -> {adjusted_team_xg:.2f}, "
                          f"Rakip xG: {opponent_xg:.2f} -> {adjusted_opponent_xg:.2f}")
            else:
                # Tersi durum
                strength_boost = 1 + (0.3 * (1 - adjustment))
                weakness_penalty = adjustment
                
                adjusted_team_xg = team_xg * weakness_penalty
                adjusted_opponent_xg = opponent_xg * strength_boost
                
                logger.info(f"Lig farkı ayarlaması: {opponent_league} lehine")
            
            return adjusted_team_xg, adjusted_opponent_xg
        
        # Lig farkı az ise minimal ayarlama
        return team_xg, opponent_xg
    
    def get_underdog_boost(self, weaker_league_strength: int, 
                          stronger_league_strength: int,
                          is_cup_match: bool = False) -> float:
        """
        Kupa maçlarında zayıf takım için sürpriz faktörü hesaplar
        
        Returns:
            float: Underdog boost faktörü (1.0-1.2 arası)
        """
        if not is_cup_match:
            return 1.0
        
        strength_diff = stronger_league_strength - weaker_league_strength
        
        # Kupa maçlarında ezeli rakip etkisi
        if strength_diff > 50:  # Çok büyük fark
            return 1.15  # %15 sürpriz şansı
        elif strength_diff > 30:  # Büyük fark
            return 1.10  # %10 sürpriz şansı
        elif strength_diff > 15:  # Orta fark
            return 1.05  # %5 sürpriz şansı
        else:
            return 1.0  # Normal
    
    def get_detailed_analysis(self, home_team: str, away_team: str,
                            home_league: str, away_league: str,
                            competition_name: Optional[str] = None,
                            home_country: Optional[str] = None,
                            away_country: Optional[str] = None) -> Dict[str, Any]:
        """
        Detaylı lig farkı analizi döndürür
        """
        analysis = self.calculate_strength_difference(home_league, away_league, competition_name, home_country, away_country)
        
        # Kupa maçı mı?
        is_cup = competition_name and any(
            cup in competition_name.lower() 
            for cup in ['cup', 'copa', 'pokal', 'coppa', 'coupe', 'kupası']
        )
        
        # xG multiplier'ları hesapla
        home_xg_multiplier = 1.0
        away_xg_multiplier = 1.0
        
        if analysis['is_cross_tier']:
            adjustment = analysis['adjustment_factor']
            
            if analysis['home_strength'] > analysis['away_strength']:
                # Ev sahibi daha güçlü ligden
                home_xg_multiplier = 1 + (0.3 * (1 - adjustment))  # Max %30 artış
                away_xg_multiplier = adjustment  # Zayıf takım cezası
            else:
                # Deplasman takımı daha güçlü ligden
                home_xg_multiplier = adjustment  # Zayıf takım cezası
                away_xg_multiplier = 1 + (0.3 * (1 - adjustment))  # Max %30 artış
        
        # Detaylı analiz ekle
        analysis['is_cup_match'] = is_cup
        analysis['home_xg_multiplier'] = home_xg_multiplier
        analysis['away_xg_multiplier'] = away_xg_multiplier
        analysis['recommendation'] = ""
        
        if analysis['is_cross_tier']:
            if analysis['home_strength'] > analysis['away_strength']:
                analysis['recommendation'] = (
                    f"{home_team} büyük favori! {home_league} çok daha güçlü bir lig. "
                    f"Form ne olursa olsun, kalite farkı belirleyici olacak."
                )
            else:
                analysis['recommendation'] = (
                    f"{away_team} büyük favori! {away_league} çok daha güçlü bir lig. "
                    f"Form ne olursa olsun, kalite farkı belirleyici olacak."
                )
            
            if is_cup:
                analysis['recommendation'] += " Ancak kupa maçı olduğu için sürpriz ihtimali var."
        else:
            analysis['recommendation'] = (
                "Benzer seviye ligler. Form ve güncel performans daha belirleyici olacak."
            )
        
        return analysis