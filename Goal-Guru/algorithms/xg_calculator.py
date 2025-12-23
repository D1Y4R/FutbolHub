"""
xG (Expected Goals) ve xGA (Expected Goals Against) Hesaplayıcı
Temel gol beklentisi hesaplamaları için kullanılır
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from math import log
from algorithms.league_context_analyzer import LeagueContextAnalyzer

logger = logging.getLogger(__name__)

class XGCalculator:
    """
    Expected Goals (xG) ve Expected Goals Against (xGA) hesaplayıcı
    """
    
    def __init__(self):
        self.weights_config = {
            'recent_matches': 999,  # Maksimum maç sayısı (sınırsız)
            'weight_distribution': [0.2, 1.0],  # Min-max ağırlık
            'home_advantage': 1.1,  # Ev sahibi avantajı
            'favorite_correction': 0.3,  # Favori takım düzeltmesi
            'days_limit': 60  # Son 60 gün - güncel veriler
        }
        # Lig bağlam analizörü
        self.league_analyzer = LeagueContextAnalyzer()
        
    def calculate_weights(self, num_matches):
        """
        Maç ağırlıklarını hesapla - son maçlara daha fazla önem
        """
        weights = np.linspace(
            self.weights_config['weight_distribution'][0],
            self.weights_config['weight_distribution'][1],
            min(num_matches, self.weights_config['recent_matches'])
        )
        return weights / weights.sum()  # Normalize
        
    def filter_last_60_days(self, matches):
        """
        Son 60 gündeki maçları filtrele - güncel veriler için
        """
        today = datetime.now()
        cutoff = today - timedelta(days=self.weights_config['days_limit'])
        filtered = []
        
        for match in matches:
            try:
                match_date = datetime.strptime(match.get('date', ''), '%Y-%m-%d')
                if match_date >= cutoff:
                    filtered.append(match)
            except:
                # Tarih parse edilemezse dahil et
                filtered.append(match)
                
        return filtered
        
    def calculate_xg_xga(self, matches, is_home=True):
        """
        Takımın xG ve xGA değerlerini hesapla
        
        Args:
            matches: Son maçların listesi (en yeni önce)
            is_home: Ev sahibi mi?
            
        Returns:
            tuple: (xG, xGA)
        """
        if not matches or len(matches) == 0:
            logger.warning("Maç verisi bulunamadı, varsayılan değerler kullanılıyor")
            return 1.3, 1.3
            
        # Son 60 gündeki maçları filtrele - güncel veriler
        filtered_matches = self.filter_last_60_days(matches)
        if not filtered_matches:
            logger.warning("Son 60 günde maç bulunamadı, varsayılan değerler kullanılıyor")
            return 1.3, 1.3
            
        # En fazla son 30 maçı al
        recent_matches = filtered_matches[:self.weights_config['recent_matches']]
        weights = self.calculate_weights(len(recent_matches))
        
        # Gol ve yenilen gol ortalamaları
        goals_scored = [m.get('goals_scored', 0) for m in recent_matches]
        goals_conceded = [m.get('goals_conceded', 0) for m in recent_matches]
        
        # Ağırlıklı ortalama
        xg = np.average(goals_scored, weights=weights) if goals_scored else 1.3
        xga = np.average(goals_conceded, weights=weights) if goals_conceded else 1.3
        
        # Ev sahibi avantajı
        if is_home:
            xg *= self.weights_config['home_advantage']
            xga *= 0.95  # Ev sahibi daha az gol yer
            
        logger.info(f"xG/xGA hesaplandı - xG: {xg:.2f}, xGA: {xga:.2f}, Ev sahibi: {is_home}")
        return xg, xga
        
    def calculate_xg_xga_with_elo(self, matches, elo_rating, opponent_elo, is_home=True):
        """
        Elo entegrasyonlu xG/xGA hesaplaması (rapordaki öneri)
        
        Args:
            matches: Maç listesi
            elo_rating: Takımın Elo rating'i
            opponent_elo: Rakibin Elo rating'i
            is_home: Ev sahibi mi?
            
        Returns:
            tuple: (xG, xGA)
        """
        # Temel xG/xGA hesapla
        base_xg, base_xga = self.calculate_xg_xga(matches, is_home)
        
        # Elo faktörü hesapla
        elo_factor = elo_rating / opponent_elo if opponent_elo != 0 else 1.0
        
        # Ev avantajı ile birleştir
        if is_home:
            elo_factor *= 1.1
        else:
            elo_factor *= 0.9
            
        # xG'yi Elo ile ayarla
        xg = base_xg * elo_factor
        
        # xGA'yı ters Elo faktörü ile ayarla
        xga_factor = opponent_elo / elo_rating if elo_rating != 0 else 1.0
        xga = base_xga * xga_factor
        
        # Sınırları kontrol et
        xg = max(0.5, min(5.0, xg))
        xga = max(0.5, min(5.0, xga))
        
        logger.info(f"Elo entegrasyonlu xG/xGA - xG: {xg:.2f}, xGA: {xga:.2f}, Elo faktör: {elo_factor:.2f}")
        return xg, xga
        
    def calculate_smart_lambda(self, team_data, opponent_data, match_context):
        """
        Kompozit akıllı lambda hesaplama sistemi
        
        Args:
            team_data: Takım verileri (xG, xGA, recent matches, etc.)
            opponent_data: Rakip takım verileri
            match_context: Maç bağlamı (ev/deplasman, h2h, motivation, etc.)
            
        Returns:
            float: Hesaplanan lambda değeri
        """
        # 1. Temel güç hesabı
        base_strength = self._calculate_base_strength(team_data, opponent_data)
        
        # 2. Form ve momentum faktörü
        form_factor = self._calculate_form_momentum_factor(team_data)
        
        # 3. Bağlamsal düzeltmeler
        context_modifier = self._calculate_context_modifier(team_data, opponent_data, match_context)
        
        # 4. Ham lambda hesapla
        lambda_raw = base_strength * form_factor * context_modifier
        
        # 5. Akıllı sınırlama
        lambda_final = self._apply_smart_limits(lambda_raw, team_data, opponent_data)
        
        logger.info(f"Kompozit Lambda - Temel: {base_strength:.2f}, Form: {form_factor:.2f}, "
                   f"Bağlam: {context_modifier:.2f}, Ham: {lambda_raw:.2f}, Final: {lambda_final:.2f}")
        
        return lambda_final
    
    def _calculate_base_strength(self, team_data, opponent_data):
        """
        Temel takım gücü hesaplama - ÇAPRAZ ÇARPMA MANTIĞI
        """
        # Takımın gol atma gücü
        team_attack = team_data.get('recent_avg_goals', team_data.get('xg', 1.2))
        
        # Rakibin savunma zayıflığı (yediği gol ortalaması)
        opponent_defense = opponent_data.get('recent_avg_conceded', opponent_data.get('xga', 1.2))
        
        # ÇAPRAZ ÇARPMA - Gerçek lambda hesabı
        base_lambda = team_attack * opponent_defense
        
        # xG etkisi (eğer mevcutsa, %30 ağırlıkla dahil et)
        if 'xg' in team_data:
            xg_factor = team_data['xg'] / team_attack if team_attack > 0 else 1.0
            # xG faktörünü sınırla (0.7-1.3 arası)
            xg_factor = max(0.7, min(1.3, xg_factor))
            base_lambda *= (0.7 + 0.3 * xg_factor)
        
        logger.info(f"Çapraz Lambda - Atak: {team_attack:.2f} × Savunma: {opponent_defense:.2f} = {base_lambda:.2f}")
        
        # Mantıklı sınırlar içinde tut (çok düşük veya çok yüksek olmasın)
        return max(0.3, min(4.0, base_lambda))
    
    def _calculate_form_momentum_factor(self, team_data):
        """
        Form ve momentum faktörü hesaplama
        """
        recent_matches = team_data.get('recent_matches', [])
        if not recent_matches:
            return 1.0
            
        # Son 5 maç sonuçları
        last_5_results = recent_matches[:5]
        form_points = 0
        
        for match in last_5_results:
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            if goals_for > goals_against:
                form_points += 3
            elif goals_for == goals_against:
                form_points += 1
        
        # Form puanını faktöre dönüştür (0-15 puan -> 0.9-1.3 faktör)
        form_percentage = form_points / 15.0
        form_factor = 0.9 + (form_percentage * 0.4)
        
        # Gol trendi analizi
        if len(recent_matches) >= 5:
            recent_goals = [m.get('goals_scored', 0) for m in recent_matches[:5]]
            older_goals = [m.get('goals_scored', 0) for m in recent_matches[5:10]] if len(recent_matches) >= 10 else recent_goals
            
            recent_avg = np.mean(recent_goals)
            older_avg = np.mean(older_goals)
            
            # Trend faktörü
            if recent_avg > older_avg * 1.2:
                trend_factor = 1.1  # Yükselen trend
            elif recent_avg < older_avg * 0.8:
                trend_factor = 0.9  # Düşen trend
            else:
                trend_factor = 1.0  # Stabil
                
            form_factor *= trend_factor
        
        # Tutarlılık faktörü
        if len(recent_matches) >= 5:
            goals_list = [m.get('goals_scored', 0) for m in recent_matches[:5]]
            consistency = 1 / (1 + np.std(goals_list) * 0.2)
            form_factor *= consistency
        
        return max(0.8, min(1.4, form_factor))
    
    def _calculate_context_modifier(self, team_data, opponent_data, match_context):
        """
        Bağlamsal düzeltme faktörü hesaplama
        """
        modifier = 1.0
        
        # Ev/Deplasman faktörü
        if match_context.get('is_home', True):
            venue_factor = 1.10  # Ev avantajı (azaltıldı)
        else:
            venue_factor = 0.95  # Deplasman dezavantajı (artırıldı)
        modifier *= venue_factor
        
        # H2H faktörü
        h2h_data = match_context.get('h2h_data', {})
        if h2h_data:
            h2h_wins = h2h_data.get('wins', 0)
            h2h_total = h2h_data.get('total', 0)
            if h2h_total > 0:
                h2h_win_rate = h2h_wins / h2h_total
                h2h_factor = 0.9 + (h2h_win_rate * 0.2)  # 0.9-1.1 arası
                modifier *= h2h_factor
        
        # Motivasyon faktörü (lig durumu)
        motivation = match_context.get('motivation_level', 'normal')
        if motivation == 'very_high':  # Şampiyonluk yarışı, küme düşme, vs.
            modifier *= 1.1
        elif motivation == 'low':  # Sezon sonu, anlamsız maç
            modifier *= 0.9
        
        # Dinlenme süresi faktörü
        rest_days = match_context.get('rest_days', 3)
        if rest_days < 3:
            modifier *= 0.95  # Yorgunluk
        elif rest_days > 7:
            modifier *= 0.98  # Ritim kaybı
        
        # Derbi/özel maç faktörü
        if match_context.get('is_derby', False):
            modifier *= 1.05  # Derbilerde daha fazla gol
        
        return max(0.7, min(1.3, modifier))
    
    def _apply_smart_limits(self, lambda_raw, team_data, opponent_data):
        """
        Akıllı lambda sınırlama
        """
        # Maç tipini belirle
        match_type = self._determine_match_type(team_data, opponent_data)
        
        if match_type == 'extreme':
            # Ekstrem maçlar için daha yüksek sınır
            return min(lambda_raw, 4.5)
        elif match_type == 'low_scoring':
            # Düşük skorlu maçlar için azaltma
            return max(lambda_raw * 0.8, 0.5)
        else:
            # Normal maçlar için standart sınırlar
            return np.clip(lambda_raw, 0.8, 3.0)
    
    def _determine_match_type(self, team_data, opponent_data):
        """
        Maç tipini belirle (normal/ekstrem/düşük skorlu)
        """
        # Takımların ortalama gol istatistikleri
        team_avg_goals = team_data.get('recent_avg_goals', 1.2)
        team_avg_conceded = team_data.get('recent_avg_conceded', 1.2)
        opp_avg_goals = opponent_data.get('recent_avg_goals', 1.2)
        opp_avg_conceded = opponent_data.get('recent_avg_conceded', 1.2)
        
        # Toplam beklenen gol
        expected_total = (team_avg_goals + opp_avg_goals + team_avg_conceded + opp_avg_conceded) / 2
        
        if expected_total > 3.5:
            return 'extreme'
        elif expected_total < 2.0:
            return 'low_scoring'
        else:
            return 'normal'
    
    def calculate_lambda_cross(self, home_xg, home_xga, away_xg, away_xga, elo_diff=0, 
                             home_team_data=None, away_team_data=None, match_context=None):
        """
        Logaritmik lambda hesaplama - Ev/Deplasman performansı ağırlıklı
        
        Args:
            home_xg: Ev sahibi xG
            home_xga: Ev sahibi xGA  
            away_xg: Deplasman xG
            away_xga: Deplasman xGA
            elo_diff: Elo farkı (home - away)
            home_team_data: Ev sahibi takım verileri (opsiyonel)
            away_team_data: Deplasman takım verileri (opsiyonel)
            match_context: Maç bağlamı (opsiyonel)
            
        Returns:
            tuple: (lambda_home, lambda_away)
        """
        logger.info("Logaritmik lambda hesaplama sistemi kullanılıyor - Ev/Deplasman performansı ağırlıklı")
        
        # EV/DEPLASMAN PERFORMANSI AĞIRLIKLI HESAPLAMA
        # Son 5 ev/deplasman maçı verilerini kullan
        venue_weight = 0.65  # Ev/deplasman performansına %65 ağırlık
        recent_weight = 0.35  # Genel son maçlara %35 ağırlık
        
        # Ev sahibi için venue-specific xG hesapla
        if home_team_data and 'venue_specific_avg_goals' in home_team_data:
            venue_home_xg = home_team_data['venue_specific_avg_goals']
            adjusted_home_xg = (venue_home_xg * venue_weight) + (home_xg * recent_weight)
            logger.info(f"Ev sahibi xG düzeltmesi: {home_xg:.2f} -> {adjusted_home_xg:.2f} (son 5 ev: {venue_home_xg:.2f})")
            home_xg = adjusted_home_xg
            
            # xGA için de aynı işlem
            venue_home_xga = home_team_data['venue_specific_avg_conceded']
            adjusted_home_xga = (venue_home_xga * venue_weight) + (home_xga * recent_weight)
            home_xga = adjusted_home_xga
        
        # Deplasman takımı için venue-specific xG hesapla
        if away_team_data and 'venue_specific_avg_goals' in away_team_data:
            venue_away_xg = away_team_data['venue_specific_avg_goals']
            adjusted_away_xg = (venue_away_xg * venue_weight) + (away_xg * recent_weight)
            logger.info(f"Deplasman xG düzeltmesi: {away_xg:.2f} -> {adjusted_away_xg:.2f} (son 5 dep: {venue_away_xg:.2f})")
            away_xg = adjusted_away_xg
            
            # xGA için de aynı işlem
            venue_away_xga = away_team_data['venue_specific_avg_conceded']
            adjusted_away_xga = (venue_away_xga * venue_weight) + (away_xga * recent_weight)
            away_xga = adjusted_away_xga
        
        # Favori takım düzeltmeleri (mevcut kod)
        if elo_diff > 0 and home_xg < away_xg:
            home_xg = min(home_xg + 0.3, away_xg * 1.2)
        elif elo_diff < 0 and away_xg < home_xg:
            away_xg = min(away_xg + 0.3, home_xg * 1.2)
            
        if elo_diff > 0 and home_xga > away_xga * 1.2:
            home_xga = max(home_xga - 0.3, away_xga * 0.8)
        elif elo_diff < 0 and away_xga > home_xga * 1.2:
            away_xga = max(away_xga - 0.3, home_xga * 0.8)
        
        # Son 5 ev/deplasman performansına ekstra bonus
        home_venue_bonus = 1.0
        away_venue_bonus = 1.0
        
        if home_team_data and 'home_performance' in home_team_data:
            home_perf = home_team_data['home_performance']
            if 'last_5_win_rate' in home_perf and home_perf['last_5_win_rate'] > 0.6:
                home_venue_bonus = 1.1  # %10 bonus
                logger.info(f"Ev sahibi son 5 ev maçı bonus: {home_perf['last_5_win_rate']:.2%} kazanma")
        
        if away_team_data and 'away_performance' in away_team_data:
            away_perf = away_team_data['away_performance']
            if 'last_5_win_rate' in away_perf and away_perf['last_5_win_rate'] > 0.4:
                away_venue_bonus = 1.05  # %5 bonus (deplasmanda kazanmak daha zor)
                logger.info(f"Deplasman son 5 dep maçı bonus: {away_perf['last_5_win_rate']:.2%} kazanma")
        
        # Lig ortalama gol faktörünü hesapla - SADECE FARKLI LİGLER İÇİN
        league_factor = 1.0  # Varsayılan (aynı lig için nötr)
        
        # Farklı liglerden mi kontrol et
        home_league = match_context.get('home_league_name') if match_context else None
        away_league = match_context.get('away_league_name') if match_context else None
        
        # Sadece farklı ligler veya kupa maçlarında lig faktörü uygula
        if home_league and away_league and home_league != away_league:
            # Farklı liglerden takımlar - lig faktörü hesapla
            home_league_context = self.league_analyzer.analyze_league_context(home_league, [])
            away_league_context = self.league_analyzer.analyze_league_context(away_league, [])
            
            # Her takım için kendi lig faktörünü uygulayacağız
            home_league_factor = home_league_context['lambda_factor']
            away_league_factor = away_league_context['lambda_factor']
            
            # Ortalama faktör (ağırlıklı ortalama için kullanılacak)
            league_factor = (home_league_factor + away_league_factor) / 2
            
            logger.info(f"Farklı ligler - Ev: {home_league} ({home_league_factor:.3f}), Dep: {away_league} ({away_league_factor:.3f})")
            logger.info(f"Ortalama lig faktörü: {league_factor:.3f}")
        elif match_context and 'is_cup_match' in match_context and match_context['is_cup_match']:
            # Kupa maçı - liglerden bağımsız olarak standart faktör
            league_factor = 1.05  # Kupa maçları genelde daha heyecanlı
            logger.info("Kupa maçı - Lig faktörü: 1.05")
        else:
            # Aynı ligden takımlar - faktör uygulanmaz
            logger.info("Aynı ligden takımlar - Lig faktörü: 1.000 (nötr)")
        
        # AĞIRLIKLI ORTALAMA YAKLAŞIMI - Daha dengeli lambda hesaplama
        # Temel lambda = xG × xGA
        base_lambda_home = home_xg * away_xga
        base_lambda_away = away_xg * home_xga
        
        # Güç oranı ve log düzeltmesi (0.9-1.1 arası)
        strength_ratio = home_xg / away_xg if away_xg > 0 else 2.0
        log_adjustment_home = 1 + 0.1 * log(strength_ratio + 1)  # 0.9-1.1 arası
        log_adjustment_away = 1 - 0.1 * log(strength_ratio + 1)  # 0.9-1.1 arası
        
        # Faktörlerin ağırlıklı ortalaması
        # Ağırlıklar: log_adj=%40, venue=%30, league=%30
        weight_log = 0.4
        weight_venue = 0.3
        weight_league = 0.3
        
        # Ev sahibi için kombine faktör
        combined_factor_home = (
            weight_log * log_adjustment_home +
            weight_venue * home_venue_bonus +
            weight_league * league_factor
        ) / (weight_log + weight_venue + weight_league)
        
        # Deplasman için kombine faktör
        combined_factor_away = (
            weight_log * log_adjustment_away +
            weight_venue * away_venue_bonus +
            weight_league * league_factor
        ) / (weight_log + weight_venue + weight_league)
        
        # Final lambda = temel × kombine faktör
        lambda_home = base_lambda_home * combined_factor_home
        lambda_away = base_lambda_away * combined_factor_away
        
        logger.info(f"Ağırlıklı Lambda - Ev: {lambda_home:.2f}, Deplasman: {lambda_away:.2f}")
        logger.info(f"  Temel lambda (Ev/Dep): {base_lambda_home:.2f}/{base_lambda_away:.2f}")
        logger.info(f"  Kombine faktör (Ev/Dep): {combined_factor_home:.3f}/{combined_factor_away:.3f}")
        logger.info(f"  - Log düzeltme: {log_adjustment_home:.3f}/{log_adjustment_away:.3f} (%40)")
        logger.info(f"  - Venue bonus: {home_venue_bonus:.3f}/{away_venue_bonus:.3f} (%30)")
        logger.info(f"  - Lig faktörü: {league_factor:.3f} (%30)")
        
        # Ekstrem maç kontrolü
        from algorithms.extreme_detector import ExtremeMatchDetector
        detector = ExtremeMatchDetector()
        
        home_stats = {'xg': home_xg, 'xga': home_xga}
        away_stats = {'xg': away_xg, 'xga': away_xga}
        
        is_extreme, _ = detector.is_extreme_match(home_stats, away_stats)
        lambda_cap = detector.get_lambda_cap(is_extreme, home_stats)
        
        lambda_home = max(0.5, min(lambda_cap, lambda_home))
        lambda_away = max(0.5, min(lambda_cap, lambda_away))
        
        return lambda_home, lambda_away