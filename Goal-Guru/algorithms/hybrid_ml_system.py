"""
Hibrit ML Rating Sistemi
Glicko-2, TrueSkill ve Deep Learning kombinasyonu
"""
import numpy as np
import logging
from datetime import datetime
import math
from .glicko2_rating import Glicko2System
from .trueskill_adapter import TrueSkillAdapter
from .xg_rating_system import XGRatingSystem

logger = logging.getLogger(__name__)

class HybridMLSystem:
    """
    Gelişmiş hibrit rating sistemi
    Glicko-2 + TrueSkill + ML kombinasyonu
    """
    
    def __init__(self):
        # Alt sistemler
        self.glicko2 = Glicko2System()
        self.trueskill = TrueSkillAdapter()
        self.xg_rating = XGRatingSystem()  # xG rating sistemi eklendi
        
        # Hibrit parametreler - xG ağırlığı artırıldı
        self.system_weights = {
            'glicko2': 0.15,     # %15 Glicko-2 (azaltıldı)
            'trueskill': 0.10,   # %10 TrueSkill (azaltıldı)
            'recent_form': 0.25, # %25 Son 5 maç formu
            'xg_rating': 0.40,   # %40 xG tabanlı rating (ARTIRILDI)
            'ml_factor': 0.10    # %10 ML düzeltmesi (azaltıldı)
        }
        
        # Cache
        self.hybrid_ratings = {}  # Takım ID -> hybrid rating
        self.performance_history = {}  # Takım ID -> performans geçmişi
        
    def calculate_hybrid_rating(self, team_id, matches):
        """
        Takım için hibrit rating hesapla
        
        Args:
            team_id: Takım ID
            matches: Maç listesi
            
        Returns:
            dict: Hibrit rating detayları
        """
        # Her sistemden rating al
        glicko2_values = self.glicko2.calculate_team_glicko2(team_id, matches)
        trueskill_values = self.trueskill.calculate_team_trueskill(team_id, matches)
        
        # xG tabanlı rating'i hesapla
        xg_strength = self._calculate_xg_rating(team_id, matches)
        
        # Takım dinamikleri analizi
        team_dynamics = self.trueskill.analyze_team_dynamics(team_id, matches[:10])
        
        # ML faktörlerini hesapla
        ml_factors = self._calculate_ml_factors(team_id, matches, glicko2_values, trueskill_values)
        
        # Son form rating'ini hesapla
        recent_form_rating = self._calculate_recent_form_rating(matches)
        
        # Hibrit rating hesapla
        # Glicko-2 ve TrueSkill'i normalize et (0-3000 aralığına)
        normalized_glicko2 = glicko2_values['rating']
        normalized_trueskill = (trueskill_values['skill'] - 3) * 60  # TrueSkill 3-47 -> 0-2640
        normalized_xg = xg_strength['overall_rating'] * 30  # xG 0-100 -> 0-3000
        
        # Ağırlıklı ortalama - xG dahil
        base_rating = (
            self.system_weights['glicko2'] * normalized_glicko2 +
            self.system_weights['trueskill'] * normalized_trueskill +
            self.system_weights['recent_form'] * recent_form_rating +
            self.system_weights['xg_rating'] * normalized_xg
        )
        
        # ML düzeltmesi ekle
        ml_adjustment = ml_factors['rating_adjustment'] * self.system_weights['ml_factor']
        hybrid_rating = base_rating + ml_adjustment
        
        # Belirsizlik hesapla (RD ve sigma kombinasyonu)
        combined_uncertainty = math.sqrt(
            (glicko2_values['rd'] / 350)**2 * self.system_weights['glicko2'] +
            (trueskill_values['uncertainty'] / 8.333)**2 * self.system_weights['trueskill']
        ) * 350  # Normalize et
        
        # Güven seviyesi (belirsizlik + volatilite + kimya)
        confidence = (
            glicko2_values['confidence'] * 0.4 +
            trueskill_values['confidence'] * 0.3 +
            trueskill_values['chemistry'] * 100 * 0.3
        )
        
        # Sonuçları birleştir
        result = {
            'hybrid_rating': hybrid_rating,
            'uncertainty': combined_uncertainty,
            'confidence': confidence,
            'volatility': glicko2_values['volatility'],
            'chemistry': trueskill_values['chemistry'],
            'form_factor': ml_factors['form_factor'],
            'consistency_factor': ml_factors['consistency_factor'],
            'momentum': team_dynamics['momentum'],
            'recent_form_rating': recent_form_rating,
            'performance_trend': self._calculate_performance_trend(matches),
            'xg_rating': xg_strength,
            'components': {
                'glicko2_rating': normalized_glicko2,
                'trueskill_rating': normalized_trueskill,
                'recent_form_rating': recent_form_rating,
                'xg_rating': normalized_xg,
                'ml_adjustment': ml_adjustment
            },
            'dynamics': team_dynamics
        }
        
        # Cache'e kaydet
        self.hybrid_ratings[team_id] = result
        self._update_performance_history(team_id, result)
        
        logger.info(f"Hibrit rating hesaplandı - Takım {team_id}: {hybrid_rating:.0f} "
                   f"(Glicko2: {normalized_glicko2:.0f}, TrueSkill: {normalized_trueskill:.0f}, "
                   f"Form: {recent_form_rating:.0f}, xG: {normalized_xg:.0f}, ML: {ml_adjustment:+.0f})")
        
        return result
    
    def get_team_rating(self, team_id, matches):
        """
        Takım rating'ini al (match_prediction.py uyumlu)
        
        Returns:
            dict: Rating detayları
        """
        # Hibrit rating hesapla
        hybrid = self.calculate_hybrid_rating(team_id, matches)
        
        # Glicko-2 ve TrueSkill değerlerini de al
        glicko2_values = self.glicko2.calculate_team_glicko2(team_id, matches)
        trueskill_values = self.trueskill.calculate_team_trueskill(team_id, matches)
        
        # match_prediction.py'nin beklediği formatta döndür
        return {
            'combined_rating': hybrid['hybrid_rating'],
            'elo': hybrid['hybrid_rating'],  # Compat için hybrid rating'i kullan
            'glicko2': glicko2_values['rating'],
            'glicko2_rd': glicko2_values['rd'],
            'trueskill': trueskill_values['skill'],
            'trueskill_sigma': trueskill_values['uncertainty'],
            'confidence': hybrid['confidence'],
            'uncertainty': hybrid['uncertainty']
        }
        
    def _calculate_ml_factors(self, team_id, matches, glicko2_values, trueskill_values):
        """
        ML tabanlı düzeltme faktörleri hesapla
        
        Returns:
            dict: ML faktörleri
        """
        if not matches:
            return {
                'rating_adjustment': 0,
                'form_factor': 1.0,
                'consistency_factor': 1.0,
                'trend_factor': 0
            }
            
        # Son 10 maçı analiz et
        recent_matches = matches[:10]
        
        # Form faktörü (son maçlardaki performans)
        form_scores = []
        for i, match in enumerate(recent_matches):
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            # Zaman ağırlıklı skor
            time_weight = 1.0 / (i + 1)  # Yeni maçlar daha önemli
            
            if goals_for > goals_against:
                score = 3 * time_weight
            elif goals_for == goals_against:
                score = 1 * time_weight
            else:
                score = 0
                
            # Gol farkı bonusu
            goal_diff = goals_for - goals_against
            score += goal_diff * 0.1 * time_weight
            
            form_scores.append(score)
            
        # Form faktörü (0.5-1.5 arası)
        form_factor = 0.5 + (sum(form_scores) / max(1, len(form_scores))) / 3
        form_factor = max(0.5, min(1.5, form_factor))
        
        # Tutarlılık faktörü
        if len(recent_matches) >= 3:
            goal_diffs = [m.get('goals_scored', 0) - m.get('goals_conceded', 0) 
                         for m in recent_matches[:5]]
            std_dev = np.std(goal_diffs)
            # NaN kontrolü
            if np.isnan(std_dev) or std_dev == 0:
                consistency_factor = 1.0
            else:
                consistency = 1.0 - min(1.0, std_dev / 3)
                consistency_factor = 0.8 + consistency * 0.4  # 0.8-1.2 arası
        else:
            consistency_factor = 1.0
            
        # Trend faktörü (momentum)
        if len(recent_matches) >= 5:
            # İlk 5 ve son 5 maç karşılaştırması
            old_avg = np.mean([m.get('goals_scored', 0) - m.get('goals_conceded', 0) 
                              for m in recent_matches[5:10]])
            new_avg = np.mean([m.get('goals_scored', 0) - m.get('goals_conceded', 0) 
                              for m in recent_matches[:5]])
            # NaN kontrolü
            if np.isnan(old_avg) or np.isnan(new_avg):
                trend_factor = 0
            else:
                trend_factor = (new_avg - old_avg) * 50  # -250 ile +250 arası
        else:
            trend_factor = 0
            
        # Rating düzeltmesi hesapla - NaN kontrolü
        rating_adjustment = (
            (form_factor - 1.0) * 200 +     # Form etkisi
            (consistency_factor - 1.0) * 100 + # Tutarlılık etkisi
            trend_factor                      # Trend etkisi
        )
        
        # Final NaN kontrolü
        if np.isnan(rating_adjustment):
            rating_adjustment = 0
        
        return {
            'rating_adjustment': rating_adjustment,
            'form_factor': form_factor,
            'consistency_factor': consistency_factor,
            'trend_factor': trend_factor
        }
        
    def _calculate_recent_form_rating(self, matches):
        """
        Son 5 maç bazlı form rating'i hesapla
        
        Returns:
            float: Form bazlı rating (1000-2000 arası)
        """
        if not matches:
            return 1500  # Varsayılan orta değer
            
        # Son 5 maçı al
        recent_matches = matches[:5]
        
        # Her maç için puan hesapla
        form_points = []
        for i, match in enumerate(recent_matches):
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            # Zaman ağırlığı (yeni maçlar daha önemli)
            time_weight = 1.0 - (i * 0.15)  # 1.0, 0.85, 0.70, 0.55, 0.40
            
            # Maç puanı
            if goals_for > goals_against:
                points = 3.0
            elif goals_for == goals_against:
                points = 1.5  # Beraberlik bonusu
            else:
                points = 0.0
                
            # Gol farkı bonusu/cezası
            goal_diff = goals_for - goals_against
            if goal_diff > 0:
                points += min(goal_diff * 0.2, 1.0)  # Max 1 puan bonus
            else:
                points += max(goal_diff * 0.1, -0.5)  # Max 0.5 puan ceza
                
            form_points.append(points * time_weight)
            
        # Ortalama form puanı (0-4 arası)
        avg_form = sum(form_points) / len(form_points) if form_points else 2.0
        
        # 1000-2000 aralığına çevir
        form_rating = 1000 + (avg_form / 4.0) * 1000
        
        return form_rating
        
    def _calculate_performance_trend(self, matches):
        """
        Performans trendini hesapla
        
        Returns:
            str: 'improving', 'stable', 'declining'
        """
        if len(matches) < 5:
            return 'stable'
            
        # Son 5 ve önceki 5 maç
        recent = matches[:5]
        previous = matches[5:10] if len(matches) >= 10 else matches[5:]
        
        if not previous:
            return 'stable'
            
        # Ortalama puanları hesapla
        recent_avg = sum([
            3 if m.get('goals_scored', 0) > m.get('goals_conceded', 0) else
            1 if m.get('goals_scored', 0) == m.get('goals_conceded', 0) else 0
            for m in recent
        ]) / len(recent)
        
        previous_avg = sum([
            3 if m.get('goals_scored', 0) > m.get('goals_conceded', 0) else
            1 if m.get('goals_scored', 0) == m.get('goals_conceded', 0) else 0
            for m in previous
        ]) / len(previous)
        
        # Trend belirleme
        diff = recent_avg - previous_avg
        if diff > 0.5:
            return 'improving'
        elif diff < -0.5:
            return 'declining'
        else:
            return 'stable'
            
    def apply_draw_correction_factor(self, home_rating, away_rating, is_derby=False):
        """
        Beraberlik düzeltme faktörünü uygula
        
        Args:
            home_rating: Ev sahibi rating
            away_rating: Deplasman rating
            is_derby: Derbi mi?
            
        Returns:
            float: Beraberlik olasılık çarpanı
        """
        rating_diff = abs(home_rating - away_rating)
        
        # Temel düzeltme faktörü
        draw_multiplier = 1.0
        
        # Rating farkına göre düzeltme
        if rating_diff < 100:
            draw_multiplier += 0.10  # %10 artış
        elif rating_diff < 200:
            draw_multiplier += 0.05  # %5 artış
            
        # Derbi düzeltmesi
        if is_derby:
            draw_multiplier += 0.15  # %15 ek artış
            
        return draw_multiplier
        
    def _update_performance_history(self, team_id, rating_data):
        """Performans geçmişini güncelle"""
        if team_id not in self.performance_history:
            self.performance_history[team_id] = []
            
        self.performance_history[team_id].append({
            'timestamp': datetime.now(),
            'rating': rating_data['hybrid_rating'],
            'confidence': rating_data['confidence'],
            'form': rating_data['form_factor']
        })
        
        # Son 100 kaydı tut
        if len(self.performance_history[team_id]) > 100:
            self.performance_history[team_id] = self.performance_history[team_id][-100:]
            
    def update_from_match(self, home_id, away_id, home_goals, away_goals):
        """
        Maç sonucuna göre tüm sistemleri güncelle
        
        Returns:
            dict: Güncelleme detayları
        """
        # Her sistemi güncelle
        glicko2_update = self.glicko2.update_ratings_from_match(
            home_id, away_id, home_goals, away_goals
        )
        trueskill_update = self.trueskill.update_ratings_from_match(
            home_id, away_id, home_goals, away_goals
        )
        
        # Hibrit rating'leri temizle (yeniden hesaplanacak)
        if home_id in self.hybrid_ratings:
            del self.hybrid_ratings[home_id]
        if away_id in self.hybrid_ratings:
            del self.hybrid_ratings[away_id]
            
        return {
            'glicko2': glicko2_update,
            'trueskill': trueskill_update
        }
        
    def get_match_prediction(self, home_id, away_id, home_matches, away_matches):
        """
        Hibrit sistem ile maç tahmini
        
        Returns:
            dict: Tahmin detayları
        """
        # Her takım için hibrit rating hesapla
        home_hybrid = self.calculate_hybrid_rating(home_id, home_matches)
        away_hybrid = self.calculate_hybrid_rating(away_id, away_matches)
        
        # Rating farkı
        rating_diff = home_hybrid['hybrid_rating'] - away_hybrid['hybrid_rating']
        
        # Glicko-2 tahminleri
        glicko2_pred = self.glicko2.get_match_prediction(home_id, away_id)
        
        # TrueSkill tahminleri
        trueskill_pred = self.trueskill.get_win_probability(home_id, away_id)
        
        # xG tabanlı tahmin hesapla
        xg_pred = self._calculate_xg_prediction(home_hybrid, away_hybrid)
        
        # Hibrit tahmin (xG ağırlığı artırıldı)
        home_win_prob = (
            glicko2_pred['home_win_prob'] * 0.20 +  # %20'ye düşürüldü
            trueskill_pred['home_win'] * 0.15 +     # %15'e düşürüldü
            xg_pred['home_win'] * 0.50 +            # %50 xG tahmini (YENİ)
            self._ml_win_probability(rating_diff, home_hybrid, away_hybrid) * 0.15
        )
        
        away_win_prob = (
            glicko2_pred['away_win_prob'] * 0.20 +
            trueskill_pred['away_win'] * 0.15 +
            xg_pred['away_win'] * 0.50 +
            self._ml_win_probability(-rating_diff, away_hybrid, home_hybrid) * 0.15
        )
        
        draw_prob = (
            (100 - glicko2_pred['home_win_prob'] - glicko2_pred['away_win_prob']) * 0.20 +
            (100 - trueskill_pred['home_win'] - trueskill_pred['away_win']) * 0.15 +
            xg_pred['draw'] * 0.50 +
            (100 - home_win_prob - away_win_prob) * 0.15
        )
        
        # Güven seviyesi
        combined_confidence = (
            home_hybrid['confidence'] * 0.5 +
            away_hybrid['confidence'] * 0.5
        )
        
        # Lambda değerleri için düzeltme faktörleri
        home_lambda_factor = self._calculate_lambda_factor(home_hybrid)
        away_lambda_factor = self._calculate_lambda_factor(away_hybrid)
        
        return {
            'rating_diff': rating_diff,
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob,
            'confidence': combined_confidence,
            'match_quality': trueskill_pred['match_quality'],
            'home_lambda_factor': home_lambda_factor,
            'away_lambda_factor': away_lambda_factor,
            'home_hybrid': home_hybrid,
            'away_hybrid': away_hybrid,
            'components': {
                'glicko2': glicko2_pred,
                'trueskill': trueskill_pred
            }
        }
        
    def _ml_win_probability(self, rating_diff, team_hybrid, opponent_hybrid):
        """ML tabanlı kazanma olasılığı"""
        # Sigmoid fonksiyonu ile rating farkını olasılığa çevir
        base_prob = 1 / (1 + math.exp(-rating_diff / 400))
        
        # Form ve momentum düzeltmesi
        form_adjustment = (team_hybrid['form_factor'] - opponent_hybrid['form_factor']) * 0.1
        momentum_adjustment = (team_hybrid['momentum'] - opponent_hybrid['momentum']) / 100 * 0.05
        
        # Kimya faktörü
        chemistry_adjustment = (team_hybrid['chemistry'] - opponent_hybrid['chemistry']) * 0.05
        
        # Final olasılık
        final_prob = base_prob + form_adjustment + momentum_adjustment + chemistry_adjustment
        
        # 0-100 aralığına sınırla
        return max(5, min(95, final_prob * 100))
        
    def _calculate_lambda_factor(self, team_hybrid):
        """
        Lambda (gol beklentisi) için düzeltme faktörü hesapla
        xG rating'in artırılmış etkisiyle
        
        Returns:
            float: Lambda çarpanı (0.5-1.5 arası)
        """
        # xG Rating'i ana faktör olarak kullan (artırılmış etki)
        xg_rating = team_hybrid['xg_rating']
        
        # xG güç puanlarından lambda faktörü hesapla
        if xg_rating and 'attacking_strength' in xg_rating:
            # Ev sahibi ve deplasman atak güçlerinin ortalaması
            attack_strength = (xg_rating['home_attack'] + xg_rating['away_attack']) / 2
            defense_weakness = (xg_rating['home_defense'] + xg_rating['away_defense']) / 2
            
            # xG tabanlı lambda faktörü (ana etki)
            xg_lambda_factor = attack_strength * defense_weakness
            xg_lambda_factor = max(0.5, min(1.5, xg_lambda_factor))
        else:
            xg_lambda_factor = 1.0
        
        # Diğer faktörler (azaltılmış etki)
        base_factor = team_hybrid['hybrid_rating'] / 1500  # 1500 ortalama
        form_effect = (team_hybrid['form_factor'] - 1.0) * 0.1  # 0.3'ten 0.1'e düşürüldü
        momentum_effect = (team_hybrid['momentum'] - 50) / 100 * 0.05  # 0.2'den 0.05'e düşürüldü
        consistency_effect = (team_hybrid['consistency_factor'] - 1.0) * 0.05  # 0.2'den 0.05'e düşürüldü
        
        # Ağırlıklı kombinasyon - xG rating'e %60 ağırlık
        lambda_factor = (
            xg_lambda_factor * 0.60 +  # xG rating %60 etki
            base_factor * 0.20 +       # Hibrit rating %20 etki
            form_effect +              # Form %10 etki
            momentum_effect +          # Momentum %5 etki
            consistency_effect         # Tutarlılık %5 etki
        )
        
        # 0.5-1.5 aralığına sınırla
        return max(0.5, min(1.5, lambda_factor))
    
    def _calculate_xg_prediction(self, home_hybrid, away_hybrid):
        """
        xG rating'lerini kullanarak 1X2 tahmini yap
        
        Returns:
            dict: home_win, draw, away_win yüzdeleri
        """
        home_xg = home_hybrid['xg_rating']
        away_xg = away_hybrid['xg_rating']
        
        # xG güç puanlarından gol beklentileri hesapla
        if home_xg and away_xg:
            # Ev sahibi lambda
            home_lambda = home_xg['home_attack'] * away_xg['away_defense']
            # Deplasman lambda
            away_lambda = away_xg['away_attack'] * home_xg['home_defense']
            
            # Poisson dağılımı ile olasılıkları hesapla
            home_win = 0.0
            draw = 0.0
            away_win = 0.0
            
            # 0-5 gol arası hesapla
            for h_goals in range(6):
                for a_goals in range(6):
                    # Poisson olasılığı
                    h_prob = (math.exp(-home_lambda) * (home_lambda ** h_goals)) / math.factorial(h_goals)
                    a_prob = (math.exp(-away_lambda) * (away_lambda ** a_goals)) / math.factorial(a_goals)
                    joint_prob = h_prob * a_prob
                    
                    if h_goals > a_goals:
                        home_win += joint_prob
                    elif h_goals == a_goals:
                        draw += joint_prob
                    else:
                        away_win += joint_prob
            
            # Beraberlik düzeltmesi - güçler yakınsa beraberlik artır
            strength_diff = abs(home_xg['overall_rating'] - away_xg['overall_rating'])
            if strength_diff < 10:  # Güçler çok yakın
                draw_boost = 0.05 * (1 - strength_diff / 10)
                draw += draw_boost
                home_win -= draw_boost / 2
                away_win -= draw_boost / 2
            
            return {
                'home_win': home_win * 100,
                'draw': draw * 100,
                'away_win': away_win * 100
            }
        else:
            # xG verisi yoksa varsayılan değerler
            return {
                'home_win': 40,
                'draw': 30,
                'away_win': 30
            }
    
    def _calculate_xg_rating(self, team_id, matches):
        """
        xG tabanlı rating hesapla
        
        Args:
            team_id: Takım ID
            matches: Maç listesi
            
        Returns:
            dict: xG rating detayları
        """
        # Maç geçmişini güncelle
        for match in matches[:10]:  # Son 10 maçı kullan
            # Ev sahibi mi deplasman mı kontrol et
            is_home = match.get('is_home', True)
            
            if is_home:
                home_id = team_id
                away_id = match.get('opponent_id', 'unknown')
                home_goals = match.get('goals_scored', 0)
                away_goals = match.get('goals_conceded', 0)
                home_xg = match.get('xg_for')
                away_xg = match.get('xg_against')
            else:
                home_id = match.get('opponent_id', 'unknown')
                away_id = team_id
                home_goals = match.get('goals_conceded', 0)
                away_goals = match.get('goals_scored', 0)
                home_xg = match.get('xg_against')
                away_xg = match.get('xg_for')
            
            # xG rating sistemini güncelle
            self.xg_rating.update_ratings(
                home_id, away_id,
                home_goals, away_goals,
                home_xg, away_xg
            )
        
        # Takım güç puanlarını al
        return self.xg_rating.get_team_strength_scores(team_id)
        
    def get_rating_comparison(self, home_id, away_id, home_matches, away_matches):
        """
        İki takımın detaylı rating karşılaştırması
        
        Returns:
            dict: Karşılaştırma detayları
        """
        home_hybrid = self.calculate_hybrid_rating(home_id, home_matches)
        away_hybrid = self.calculate_hybrid_rating(away_id, away_matches)
        
        return {
            'home': {
                'team_id': home_id,
                'hybrid_rating': home_hybrid['hybrid_rating'],
                'glicko2': home_hybrid['components']['glicko2_rating'],
                'trueskill': home_hybrid['components']['trueskill_rating'],
                'form': home_hybrid['dynamics']['form_trend'],
                'chemistry': home_hybrid['chemistry'],
                'confidence': home_hybrid['confidence']
            },
            'away': {
                'team_id': away_id,
                'hybrid_rating': away_hybrid['hybrid_rating'],
                'glicko2': away_hybrid['components']['glicko2_rating'],
                'trueskill': away_hybrid['components']['trueskill_rating'],
                'form': away_hybrid['dynamics']['form_trend'],
                'chemistry': away_hybrid['chemistry'],
                'confidence': away_hybrid['confidence']
            },
            'advantage': {
                'rating_diff': home_hybrid['hybrid_rating'] - away_hybrid['hybrid_rating'],
                'form_diff': home_hybrid['form_factor'] - away_hybrid['form_factor'],
                'chemistry_diff': home_hybrid['chemistry'] - away_hybrid['chemistry'],
                'momentum_diff': home_hybrid['momentum'] - away_hybrid['momentum']
            }
        }