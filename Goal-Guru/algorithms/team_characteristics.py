"""
Team Characteristics Analyzer
Takımların oyun stillerini ve karakteristik özelliklerini analiz eden modül
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TeamCharacteristicsAnalyzer:
    """
    Takım karakteristiklerini ve oyun stillerini analiz eden sınıf
    """
    
    def __init__(self):
        # Karakteristik eşik değerleri
        self.thresholds = {
            'high_attack': 1.8,      # Yüksek atak gücü eşiği
            'low_attack': 1.0,        # Düşük atak gücü eşiği
            'solid_defense': 1.0,     # Sağlam savunma eşiği
            'weak_defense': 1.5,      # Zayıf savunma eşiği
            'high_tempo': 3.0,        # Yüksek tempo eşiği (toplam gol)
            'low_tempo': 2.0,         # Düşük tempo eşiği
            'high_possession': 0.55,  # Yüksek topa sahip olma
            'low_possession': 0.45    # Düşük topa sahip olma
        }
        
    def analyze_team_style(self, team_features: Dict, opponent_features: Dict = None) -> Dict:
        """
        Takımın oyun stilini analiz et
        
        Args:
            team_features: Takım özellikleri
            opponent_features: Rakip takım özellikleri (opsiyonel)
            
        Returns:
            Takım stil analizi
        """
        try:
            # Temel karakteristikleri belirle
            attack_profile = self._analyze_attack_style(team_features)
            defense_profile = self._analyze_defense_style(team_features)
            tempo_profile = self._analyze_game_tempo(team_features)
            tactical_profile = self._analyze_tactical_approach(team_features)
            
            # Güçlü ve zayıf yönleri belirle
            strengths_weaknesses = self._identify_strengths_weaknesses(
                attack_profile, defense_profile, tempo_profile
            )
            
            # Rakibe karşı matchup analizi
            matchup_analysis = None
            if opponent_features:
                matchup_analysis = self._analyze_matchup(
                    team_features, opponent_features,
                    attack_profile, defense_profile
                )
            
            return {
                'attack_profile': attack_profile,
                'defense_profile': defense_profile,
                'tempo_profile': tempo_profile,
                'tactical_profile': tactical_profile,
                'strengths': strengths_weaknesses['strengths'],
                'weaknesses': strengths_weaknesses['weaknesses'],
                'matchup_analysis': matchup_analysis,
                'style_summary': self._generate_style_summary(
                    attack_profile, defense_profile, tempo_profile, tactical_profile
                )
            }
            
        except Exception as e:
            logger.error(f"Takım stil analizi hatası: {e}")
            return self._get_default_style()
    
    def _analyze_attack_style(self, features: Dict) -> Dict:
        """
        Atak stilini analiz et
        """
        avg_goals = features.get('avg_goals', 1.2)
        scoring_consistency = features.get('scoring_consistency', 0.5)
        attack_strength = features.get('attack_strength', 0.5)
        
        # Atak tipi belirleme
        if avg_goals > self.thresholds['high_attack']:
            if scoring_consistency > 0.7:
                attack_type = 'clinical_finisher'  # Tutarlı yüksek skor
            else:
                attack_type = 'explosive_attacker'  # Patlamalı atak
        elif avg_goals > self.thresholds['low_attack']:
            attack_type = 'balanced_attacker'  # Dengeli atak
        else:
            if scoring_consistency > 0.6:
                attack_type = 'efficient_scorer'  # Az ama etkili
            else:
                attack_type = 'struggling_attacker'  # Zayıf atak
        
        # Atak pattern'i belirleme
        home_goals = features.get('venue_avg_goals', avg_goals)
        away_goals = features.get('general_avg_goals', avg_goals)
        
        if home_goals > away_goals * 1.3:
            attack_pattern = 'home_dominant'  # Evde güçlü
        elif away_goals > home_goals * 1.2:
            attack_pattern = 'away_specialist'  # Deplasmanda güçlü
        else:
            attack_pattern = 'consistent'  # Tutarlı
        
        return {
            'type': attack_type,
            'pattern': attack_pattern,
            'avg_goals': avg_goals,
            'consistency': scoring_consistency,
            'strength_score': attack_strength,
            'effectiveness': self._calculate_attack_effectiveness(features)
        }
    
    def _analyze_defense_style(self, features: Dict) -> Dict:
        """
        Savunma stilini analiz et
        """
        avg_conceded = features.get('avg_conceded', 1.2)
        defensive_stability = features.get('defensive_stability', 0.5)
        clean_sheet_rate = features.get('clean_sheet_rate', 0.2)
        defense_strength = features.get('defense_strength', 0.5)
        
        # Savunma tipi belirleme
        if avg_conceded < self.thresholds['solid_defense']:
            if clean_sheet_rate > 0.4:
                defense_type = 'fortress'  # Kale gibi savunma
            else:
                defense_type = 'solid_defender'  # Sağlam savunma
        elif avg_conceded < self.thresholds['weak_defense']:
            defense_type = 'average_defender'  # Ortalama savunma
        else:
            if defensive_stability > 0.6:
                defense_type = 'leaky_but_organized'  # Organize ama gevşek
            else:
                defense_type = 'vulnerable_defender'  # Zayıf savunma
        
        # Savunma stili
        if clean_sheet_rate > 0.35 and avg_conceded < 1.2:
            defense_style = 'low_block'  # Derin savunma
        elif avg_conceded > 1.5 and features.get('btts_rate', 0.5) > 0.6:
            defense_style = 'high_line'  # Yüksek savunma hattı
        else:
            defense_style = 'balanced_defense'  # Dengeli savunma
        
        return {
            'type': defense_type,
            'style': defense_style,
            'avg_conceded': avg_conceded,
            'stability': defensive_stability,
            'clean_sheet_rate': clean_sheet_rate,
            'strength_score': defense_strength,
            'vulnerability': self._calculate_defensive_vulnerability(features)
        }
    
    def _analyze_game_tempo(self, features: Dict) -> Dict:
        """
        Oyun temposunu analiz et
        """
        total_goals = features.get('avg_goals', 1.2) + features.get('avg_conceded', 1.2)
        over_2_5_rate = features.get('over_2_5_rate', 0.5)
        tempo_factor = features.get('tempo_factor', 0.5)
        
        # Tempo kategorisi
        if total_goals > self.thresholds['high_tempo']:
            if over_2_5_rate > 0.65:
                tempo_type = 'ultra_high_tempo'  # Çok yüksek tempo
            else:
                tempo_type = 'high_tempo'  # Yüksek tempo
        elif total_goals > self.thresholds['low_tempo']:
            tempo_type = 'medium_tempo'  # Orta tempo
        else:
            if over_2_5_rate < 0.35:
                tempo_type = 'ultra_low_tempo'  # Çok düşük tempo
            else:
                tempo_type = 'low_tempo'  # Düşük tempo
        
        # Tempo tutarlılığı
        goals_std = features.get('general_std_goals', 0.5)
        conceded_std = features.get('general_std_conceded', 0.5)
        tempo_consistency = 1 / (1 + (goals_std + conceded_std) / 2)
        
        return {
            'type': tempo_type,
            'avg_total_goals': total_goals,
            'over_2_5_rate': over_2_5_rate,
            'tempo_score': tempo_factor,
            'consistency': tempo_consistency,
            'match_pace': self._categorize_match_pace(total_goals, over_2_5_rate)
        }
    
    def _analyze_tactical_approach(self, features: Dict) -> Dict:
        """
        Taktiksel yaklaşımı analiz et
        """
        win_rate = features.get('win_rate', 0.33)
        draw_rate = features.get('general_draw_rate', 0.33)
        risk_factor = features.get('risk_factor', 0.5)
        adaptability = features.get('adaptability', 1.0)
        
        # Risk yaklaşımı
        if risk_factor > 0.7:
            if win_rate > 0.5:
                risk_approach = 'calculated_aggression'  # Hesaplı agresiflik
            else:
                risk_approach = 'reckless_aggression'  # Kontrolsüz agresiflik
        elif risk_factor < 0.3:
            if draw_rate > 0.4:
                risk_approach = 'ultra_conservative'  # Aşırı muhafazakar
            else:
                risk_approach = 'conservative'  # Muhafazakar
        else:
            risk_approach = 'balanced_risk'  # Dengeli risk
        
        # Oyun planı
        if win_rate > 0.5 and features.get('avg_goals', 1.2) > 1.5:
            game_plan = 'attacking_dominance'  # Atak baskınlığı
        elif draw_rate > 0.4:
            game_plan = 'control_oriented'  # Kontrol odaklı
        elif features.get('avg_conceded', 1.2) < 1.0:
            game_plan = 'defensive_solidity'  # Savunma öncelikli
        else:
            game_plan = 'opportunistic'  # Fırsatçı
        
        # Esneklik
        if adaptability > 1.05:
            flexibility = 'highly_adaptable'  # Çok esnek
        elif adaptability > 0.95:
            flexibility = 'moderately_adaptable'  # Orta esnek
        else:
            flexibility = 'rigid'  # Katı
        
        return {
            'risk_approach': risk_approach,
            'game_plan': game_plan,
            'flexibility': flexibility,
            'win_mentality': win_rate,
            'draw_tendency': draw_rate,
            'adaptability_score': adaptability
        }
    
    def _identify_strengths_weaknesses(self, attack: Dict, defense: Dict, tempo: Dict) -> Dict:
        """
        Güçlü ve zayıf yönleri belirle
        """
        strengths = []
        weaknesses = []
        
        # Atak güçlü yönler/zayıflıklar
        if attack['avg_goals'] > 1.8:
            strengths.append('high_scoring_ability')
        elif attack['avg_goals'] < 1.0:
            weaknesses.append('poor_scoring_ability')
            
        if attack['consistency'] > 0.7:
            strengths.append('consistent_scoring')
        elif attack['consistency'] < 0.4:
            weaknesses.append('inconsistent_scoring')
        
        # Savunma güçlü yönler/zayıflıklar
        if defense['avg_conceded'] < 1.0:
            strengths.append('strong_defense')
        elif defense['avg_conceded'] > 1.5:
            weaknesses.append('weak_defense')
            
        if defense['clean_sheet_rate'] > 0.35:
            strengths.append('clean_sheet_specialist')
        elif defense['clean_sheet_rate'] < 0.15:
            weaknesses.append('rarely_keeps_clean_sheets')
        
        # Tempo güçlü yönler/zayıflıklar
        if tempo['consistency'] > 0.7:
            strengths.append('predictable_game_flow')
        elif tempo['consistency'] < 0.4:
            weaknesses.append('unpredictable_performance')
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses
        }
    
    def _analyze_matchup(self, team_features: Dict, opponent_features: Dict,
                        team_attack: Dict, team_defense: Dict) -> Dict:
        """
        Rakibe karşı matchup analizi
        """
        # Atak vs Savunma matchup
        team_attack_score = team_attack['strength_score']
        opp_defense_score = opponent_features.get('defense_strength', 0.5)
        attack_advantage = team_attack_score - opp_defense_score
        
        # Savunma vs Atak matchup
        team_defense_score = team_defense['strength_score']
        opp_attack_score = opponent_features.get('attack_strength', 0.5)
        defense_advantage = team_defense_score - opp_attack_score
        
        # Tempo uyumu
        team_tempo = team_features.get('tempo_factor', 0.5)
        opp_tempo = opponent_features.get('tempo_factor', 0.5)
        tempo_clash = abs(team_tempo - opp_tempo)
        
        # Stil uyumu değerlendirmesi
        if attack_advantage > 0.2:
            attack_matchup = 'favorable'
        elif attack_advantage < -0.2:
            attack_matchup = 'unfavorable'
        else:
            attack_matchup = 'balanced'
            
        if defense_advantage > 0.2:
            defense_matchup = 'favorable'
        elif defense_advantage < -0.2:
            defense_matchup = 'unfavorable'
        else:
            defense_matchup = 'balanced'
        
        # Tempo çatışması
        if tempo_clash > 0.3:
            tempo_matchup = 'contrasting_styles'  # Zıt stiller
        else:
            tempo_matchup = 'similar_styles'  # Benzer stiller
        
        # Genel matchup skoru
        overall_advantage = (attack_advantage + defense_advantage) / 2
        
        return {
            'attack_matchup': attack_matchup,
            'defense_matchup': defense_matchup,
            'tempo_matchup': tempo_matchup,
            'attack_advantage': attack_advantage,
            'defense_advantage': defense_advantage,
            'overall_advantage': overall_advantage,
            'style_clash_factor': tempo_clash,
            'predicted_game_flow': self._predict_game_flow(
                attack_advantage, defense_advantage, tempo_clash
            )
        }
    
    def _predict_game_flow(self, attack_adv: float, defense_adv: float, tempo_clash: float) -> str:
        """
        Maç akışını tahmin et
        """
        if attack_adv > 0.3 and defense_adv > 0.3:
            return 'dominant_performance_expected'  # Baskın performans bekleniyor
        elif attack_adv < -0.3 and defense_adv < -0.3:
            return 'difficult_match_expected'  # Zor maç bekleniyor
        elif tempo_clash > 0.4:
            return 'tactical_battle_expected'  # Taktiksel savaş bekleniyor
        elif attack_adv > 0.2 and defense_adv < -0.2:
            return 'open_game_expected'  # Açık oyun bekleniyor
        elif attack_adv < -0.2 and defense_adv > 0.2:
            return 'defensive_game_expected'  # Savunma ağırlıklı oyun
        else:
            return 'balanced_game_expected'  # Dengeli maç bekleniyor
    
    def _calculate_attack_effectiveness(self, features: Dict) -> float:
        """
        Atak etkinliğini hesapla
        """
        goals = features.get('avg_goals', 1.2)
        consistency = features.get('scoring_consistency', 0.5)
        win_rate = features.get('win_rate', 0.33)
        
        # Etkinlik formülü
        effectiveness = (goals / 3.0) * 0.4 + consistency * 0.3 + win_rate * 0.3
        return min(1.0, max(0.0, effectiveness))
    
    def _calculate_defensive_vulnerability(self, features: Dict) -> float:
        """
        Savunma zayıflığını hesapla
        """
        conceded = features.get('avg_conceded', 1.2)
        stability = features.get('defensive_stability', 0.5)
        clean_sheets = features.get('clean_sheet_rate', 0.2)
        
        # Zayıflık formülü (ters mantık - yüksek değer = zayıf savunma)
        vulnerability = (conceded / 3.0) * 0.4 + (1 - stability) * 0.3 + (1 - clean_sheets) * 0.3
        return min(1.0, max(0.0, vulnerability))
    
    def _categorize_match_pace(self, total_goals: float, over_rate: float) -> str:
        """
        Maç hızını kategorize et
        """
        if total_goals > 3.5 and over_rate > 0.7:
            return 'frantic_pace'  # Çılgın tempo
        elif total_goals > 2.8 and over_rate > 0.55:
            return 'fast_pace'  # Hızlı tempo
        elif total_goals > 2.2:
            return 'moderate_pace'  # Orta tempo
        elif total_goals > 1.8:
            return 'slow_pace'  # Yavaş tempo
        else:
            return 'very_slow_pace'  # Çok yavaş tempo
    
    def _generate_style_summary(self, attack: Dict, defense: Dict, 
                               tempo: Dict, tactical: Dict) -> str:
        """
        Takım stili özeti oluştur
        """
        # Stil bileşenleri
        attack_desc = {
            'clinical_finisher': 'Klinik bitirici',
            'explosive_attacker': 'Patlamalı hücumcu',
            'balanced_attacker': 'Dengeli hücumcu',
            'efficient_scorer': 'Verimli golcü',
            'struggling_attacker': 'Zayıf hücumcu'
        }.get(attack['type'], 'Standart hücumcu')
        
        defense_desc = {
            'fortress': 'Kale savunma',
            'solid_defender': 'Sağlam savunma',
            'average_defender': 'Ortalama savunma',
            'leaky_but_organized': 'Organize ama açık verici',
            'vulnerable_defender': 'Zayıf savunma'
        }.get(defense['type'], 'Standart savunma')
        
        tempo_desc = {
            'ultra_high_tempo': 'Çok yüksek tempolu',
            'high_tempo': 'Yüksek tempolu',
            'medium_tempo': 'Orta tempolu',
            'low_tempo': 'Düşük tempolu',
            'ultra_low_tempo': 'Çok düşük tempolu'
        }.get(tempo['type'], 'Normal tempolu')
        
        risk_desc = {
            'calculated_aggression': 'hesaplı agresif',
            'reckless_aggression': 'kontrolsüz agresif',
            'ultra_conservative': 'aşırı muhafazakar',
            'conservative': 'muhafazakar',
            'balanced_risk': 'dengeli risk alan'
        }.get(tactical['risk_approach'], 'dengeli')
        
        # Özet oluştur
        summary = f"{attack_desc}, {defense_desc}, {tempo_desc} ve {risk_desc} bir takım profili"
        
        return summary
    
    def _get_default_style(self) -> Dict:
        """
        Varsayılan stil profili döndür
        """
        return {
            'attack_profile': {
                'type': 'balanced_attacker',
                'pattern': 'consistent',
                'avg_goals': 1.2,
                'consistency': 0.5,
                'strength_score': 0.5,
                'effectiveness': 0.5
            },
            'defense_profile': {
                'type': 'average_defender',
                'style': 'balanced_defense',
                'avg_conceded': 1.2,
                'stability': 0.5,
                'clean_sheet_rate': 0.2,
                'strength_score': 0.5,
                'vulnerability': 0.5
            },
            'tempo_profile': {
                'type': 'medium_tempo',
                'avg_total_goals': 2.4,
                'over_2_5_rate': 0.5,
                'tempo_score': 0.5,
                'consistency': 0.5,
                'match_pace': 'moderate_pace'
            },
            'tactical_profile': {
                'risk_approach': 'balanced_risk',
                'game_plan': 'opportunistic',
                'flexibility': 'moderately_adaptable',
                'win_mentality': 0.33,
                'draw_tendency': 0.33,
                'adaptability_score': 1.0
            },
            'strengths': [],
            'weaknesses': [],
            'matchup_analysis': None,
            'style_summary': 'Dengeli hücumcu, Ortalama savunma, Orta tempolu ve dengeli risk alan bir takım profili'
        }