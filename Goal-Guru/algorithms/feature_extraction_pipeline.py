"""
Feature Extraction Pipeline
Takım verilerini işleyip karakteristik özellikleri çıkaran süzgeç sistemi
%65 venue-specific performans, %35 genel performans ağırlıklı
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class FeatureExtractionPipeline:
    """
    Takım karakteristik özelliklerini çıkaran pipeline
    """
    
    def __init__(self):
        self.venue_weight = 0.65  # Ev/Deplasman performans ağırlığı
        self.general_weight = 0.35  # Genel performans ağırlığı
        self.scaler = StandardScaler()
        
        # ML modelleri - pattern recognition için
        self.rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.knn_model = KNeighborsRegressor(n_neighbors=5)
        
        # Özellik isimleri
        self.feature_names = [
            'avg_goals', 'avg_conceded', 'win_rate', 'draw_rate', 'loss_rate',
            'clean_sheet_rate', 'btts_rate', 'over_2_5_rate', 'scoring_consistency',
            'defensive_stability', 'form_momentum', 'goal_difference_avg'
        ]
        
    def extract_features(self, team_data: Dict, is_home: bool = True) -> Dict:
        """
        Takım verilerinden özellik çıkarımı
        
        Args:
            team_data: Takım performans verileri
            is_home: Ev sahibi mi?
            
        Returns:
            Zenginleştirilmiş özellik vektörü
        """
        try:
            # Venue-specific özellikler (%65 ağırlık)
            venue_features = self._extract_venue_features(team_data, is_home)
            
            # Genel performans özellikleri (%35 ağırlık)
            general_features = self._extract_general_features(team_data)
            
            # Ağırlıklı birleştirme
            combined_features = self._combine_features(venue_features, general_features)
            
            # Karakteristik profil belirleme
            team_profile = self._determine_team_profile(combined_features)
            
            # ML ile pattern tanıma ve zenginleştirme
            enriched_features = self._enrich_with_ml(combined_features, team_profile)
            
            # Normalize et
            normalized_features = self._normalize_features(enriched_features)
            
            return {
                'raw_features': combined_features,
                'team_profile': team_profile,
                'enriched_features': enriched_features,
                'normalized_features': normalized_features,
                'venue_weight_used': self.venue_weight if venue_features else 0,
                'feature_quality_score': self._calculate_feature_quality(team_data)
            }
            
        except Exception as e:
            logger.error(f"Feature extraction hatası: {e}")
            return self._get_default_features()
    
    def _extract_venue_features(self, team_data: Dict, is_home: bool) -> Dict:
        """
        Venue-specific (ev/deplasman) özellikler çıkar
        """
        features = {}
        
        if is_home and 'home_performance' in team_data:
            perf = team_data['home_performance']
            # Son 5 ev maçı verileri
            features['venue_avg_goals'] = perf.get('last_5_avg_goals', perf.get('avg_goals', 1.3))
            features['venue_avg_conceded'] = perf.get('last_5_avg_conceded', perf.get('avg_conceded', 1.2))
            features['venue_win_rate'] = perf.get('last_5_win_rate', 0.4)
            features['venue_form'] = self._encode_form(perf.get('last_5_form', ''))
            features['venue_matches'] = perf.get('last_5_matches', 0)
            
        elif not is_home and 'away_performance' in team_data:
            perf = team_data['away_performance']
            # Son 5 deplasman maçı verileri
            features['venue_avg_goals'] = perf.get('last_5_avg_goals', perf.get('avg_goals', 1.0))
            features['venue_avg_conceded'] = perf.get('last_5_avg_conceded', perf.get('avg_conceded', 1.3))
            features['venue_win_rate'] = perf.get('last_5_win_rate', 0.3)
            features['venue_form'] = self._encode_form(perf.get('last_5_form', ''))
            features['venue_matches'] = perf.get('last_5_matches', 0)
        else:
            # Varsayılan değerler
            features = {
                'venue_avg_goals': 1.2 if is_home else 1.0,
                'venue_avg_conceded': 1.2,
                'venue_win_rate': 0.4 if is_home else 0.3,
                'venue_form': 0.5,
                'venue_matches': 0
            }
        
        # Venue bonus hesapla
        if features['venue_win_rate'] > 0.6 and is_home:
            features['venue_dominance'] = 1.1  # Ev sahibi güçlü
        elif features['venue_win_rate'] > 0.4 and not is_home:
            features['venue_resilience'] = 1.05  # Deplasmanda dirençli
        else:
            features['venue_dominance'] = 1.0
            features['venue_resilience'] = 1.0
            
        return features
    
    def _extract_general_features(self, team_data: Dict) -> Dict:
        """
        Genel performans özellikleri çıkar (son 10 maç)
        """
        features = {}
        
        if 'recent_matches' in team_data and team_data['recent_matches']:
            matches = team_data['recent_matches'][:10]  # Son 10 maç
            
            # Temel istatistikler
            goals_scored = [m.get('goals_scored', 0) for m in matches]
            goals_conceded = [m.get('goals_conceded', 0) for m in matches]
            
            features['general_avg_goals'] = np.mean(goals_scored) if goals_scored else 1.2
            features['general_avg_conceded'] = np.mean(goals_conceded) if goals_conceded else 1.2
            features['general_std_goals'] = np.std(goals_scored) if len(goals_scored) > 1 else 0.5
            features['general_std_conceded'] = np.std(goals_conceded) if len(goals_conceded) > 1 else 0.5
            
            # Maç sonuçları
            wins = sum(1 for m in matches if m.get('goals_scored', 0) > m.get('goals_conceded', 0))
            draws = sum(1 for m in matches if m.get('goals_scored', 0) == m.get('goals_conceded', 0))
            losses = sum(1 for m in matches if m.get('goals_scored', 0) < m.get('goals_conceded', 0))
            
            total = len(matches)
            features['general_win_rate'] = wins / total if total > 0 else 0.33
            features['general_draw_rate'] = draws / total if total > 0 else 0.33
            features['general_loss_rate'] = losses / total if total > 0 else 0.34
            
            # Özel metriklier
            features['clean_sheet_rate'] = sum(1 for m in matches if m.get('goals_conceded', 0) == 0) / total if total > 0 else 0.2
            features['btts_rate'] = sum(1 for m in matches if m.get('goals_scored', 0) > 0 and m.get('goals_conceded', 0) > 0) / total if total > 0 else 0.5
            features['over_2_5_rate'] = sum(1 for m in matches if (m.get('goals_scored', 0) + m.get('goals_conceded', 0)) > 2.5) / total if total > 0 else 0.5
            
            # Form momentum (son 5 vs önceki 5)
            if len(matches) >= 10:
                recent_5_points = sum(3 if m.get('goals_scored', 0) > m.get('goals_conceded', 0) else (1 if m.get('goals_scored', 0) == m.get('goals_conceded', 0) else 0) for m in matches[:5])
                older_5_points = sum(3 if m.get('goals_scored', 0) > m.get('goals_conceded', 0) else (1 if m.get('goals_scored', 0) == m.get('goals_conceded', 0) else 0) for m in matches[5:10])
                features['form_momentum'] = (recent_5_points - older_5_points) / 15.0  # Normalize -1 to 1
            else:
                features['form_momentum'] = 0
                
        else:
            # Varsayılan değerler
            features = {
                'general_avg_goals': 1.2,
                'general_avg_conceded': 1.2,
                'general_std_goals': 0.5,
                'general_std_conceded': 0.5,
                'general_win_rate': 0.33,
                'general_draw_rate': 0.33,
                'general_loss_rate': 0.34,
                'clean_sheet_rate': 0.2,
                'btts_rate': 0.5,
                'over_2_5_rate': 0.5,
                'form_momentum': 0
            }
            
        return features
    
    def _combine_features(self, venue_features: Dict, general_features: Dict) -> Dict:
        """
        Venue ve genel özellikleri ağırlıklı olarak birleştir
        """
        combined = {}
        
        # Ortak metrikleri ağırlıklı birleştir
        metrics_to_combine = ['avg_goals', 'avg_conceded', 'win_rate']
        
        for metric in metrics_to_combine:
            venue_key = f'venue_{metric}'
            general_key = f'general_{metric}'
            
            if venue_key in venue_features and general_key in general_features:
                # %65 venue, %35 genel
                combined[metric] = (venue_features[venue_key] * self.venue_weight + 
                                  general_features[general_key] * self.general_weight)
            elif venue_key in venue_features:
                combined[metric] = venue_features[venue_key]
            elif general_key in general_features:
                combined[metric] = general_features[general_key]
        
        # Diğer özellikleri ekle
        combined.update({k: v for k, v in venue_features.items() if not k.startswith('venue_')})
        combined.update({k: v for k, v in general_features.items() if not k.startswith('general_')})
        
        # Hesaplanmış metrikler
        combined['goal_difference'] = combined.get('avg_goals', 1.2) - combined.get('avg_conceded', 1.2)
        combined['scoring_consistency'] = 1 / (1 + general_features.get('general_std_goals', 0.5))
        combined['defensive_stability'] = 1 / (1 + general_features.get('general_std_conceded', 0.5))
        
        return combined
    
    def _determine_team_profile(self, features: Dict) -> Dict:
        """
        Takım karakteristik profilini belirle
        """
        profile = {}
        
        # Atak profili
        avg_goals = features.get('avg_goals', 1.2)
        if avg_goals > 2.0:
            profile['attack_style'] = 'highly_offensive'
            profile['attack_score'] = 0.9
        elif avg_goals > 1.5:
            profile['attack_style'] = 'offensive'
            profile['attack_score'] = 0.7
        elif avg_goals > 1.0:
            profile['attack_style'] = 'balanced'
            profile['attack_score'] = 0.5
        else:
            profile['attack_style'] = 'defensive'
            profile['attack_score'] = 0.3
        
        # Savunma profili
        avg_conceded = features.get('avg_conceded', 1.2)
        clean_sheet_rate = features.get('clean_sheet_rate', 0.2)
        
        if avg_conceded < 0.8 and clean_sheet_rate > 0.4:
            profile['defense_style'] = 'very_solid'
            profile['defense_score'] = 0.9
        elif avg_conceded < 1.2:
            profile['defense_style'] = 'solid'
            profile['defense_score'] = 0.7
        elif avg_conceded < 1.5:
            profile['defense_style'] = 'average'
            profile['defense_score'] = 0.5
        else:
            profile['defense_style'] = 'weak'
            profile['defense_score'] = 0.3
        
        # Oyun temposu
        total_goals_avg = features.get('avg_goals', 1.2) + features.get('avg_conceded', 1.2)
        over_2_5_rate = features.get('over_2_5_rate', 0.5)
        
        if total_goals_avg > 3.5 and over_2_5_rate > 0.6:
            profile['game_tempo'] = 'very_high'
            profile['tempo_score'] = 0.9
        elif total_goals_avg > 2.5:
            profile['game_tempo'] = 'high'
            profile['tempo_score'] = 0.7
        elif total_goals_avg > 2.0:
            profile['game_tempo'] = 'medium'
            profile['tempo_score'] = 0.5
        else:
            profile['game_tempo'] = 'low'
            profile['tempo_score'] = 0.3
        
        # Risk profili
        win_rate = features.get('win_rate', 0.33)
        draw_rate = features.get('general_draw_rate', 0.33)
        
        if win_rate > 0.5 and draw_rate < 0.2:
            profile['risk_appetite'] = 'aggressive'
            profile['risk_score'] = 0.8
        elif draw_rate > 0.4:
            profile['risk_appetite'] = 'conservative'
            profile['risk_score'] = 0.3
        else:
            profile['risk_appetite'] = 'balanced'
            profile['risk_score'] = 0.5
        
        # Form durumu
        form_momentum = features.get('form_momentum', 0)
        if form_momentum > 0.3:
            profile['current_form'] = 'improving'
        elif form_momentum < -0.3:
            profile['current_form'] = 'declining'
        else:
            profile['current_form'] = 'stable'
            
        return profile
    
    def _enrich_with_ml(self, features: Dict, profile: Dict) -> Dict:
        """
        ML modelleri ile özellikleri zenginleştir
        """
        enriched = features.copy()
        
        try:
            # Feature vektörünü hazırla
            feature_vector = np.array([
                features.get('avg_goals', 1.2),
                features.get('avg_conceded', 1.2),
                features.get('win_rate', 0.33),
                features.get('goal_difference', 0),
                features.get('scoring_consistency', 0.5),
                features.get('defensive_stability', 0.5),
                features.get('clean_sheet_rate', 0.2),
                features.get('btts_rate', 0.5),
                features.get('over_2_5_rate', 0.5),
                features.get('form_momentum', 0)
            ]).reshape(1, -1)
            
            # Profile skorlarını ekle
            enriched['attack_strength'] = profile.get('attack_score', 0.5)
            enriched['defense_strength'] = profile.get('defense_score', 0.5)
            enriched['tempo_factor'] = profile.get('tempo_score', 0.5)
            enriched['risk_factor'] = profile.get('risk_score', 0.5)
            
            # Adaptasyon faktörü hesapla (takımın farklı koşullara uyumu)
            venue_dominance = features.get('venue_dominance', 1.0)
            venue_resilience = features.get('venue_resilience', 1.0)
            enriched['adaptability'] = (venue_dominance + venue_resilience) / 2
            
            # Tutarlılık faktörü
            enriched['consistency_factor'] = (features.get('scoring_consistency', 0.5) + 
                                             features.get('defensive_stability', 0.5)) / 2
            
            # Momentum faktörü
            enriched['momentum_factor'] = 0.5 + features.get('form_momentum', 0) * 0.5
            
        except Exception as e:
            logger.warning(f"ML enrichment hatası: {e}")
            
        return enriched
    
    def _normalize_features(self, features: Dict) -> np.ndarray:
        """
        Özellikleri normalize et (0-1 aralığına)
        """
        # Normalize edilecek özellikler
        keys_to_normalize = [
            'avg_goals', 'avg_conceded', 'win_rate', 'goal_difference',
            'attack_strength', 'defense_strength', 'tempo_factor', 'risk_factor',
            'adaptability', 'consistency_factor', 'momentum_factor'
        ]
        
        feature_vector = []
        for key in keys_to_normalize:
            value = features.get(key, 0.5)
            # Min-max normalization
            if key in ['avg_goals', 'avg_conceded']:
                normalized = min(1.0, value / 3.0)  # 3 gol üstü = 1.0
            elif key in ['goal_difference']:
                normalized = 0.5 + (value / 4.0)  # -2 to +2 range -> 0 to 1
                normalized = max(0, min(1, normalized))
            elif key in ['win_rate', 'attack_strength', 'defense_strength', 'tempo_factor', 
                        'risk_factor', 'adaptability', 'consistency_factor', 'momentum_factor']:
                normalized = value  # Zaten 0-1 aralığında
            else:
                normalized = 0.5
                
            feature_vector.append(normalized)
            
        return np.array(feature_vector)
    
    def _encode_form(self, form_string: str) -> float:
        """
        Form string'ini (W/D/L) sayısal değere çevir
        """
        if not form_string:
            return 0.5
            
        points = 0
        for char in form_string[-5:]:  # Son 5 maç
            if char == 'W':
                points += 3
            elif char == 'D':
                points += 1
                
        return points / 15.0  # Max 15 puan, normalize to 0-1
    
    def _calculate_feature_quality(self, team_data: Dict) -> float:
        """
        Veri kalitesi skoru hesapla
        """
        quality_score = 0.0
        
        # Veri mevcudiyeti kontrolleri
        if 'recent_matches' in team_data and team_data['recent_matches']:
            quality_score += 0.3
            if len(team_data['recent_matches']) >= 10:
                quality_score += 0.2
                
        if 'home_performance' in team_data and 'last_5_matches' in team_data['home_performance']:
            if team_data['home_performance']['last_5_matches'] >= 5:
                quality_score += 0.25
                
        if 'away_performance' in team_data and 'last_5_matches' in team_data['away_performance']:
            if team_data['away_performance']['last_5_matches'] >= 5:
                quality_score += 0.25
                
        return min(1.0, quality_score)
    
    def _get_default_features(self) -> Dict:
        """
        Varsayılan özellik seti döndür
        """
        return {
            'raw_features': {
                'avg_goals': 1.2,
                'avg_conceded': 1.2,
                'win_rate': 0.33,
                'goal_difference': 0,
                'scoring_consistency': 0.5,
                'defensive_stability': 0.5
            },
            'team_profile': {
                'attack_style': 'balanced',
                'defense_style': 'average',
                'game_tempo': 'medium',
                'risk_appetite': 'balanced',
                'current_form': 'stable'
            },
            'enriched_features': {
                'attack_strength': 0.5,
                'defense_strength': 0.5,
                'tempo_factor': 0.5,
                'risk_factor': 0.5,
                'adaptability': 1.0,
                'consistency_factor': 0.5,
                'momentum_factor': 0.5
            },
            'normalized_features': np.array([0.5] * 11),
            'venue_weight_used': 0,
            'feature_quality_score': 0
        }
    
    def adjust_weights_dynamically(self, team_data: Dict, match_context: Dict = None) -> None:
        """
        Takım karakteristiğine göre venue/general ağırlıklarını dinamik ayarla
        """
        # Deplasmanda güçlü takımlar için ağırlığı ayarla
        if 'away_performance' in team_data:
            away_win_rate = team_data['away_performance'].get('last_5_win_rate', 0.3)
            if away_win_rate > 0.5:  # Deplasmanda çok güçlü
                self.venue_weight = 0.5  # %50-%50 yap
                self.general_weight = 0.5
                logger.info(f"Deplasmanda güçlü takım tespit edildi, ağırlıklar %50-%50 yapıldı")
        
        # Evde zayıf takımlar için ağırlığı ayarla
        if 'home_performance' in team_data:
            home_win_rate = team_data['home_performance'].get('last_5_win_rate', 0.4)
            if home_win_rate < 0.3:  # Evde zayıf
                self.venue_weight = 0.45  # Venue ağırlığını azalt
                self.general_weight = 0.55
                logger.info(f"Evde zayıf takım tespit edildi, ağırlıklar %45-%55 yapıldı")
        
        # Maç bağlamına göre ayarlama
        if match_context:
            if match_context.get('is_derby', False):
                # Derbilerde form daha az önemli, venue daha önemli
                self.venue_weight = 0.75
                self.general_weight = 0.25
            elif match_context.get('is_cup_match', False):
                # Kupa maçlarında genel form daha önemli
                self.venue_weight = 0.4
                self.general_weight = 0.6