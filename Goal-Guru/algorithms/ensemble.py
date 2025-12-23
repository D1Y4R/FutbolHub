"""
Advanced Ensemble Tahmin Birleştirici
Tüm modellerin ağırlıklı ortalamasını alarak nihai tahmin üretir
Genetik algoritma ile optimize edilmiş dinamik ağırlık sistemi ile çalışır
"""
import numpy as np
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json

# Proje root'a ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """
    Farklı modellerin tahminlerini birleştiren ensemble sistem
    Dinamik ağırlık hesaplama özelliği ile
    """
    
    def __init__(self, enable_genetic_optimization=True):
        # Dinamik ağırlık hesaplayıcıyı başlat
        try:
            from dynamic_weight_calculator import DynamicWeightCalculator
            self.dynamic_calculator = DynamicWeightCalculator()
            self.use_dynamic_weights = True
            logger.info("Dinamik ağırlık sistemi aktif")
        except Exception as e:
            logger.warning(f"Dinamik ağırlık sistemi yüklenemedi: {e}")
            self.dynamic_calculator = None
            self.use_dynamic_weights = False
        
        # Meta-Learning Layer'ı başlat - Öncelikli sistem
        self.meta_learning_layer = None
        self.use_meta_learning = False
        
        try:
            from algorithms.meta_learning_layer import MetaLearningLayer
            self.meta_learning_layer = MetaLearningLayer()
            self.use_meta_learning = True
            logger.info("🧠 Meta-Learning Layer aktif - Akıllı model seçimi etkin")
        except Exception as e:
            logger.warning(f"Meta-Learning Layer yüklenemedi: {e}")
            self.meta_learning_layer = None
            self.use_meta_learning = False
        
        # League Strength Analyzer'ı başlat - Cross-league matches için
        try:
            from algorithms.league_strength_analyzer import LeagueStrengthAnalyzer
            self.league_strength_analyzer = LeagueStrengthAnalyzer()
            self.use_league_strength_adjustment = True
            logger.info("🌍 League Strength Analyzer aktif - Cross-league düzenlemesi etkin")
        except Exception as e:
            logger.warning(f"League Strength Analyzer yüklenemedi: {e}")
            self.league_strength_analyzer = None
            self.use_league_strength_adjustment = False
        
        # Genetic Algorithm Optimizer'ı başlat
        self.genetic_optimizer = None
        self.context_aware_optimizer = None
        self.use_genetic_optimization = False
        
        if enable_genetic_optimization:
            try:
                from algorithms.genetic_ensemble_optimizer import (
                    GeneticEnsembleOptimizer, ContextAwareOptimizer, EvolutionConfig
                )
                
                # Genetic optimizer config
                config = EvolutionConfig(
                    population_size=30,  # Küçük population hızlı sonuç için
                    elite_size=6,
                    max_generations=50,  # Daha az generation
                    adaptive_parameters=True
                )
                
                self.genetic_optimizer = GeneticEnsembleOptimizer(config)
                self.context_aware_optimizer = ContextAwareOptimizer(self.genetic_optimizer)
                self.use_genetic_optimization = True
                
                # Previous optimization state'i yükle
                self.genetic_optimizer.load_evolution_state()
                
                logger.info("🧬 Genetik Algoritma Optimizasyonu aktif")
                
            except Exception as e:
                logger.warning(f"Genetik algoritma yüklenemedi: {e}")
                self.genetic_optimizer = None
                self.use_genetic_optimization = False
        
        # Optimization cache and tracking
        self.optimization_cache = {}
        self.cache_timeout = 3600  # 1 hour cache
        self.optimization_history = []
        self.performance_tracking = {}
        
        # Advanced Prediction Confidence System
        self.prediction_confidence_system = None
        self.use_advanced_confidence = False
        
        try:
            from algorithms.prediction_confidence_system import (
                PredictionConfidenceSystem, ModelPredictionInput, 
                MatchContext, PredictionType
            )
            self.prediction_confidence_system = PredictionConfidenceSystem()
            self.use_advanced_confidence = True
            logger.info("🎯 Gelişmiş Güven Sistemi aktif - Comprehensive confidence analysis enabled")
        except Exception as e:
            logger.warning(f"Gelişmiş güven sistemi yüklenemedi: {e}")
            self.prediction_confidence_system = None
            self.use_advanced_confidence = False
            
        # Varsayılan model ağırlıkları (fallback)
        self.weights = {
            'poisson': 0.25,     # Temel model, güvenilir
            'dixon_coles': 0.18, # Düşük skorlar için iyi
            'xgboost': 0.12,     # ML gücü
            'monte_carlo': 0.15, # Belirsizlik için
            'crf': 0.15,         # CRF tahmin modeli
            'neural_network': 0.15  # Neural Network modeli
        }
        
        # Ekstrem maçlar için ağırlıklar (fallback)
        self.extreme_weights = {
            'poisson': 0.35,     # Yüksek skorları daha iyi modeller
            'dixon_coles': 0.08, # Düşük skor eğilimini azalt
            'xgboost': 0.18,     # Veri tabanlı tahmin
            'monte_carlo': 0.15, # Simülasyon
            'crf': 0.12,         # CRF modeli
            'neural_network': 0.12  # Neural Network modeli
        }
        
        # Durum bazlı ağırlık ayarları (fallback)
        self.adjustments = {
            'low_scoring': {'dixon_coles': +0.10, 'poisson': -0.10},
            'high_elo_diff': {'xgboost': +0.10, 'monte_carlo': -0.05},
            'close_match': {'monte_carlo': +0.05, 'xgboost': -0.05}
        }
        
    def _fallback_weight_calculation(self, match_context, algorithm_weights=None):
        """
        Eski ağırlık hesaplama sistemi (fallback)
        """
        # Ekstrem maç kontrolü
        from algorithms.extreme_detector import ExtremeMatchDetector
        detector = ExtremeMatchDetector()
        
        home_stats = match_context.get('home_stats', {})
        away_stats = match_context.get('away_stats', {})
        
        is_extreme, extreme_details = detector.is_extreme_match(home_stats, away_stats)
        
        # Dinamik ağırlıklar sağlanmışsa öncelik ver
        if algorithm_weights:
            adjusted_weights = algorithm_weights.copy()
            logger.info(f"Self-learning dinamik ağırlıklar kullanılıyor: {adjusted_weights}")
        # Ekstrem maç ise özel ağırlıkları kullan
        elif is_extreme:
            adjusted_weights = self.extreme_weights.copy()
            logger.info(f"Ekstrem maç algılandı, özel ağırlıklar kullanılıyor")
        else:
            adjusted_weights = self.weights.copy()
        
        # Bağlama göre ağırlık ayarla
        adjusted_weights = self._adjust_weights_by_context(adjusted_weights, match_context)
        
        # Normalize et
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
        
    def combine_predictions(self, model_predictions, match_context, algorithm_weights=None, force_genetic=False):
        """
        Farklı model tahminlerini birleştir
        
        Args:
            model_predictions: dict - Her modelin tahminleri
            match_context: dict - Maç bağlamı (lambda'lar, elo farkı vb.)
            algorithm_weights: dict - Opsiyonel dinamik ağırlıklar (self-learning'den)
            force_genetic: bool - Genetic optimization'ı zorla kullan
            
        Returns:
            dict: Birleştirilmiş tahmin
        """
        start_time = datetime.now()
        
        # Meta-Learning Layer - Öncelikli akıllı sistem
        if self.use_meta_learning and self.meta_learning_layer:
            try:
                # Available models for selection
                available_models = list(model_predictions.keys())
                
                # Get optimal model weights from meta-learning
                optimal_models = self.meta_learning_layer.select_optimal_models(match_context, available_models)
                
                # Convert to weight dictionary
                adjusted_weights = {}
                for model_name, weight in optimal_models:
                    adjusted_weights[model_name] = weight
                
                optimization_method = "🧠 Meta-Learning Akıllı Seçim"
                
                logger.info(f"Meta-learning model seçimi: {[(m, f'{w:.3f}') for m, w in optimal_models[:3]]}")
                
            except Exception as e:
                logger.warning(f"Meta-learning optimizasyon hatası: {e}")
                adjusted_weights = None
                optimization_method = "🧠 Meta-Learning (HATA)"
        
        # Genetic Algorithm Optimization - İkinci seçenek
        elif (self.use_genetic_optimization and 
            (force_genetic or self._should_use_genetic_optimization(match_context))):
            try:
                adjusted_weights = self._get_genetic_optimized_weights(match_context)
                optimization_method = "🧬 Genetik Algoritma"
                
            except Exception as e:
                logger.warning(f"Genetik optimizasyon hatası: {e}")
                adjusted_weights = None
                optimization_method = "🧬 Genetik Algoritma (HATA)"
        
        # Dynamic Weight Calculator - Üçüncü seçenek
        elif self.use_dynamic_weights and self.dynamic_calculator:
            try:
                # Maç bilgilerini hazırla
                match_info = {
                    'league': match_context.get('league', ''),
                    'home_team': match_context.get('home_team', ''),
                    'away_team': match_context.get('away_team', ''),
                    'elo_diff': match_context.get('elo_diff', 0),
                    'home_stats': match_context.get('home_stats', {}),
                    'away_stats': match_context.get('away_stats', {}),
                    'date': match_context.get('date', ''),
                    'home_position': match_context.get('home_position', 10),
                    'away_position': match_context.get('away_position', 10)
                }
                
                # Dinamik ağırlıkları hesapla
                adjusted_weights = self.dynamic_calculator.calculate_weights(match_info)
                optimization_method = "⚡ Dinamik Ağırlık Hesaplayıcı"
                
            except Exception as e:
                logger.error(f"Dinamik ağırlık hesaplama hatası: {e}")
                adjusted_weights = None
                optimization_method = "⚡ Dinamik Ağırlık (HATA)"
        else:
            adjusted_weights = None
            optimization_method = "📊 Fallback Sistem"
        
        # Fallback sistem - Hiçbir gelişmiş sistem çalışmazsa
        if adjusted_weights is None:
            adjusted_weights = self._fallback_weight_calculation(match_context, algorithm_weights)
            if optimization_method.endswith("(HATA)") or optimization_method == "📊 Fallback Sistem":
                optimization_method = "📊 Fallback Sistem"
        
        # Performance tracking
        optimization_time = (datetime.now() - start_time).total_seconds()
        self._track_optimization_performance(match_context, adjusted_weights, optimization_method, optimization_time)
        
        logger.info(f"Ensemble ağırlıkları ({optimization_method}): {adjusted_weights}")
        logger.debug(f"Optimizasyon süresi: {optimization_time:.3f}s")
        
        # Birleştirilmiş tahminler - ÖNCE TANIMLANMALI
        combined = {
            'home_win': 0.0,
            'draw': 0.0,
            'away_win': 0.0,
            'over_2_5': 0.0,
            'under_2_5': 0.0,
            'btts_yes': 0.0,
            'btts_no': 0.0,
            'expected_goals': {'home': 0.0, 'away': 0.0},
            'most_likely_scores': {},
            'confidence': 0.0
        }
        
        # Her modelin katkısını ekle
        for model_name, predictions in model_predictions.items():
            if model_name not in adjusted_weights:
                continue
                
            weight = adjusted_weights[model_name]
            
            # 1X2 tahminleri
            combined['home_win'] += predictions.get('home_win', 0) * weight
            combined['draw'] += predictions.get('draw', 0) * weight
            combined['away_win'] += predictions.get('away_win', 0) * weight
            
            # Gol tahminleri
            combined['over_2_5'] += predictions.get('over_2_5', 0) * weight
            combined['under_2_5'] += predictions.get('under_2_5', 0) * weight
            combined['btts_yes'] += predictions.get('btts_yes', 0) * weight
            combined['btts_no'] += predictions.get('btts_no', 0) * weight
            
            # Beklenen goller
            if 'expected_goals' in predictions:
                combined['expected_goals']['home'] += predictions['expected_goals'].get('home', 0) * weight
                combined['expected_goals']['away'] += predictions['expected_goals'].get('away', 0) * weight
                
            # Güven seviyesi - modelin kendi güveni ve tahmin keskinliği
            model_confidence = predictions.get('confidence', 70)  # Varsayılan %70
            # En yüksek olasılığa göre güven ayarla
            max_prob = max(predictions.get('home_win', 0), predictions.get('draw', 0), predictions.get('away_win', 0))
            # Güven değerini hesapla (zaten yüzde olarak geliyor)
            adjusted_confidence = model_confidence * weight
            combined['confidence'] += adjusted_confidence
            
            # Debug: Model güven değerlerini logla
            logger.debug(f"Model {model_name}: confidence={model_confidence}, max_prob={max_prob}, adjusted={adjusted_confidence}, weight={weight}")
            
        # Beraberlik düzeltme faktörünü uygula
        # Rating farkını kontrol et
        if 'elo_diff' in match_context:
            rating_diff = abs(match_context['elo_diff'])
            is_derby = match_context.get('is_derby', False)
            
            # Beraberlik düzeltme çarpanını hesapla
            draw_multiplier = 1.0
            
            # Rating farkına göre düzeltme (Güçlendirildi - matematiksel bias için)
            if rating_diff < 100:
                draw_multiplier += 0.35  # %35 artış (denk takımlar)
            elif rating_diff < 200:
                draw_multiplier += 0.25  # %25 artış (yakın takımlar)
            elif rating_diff < 300:
                draw_multiplier += 0.15  # %15 artış (orta fark)
                
            # Derbi düzeltmesi (güçlendirildi)
            if is_derby:
                draw_multiplier += 0.25  # %25 ek artış (derbiler daha çok beraberlik)
                
            # Beraberlik olasılığını düzelt
            if draw_multiplier > 1.0:
                # Orijinal beraberlik olasılığı
                original_draw = combined['draw']
                
                # Yeni beraberlik olasılığı
                new_draw = min(100, original_draw * draw_multiplier)
                
                # Farkı diğer sonuçlardan çıkar
                draw_increase = new_draw - original_draw
                if draw_increase > 0:
                    # Ev ve deplasman olasılıklarını orantılı olarak azalt
                    home_ratio = combined['home_win'] / (combined['home_win'] + combined['away_win']) if (combined['home_win'] + combined['away_win']) > 0 else 0.5
                    away_ratio = 1 - home_ratio
                    
                    combined['home_win'] = max(0, combined['home_win'] - draw_increase * home_ratio)
                    combined['away_win'] = max(0, combined['away_win'] - draw_increase * away_ratio)
                    combined['draw'] = new_draw
                    
                logger.info(f"Beraberlik düzeltmesi uygulandı: {original_draw:.1f}% -> {new_draw:.1f}% (x{draw_multiplier:.2f})")
        
        # Kesin skor tahminleri (matrislerden)
        combined['most_likely_scores'] = self._combine_score_predictions(model_predictions, adjusted_weights)
        
        # Kesin skorla 1X2 manipülasyonu KALDIRILDI
        # Kullanıcı istediği gibi matematiksel bias giderildi, kesin skor müdahalesi yok
        # Sadece istatistiksel tahminler korundu
        
        # En olası sonucu belirle - TUTARLILIK KONTROLÜ EKLENDI
        outcomes = {
            'HOME_WIN': combined['home_win'],
            'DRAW': combined['draw'],
            'AWAY_WIN': combined['away_win']
        }
        
        # Önce standart maksimum hesapla
        most_likely_outcome = max(outcomes, key=outcomes.get)
        
        # TUTARLILIK KONTROLÜ: En olası skor ile 1X2 uyumunu kontrol et
        if combined['most_likely_scores']:
            top_score = combined['most_likely_scores'][0]
            score_parts = top_score['score'].split('-')
            
            if len(score_parts) == 2:
                home_goals = int(score_parts[0])
                away_goals = int(score_parts[1])
                
                # En olası skorun sonucu ne?
                score_outcome = 'DRAW' if home_goals == away_goals else ('HOME_WIN' if home_goals > away_goals else 'AWAY_WIN')
                
                # Tutarlılık kontrolü - En olası skor ile 1X2 uyuşmuyor mu?
                if score_outcome != most_likely_outcome and top_score['probability'] > 3.0:
                    # 1X2 olasılıkları arasındaki fark küçükse (<%10), en olası skora göre ayarla
                    max_prob_diff = max(abs(outcomes['HOME_WIN'] - outcomes['DRAW']),
                                       abs(outcomes['HOME_WIN'] - outcomes['AWAY_WIN']),
                                       abs(outcomes['DRAW'] - outcomes['AWAY_WIN']))
                    
                    if max_prob_diff < 10.0:  # Fark %10'dan azsa
                        logger.info(f"🔄 TUTARLILIK DÜZELTMESİ: En olası skor {top_score['score']} (%{top_score['probability']:.1f}) → {score_outcome}")
                        logger.info(f"   1X2 eski: Home {outcomes['HOME_WIN']:.1f}%, Draw {outcomes['DRAW']:.1f}%, Away {outcomes['AWAY_WIN']:.1f}%")
                        
                        most_likely_outcome = score_outcome
                        
                        # Olasılıkları da güncelle (skor sonucu lehine)
                        adjustment = 8  # %8 artış
                        
                        if score_outcome == 'DRAW':
                            combined['draw'] = min(100, combined['draw'] + adjustment)
                            combined['home_win'] = max(0, combined['home_win'] - adjustment/2)
                            combined['away_win'] = max(0, combined['away_win'] - adjustment/2)
                        elif score_outcome == 'HOME_WIN':
                            combined['home_win'] = min(100, combined['home_win'] + adjustment)
                            combined['draw'] = max(0, combined['draw'] - adjustment/2)
                            combined['away_win'] = max(0, combined['away_win'] - adjustment/2)
                        else:  # AWAY_WIN
                            combined['away_win'] = min(100, combined['away_win'] + adjustment)
                            combined['draw'] = max(0, combined['draw'] - adjustment/2)
                            combined['home_win'] = max(0, combined['home_win'] - adjustment/2)
                        
                        logger.info(f"   1X2 yeni: Home {combined['home_win']:.1f}%, Draw {combined['draw']:.1f}%, Away {combined['away_win']:.1f}%")
        
        combined['most_likely_outcome'] = most_likely_outcome
        
        # ADVANCED CONFIDENCE SYSTEM - Gelişmiş güven hesaplama
        if self.use_advanced_confidence and self.prediction_confidence_system:
            try:
                # Model prediction inputs oluştur
                model_prediction_inputs = []
                for model_name, predictions in model_predictions.items():
                    if model_name not in adjusted_weights:
                        continue
                    
                    # Convert predictions to expected format
                    model_prediction = {
                        'home_win': predictions.get('home_win', 0),
                        'draw': predictions.get('draw', 0),
                        'away_win': predictions.get('away_win', 0)
                    }
                    
                    # Historical accuracy için varsayılan değer
                    historical_accuracy = 0.7  # Varsayılan %70
                    if hasattr(self, 'model_performance_history'):
                        historical_accuracy = self.model_performance_history.get(model_name, 0.7)
                    
                    # Create model prediction input
                    from algorithms.prediction_confidence_system import ModelPredictionInput
                    model_input = ModelPredictionInput(
                        model_name=model_name,
                        prediction=model_prediction,
                        confidence=predictions.get('confidence', 70),
                        historical_accuracy=historical_accuracy,
                        context_performance=adjusted_weights[model_name],  # Model weight as performance indicator
                        data_quality=match_context.get('data_completeness', 0.8),
                        features_used=list(predictions.keys()),
                        uncertainty=max(0.1, 1 - predictions.get('confidence', 70) / 100)
                    )
                    model_prediction_inputs.append(model_input)
                
                # Match context oluştur
                from algorithms.prediction_confidence_system import MatchContext
                confidence_match_context = MatchContext(
                    league=match_context.get('league', 'Unknown'),
                    teams=(match_context.get('home_team', 'Home'), match_context.get('away_team', 'Away')),
                    team_strengths=(match_context.get('home_strength', 50), match_context.get('away_strength', 50)),
                    recent_form=(match_context.get('home_form_score', 0.5), match_context.get('away_form_score', 0.5)),
                    head_to_head_history=match_context.get('h2h_matches', 5),
                    data_completeness=match_context.get('data_completeness', 0.8),
                    match_importance=match_context.get('match_importance', 0.5),
                    seasonal_period=match_context.get('seasonal_period', 'mid_season'),
                    venue_type=match_context.get('venue_type', 'home'),
                    weather_conditions=match_context.get('weather', None),
                    fixture_congestion=match_context.get('fixture_congestion', 0.3)
                )
                
                # Comprehensive confidence calculation
                from algorithms.prediction_confidence_system import PredictionType
                confidence_metrics = self.prediction_confidence_system.calculate_comprehensive_confidence(
                    model_predictions=model_prediction_inputs,
                    match_context=confidence_match_context,
                    prediction_type=PredictionType.WIN_DRAW_LOSS
                )
                
                # Use comprehensive confidence metrics
                combined['confidence'] = confidence_metrics.overall_confidence
                combined['confidence_details'] = {
                    'model_agreement': confidence_metrics.model_agreement,
                    'prediction_variance': confidence_metrics.prediction_variance,
                    'historical_accuracy': confidence_metrics.historical_accuracy,
                    'data_quality': confidence_metrics.data_quality_score,
                    'context_familiarity': confidence_metrics.context_familiarity,
                    'stability_score': confidence_metrics.stability_score,
                    'uncertainty_interval': confidence_metrics.uncertainty_interval,
                    'risk_adjusted_confidence': confidence_metrics.risk_adjusted_confidence,
                    'recommendation_strength': confidence_metrics.recommendation_strength,
                    'explanation': confidence_metrics.explanation,
                    'alert_level': confidence_metrics.alert_level
                }
                
                logger.info(f"🎯 Gelişmiş Güven Sistemi: {confidence_metrics.overall_confidence:.1f}% ({confidence_metrics.alert_level})")
                logger.info(f"📊 Model Anlaşması: {confidence_metrics.model_agreement:.3f}, Varyans: {confidence_metrics.prediction_variance:.3f}")
                logger.info(f"💡 Açıklama: {confidence_metrics.explanation}")
                
            except Exception as e:
                logger.warning(f"Gelişmiş güven sistemi hatası: {e}")
                # Fallback to basic confidence calculation
                combined['confidence'] = self._calculate_basic_confidence(outcomes, model_predictions)
                
        else:
            # Fallback: Basic confidence calculation
            combined['confidence'] = self._calculate_basic_confidence(outcomes, model_predictions)
        
        # CROSS-LEAGUE ADJUSTMENT - Farklı lig takımları için düzeltme
        # BU NORMALIZE EDİLMEMİŞ OLASILIKLARLA ÇALIŞMALI (normalize edilmeden önce)
        league_strength_info = self._apply_league_strength_adjustment(match_context, combined, model_predictions)
        
        # BERABERLIK MINIMUM SINIR KONTROLÜ - Lig istatistiklerine göre
        # Ortalama futbol maçlarında beraberlik oranı %25-28 civarındadır
        # Minimum %15 sınırı mantıklı bir alt limit
        MIN_DRAW_PROBABILITY = 15.0  # %15 minimum beraberlik
        MAX_WIN_PROBABILITY = 75.0   # %75 maksimum tek sonuç
        
        # Beraberlik çok düşükse düzelt
        if combined['draw'] < MIN_DRAW_PROBABILITY:
            draw_deficit = MIN_DRAW_PROBABILITY - combined['draw']
            combined['draw'] = MIN_DRAW_PROBABILITY
            
            # Eksik miktarı ev/deplasman'dan orantılı olarak çıkar
            # GÜVENLIK: Negatif değerleri önlemek için sınırlandır
            total_wins = combined['home_win'] + combined['away_win']
            if total_wins > 0:
                # Çıkarılabilecek maksimum miktar = toplam kazanç olasılıkları
                # Her iki sonuç da minimum %5 kalmalı
                min_win_threshold = 5.0
                available_for_draw = max(0, total_wins - 2 * min_win_threshold)
                actual_draw_deficit = min(draw_deficit, available_for_draw)
                
                if actual_draw_deficit > 0:
                    home_ratio = combined['home_win'] / total_wins
                    away_ratio = combined['away_win'] / total_wins
                    # Her sonuçtan çıkarılacak miktarı sınırla
                    home_deduction = min(actual_draw_deficit * home_ratio, combined['home_win'] - min_win_threshold)
                    away_deduction = min(actual_draw_deficit * away_ratio, combined['away_win'] - min_win_threshold)
                    combined['home_win'] = max(min_win_threshold, combined['home_win'] - max(0, home_deduction))
                    combined['away_win'] = max(min_win_threshold, combined['away_win'] - max(0, away_deduction))
            
            logger.info(f"⚖️ Beraberlik düzeltmesi: minimum %{MIN_DRAW_PROBABILITY} uygulandı (eksik: {draw_deficit:.1f}%)")
        
        # Tek bir sonuç çok yüksekse sınırla
        for outcome in ['home_win', 'away_win']:
            if combined[outcome] > MAX_WIN_PROBABILITY:
                excess = combined[outcome] - MAX_WIN_PROBABILITY
                combined[outcome] = MAX_WIN_PROBABILITY
                combined['draw'] += excess * 0.6  # Fazlalığın %60'ını beraberliğe
                other_outcome = 'away_win' if outcome == 'home_win' else 'home_win'
                combined[other_outcome] += excess * 0.4  # %40'ını diğer sonuca
                logger.info(f"⚖️ {outcome} sınırlandı: max %{MAX_WIN_PROBABILITY} (fazla: {excess:.1f}%)")
        
        # 1X2 olasılıklarını normalize et (toplamı 100'e tamamla)
        match_outcome_total = combined['home_win'] + combined['draw'] + combined['away_win']
        if match_outcome_total > 0:
            combined['home_win'] = (combined['home_win'] / match_outcome_total) * 100
            combined['draw'] = (combined['draw'] / match_outcome_total) * 100
            combined['away_win'] = (combined['away_win'] / match_outcome_total) * 100
        else:
            # Fallback durumu
            combined['home_win'] = 33.3
            combined['draw'] = 33.3
            combined['away_win'] = 33.4
            
        # Son kontrol: Normalize sonrası beraberlik hala çok düşükse tekrar düzelt
        # GÜVENLIK: Negatif değerleri önlemek için sınırlandır
        if combined['draw'] < MIN_DRAW_PROBABILITY:
            logger.warning(f"⚠️ Normalize sonrası beraberlik hala düşük: {combined['draw']:.1f}%")
            draw_boost_needed = MIN_DRAW_PROBABILITY - combined['draw']
            combined['draw'] = MIN_DRAW_PROBABILITY
            # Orantılı olarak diğerlerinden çıkar - ama sınırlı
            total_wins = combined['home_win'] + combined['away_win']
            min_win_threshold = 5.0
            available = max(0, total_wins - 2 * min_win_threshold)
            actual_boost = min(draw_boost_needed, available)
            
            if total_wins > 0 and actual_boost > 0:
                home_ratio = combined['home_win'] / total_wins
                home_deduction = min(actual_boost * home_ratio, combined['home_win'] - min_win_threshold)
                away_deduction = min(actual_boost * (1 - home_ratio), combined['away_win'] - min_win_threshold)
                combined['home_win'] = max(min_win_threshold, combined['home_win'] - max(0, home_deduction))
                combined['away_win'] = max(min_win_threshold, combined['away_win'] - max(0, away_deduction))
            
        # BTTS değerlerini normalize et (toplamı 100'e tamamla)
        btts_total = combined['btts_yes'] + combined['btts_no']
        if btts_total > 0:
            combined['btts_yes'] = (combined['btts_yes'] / btts_total) * 100
            combined['btts_no'] = (combined['btts_no'] / btts_total) * 100
        else:
            # Fallback durumu
            combined['btts_yes'] = 50.0
            combined['btts_no'] = 50.0
            
        # Over/Under değerlerini normalize et (toplamı 100'e tamamla)
        ou_total = combined['over_2_5'] + combined['under_2_5']
        if ou_total > 0:
            combined['over_2_5'] = (combined['over_2_5'] / ou_total) * 100
            combined['under_2_5'] = (combined['under_2_5'] / ou_total) * 100
        else:
            # Fallback durumu
            combined['over_2_5'] = 45.0
            combined['under_2_5'] = 55.0
        
        return combined
        
    def _adjust_weights_by_context(self, weights, context):
        """
        Maç bağlamına göre ağırlıkları ayarla
        """
        lambda_home = context.get('lambda_home', 1.5)
        lambda_away = context.get('lambda_away', 1.5)
        elo_diff = context.get('elo_diff', 0)
        
        # Düşük skorlu maç (toplam lambda < 2.5)
        if lambda_home + lambda_away < 2.5:
            logger.debug("Düşük skorlu maç tespit edildi")
            for model, adjustment in self.adjustments['low_scoring'].items():
                if model in weights:
                    weights[model] = max(0, weights[model] + adjustment)
                    
        # Yüksek Elo farkı (favori var)
        if abs(elo_diff) > 300:
            logger.debug(f"Yüksek Elo farkı: {elo_diff}")
            for model, adjustment in self.adjustments['high_elo_diff'].items():
                if model in weights:
                    weights[model] = max(0, weights[model] + adjustment)
                    
        # Yakın maç
        elif abs(elo_diff) < 100:
            logger.debug("Yakın maç tespit edildi")
            for model, adjustment in self.adjustments['close_match'].items():
                if model in weights:
                    weights[model] = max(0, weights[model] + adjustment)
                    
        return weights
    
    def _should_use_genetic_optimization(self, match_context: Dict) -> bool:
        """
        Genetic optimization'ın kullanılıp kullanılmayacağını belirle
        
        Args:
            match_context: Maç bağlamı
            
        Returns:
            bool: Genetic optimization kullanılmalı mı?
        """
        # Genetic optimization kriterleri
        criteria = {
            'high_stakes_match': False,
            'complex_context': False,
            'performance_gap': False,
            'special_circumstances': False
        }
        
        # 1. Yüksek risk maçlar (derbi, kupalar, play-off vb.)
        league = match_context.get('league', '').lower()
        match_type = match_context.get('match_type', 'balanced')
        
        high_stakes_keywords = ['champions_league', 'europa_league', 'cup', 'final', 'derby', 'playoff']
        if any(keyword in league for keyword in high_stakes_keywords) or match_type == 'derby':
            criteria['high_stakes_match'] = True
        
        # 2. Karmaşık context (büyük elo farkı, özel durumlar)
        elo_diff = abs(match_context.get('elo_diff', 0))
        if elo_diff > 200 or match_type in ['heavy_favorite', 'extreme']:
            criteria['complex_context'] = True
        
        # 3. Model performans açığı varsa
        recent_performance = self._get_recent_ensemble_performance()
        if recent_performance and recent_performance < 0.65:  # %65'in altında
            criteria['performance_gap'] = True
        
        # 4. Özel koşullar (sezon sonu, transfer dönemi vb.)
        current_month = datetime.now().month
        if current_month in [5, 6, 7, 8]:  # Transfer dönemleri ve sezon geçişleri
            criteria['special_circumstances'] = True
        
        # En az 2 kriter sağlanırsa genetic optimization kullan
        active_criteria = sum(criteria.values())
        should_use = active_criteria >= 2
        
        if should_use:
            logger.info(f"🧬 Genetic optimization tetiklendi: {criteria}")
        
        return should_use
    
    def _calculate_basic_confidence(self, outcomes: dict, model_predictions: dict) -> float:
        """
        Fallback: Basic confidence calculation (legacy system)
        """
        max_outcome_prob = max(outcomes.values())
        
        # Yeni dinamik güven hesaplama - tahmin keskinliğine göre (YÜZDE OLARAK)
        if max_outcome_prob > 60:  # Çok net favori
            # %60+ için güven %75-90 arası
            confidence_boost = (max_outcome_prob - 60) / 40  # 0-1 arası
            confidence = 75 + (confidence_boost * 15)  # %75-90
        elif max_outcome_prob > 45:  # Orta düzey favori  
            # %45-60 için güven %60-75 arası
            confidence_boost = (max_outcome_prob - 45) / 15  # 0-1 arası
            confidence = 60 + (confidence_boost * 15)  # %60-75
        elif max_outcome_prob > 35:  # Hafif favori
            # %35-45 için güven %50-60 arası
            confidence_boost = (max_outcome_prob - 35) / 10  # 0-1 arası
            confidence = 50 + (confidence_boost * 10)  # %50-60
        else:  # Çok dengeli maç
            # %35 altı için güven %45-50 arası
            confidence_boost = max(0, (max_outcome_prob - 25) / 10)  # 0-1 arası
            confidence = 45 + (confidence_boost * 5)  # %45-50
            
        # Model güven değerlerinin ortalamasını da hesaba kat
        model_confidence_avg = 0
        model_count = 0
        for model_name, predictions in model_predictions.items():
            if 'confidence' in predictions:
                model_confidence_avg += predictions['confidence']
                model_count += 1
        
        if model_count > 0:
            model_confidence_avg /= model_count
            # Final güven = %70 tahmin keskinliği + %30 model ortalaması
            confidence = (confidence * 0.7) + (model_confidence_avg * 0.3)
            
        # Güven değerini sınırla (YÜZDE OLARAK)
        confidence = max(45, min(90, confidence))
        
        logger.info(f"📊 Fallback güven hesaplama: {confidence:.1f}% (Max prob: {max_outcome_prob:.1f}%)")
        
        return confidence
    
    def _get_genetic_optimized_weights(self, match_context: Dict) -> Dict[str, float]:
        """
        Context-aware genetic optimization ile ağırlıkları optimize et
        
        Args:
            match_context: Maç bağlamı
            
        Returns:
            Dict[str, float]: Optimize edilmiş ağırlıklar
        """
        # Cache kontrolü
        context_key = self._generate_context_key(match_context)
        
        if context_key in self.optimization_cache:
            cached_result, timestamp = self.optimization_cache[context_key]
            if (datetime.now() - timestamp).seconds < self.cache_timeout:
                logger.debug("🧬 Genetic optimization sonucu cache'den alındı")
                return cached_result
        
        # Context-specific optimization
        if self.context_aware_optimizer:
            try:
                # Context listesi oluştur
                context_contexts = [match_context]
                
                # Context type belirle
                league = match_context.get('league', 'unknown')
                match_type = match_context.get('match_type', 'balanced')
                context_type = f"{league}_{match_type}"
                
                # Optimize et
                optimized_weights = self.context_aware_optimizer.optimize_for_context(
                    context_type=context_type,
                    match_contexts=context_contexts,
                    cache_timeout=self.cache_timeout
                )
                
                # Cache'e kaydet
                self.optimization_cache[context_key] = (optimized_weights, datetime.now())
                
                logger.info(f"🧬 Genetic optimization tamamlandı: {context_type}")
                return optimized_weights
                
            except Exception as e:
                logger.error(f"Context-aware optimization hatası: {e}")
                raise
        
        # Fallback: Direct genetic optimization
        if self.genetic_optimizer:
            try:
                optimized_weights = self.genetic_optimizer.optimize_weights(
                    match_contexts=[match_context]
                )
                
                # Cache'e kaydet
                self.optimization_cache[context_key] = (optimized_weights, datetime.now())
                
                logger.info("🧬 Direct genetic optimization tamamlandı")
                return optimized_weights
                
            except Exception as e:
                logger.error(f"Direct genetic optimization hatası: {e}")
                raise
        
        raise Exception("Genetic optimizer mevcut değil")
    
    def _generate_context_key(self, match_context: Dict) -> str:
        """Context için cache key oluştur"""
        import math
        
        league = match_context.get('league', 'unknown')
        match_type = match_context.get('match_type', 'balanced')
        
        # NaN kontrolü - geçersiz değerleri 0 ile değiştir
        elo_diff = match_context.get('elo_diff', 0)
        if math.isnan(elo_diff) or elo_diff is None:
            elo_diff = 0
        
        elo_diff_range = int(abs(elo_diff) / 100) * 100  # 100'lük gruplar
        
        return f"{league}_{match_type}_{elo_diff_range}"
    
    def _get_recent_ensemble_performance(self) -> Optional[float]:
        """Son zamanlardaki ensemble performansını al"""
        if not self.optimization_history:
            return None
        
        # Son 10 optimization'ın ortalaması
        recent_performances = [
            entry.get('performance', 0.5) 
            for entry in self.optimization_history[-10:]
        ]
        
        if recent_performances:
            return sum(recent_performances) / len(recent_performances)
        
        return None
    
    def _track_optimization_performance(self, 
                                      match_context: Dict, 
                                      weights: Dict[str, float], 
                                      method: str, 
                                      optimization_time: float):
        """Optimization performansını takip et"""
        # Performance entry oluştur
        performance_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'optimization_time': optimization_time,
            'context': {
                'league': match_context.get('league', ''),
                'match_type': match_context.get('match_type', ''),
                'elo_diff': match_context.get('elo_diff', 0)
            },
            'weights': weights.copy(),
            'performance': 0.5  # Bu gerçek sonuçlarla güncellenecek
        }
        
        # History'e ekle
        self.optimization_history.append(performance_entry)
        
        # Son 100 entry'yi tut
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        # Performance tracking
        context_key = self._generate_context_key(match_context)
        if context_key not in self.performance_tracking:
            self.performance_tracking[context_key] = {
                'count': 0,
                'total_time': 0.0,
                'methods': defaultdict(int)
            }
        
        self.performance_tracking[context_key]['count'] += 1
        self.performance_tracking[context_key]['total_time'] += optimization_time
        self.performance_tracking[context_key]['methods'][method] += 1
        
        # Log performance statistics
        if self.performance_tracking[context_key]['count'] % 10 == 0:
            self._log_performance_statistics(context_key)
    
    def _log_performance_statistics(self, context_key: str):
        """Performance statistics'leri logla"""
        stats = self.performance_tracking.get(context_key, {})
        
        if stats.get('count', 0) > 0:
            avg_time = stats['total_time'] / stats['count']
            methods = dict(stats['methods'])
            
            logger.info(f"📊 Performance Stats [{context_key}]:")
            logger.info(f"   Toplam optimizasyon: {stats['count']}")
            logger.info(f"   Ortalama süre: {avg_time:.3f}s")
            logger.info(f"   Kullanılan metodlar: {methods}")
    
    def get_optimization_analysis(self) -> Dict:
        """
        Optimization analizi ve istatistikleri
        
        Returns:
            Dict: Detaylı analiz raporu
        """
        if not self.optimization_history:
            return {"error": "Henüz optimization geçmişi yok"}
        
        # Method distribution
        method_counts = defaultdict(int)
        total_time = 0.0
        
        for entry in self.optimization_history:
            method_counts[entry['method']] += 1
            total_time += entry['optimization_time']
        
        # Context analysis
        context_performance = defaultdict(list)
        for entry in self.optimization_history:
            context_key = f"{entry['context']['league']}_{entry['context']['match_type']}"
            context_performance[context_key].append(entry['optimization_time'])
        
        analysis = {
            'total_optimizations': len(self.optimization_history),
            'method_distribution': dict(method_counts),
            'average_optimization_time': total_time / len(self.optimization_history),
            'context_performance': {
                context: {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
                for context, times in context_performance.items()
            },
            'genetic_optimization_usage': method_counts.get('🧬 Genetik Algoritma', 0),
            'cache_efficiency': len(self.optimization_cache),
            'recent_performance_trend': self._analyze_recent_trend()
        }
        
        return analysis
    
    def _analyze_recent_trend(self) -> str:
        """Son zamanlardaki trend analizi"""
        if len(self.optimization_history) < 10:
            return "Yetersiz veri"
        
        recent_times = [entry['optimization_time'] for entry in self.optimization_history[-10:]]
        older_times = [entry['optimization_time'] for entry in self.optimization_history[-20:-10]]
        
        if not older_times:
            return "Karşılaştırma için yetersiz veri"
        
        recent_avg = sum(recent_times) / len(recent_times)
        older_avg = sum(older_times) / len(older_times)
        
        if recent_avg < older_avg * 0.9:
            return "İyileşme"
        elif recent_avg > older_avg * 1.1:
            return "Yavaşlama"
        else:
            return "Stabil"
    
    def force_genetic_optimization(self, match_context: Dict) -> Dict[str, float]:
        """
        Genetic optimization'ı zorla çalıştır (test/debug amaçlı)
        
        Args:
            match_context: Maç bağlamı
            
        Returns:
            Dict[str, float]: Optimize edilmiş ağırlıklar
        """
        if not self.use_genetic_optimization:
            raise Exception("Genetic optimization aktif değil")
        
        logger.info("🧬 ZORLANMIŞ Genetic optimization başlatılıyor...")
        
        return self._get_genetic_optimized_weights(match_context)
    
    def update_optimization_performance(self, 
                                      match_context: Dict, 
                                      predicted_outcome: str, 
                                      actual_outcome: str, 
                                      accuracy: float):
        """
        Gerçek sonuçlarla optimization performansını güncelle
        
        Args:
            match_context: Maç bağlamı
            predicted_outcome: Tahmin edilen sonuç
            actual_outcome: Gerçek sonuç
            accuracy: Doğruluk oranı
        """
        context_key = self._generate_context_key(match_context)
        
        # Son optimization entry'yi bul ve güncelle
        for entry in reversed(self.optimization_history):
            entry_context_key = self._generate_context_key(entry['context'])
            if entry_context_key == context_key:
                entry['performance'] = accuracy
                entry['predicted_outcome'] = predicted_outcome
                entry['actual_outcome'] = actual_outcome
                entry['accuracy'] = accuracy
                break
        
        # Genetic optimizer'ın meta-learning'ine bildir
        if (self.genetic_optimizer and 
            hasattr(self.genetic_optimizer, 'meta_learner') and
            self.genetic_optimizer.meta_learner):
            
            try:
                self.genetic_optimizer.meta_learner.learn_from_optimization(
                    context=match_context,
                    optimization_result=entry.get('weights', {}),
                    performance=accuracy
                )
            except Exception as e:
                logger.warning(f"Meta-learning güncelleme hatası: {e}")
        
        logger.debug(f"Optimization performansı güncellendi: {accuracy:.3f} ({context_key})")
        
    def _combine_score_predictions(self, model_predictions, weights):
        """
        Farklı modellerin skor tahminlerini birleştir
        """
        combined_scores = {}
        
        # Her modelin skor tahminlerini topla
        for model_name, predictions in model_predictions.items():
            if model_name not in weights or 'score_probabilities' not in predictions:
                continue
                
            weight = weights[model_name]
            
            for score_pred in predictions['score_probabilities']:
                score = score_pred['score']
                prob = score_pred['probability'] * weight
                
                if score in combined_scores:
                    combined_scores[score] += prob
                else:
                    combined_scores[score] = prob
                    
        # En olası 5 skoru sırala
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [{'score': s[0], 'probability': s[1]} for s in sorted_scores]
    
    def record_prediction_feedback(self, prediction_result: Dict, actual_result: Dict, match_context: Dict):
        """
        Tahmin sonuçlarını meta-learning layer'a kaydet
        
        Args:
            prediction_result: Ensemble tahmin sonucu
            actual_result: Gerçek maç sonucu
            match_context: Maç bağlamı
        """
        try:
            if not self.use_meta_learning or not self.meta_learning_layer:
                logger.debug("Meta-learning aktif değil, feedback kaydedilmedi")
                return
            
            # Extract model predictions from prediction context
            prediction_context = prediction_result.get('_prediction_context', {})
            model_predictions = prediction_context.get('model_predictions', {})
            
            if not model_predictions:
                logger.warning("Model predictions not found in prediction context")
                return
            
            # Record feedback for meta-learning
            self.meta_learning_layer.record_prediction_feedback(
                model_predictions=model_predictions,
                actual_result=actual_result,
                match_context=match_context,
                ensemble_result=prediction_result
            )
            
            logger.info(f"🧠 Meta-learning feedback kaydedildi: {len(model_predictions)} model")
            
        except Exception as e:
            logger.error(f"Meta-learning feedback kaydetme hatası: {e}")
    
    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """
        Meta-learning sisteminden öğrenme içgörülerini al
        
        Returns:
            Dict: Learning insights and recommendations
        """
        try:
            if not self.use_meta_learning or not self.meta_learning_layer:
                return {'error': 'Meta-learning aktif değil'}
            
            insights = self.meta_learning_layer.get_learning_insights()
            
            # Add ensemble-specific insights
            ensemble_insights = {
                'ensemble_integration': {
                    'meta_learning_enabled': self.use_meta_learning,
                    'genetic_optimization_enabled': self.use_genetic_optimization,
                    'dynamic_weights_enabled': self.use_dynamic_weights,
                    'optimization_cache_size': len(self.optimization_cache),
                    'optimization_history_size': len(self.optimization_history)
                }
            }
            
            insights['ensemble_integration'] = ensemble_insights['ensemble_integration']
            
            return insights
            
        except Exception as e:
            logger.error(f"Meta-learning insights alma hatası: {e}")
            return {'error': str(e)}
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        Meta-learning sisteminden model performans özetini al
        
        Returns:
            Dict: Model performance summary
        """
        try:
            if not self.use_meta_learning or not self.meta_learning_layer:
                return {'error': 'Meta-learning aktif değil'}
            
            insights = self.meta_learning_layer.get_learning_insights()
            performance_summary = insights.get('performance_summary', {})
            
            # Add ranking information
            model_rankings = insights.get('model_rankings', {})
            
            # Combine performance and ranking data
            summary = {
                'model_performance': performance_summary,
                'context_rankings': model_rankings,
                'learning_stats': insights.get('meta_learning_stats', {}),
                'error_insights': insights.get('error_patterns', {}),
                'recommendations': insights.get('learning_recommendations', [])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Model performans özeti alma hatası: {e}")
            return {'error': str(e)}
    
    def force_meta_learning_adaptation(self, context: str = "manual_trigger"):
        """
        Meta-learning sistemini manuel olarak adapte et
        
        Args:
            context: Adaptation tetikleyici bağlamı
        """
        try:
            if not self.use_meta_learning or not self.meta_learning_layer:
                logger.warning("Meta-learning aktif değil, adaptation yapılamadı")
                return False
            
            # Check for concept drift
            self.meta_learning_layer._check_concept_drift()
            
            # Update model rankings
            self.meta_learning_layer._update_model_rankings()
            
            # Save updated state
            self.meta_learning_layer._save_persistent_data()
            
            logger.info(f"🔄 Meta-learning manuel adaptation tamamlandı: {context}")
            return True
            
        except Exception as e:
            logger.error(f"Meta-learning adaptation hatası: {e}")
            return False
    
    def get_optimal_models_for_context(self, match_context: Dict) -> List[Tuple[str, float]]:
        """
        Belirli bir bağlam için optimal modelleri al
        
        Args:
            match_context: Maç bağlamı
            
        Returns:
            List[Tuple[str, float]]: (model_name, expected_performance) listesi
        """
        try:
            if not self.use_meta_learning or not self.meta_learning_layer:
                logger.warning("Meta-learning aktif değil, fallback ağırlıklar kullanılıyor")
                return [(model, weight) for model, weight in self.weights.items()]
            
            # Available models
            available_models = list(self.weights.keys())
            
            # Get optimal models
            optimal_models = self.meta_learning_layer.select_optimal_models(match_context, available_models)
            
            return optimal_models
            
        except Exception as e:
            logger.error(f"Optimal model seçimi hatası: {e}")
            return [(model, weight) for model, weight in self.weights.items()]    
    def _apply_league_strength_adjustment(self, match_context: Dict, combined: Dict, model_predictions: Dict) -> Optional[Dict]:
        """
        Cross-league maçlar için lig gücü adjustment'ı uygula
        
        Örnek: Galatasaray (Süper Lig) vs Liverpool (Premier League)
        - Galatasaray'ın domestic form'u daha düşük lig gücü nedeniyle ayarlanır
        - Liverpool'un form'u daha güçlü lig nedeniyle değerli sayılır
        
        Args:
            match_context: Maç bağlamı (league, home_league, away_league vb.)
            combined: Birleştirilmiş tahminler (düzenlenecek)
            model_predictions: Model tahminleri
            
        Returns:
            League strength bilgileri
        """
        if not self.use_league_strength_adjustment or not self.league_strength_analyzer:
            return None
        
        try:
            # CRITICAL: Check cross_league flag first
            if not match_context.get('cross_league', False):
                # Same league, no adjustment needed
                return None
            
            # Get league strength context (already computed in predict_match)
            league_strength_context = match_context.get('league_strength_context')
            if not league_strength_context:
                logger.warning("Cross-league flag set but no league_strength_context found!")
                return None
            
            # Extract domestic league info
            home_info = league_strength_context.get('home', {})
            away_info = league_strength_context.get('away', {})
            
            home_league = home_info.get('league_name', 'Unknown')
            away_league = away_info.get('league_name', 'Unknown')
            home_strength = home_info.get('strength_score', 50)
            away_strength = away_info.get('strength_score', 50)
            
            # Get multipliers from league strength
            home_multiplier = self.league_strength_analyzer.get_league_multiplier(home_league)
            away_multiplier = self.league_strength_analyzer.get_league_multiplier(away_league)
            
            # Calculate strength difference
            strength_diff = abs(home_strength - away_strength)
            
            if strength_diff < 10:
                # Çok küçük fark, adjustment gereksiz
                return None
            
            # Get UEFA competition flag from league_strength_context
            is_uefa_competition = league_strength_context.get('is_uefa_competition', False)
            uefa_adjustment_factor = league_strength_context.get('uefa_adjustment_factor', 0.5)
            
            if is_uefa_competition:
                competition_league_id = match_context.get('competition_league_id')
                uefa_name = "ŞAMPIYONLAR LİGİ" if competition_league_id == 3 else \
                           "AVRUPA LİGİ" if competition_league_id == 4 else "CONFERENCE LİGİ"
                logger.info(f"🏆 {uefa_name} (ID: {competition_league_id}) - Ultra agresif cross-league adjustment (120%) aktif!")
            
            logger.info(f"🌍 CROSS-LEAGUE ADJUSTMENT: {home_league} ({home_strength}) vs {away_league} ({away_strength})")
            logger.info(f"   Strength gap: {strength_diff:.1f} points, UEFA factor: {uefa_adjustment_factor}")
            
            # Güç oranını hesapla - GÜÇLÜ AYARLAMA
            strength_ratio = max(home_strength, away_strength) / min(home_strength, away_strength)
            
            # Lig farkına göre adjustment gücü (çok büyük farklar için daha agresif)
            # UEFA KOMPETİSYONLARI İÇİN 2-3X DAHA AGRESİF!
            if is_uefa_competition:
                # UEFA KOMPETİSYONLARI - ULTRA AGRESİF AYARLAMA
                if strength_diff > 40:  # Elite vs Medium (örn: Premier League vs Süper Lig)
                    base_adjustment = 120  # %120 max adjustment (3x normal)
                elif strength_diff > 25:  # Strong vs Lower
                    base_adjustment = 85   # 2.5x normal
                elif strength_diff > 15:  # Medium vs Amateur
                    base_adjustment = 60   # 2.4x normal
                else:
                    base_adjustment = 40   # 2.6x normal
            else:
                # NORMAL LİG MAÇLARI - complement home advantage reduction
                if strength_diff > 40:  # Elite vs Medium (örn: Premier League vs Süper Lig)
                    base_adjustment = 70  # %70 max adjustment (increased to work with 70% home reduction)
                elif strength_diff > 25:  # Strong vs Lower
                    base_adjustment = 50
                elif strength_diff > 15:  # Medium vs Amateur
                    base_adjustment = 35
                else:
                    base_adjustment = 20
            
            if home_strength > away_strength:
                # Ev sahibi daha güçlü ligden
                # Deplasman takımının domestic form'u aslında daha az değerli
                adjustment_factor = away_multiplier / home_multiplier
                
                # Away team'in kazanma şansını azalt, home'u artır
                away_penalty = (1.0 - adjustment_factor) * base_adjustment
                home_boost = away_penalty * 0.7  # Ev sahibine %70'ini ver
                
                logger.info(f"   → Ev sahibi lehine düzeltme (Güç farkı: {strength_diff:.1f}): Home +{home_boost:.1f}%, Away -{away_penalty:.1f}%")
                
                combined['home_win'] = min(100, combined['home_win'] + home_boost)
                combined['away_win'] = max(0, combined['away_win'] - away_penalty)
                
            else:
                # Deplasman takımı daha güçlü ligden (örn: Liverpool)
                # Ev sahibinin domestic form'u daha az değerli
                adjustment_factor = home_multiplier / away_multiplier
                
                # Home team'in kazanma şansını azalt, away'i artır
                home_penalty = (1.0 - adjustment_factor) * base_adjustment
                away_boost = home_penalty * 0.7  # Deplasman'a %70'ini ver
                
                logger.info(f"   → Deplasman lehine düzeltme (Güç farkı: {strength_diff:.1f}): Away +{away_boost:.1f}%, Home -{home_penalty:.1f}%")
                
                combined['away_win'] = min(100, combined['away_win'] + away_boost)
                combined['home_win'] = max(0, combined['home_win'] - home_penalty)
            
            # Normalize et (toplam 100 olsun)
            total_prob = combined['home_win'] + combined['draw'] + combined['away_win']
            if total_prob > 0:
                combined['home_win'] = (combined['home_win'] / total_prob) * 100
                combined['draw'] = (combined['draw'] / total_prob) * 100
                combined['away_win'] = (combined['away_win'] / total_prob) * 100
            
            # League strength bilgisini kaydet
            league_info = {
                'home_league': home_league,
                'away_league': away_league,
                'home_strength': home_strength,
                'away_strength': away_strength,
                'adjustment_applied': True,
                'strength_difference': strength_diff
            }
            
            # Match context'e ekle (daha sonra frontend'de gösterebiliriz)
            match_context['league_strength_info'] = league_info
            
            return league_info
            
        except Exception as e:
            logger.error(f"League strength adjustment hatası: {e}")
            return None
