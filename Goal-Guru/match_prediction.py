import logging
import json
import os
import math
from datetime import datetime
import requests
import time
import numpy as np

# Algoritmalar
from algorithms import (
    XGCalculator,
    EloSystem,
    HybridMLSystem,
    PoissonModel,
    DixonColesModel,
    XGBoostModel,
    MonteCarloSimulator,
    EnsemblePredictor,
    CRFPredictor,
    SelfLearningModel,
    PsychologicalProfiler
)

# Yeni tahmin algoritmaları
from algorithms.halftime_predictor import HalfTimeFullTimePredictor
from algorithms.handicap_predictor import HandicapPredictor
from algorithms.goal_range_predictor import GoalRangePredictor
from algorithms.double_chance_predictor import DoubleChancePredictor
from algorithms.team_goals_predictor import TeamGoalsPredictor

# Yeni geliştirme modülleri
from model_evaluator import ModelEvaluator
from continuous_learner import ContinuousLearner
from advanced_features import AdvancedFeatureEngineer
from distributed_trainer import DistributedTrainer
from model_validator import ComprehensiveValidator
from explainable_ai import PredictionExplainer
from performance_optimizer import (
    prediction_cache, performance_monitor, 
    batch_processor, query_optimizer
)
from async_data_fetcher import AsyncDataFetcher
from dynamic_team_analyzer import DynamicTeamAnalyzer

# Phase 3 modülleri
from algorithms.form_trend_analyzer import FormTrendAnalyzer
from algorithms.feature_engineering import FeatureEngineer
from algorithms.league_strength_analyzer import LeagueStrengthAnalyzer
from algorithms.momentum_shift_detector import MomentumShiftDetector
from algorithms.seasonal_performance_analyzer import SeasonalPerformanceAnalyzer

# Yeni Feature Extraction Pipeline modülleri
from algorithms.feature_extraction_pipeline import FeatureExtractionPipeline
from algorithms.team_characteristics import TeamCharacteristicsAnalyzer
from algorithms.league_context_analyzer import LeagueContextAnalyzer
from algorithms.league_normalization_engine import LeagueNormalizationEngine

# Advanced Analysis Systems
from algorithms.dynamic_time_analyzer import DynamicTimeAnalyzer

# API config
from api_config import APIConfig

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# League ID helper functions
def load_league_ids():
    """Load league ID mappings from config"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'league_ids.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load league IDs: {e}")
        return {}

class MatchPredictor:
    """
    Gelişmiş futbol maç tahmin sistemi
    Çoklu algoritma ve ensemble yaklaşımı
    """
    
    def __init__(self):
        """Tahmin sınıfını ve algoritmalarını başlat"""
        
        # Load league ID mappings
        self.league_ids = self._load_league_ids()
        logger.info("MatchPredictor gelişmiş sürüm başlatılıyor...")
        
        # API anahtarını al
        api_config = APIConfig()
        self.api_key = api_config.current_api_key
        
        # Algoritmaları başlat
        self.xg_calculator = XGCalculator()
        self.hybrid_ml_system = HybridMLSystem()
        self.poisson_model = PoissonModel()
        self.dixon_coles = DixonColesModel()
        self.xgboost_model = XGBoostModel()
        
        # Feature Extraction Pipeline ve analizörler
        self.feature_pipeline = FeatureExtractionPipeline()
        self.team_analyzer = TeamCharacteristicsAnalyzer()
        self.league_analyzer = LeagueContextAnalyzer()
        self.monte_carlo = MonteCarloSimulator()
        self.ensemble = EnsemblePredictor()
        self.crf_predictor = CRFPredictor()
        self.self_learning = SelfLearningModel()
        
        # Neural Network modelini ekle
        from algorithms.neural_network import NeuralNetworkModel
        self.neural_network = NeuralNetworkModel()
        
        # Yeni tahmin algoritmaları
        self.htft_predictor = HalfTimeFullTimePredictor()
        self.handicap_predictor = HandicapPredictor()
        self.goal_range_predictor = GoalRangePredictor()
        self.double_chance_predictor = DoubleChancePredictor()
        self.team_goals_predictor = TeamGoalsPredictor()
        
        # Geliştirme modülleri
        self.model_evaluator = ModelEvaluator()
        self.continuous_learner = ContinuousLearner()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.distributed_trainer = DistributedTrainer()
        self.model_validator = ComprehensiveValidator()
        self.prediction_explainer = PredictionExplainer()
        self.async_fetcher = AsyncDataFetcher()
        self.dynamic_team_analyzer = DynamicTeamAnalyzer()
        
        # Phase 3 modülleri
        self.form_trend_analyzer = FormTrendAnalyzer()
        self.enhanced_feature_engineer = FeatureEngineer()
        self.league_strength_analyzer = LeagueStrengthAnalyzer()
        
        # Momentum Shift Detector - Advanced momentum analysis
        self.momentum_shift_detector = MomentumShiftDetector()
        
        # League Normalization Engine
        self.league_normalization_engine = LeagueNormalizationEngine()
        
        # Seasonal Performance Analyzer - Comprehensive seasonal analysis
        self.seasonal_performance_analyzer = SeasonalPerformanceAnalyzer()
        
        # Dynamic Time Analyzer - Time-weighted features
        self.dynamic_time_analyzer = DynamicTimeAnalyzer()
        
        # Fixture Congestion Analyzer
        from algorithms.fixture_congestion_analyzer import FixtureCongestionAnalyzer
        self.fixture_congestion_analyzer = FixtureCongestionAnalyzer()
        
        # Venue Performance Optimizer
        from algorithms.venue_performance_optimizer import VenuePerformanceOptimizer
        self.venue_performance_optimizer = VenuePerformanceOptimizer()
        
        # Psychological Profiler
        self.psychological_profiler = PsychologicalProfiler()
        
        # Meta-Learning Layer ve Prediction Confidence System
        try:
            from algorithms.meta_learning_layer import MetaLearningLayer
            from algorithms.prediction_confidence_system import PredictionConfidenceSystem
            self.meta_learning_layer = MetaLearningLayer()
            self.prediction_confidence_system = PredictionConfidenceSystem()
            logger.info("Meta-Learning Layer ve Prediction Confidence System yüklendi")
        except Exception as e:
            logger.warning(f"Meta-Learning/Confidence sistemi yüklenemedi: {e}")
            self.meta_learning_layer = None
            self.prediction_confidence_system = None
        
        # Tek JSON dosyası kullan
        self.cache_file = 'predictions_cache.json'
        self.cache_data = self._load_cache()
            
        logger.info("Tüm algoritmalar ve geliştirme modülleri başlatıldı")
        
    def _load_cache(self):
        """Önbellek dosyasını yükle"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
        
    def predict_match(self, home_team_id, away_team_id, home_name="Ev Sahibi", away_name="Deplasman", force_update=False):
        """
        Gelişmiş maç tahmini - tüm algoritmaları kullanır
        
        Args:
            home_team_id: Ev sahibi takım ID
            away_team_id: Deplasman takım ID
            home_name: Ev sahibi takım adı
            away_name: Deplasman takım adı
            force_update: Önbelleği yoksay
            
        Returns:
            dict: Tahmin sonuçları
        """
        start_time = time.time()
        logger.info(f"Tahmin başlatılıyor: {home_name} vs {away_name}")
        
        # Performans optimizasyonu - Önbellek kontrolü
        cache_key = f"{home_team_id}_{away_team_id}"
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        if not force_update:
            # Gelişmiş önbellek kontrolü
            cached = prediction_cache.get_prediction(home_team_id, away_team_id, date_str)
            if cached:
                performance_monitor.record_cache_access(hit=True)
                logger.info("Önbellekten tahmin döndürülüyor")
                return cached
            performance_monitor.record_cache_access(hit=False)
                
        try:
            # 1. ÖNCE LİG CONTEXT'İNİ BELİRLE (UEFA Competition mu? - League ID bazlı)
            # Takım verilerini geçici olarak al - lig bilgisi için
            temp_home_data = self._get_team_data(home_team_id, home_name, is_home=True)
            temp_away_data = self._get_team_data(away_team_id, away_name, is_home=False)
            
            # Takımların son maçlarından competition context'i belirle (LEAGUE ID bazlı)
            is_uefa_competition = False
            competition_name = ''
            competition_league_id = None
            
            # Her iki takımın da son maçlarına bak
            all_recent_matches = temp_home_data.get('recent_matches', []) + temp_away_data.get('recent_matches', [])
            
            for match in all_recent_matches[:10]:  # Son 10 maçı kontrol et
                league = match.get('league', '') or match.get('league_name', '')
                league_id = match.get('league_id')
                
                # LEAGUE ID bazlı kontrol (çok daha güvenilir!)
                if league_id and self._is_uefa_competition(league_id):
                    is_uefa_competition = True
                    competition_name = league
                    competition_league_id = league_id
                    break
            
            if is_uefa_competition:
                uefa_type = "ŞAMPIYONLAR LİGİ" if competition_league_id == 3 else \
                           "UEFA AVRUPA LİGİ" if competition_league_id == 4 else \
                           "UEFA CONFERENCE LİGİ" if competition_league_id == 683 else "UEFA"
                logger.info(f"🏆 {uefa_type} MAÇI TESPİT EDİLDİ (League ID: {competition_league_id}): {competition_name}")
                logger.info(f"   → UEFA maçlarına %90 ağırlık verilecek, ulusal lig verisi minimize edilecek")
            
            # 1. Takım verilerini al (şimdi UEFA context'i ile)
            home_data = self._get_team_data(home_team_id, home_name, is_home=True, 
                                           champions_league_context=is_uefa_competition,
                                           uefa_league_id=competition_league_id)
            away_data = self._get_team_data(away_team_id, away_name, is_home=False,
                                          champions_league_context=is_uefa_competition,
                                          uefa_league_id=competition_league_id)
            
            # 1.2. Form Trend Analysis (Phase 3.1)
            home_form_analysis = self.form_trend_analyzer.analyze_team_form(
                home_data.get('recent_matches', []), 
                int(home_team_id)
            )
            away_form_analysis = self.form_trend_analyzer.analyze_team_form(
                away_data.get('recent_matches', []), 
                int(away_team_id)
            )
            form_comparison = self.form_trend_analyzer.compare_team_forms(home_form_analysis, away_form_analysis)
            
            # Form analizini takım verilerine ekle
            home_data['form_analysis'] = home_form_analysis
            away_data['form_analysis'] = away_form_analysis
            home_data['form_score'] = home_form_analysis['overall_form_score']
            away_data['form_score'] = away_form_analysis['overall_form_score']
            
            # 1.4. Get DOMESTIC league information from team_data (already extracted in _get_team_data)
            home_league = home_data.get('domestic_league_name', '')
            away_league = away_data.get('domestic_league_name', '')
            home_league_id = home_data.get('domestic_league_id')
            away_league_id = away_data.get('domestic_league_id')
            
            logger.info(f"🏟️  Domestic Lig bilgileri - Ev: {home_league} (ID: {home_league_id}), Deplasman: {away_league} (ID: {away_league_id})")
            
            # CRITICAL: Detect cross-league match
            is_cross_league = False
            if home_league_id and away_league_id and home_league_id != away_league_id:
                is_cross_league = True
                home_strength = self._get_league_strength_score(home_league_id)
                away_strength = self._get_league_strength_score(away_league_id)
                strength_gap = abs(home_strength - away_strength)
                logger.info(f"🔀 CROSS-LEAGUE MATCH DETECTED! {home_name} vs {away_name}")
                logger.info(f"   {home_league} (strength: {home_strength}) vs {away_league} (strength: {away_strength})")
                logger.info(f"   Strength gap: {strength_gap} points")
                if is_uefa_competition:
                    logger.info(f"   ⚡ UEFA Context: 120% ultra-aggressive adjustment will be applied")
                else:
                    logger.info(f"   → Normal context: 50% standard adjustment will be applied")
            else:
                logger.info(f"✓ Same league match or missing league IDs - no cross-league adjustment")
            
            # Get competition name from league_name (API sends it as league_name, not competition_name)
            competition_name = home_data.get('league', home_data.get('league_name', ''))
            logger.info(f"🔍 DEBUG: competition_name from home_data: '{competition_name}'")
            
            logger.info(f"Lig bilgileri - Ev: {home_league}, Deplasman: {away_league}")
            
            # 1.4. Fixture Congestion Analysis
            logger.info("Fixture Congestion Analysis başlatılıyor...")
            home_congestion_analysis = self.fixture_congestion_analyzer.analyze_fixture_congestion(
                int(home_team_id), 
                home_data.get('recent_matches', []),
                upcoming_match_date=datetime.now(),
                league_id=str(home_league.get('id', '')) if isinstance(home_league, dict) else (str(home_league) if home_league else None)
            )
            away_congestion_analysis = self.fixture_congestion_analyzer.analyze_fixture_congestion(
                int(away_team_id), 
                away_data.get('recent_matches', []),
                upcoming_match_date=datetime.now(),
                league_id=str(away_league.get('id', '')) if isinstance(away_league, dict) else (str(away_league) if away_league else None)
            )
            
            # Fatigue comparison between teams
            fatigue_comparison = self.fixture_congestion_analyzer.compare_team_fatigue(
                home_congestion_analysis, away_congestion_analysis
            )
            
            # Add fatigue data to team data
            home_data['congestion_analysis'] = home_congestion_analysis
            away_data['congestion_analysis'] = away_congestion_analysis
            home_data['fatigue_score'] = home_congestion_analysis.get('fatigue_score', {}).get('overall_fatigue_score', 50)
            away_data['fatigue_score'] = away_congestion_analysis.get('fatigue_score', {}).get('overall_fatigue_score', 50)
            
            logger.info(f"Fatigue Scores - Home: {home_data['fatigue_score']:.1f}, Away: {away_data['fatigue_score']:.1f}")
            logger.info(f"Fatigue Advantage: {fatigue_comparison.get('advantage', 'balanced')}")
            
            # 1.3. Dynamic Team Analyzer ile takım analizleri
            home_team_analysis = None
            away_team_analysis = None
            team_comparison = None
            
            try:
                # Takım bilgilerini hazırla
                home_team_info = {
                    'position': home_data.get('league_position', 10),
                    'recent_form': home_data.get('recent_form', 'DDDDD'),
                    'matches_played': len(home_data.get('recent_matches', [])),
                    'total_matches': 38  # Varsayılan
                }
                
                away_team_info = {
                    'position': away_data.get('league_position', 10),
                    'recent_form': away_data.get('recent_form', 'DDDDD'),
                    'matches_played': len(away_data.get('recent_matches', [])),
                    'total_matches': 38  # Varsayılan
                }
                
                # Takım analizlerini yap
                home_team_analysis = self.dynamic_team_analyzer.analyze_team(
                    team_id=home_team_id,
                    team_matches=home_data.get('recent_matches', []),
                    team_info=home_team_info,
                    is_home=True
                )
                
                away_team_analysis = self.dynamic_team_analyzer.analyze_team(
                    team_id=away_team_id,
                    team_matches=away_data.get('recent_matches', []),
                    team_info=away_team_info,
                    is_home=False
                )
                
                # Takımları karşılaştır
                team_comparison = self.dynamic_team_analyzer.compare_teams(
                    home_team_analysis,
                    away_team_analysis
                )
                
                logger.info(f"Dynamic Team Analyzer tamamlandı - Ev: {home_team_analysis['overall_score']}, Dep: {away_team_analysis['overall_score']}")
                logger.info(f"Momentum avantajı: {team_comparison['momentum_advantage']}")
                
            except Exception as e:
                logger.warning(f"Dynamic Team Analyzer hatası: {e}")
            
            # 1.3. Psychological Profiler Analysis (Enhanced)
            psychological_analysis = None
            try:
                # Maç bağlamını hazırla
                match_context = {
                    'league': None,  # league_data tanımlı değil, None kullan
                    'league_table': None,  # league_table tanımlı değil, None kullan
                    'h2h_data': h2h_data if 'h2h_data' in locals() else None,
                    'home_team': home_name,
                    'away_team': away_name,
                    'competition': 'League',  # Bu bilgiyi API'den alabilirsiniz
                    'round': 'Regular Season',  # Bu bilgiyi API'den alabilirsiniz
                    'date': datetime.now()
                }
                
                # Psikolojik profil analizi
                psychological_analysis = self.psychological_profiler.analyze_psychological_profile(
                    home_data, away_data, match_context
                )
                
                # Psikolojik analiz sonuçlarını logla
                logger.info(f"Psikolojik Analiz tamamlandı:")
                logger.info(f"  Ev takımı motivasyon: {psychological_analysis['motivation_analysis']['home_team']['total_motivation']}")
                logger.info(f"  Deplasman takımı motivasyon: {psychological_analysis['motivation_analysis']['away_team']['total_motivation']}")
                logger.info(f"  Maç önem skoru: {psychological_analysis['match_importance_score']:.1f}/10")
                logger.info(f"  Psikolojik avantaj: {psychological_analysis['psychological_advantage']}")
                
                # Kritik maç tespiti
                if psychological_analysis['critical_match_analysis']['is_critical_match']:
                    critical_types = ', '.join(psychological_analysis['critical_match_analysis']['critical_types'])
                    logger.info(f"  KRİTİK MAÇ: {critical_types}")
                
            except Exception as e:
                logger.warning(f"Psychological Profiler hatası: {e}")
                psychological_analysis = None
            
            # 1.6. Venue Performance Optimizer Analysis (New)
            venue_analysis = None
            try:
                logger.info("Venue Performance Optimizer başlatılıyor...")
                
                # Venue bilgilerini hazırla
                venue_info = self._prepare_venue_info(home_data, away_data, home_league)
                
                # Match context hazırla
                match_context = {
                    'date': datetime.now(),
                    'time': '15:00',  # Default time
                    'season': '2024-25',
                    'competition': competition_name or 'League'
                }
                
                # Historical matches combine et
                historical_matches = home_data.get('recent_matches', []) + away_data.get('recent_matches', [])
                
                # Venue analizi yap
                venue_analysis = self.venue_performance_optimizer.analyze_comprehensive_venue_performance(
                    home_team_id=int(home_team_id),
                    away_team_id=int(away_team_id),
                    venue_info=venue_info,
                    match_context=match_context,
                    historical_matches=historical_matches
                )
                
                # Venue analiz sonuçlarını logla
                logger.info(f"Venue Performance Analizi tamamlandı:")
                logger.info(f"  Ev sahibi avantaj katsayısı: {venue_analysis['home_advantage_analysis']['final_coefficient']:.3f}")
                logger.info(f"  Venue zorluk skoru: {venue_analysis['venue_difficulty_score']}/100")
                logger.info(f"  Seyahat etkisi: {venue_analysis['travel_impact_assessment']['overall_travel_penalty']:.3f}")
                logger.info(f"  Home team boost: {venue_analysis['performance_predictions']['home_team_boost']:.3f}")
                logger.info(f"  Away team penalty: {venue_analysis['performance_predictions']['away_team_penalty']:.3f}")
                
            except Exception as e:
                logger.warning(f"Venue Performance Optimizer hatası: {e}")
                venue_analysis = None
            
            # 1.5. H2H verilerini al
            h2h_data = None
            try:
                # API anahtarını al
                from api_config import APIConfig
                api_config = APIConfig()
                api_key = api_config.get_api_key()
                
                # Asenkron veri çekme
                import asyncio
                async def fetch_h2h():
                    async with self.async_fetcher as fetcher:
                        return await fetcher.fetch_h2h_data(home_team_id, away_team_id, api_key, home_name, away_name)
                
                # H2H verilerini çek
                h2h_data = asyncio.run(fetch_h2h())
                logger.info(f"H2H verileri başarıyla alındı: {home_name} vs {away_name}")
                # H2H veri yapısını logla
                if h2h_data:
                    logger.info(f"H2H veri yapısı anahtarları: {list(h2h_data.keys())[:5]}")
                    if isinstance(h2h_data, dict) and 'firstTeam_VS_secondTeam' in h2h_data:
                        logger.info(f"H2H maç sayısı: {len(h2h_data['firstTeam_VS_secondTeam'])}")
                    elif isinstance(h2h_data, list):
                        logger.info(f"H2H doğrudan liste, maç sayısı: {len(h2h_data)}")
            except Exception as e:
                logger.warning(f"H2H verileri alınamadı: {e}")
                h2h_data = None
            
            # 2. Hybrid ML rating hesapla
            home_rating = self.hybrid_ml_system.get_team_rating(
                home_team_id, home_data.get('recent_matches', [])
            )
            away_rating = self.hybrid_ml_system.get_team_rating(
                away_team_id, away_data.get('recent_matches', [])
            )
            # Combined rating'i kullan (Elo, Glicko-2 ve TrueSkill ortalaması)
            home_elo = home_rating.get('combined_rating', 1500)
            away_elo = away_rating.get('combined_rating', 1500)
            elo_diff = home_elo - away_elo
            
            # League info already extracted above (line 205-206)
            
            # 2.8. Feature Extraction Pipeline - Takım özelliklerini çıkar
            logger.info("Feature Extraction Pipeline başlatılıyor...")
            
            # Ev sahibi takım özellikleri
            home_features = self.feature_pipeline.extract_features(home_data, is_home=True)
            logger.info(f"Ev sahibi özellikleri çıkarıldı - Veri kalitesi: {home_features['feature_quality_score']:.2f}")
            
            # Deplasman takımı özellikleri
            away_features = self.feature_pipeline.extract_features(away_data, is_home=False)
            logger.info(f"Deplasman özellikleri çıkarıldı - Veri kalitesi: {away_features['feature_quality_score']:.2f}")
            
            # Takım karakteristik analizi
            home_style = self.team_analyzer.analyze_team_style(
                home_features['enriched_features'], 
                away_features['enriched_features']
            )
            away_style = self.team_analyzer.analyze_team_style(
                away_features['enriched_features'],
                home_features['enriched_features']
            )
            
            logger.info(f"Ev sahibi stili: {home_style['style_summary']}")
            logger.info(f"Deplasman stili: {away_style['style_summary']}")
            
            # 3. xG/xGA hesapla - Elo entegrasyonu ile (rapordaki öneri)
            home_xg, home_xga = self.xg_calculator.calculate_xg_xga_with_elo(
                home_data.get('recent_matches', []), 
                home_elo, 
                away_elo,
                is_home=True
            )
            away_xg, away_xga = self.xg_calculator.calculate_xg_xga_with_elo(
                away_data.get('recent_matches', []),
                away_elo,
                home_elo, 
                is_home=False
            )
            
            # 3.4. Apply venue effects to xG calculations (if venue analysis available)
            if venue_analysis:
                home_xg, away_xg = self._apply_venue_effects_to_xg(home_xg, away_xg, venue_analysis)
                home_xga, away_xga = self._apply_venue_effects_to_xg(home_xga, away_xga, venue_analysis)
            
            # 3.5. Lig farkı analizini uygula
            league_analysis = None  # Initialize outside if block
            if home_league and away_league:
                # Ülke bilgilerini çıkar
                home_country = home_data.get('country_name', '')
                away_country = away_data.get('country_name', '')
                
                # Lig isimlerini string'e çevir (dict ise)
                home_league_str = self._extract_league_name(home_league) if home_league else 'Unknown'
                away_league_str = self._extract_league_name(away_league) if away_league else 'Unknown'
                
                # Lig farkı analizi
                league_analysis = self.league_strength_analyzer.get_detailed_analysis(
                    home_name, away_name, home_league_str, away_league_str, competition_name, home_country, away_country
                )
                
                # xG değerlerini lig farkına göre ayarla
                adjusted_home_xg, adjusted_away_xg = self.league_strength_analyzer.adjust_team_strength(
                    home_xg, away_xg, home_league_str, away_league_str, competition_name, home_country, away_country
                )
                
                # xGA değerlerini de ayarla
                adjusted_home_xga, adjusted_away_xga = self.league_strength_analyzer.adjust_team_strength(
                    home_xga, away_xga, home_league_str, away_league_str, competition_name, home_country, away_country
                )
                
                # Lig farkı büyükse ayarlanmış değerleri kullan
                if league_analysis['is_cross_tier']:
                    logger.info(f"Lig farkı analizi uygulandı: {league_analysis['analysis']}")
                    logger.info(f"xG ayarlaması - Ev: {home_xg:.2f} -> {adjusted_home_xg:.2f}, "
                              f"Deplasman: {away_xg:.2f} -> {adjusted_away_xg:.2f}")
                    home_xg, away_xg = adjusted_home_xg, adjusted_away_xg
                    home_xga, away_xga = adjusted_home_xga, adjusted_away_xga
            
            # 4. Lambda değerlerini hesapla - Kompozit akıllı sistem
            # Maç bağlamını hazırla - Lig bilgilerini dahil et
            match_context_for_lambda = {
                'is_derby': False,  # TODO: Derbi kontrolü eklenebilir
                'rest_days': 3,  # TODO: Gerçek dinlenme günleri hesaplanabilir
                'motivation_level': 'normal',  # TODO: Lig durumuna göre ayarlanabilir
                'h2h_data': {},  # H2H verileri aşağıda eklenecek
                'league_name': home_league,  # Lig adı - lambda faktörü için
                'recent_league_matches': home_data.get('recent_matches', [])  # Lig maçları
            }
            
            # H2H verilerini ekle
            if h2h_data:
                h2h_matches = h2h_data if isinstance(h2h_data, list) else h2h_data.get('firstTeam_VS_secondTeam', [])
                if h2h_matches and isinstance(h2h_matches, list):
                    home_wins = 0
                    for m in h2h_matches:
                        if isinstance(m, dict):
                            # Support both old and new API formats
                            if 'fixture' in m:
                                # New API format (nested)
                                goals = m.get('goals', {})
                                teams = m.get('teams', {})
                                home_score = goals.get('home', 0) if goals.get('home') is not None else 0
                                away_score = goals.get('away', 0) if goals.get('away') is not None else 0
                                home_team_id_from_match = str(teams.get('home', {}).get('id', ''))
                            else:
                                # Old API format (flat) - fallback
                                home_score = int(m.get('match_hometeam_score', 0)) if str(m.get('match_hometeam_score', '')).isdigit() else 0
                                away_score = int(m.get('match_awayteam_score', 0)) if str(m.get('match_awayteam_score', '')).isdigit() else 0
                                home_team_id_from_match = str(m.get('match_hometeam_id', ''))
                            
                            if home_score > away_score and home_team_id_from_match == str(home_team_id):
                                home_wins += 1
                    match_context_for_lambda['h2h_data'] = {
                        'wins': home_wins,
                        'total': len(h2h_matches)
                    }
            
            # Kompozit lambda hesaplama
            lambda_home, lambda_away = self.xg_calculator.calculate_lambda_cross(
                home_xg, home_xga, away_xg, away_xga, elo_diff,
                home_team_data=home_data,
                away_team_data=away_data,
                match_context=match_context_for_lambda
            )
            
            # Maç bağlamı - Ekstrem maç bilgilerini ekle
            match_context = {
                'lambda_home': lambda_home,
                'lambda_away': lambda_away,
                'elo_diff': elo_diff,
                'home_xg': home_xg,
                'home_xga': home_xga,
                'away_xg': away_xg,
                'away_xga': away_xga,
                # Cross-league adjustment için lig bilgileri
                'home_league': home_league if home_league else 'Unknown',
                'away_league': away_league if away_league else 'Unknown',
                # UEFA COMPETITION DETECTION için competition bilgisi (league ID bazlı)
                'competition': competition_name if competition_name else '',
                'competition_league_id': competition_league_id,  # UEFA detection için league ID
                # DEBUG
                'league': competition_name if competition_name else 'Unknown League',
                # CRITICAL: League strength context for ensemble predictor
                'cross_league': is_cross_league,  # Flag to trigger cross-league adjustment
                'league_strength_context': {
                    'home': {
                        'league_name': home_league,
                        'league_id': home_league_id,
                        'strength_score': self._get_league_strength_score(home_league_id) if home_league_id else 50
                    },
                    'away': {
                        'league_name': away_league,
                        'league_id': away_league_id,
                        'strength_score': self._get_league_strength_score(away_league_id) if away_league_id else 50
                    },
                    'is_uefa_competition': is_uefa_competition,
                    'uefa_adjustment_factor': 1.2 if is_uefa_competition else 0.5  # 120% vs 50%
                },
                # Ekstrem maç için istatistikler
                'home_stats': {
                    'xg': home_xg,
                    'xga': home_xga,
                    'avg_goals_scored': home_data.get('home_performance', {}).get('avg_goals', 1.5),
                    'avg_goals_conceded': home_data.get('home_performance', {}).get('avg_conceded', 1.0),
                    'form': [m.get('goals_scored', 0) for m in home_data.get('recent_matches', [])[:5]]
                },
                'away_stats': {
                    'xg': away_xg,
                    'xga': away_xga,
                    'avg_goals_scored': away_data.get('away_performance', {}).get('avg_goals', 1.2),
                    'avg_goals_conceded': away_data.get('away_performance', {}).get('avg_conceded', 1.3),
                    'form': [m.get('goals_scored', 0) for m in away_data.get('recent_matches', [])[:5]]
                }
            }
            
            # 4.5. Gelişmiş özellik mühendisliği (Phase 3.2)
            # Enhanced match context for feature engineering
            enhanced_match_context = {
                **match_context,
                'datetime': datetime.now(),
                'league_id': home_data.get('league_id', 203),  # Default Süper Lig
                'h2h_data': h2h_data,
                'is_derby': False,  # TODO: Implement derby detection
                'competition_type': 'league',
                'importance_score': 0.5,  # TODO: Calculate based on league position
            }
            
            # Phase 3.2 Enhanced Feature Engineering
            enhanced_features = self.enhanced_feature_engineer.engineer_features(
                home_data,
                away_data,
                enhanced_match_context
            )
            
            # Keep backward compatibility with old feature structure
            advanced_features = self.feature_engineer.extract_all_features(
                home_data, 
                away_data, 
                match_context
            )
            
            # Merge enhanced features into advanced features
            advanced_features.update(enhanced_features)
            
            # CRITICAL: Apply cross-league adjustments to λ values BEFORE Poisson/Dixon-Coles generation
            if is_cross_league and home_league_id and away_league_id:
                home_strength = self._get_league_strength_score(home_league_id)
                away_strength = self._get_league_strength_score(away_league_id)
                strength_gap = abs(home_strength - away_strength)
                
                if strength_gap > 15:  # Significant strength difference
                    # Calculate adjustment multipliers based on architect's recommendation
                    # Apply same adjustment factors used in ensemble for consistency
                    uefa_factor = 1.2 if is_uefa_competition else 0.5
                    
                    if strength_gap > 40:
                        base_adjustment = 0.70 if not is_uefa_competition else 1.20
                    elif strength_gap > 25:
                        base_adjustment = 0.50 if not is_uefa_competition else 0.80
                    elif strength_gap > 15:
                        base_adjustment = 0.35 if not is_uefa_competition else 0.60
                    else:
                        base_adjustment = 0.20 if not is_uefa_competition else 0.40
                    
                    # Apply to λ values (multiplicative scaling) - ULTRA AGGRESSIVE for large gaps
                    if away_strength > home_strength:
                        # Stronger away team: reduce home λ, boost away λ
                        # Use gap/50 instead of gap/100 for 2x more aggressive adjustment
                        home_multiplier = 1.0 - (base_adjustment * (strength_gap / 50.0))
                        away_multiplier = 1.0 + (base_adjustment * (strength_gap / 50.0))
                        
                        # CRITICAL: Clamp multipliers to prevent negative λ values
                        home_multiplier = max(0.15, min(1.8, home_multiplier))
                        away_multiplier = max(0.15, min(2.5, away_multiplier))
                        
                        original_lambda_home = lambda_home
                        original_lambda_away = lambda_away
                        
                        lambda_home = lambda_home * home_multiplier
                        lambda_away = lambda_away * away_multiplier
                        
                        logger.info(f"🎯 CROSS-LEAGUE λ ADJUSTMENT (Pre-Poisson/Dixon-Coles):")
                        logger.info(f"   Strength gap: {strength_gap} points (Away team stronger)")
                        logger.info(f"   λ_home: {original_lambda_home:.2f} → {lambda_home:.2f} (x{home_multiplier:.2f})")
                        logger.info(f"   λ_away: {original_lambda_away:.2f} → {lambda_away:.2f} (x{away_multiplier:.2f})")
                        logger.info(f"   Base adjustment: {base_adjustment:.2f}, UEFA factor: {uefa_factor}")
                    else:
                        # Stronger home team: boost home λ, reduce away λ
                        # Use gap/50 instead of gap/100 for 2x more aggressive adjustment
                        home_multiplier = 1.0 + (base_adjustment * (strength_gap / 50.0))
                        away_multiplier = 1.0 - (base_adjustment * (strength_gap / 50.0))
                        
                        # CRITICAL: Clamp multipliers to prevent negative λ values
                        home_multiplier = max(0.15, min(2.5, home_multiplier))
                        away_multiplier = max(0.15, min(1.8, away_multiplier))
                        
                        original_lambda_home = lambda_home
                        original_lambda_away = lambda_away
                        
                        lambda_home = lambda_home * home_multiplier
                        lambda_away = lambda_away * away_multiplier
                        
                        logger.info(f"🎯 CROSS-LEAGUE λ ADJUSTMENT (Pre-Poisson/Dixon-Coles):")
                        logger.info(f"   Strength gap: {strength_gap} points (Home team stronger)")
                        logger.info(f"   λ_home: {original_lambda_home:.2f} → {lambda_home:.2f} (x{home_multiplier:.2f})")
                        logger.info(f"   λ_away: {original_lambda_away:.2f} → {lambda_away:.2f} (x{away_multiplier:.2f})")
                        logger.info(f"   Base adjustment: {base_adjustment:.2f}, UEFA factor: {uefa_factor}")
            
            # 5. Tüm modelleri çalıştır
            model_predictions = {}
            
            # Poisson Model
            poisson_matrix = self.poisson_model.calculate_probability_matrix(
                lambda_home, lambda_away, elo_diff
            )
            model_predictions['poisson'] = self._process_poisson_results(poisson_matrix, lambda_home, lambda_away)
            
            # Dixon-Coles Model
            dc_matrix = self.dixon_coles.calculate_probability_matrix(
                lambda_home, lambda_away, elo_diff
            )
            model_predictions['dixon_coles'] = self._process_dixon_coles_results(dc_matrix, lambda_home, lambda_away)
            
            # XGBoost Model
            xg_features = self.xgboost_model.prepare_features(home_data, away_data, match_context)
            model_predictions['xgboost'] = self.xgboost_model.predict(xg_features)
            
            # Monte Carlo Simülasyonu - takım ID'leri ile
            mc_results = self.monte_carlo.run_simulations(
                lambda_home, lambda_away, elo_diff, 
                home_id=home_team_id, away_id=away_team_id
            )
            model_predictions['monte_carlo'] = self._process_monte_carlo_results(mc_results)
            
            # CRF Model
            crf_features = self.crf_predictor.prepare_features(
                home_data, away_data, lambda_home, lambda_away, elo_diff
            )
            model_predictions['crf'] = self.crf_predictor.predict(crf_features)
            
            # Neural Network Model
            nn_features = self.neural_network.prepare_features(
                home_data, away_data, match_context, match_context
            )
            model_predictions['neural_network'] = self.neural_network.predict(nn_features)
            
            # Self-Learning model context'i kullanarak ağırlıkları al
            is_extreme = lambda_home + lambda_away > 5.0
            dynamic_context = {
                'is_extreme': is_extreme,
                'expected_total_goals': lambda_home + lambda_away,
                'elo_diff': elo_diff
            }
            
            # 5.1. Venue Performance Analysis
            try:
                # Prepare venue info
                venue_info_for_analysis = self._prepare_venue_info(home_data, away_data, home_league)
                historical_matches_combined = home_data.get('recent_matches', []) + away_data.get('recent_matches', [])
                
                venue_analysis = self.venue_performance_optimizer.analyze_comprehensive_venue_performance(
                    home_team_id=int(home_team_id),
                    away_team_id=int(away_team_id),
                    venue_info=venue_info_for_analysis,
                    match_context=match_context,
                    historical_matches=historical_matches_combined
                )
                logger.info(f"Venue analysis: Home advantage: {venue_analysis.get('home_advantage_factor', 1.0):.2f}")
            except Exception as e:
                logger.warning(f"Venue analysis failed: {e}")
                venue_analysis = {'home_advantage_factor': 1.0}
            
            # 5.2. Seasonal Performance Analysis
            try:
                home_matches = home_data.get('recent_matches', [])
                seasonal_analysis = self.seasonal_performance_analyzer.analyze_seasonal_performance(
                    home_matches, match_context
                )
                logger.info(f"Seasonal analysis: Home phase: {seasonal_analysis.get('home_seasonal_phase', 'unknown')}")
            except Exception as e:
                logger.warning(f"Seasonal analysis failed: {e}")
                seasonal_analysis = {'seasonal_adjustment_factor': 1.0}
            
            # 5.3. Dynamic Time-weighted Features
            try:
                temporal_features = self.dynamic_time_analyzer.analyze_temporal_features(
                    {'team_id': home_team_id}, match_context
                )
                logger.info(f"Temporal analysis: Features generated: {len(temporal_features.get('features', []))}")
            except Exception as e:
                logger.warning(f"Temporal analysis failed: {e}")
                temporal_features = {'time_weighted_score': 0.5}

            # 6. Ensemble birleştirme - dinamik ağırlıklarla
            algorithm_weights = self.self_learning.get_dynamic_weights(dynamic_context)
            
            # 6.1. Meta-Learning Layer Integration
            if hasattr(self, 'meta_learning_layer') and self.meta_learning_layer:
                try:
                    meta_context = {
                        'home_team': home_team_id,
                        'away_team': away_team_id,
                        'league': match_context.get('league', 'unknown'),
                        'venue_analysis': venue_analysis,
                        'seasonal_analysis': seasonal_analysis,
                        'temporal_features': temporal_features
                    }
                    optimal_weights = self.meta_learning_layer.optimize_model_weights(
                        model_predictions, meta_context
                    )
                    algorithm_weights.update(optimal_weights)
                    logger.info("Meta-learning optimization applied")
                except Exception as e:
                    logger.warning(f"Meta-learning failed: {e}")
            
            final_prediction = self.ensemble.combine_predictions(
                model_predictions, match_context, algorithm_weights
            )
            
            # 6.2. Prediction Confidence System Integration
            if hasattr(self, 'prediction_confidence_system') and self.prediction_confidence_system:
                try:
                    confidence_data = self.prediction_confidence_system.calculate_comprehensive_confidence(
                        model_predictions, match_context, final_prediction
                    )
                    final_prediction['confidence'] = confidence_data.get('overall_confidence', final_prediction.get('confidence', 50))
                    final_prediction['confidence_details'] = confidence_data
                    logger.info(f"Confidence system applied: {final_prediction['confidence']:.1f}%")
                except Exception as e:
                    logger.warning(f"Confidence system failed: {e}")
            
            # 6.1. Psychological Adjustments to Predictions
            if psychological_analysis:
                try:
                    # Psikolojik faktörlerin tahminlere etkisini uygula
                    psychological_impact = psychological_analysis['overall_assessment']['psychological_prediction_impact']
                    
                    # Motivasyon avantajını belirle (tüm scope'larda kullanılabilmesi için)
                    motivation_diff = psychological_analysis['motivation_analysis']['motivation_differential']
                    momentum_advantage = psychological_analysis['momentum_analysis']['momentum_advantage']
                    
                    # 1X2 olasılıklarını ayarla - BERABERLIK KORUMALI
                    outcome_adjustment = psychological_impact.get('outcome_probability_adjustment', 0)
                    if abs(outcome_adjustment) > 0.05:  # Anlamlı bir ayar varsa
                        
                        # Beraberlik için minimum sınır - asla %12'nin altına düşmemeli
                        min_draw_threshold = 12.0
                        
                        # Ev sahibi avantajında ise
                        if motivation_diff > 10 or 'home' in momentum_advantage:
                            adjustment_factor = min(0.10, outcome_adjustment)  # Maksimum %10 (eskiden %15)
                            final_prediction['home_win'] += adjustment_factor * 100
                            # Beraberlikten daha az çıkar, asıl rakipten çıkar
                            final_prediction['away_win'] -= (adjustment_factor * 0.8) * 100  # %80 rakipten
                            draw_reduction = (adjustment_factor * 0.2) * 100  # %20 beraberlikten
                            # Beraberlik minimum sınırın altına düşmesin
                            if final_prediction['draw'] - draw_reduction >= min_draw_threshold:
                                final_prediction['draw'] -= draw_reduction
                            
                        # Deplasman avantajında ise  
                        elif motivation_diff < -10 or 'away' in momentum_advantage:
                            adjustment_factor = min(0.10, outcome_adjustment)  # Maksimum %10 (eskiden %15)
                            final_prediction['away_win'] += adjustment_factor * 100
                            # Beraberlikten daha az çıkar, asıl rakipten çıkar
                            final_prediction['home_win'] -= (adjustment_factor * 0.8) * 100  # %80 rakipten
                            draw_reduction = (adjustment_factor * 0.2) * 100  # %20 beraberlikten
                            # Beraberlik minimum sınırın altına düşmesin
                            if final_prediction['draw'] - draw_reduction >= min_draw_threshold:
                                final_prediction['draw'] -= draw_reduction
                    
                    # Beklenen golleri ayarla
                    goal_adjustment = psychological_impact.get('goal_expectation_adjustment', 0)
                    if abs(goal_adjustment) > 0.05:
                        if motivation_diff > 15:  # Güçlü ev avantajı
                            final_prediction['expected_goals']['home'] += goal_adjustment
                        elif motivation_diff < -15:  # Güçlü deplasman avantajı
                            final_prediction['expected_goals']['away'] += goal_adjustment
                    
                    # Güven seviyesini ayarla - NaN kontrolü
                    confidence_adjustment = psychological_impact.get('confidence_adjustment', 0)
                    
                    # NaN ve geçersiz değer kontrolü
                    if math.isnan(confidence_adjustment):
                        confidence_adjustment = 0
                    
                    # Confidence'a ekle (confidence_adjustment zaten -1 ile +1 arası, yüzde olarak ekle)
                    # final_prediction['confidence'] 0-100 arası, adjustment'ı doğrudan ekle
                    final_prediction['confidence'] += confidence_adjustment
                    
                    # NaN kontrolü ve sınırlandırma
                    if math.isnan(final_prediction['confidence']) or final_prediction['confidence'] is None:
                        final_prediction['confidence'] = 70  # Varsayılan
                    else:
                        final_prediction['confidence'] = max(45, min(90, final_prediction['confidence']))
                    
                    # Kritik maç varsa güven seviyesini biraz düşür (belirsizlik artar)
                    if psychological_analysis['critical_match_analysis']['is_critical_match']:
                        final_prediction['confidence'] *= 0.95
                    
                    logger.info(f"Psikolojik ayarlamalar uygulandı - Yeni güven: {final_prediction['confidence']:.1f}%")
                    
                except Exception as e:
                    logger.warning(f"Psikolojik ayarlama hatası: {e}")
            
            # 6.5. Dynamic Team Analyzer ayarlamalarını uygula
            if team_comparison:
                adjustments = team_comparison['combined_adjustments']
                
                # Lambda değerlerini ayarla
                original_lambda_home = lambda_home
                original_lambda_away = lambda_away
                lambda_home += lambda_home * adjustments['total_goals_modifier']
                lambda_away += lambda_away * adjustments['total_goals_modifier']
                
                # BTTS (KG) tahminini ayarla
                if 'both_teams_to_score' in final_prediction:
                    btts_prob = final_prediction['both_teams_to_score']['yes']
                    btts_adjustment = adjustments['btts_modifier'] / 100.0
                    new_btts_yes = max(0, min(100, btts_prob + btts_adjustment))
                    final_prediction['both_teams_to_score']['yes'] = new_btts_yes
                    final_prediction['both_teams_to_score']['no'] = 100 - new_btts_yes
                
                # Over/Under tahminlerini ayarla
                if 'over_under' in final_prediction:
                    ou_adjustment = adjustments['over_2_5_modifier'] / 100.0
                    for market in final_prediction['over_under']:
                        if market['threshold'] == 2.5:
                            over_prob = market['over']
                            new_over = max(0, min(100, over_prob + ou_adjustment))
                            market['over'] = new_over
                            market['under'] = 100 - new_over
                
                # Güven skorunu ayarla
                if 'confidence' in final_prediction:
                    conf_adjustment = adjustments['confidence_modifier']
                    final_prediction['confidence'] = max(0, min(100, 
                        final_prediction['confidence'] + conf_adjustment))
                
                # Volatilite faktörünü kaydet
                final_prediction['volatility_factor'] = adjustments['volatility_factor']
                
                logger.info(f"Dynamic Team Analyzer ayarlamaları uygulandı:")
                logger.info(f"  Lambda ayarı: {adjustments['total_goals_modifier']:+.2f}")
                logger.info(f"  BTTS ayarı: {adjustments['btts_modifier']:+.0f}%")
                logger.info(f"  O/U 2.5 ayarı: {adjustments['over_2_5_modifier']:+.0f}%")
                logger.info(f"  Güven ayarı: {adjustments['confidence_modifier']:+.0f}%")
            
            # 7. Yeni tahmin türlerini hesapla
            # HT/FT tahminleri
            htft_predictions = self.htft_predictor.predict_htft(
                home_data, away_data, lambda_home, lambda_away, elo_diff
            )
            
            # İlk yarı gol tahminleri
            halftime_goals = self.htft_predictor.predict_halftime_goals(
                home_data, away_data, lambda_home, lambda_away
            )
            
            # Handikap tahminleri
            asian_handicap = self.handicap_predictor.predict_asian_handicap(
                home_xg, away_xg, elo_diff,
                ''.join(self._analyze_form(home_data.get('recent_matches', [])[:5])),
                ''.join(self._analyze_form(away_data.get('recent_matches', [])[:5]))
            )
            
            european_handicap = self.handicap_predictor.predict_european_handicap(
                home_xg, away_xg, elo_diff, final_prediction
            )
            
            # Gol aralığı tahminleri
            goal_ranges = self.goal_range_predictor.predict_goal_ranges(
                lambda_home, lambda_away, match_context
            )
            
            # Toplam gol marketleri
            total_goals_markets = self.goal_range_predictor.predict_total_goals_markets(
                lambda_home, lambda_away
            )
            
            # Çifte şans tahminleri
            double_chance = self.double_chance_predictor.predict_double_chance(final_prediction)
            
            # Takım gol tahminleri
            # Savunma gücü hesaplama: xGA/xG oranı (1'den küçük = iyi savunma, 1'den büyük = kötü savunma)
            # Min 0.5, Max 2.0 sınırları ile
            # Ev sahibi savunması: home_xga/home_xg
            # Deplasman savunması: away_xga/away_xg
            home_defense_strength = max(0.5, min(2.0, home_xga / home_xg)) if home_xg > 0 else 1.0
            away_defense_strength = max(0.5, min(2.0, away_xga / away_xg)) if away_xg > 0 else 1.0
            
            # Debug log
            logger.info(f"Savunma gücü hesaplama:")
            logger.info(f"  - Ev sahibi xG: {home_xg:.2f}, xGA: {home_xga:.2f}")
            logger.info(f"  - Deplasman xG: {away_xg:.2f}, xGA: {away_xga:.2f}")
            logger.info(f"  - Ev sahibi savunma gücü: {home_defense_strength:.2f}")
            logger.info(f"  - Deplasman savunma gücü: {away_defense_strength:.2f}")
            
            team_goals = self.team_goals_predictor.predict_both_teams_goals(
                lambda_home, lambda_away, home_name, away_name,
                home_defense=home_defense_strength,  # Ev sahibi savunması
                away_defense=away_defense_strength   # Deplasman savunması
            )
            
            # Tahminleri final_prediction'a ekle
            final_prediction['advanced_predictions'] = {
                'htft': htft_predictions,
                'halftime_goals': halftime_goals,
                'asian_handicap': asian_handicap,
                'european_handicap': european_handicap,
                'goal_ranges': goal_ranges,
                'total_goals_markets': total_goals_markets,
                'double_chance': double_chance,
                'team_goals': team_goals,
                'fatigue_analysis': {
                    'home_fatigue_score': home_data['fatigue_score'],
                    'away_fatigue_score': away_data['fatigue_score'],
                    'fatigue_comparison': fatigue_comparison,
                    'fatigue_advantage': fatigue_comparison.get('advantage', 'balanced'),
                    'home_risk_level': home_congestion_analysis.get('risk_level', 'moderate'),
                    'away_risk_level': away_congestion_analysis.get('risk_level', 'moderate'),
                    'home_congestion_analysis': home_congestion_analysis,
                    'away_congestion_analysis': away_congestion_analysis
                }
            }
            
            # 7. Ekstrem maç kontrolü ve düzeltme
            from algorithms.extreme_detector import ExtremeMatchDetector
            detector = ExtremeMatchDetector()
            
            is_extreme, extreme_details = detector.is_extreme_match(
                match_context['home_stats'], 
                match_context['away_stats']
            )
            
            if is_extreme:
                # Ekstrem maç tahminlerini validate et
                final_prediction = detector.validate_extreme_prediction(
                    final_prediction,
                    match_context['home_stats'],
                    match_context['away_stats']
                )
                logger.info(f"Ekstrem maç düzeltmesi uygulandı: {extreme_details['indicators']}")
            
            # 7. Sonuç formatla
            prediction = self._format_prediction(
                final_prediction, match_context, home_name, away_name, 
                home_team_id, away_team_id, home_data, away_data, h2h_data,
                home_team_analysis, away_team_analysis, team_comparison,
                form_comparison, enhanced_features, league_analysis, 
                psychological_analysis
            )
            
            # 8. Açıklanabilir AI
            try:
                # Model ve özellik vektörü hazırla
                features = np.array([
                    home_xg,
                    away_xg,
                    home_xga,
                    away_xga,
                    elo_diff,
                    advanced_features.get('form_momentum', {}).get('home', {}).get('composite_score', 0),
                    advanced_features.get('form_momentum', {}).get('away', {}).get('composite_score', 0),
                    advanced_features.get('form_momentum', {}).get('differential', 0),
                    advanced_features.get('goal_dynamics', {}).get('home', {}).get('scoring_trend', 0),
                    advanced_features.get('advanced_context', {}).get('match_importance', 0.5)
                ]).reshape(1, -1)
                
                explanation = self.prediction_explainer.explain_prediction(
                    prediction['predictions'],
                    model=self.xgboost_model.model_1x2 if hasattr(self.xgboost_model, 'model_1x2') else None,
                    features=features
                )
                prediction['explanation'] = explanation
            except Exception as e:
                logger.warning(f"Açıklama oluşturulamadı: {e}")
            
            # 9. Sürekli öğrenme (gerçek sonuç geldiğinde çalışacak)
            
            # Hesaplama süresi
            prediction['calculation_time'] = round(time.time() - start_time, 2)
            
            # Performans kayıt
            performance_monitor.record_prediction_time('ensemble', prediction['calculation_time'])
            
            # Gelişmiş önbelleğe kaydet
            prediction_cache.set_prediction(home_team_id, away_team_id, date_str, prediction)
            
            logger.info(f"Tahmin tamamlandı ({prediction['calculation_time']}s): {prediction['predictions']['most_likely_outcome']}")
            return prediction
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}", exc_info=True)
            # Hata durumunda basit tahmin döndür
            return self._get_fallback_prediction(home_team_id, away_team_id, home_name, away_name)
            
    def _get_team_data(self, team_id, team_name, is_home=True, champions_league_context=False, uefa_league_id=None):
        """
        Takım verilerini API'den al veya varsayılan kullan
        
        Args:
            team_id: Takım ID
            team_name: Takım adı
            is_home: Ev sahibi mi?
            champions_league_context: UEFA maçı mı? (Eğer True ise UEFA performansı %90 ağırlık alır)
            uefa_league_id: UEFA lig ID (3: CL, 4: EL, 683: Conference) - sadece bu ligden veri çekilir
        """
        try:
            # API'den gerçek takım verilerini almayı dene
            import requests
            from datetime import datetime, timedelta
            from api_config import APIConfig
            
            # API anahtarını config'den al
            api_config = APIConfig()
            api_key = api_config.get_api_key()
            
            if not api_key:
                logger.warning("API anahtarı bulunamadı")
                raise Exception("API anahtarı yok")
                
            url = "https://v3.football.api-sports.io/fixtures"
            headers = {'x-apisports-key': api_key}
            
            # Last 120 days data (2025 data)
            date_from = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
            date_to = datetime.now().strftime('%Y-%m-%d')
            
            # Determine current season (e.g., 2024 for 2024-2025 season)
            current_month = datetime.now().month
            current_year = datetime.now().year
            # Football seasons typically run from August to May
            # If we're in Jan-May, the season started last year
            season_year = current_year if current_month >= 7 else current_year - 1
            
            # Debugging: log date range
            logger.info(f"Fetching data for team {team_id}: {date_from} to {date_to} (season: {season_year})")
            
            params = {
                'team': team_id,
                'season': season_year,  # CRITICAL: API requires season parameter
                'timezone': 'Europe/Istanbul'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Debug: Log the response structure
                if isinstance(data, dict):
                    logger.info(f"API response keys for team {team_id}: {list(data.keys())}")
                    if 'errors' in data and data['errors']:
                        logger.error(f"API errors for team {team_id}: {data['errors']}")
                    if 'results' in data:
                        logger.info(f"API results count for team {team_id}: {data.get('results', 0)}")
                matches = data.get('response', []) if isinstance(data, dict) else []
                logger.info(f"API response for team {team_id}: {len(matches) if isinstance(matches, list) else 0} matches")
                # Debug: Log first match structure if available
                if isinstance(matches, list) and len(matches) > 0:
                    logger.info(f"First match keys for team {team_id}: {list(matches[0].keys()) if isinstance(matches[0], dict) else 'not a dict'}")
                if isinstance(matches, list) and len(matches) > 0:
                    # Gerçek maç verilerini işle
                    recent_matches = []
                    home_goals = []
                    home_conceded = []
                    away_goals = []
                    away_conceded = []
                    
                    # Son maçları tarih sırasına göre filtrele (en yeniler önce)
                    sorted_matches = sorted(matches, key=lambda x: x.get('fixture', {}).get('date', ''), reverse=True)
                    
                    # 2025 verilerine odaklan ve eski verileri filtrele
                    current_year = datetime.now().year
                    filtered_matches = []
                    for match in sorted_matches:
                        match_date = match.get('fixture', {}).get('date', '')
                        if match_date and str(current_year) in match_date:  # 2025 verisi kontrolü
                            filtered_matches.append(match)
                    
                    logger.info(f"Toplam {len(matches)} maçtan {len(filtered_matches)} tanesi 2025 verisi")
                    
                    # İlk maçın veri yapısını logla
                    if filtered_matches:
                        first_match = filtered_matches[0]
                        logger.info(f"API'den gelen maç verisi örneği - anahtarlar: {list(first_match.keys())[:15]}")
                        # Lig bilgisi içeren alanları kontrol et
                        league_fields = ['league_name', 'league_id', 'country_name', 'match_league']
                        for field in league_fields:
                            if field in first_match:
                                logger.info(f"  {field}: {first_match[field]}")
                        
                    for match in filtered_matches[:30]:  # En fazla 30 güncel maç al
                        # KRİTİK FİLTRE: Sadece TAMAMLANMIŞ maçları al!
                        fixture = match.get('fixture', {})
                        teams = match.get('teams', {})
                        goals = match.get('goals', {})
                        league = match.get('league', {})
                        
                        match_status = fixture.get('status', {}).get('short', '').strip()
                        
                        # Henüz oynanmamış veya devam eden maçları atla
                        if match_status not in ['FT', 'AET', 'PEN']:
                            home_name = teams.get('home', {}).get('name', 'N/A')
                            away_name = teams.get('away', {}).get('name', 'N/A')
                            logger.info(f"Tamamlanmamış maç atlandı: '{match_status}' - {fixture.get('date', 'N/A')} {home_name} vs {away_name}")
                            continue
                        
                        # Skorları güvenli şekilde al
                        home_score_raw = goals.get('home')
                        away_score_raw = goals.get('away')
                        
                        # Skorlar geçerli mi kontrol et
                        if (home_score_raw is None or away_score_raw is None):
                            home_name = teams.get('home', {}).get('name', 'N/A')
                            away_name = teams.get('away', {}).get('name', 'N/A')
                            logger.info(f"Geçersiz skor atlandı: '{home_score_raw}'-'{away_score_raw}' - {fixture.get('date', 'N/A')} {home_name} vs {away_name}")
                            continue
                            
                        try:
                            home_score = int(home_score_raw)
                            away_score = int(away_score_raw)
                        except (ValueError, TypeError):
                            logger.debug(f"Sayıya çevrilemeyen skor atlandı: {home_score_raw}-{away_score_raw}")
                            continue
                        
                        # Extract date from ISO format (2025-12-23T20:00:00+03:00 -> 2025-12-23)
                        match_date_iso = fixture.get('date', '')
                        match_date = match_date_iso.split('T')[0] if 'T' in match_date_iso else match_date_iso
                        
                        # Bu takım ev sahibi mi deplasman mı?
                        home_team_id = str(teams.get('home', {}).get('id', ''))
                        away_team_id = str(teams.get('away', {}).get('id', ''))
                        
                        if home_team_id == str(team_id):
                            # Ev sahibi maçı
                            recent_matches.append({
                                'goals_scored': home_score,
                                'goals_conceded': away_score,
                                'date': match_date,
                                'is_home': True,
                                'match_id': str(fixture.get('id', '')),
                                'opponent': teams.get('away', {}).get('name', 'Bilinmeyen'),
                                'status': match_status,
                                'league': league.get('name', 'Unknown'),
                                'league_id': league.get('id')  # League ID eklendi
                            })
                            home_goals.append(home_score)
                            home_conceded.append(away_score)
                        elif away_team_id == str(team_id):
                            # Deplasman maçı
                            recent_matches.append({
                                'goals_scored': away_score,
                                'goals_conceded': home_score,
                                'date': match_date,
                                'is_home': False,
                                'match_id': str(fixture.get('id', '')),
                                'opponent': teams.get('home', {}).get('name', 'Bilinmeyen'),
                                'status': match_status,
                                'league': league.get('name', 'Unknown'),
                                'league_id': league.get('id')  # League ID eklendi
                            })
                            away_goals.append(away_score)
                            away_conceded.append(home_score)
                    
                    # Performans istatistikleri hesapla
                    home_avg_goals = sum(home_goals) / len(home_goals) if home_goals else 1.3
                    home_avg_conceded = sum(home_conceded) / len(home_conceded) if home_conceded else 1.3
                    away_avg_goals = sum(away_goals) / len(away_goals) if away_goals else 1.0
                    away_avg_conceded = sum(away_conceded) / len(away_conceded) if away_conceded else 1.3
                    
                    # SON 5-10 EV/DEPLASMAN MAÇLARINA ÖZEL ANALİZ
                    # ÖNEMLİ: UEFA maçıysa, SADECE UEFA maçlarını kullan (çok agresif!)
                    if champions_league_context and uefa_league_id:
                        # League ID bazlı filtreleme (çok daha güvenilir!)
                        uefa_matches = []
                        for m in recent_matches:
                            # Match'ten league_id'yi al
                            match_league_id = m.get('league_id')
                            # Eğer league ID eşleşiyorsa (CL/EL/Conference)
                            if match_league_id and (match_league_id == uefa_league_id or self._is_uefa_competition(match_league_id)):
                                uefa_matches.append(m)
                        
                        logger.info(f"🏆 UEFA Context (League ID: {uefa_league_id}): Takım {team_id} için {len(uefa_matches)} UEFA maçı bulundu (toplam {len(recent_matches)} maç)")
                        
                        if len(uefa_matches) >= 1:  # En az 1 UEFA maçı varsa
                            # UEFA maçlarına %90 ağırlık ver, ulusal lige minimize et
                            weighted_matches = (uefa_matches * 9) + recent_matches  # 90% UEFA, 10% ulusal
                            recent_matches = weighted_matches[:30]  # İlk 30'u al
                            logger.info(f"   → UEFA maçlarına %90 ağırlık verildi: {len(uefa_matches)} UEFA maçı x9 + minimal ulusal lig")
                        else:
                            logger.info(f"   → Yetersiz UEFA maçı ({len(uefa_matches)}), ulusal lig verisi kullanılıyor")
                    
                    # Ev sahibi maçları filtrele
                    home_matches = [m for m in recent_matches if m['is_home']][:10]  # Son 10 ev maçı
                    away_matches = [m for m in recent_matches if not m['is_home']][:10]  # Son 10 deplasman maçı
                    
                    # Son 5 ev/deplasman maçı için detaylı analiz
                    last_5_home = home_matches[:5]
                    last_5_away = away_matches[:5]
                    
                    # Son 5 ev maçı istatistikleri
                    if last_5_home:
                        last_5_home_goals = [m['goals_scored'] for m in last_5_home]
                        last_5_home_conceded = [m['goals_conceded'] for m in last_5_home]
                        last_5_home_avg_goals = sum(last_5_home_goals) / len(last_5_home_goals)
                        last_5_home_avg_conceded = sum(last_5_home_conceded) / len(last_5_home_conceded)
                        last_5_home_form = self._analyze_form(last_5_home)
                        last_5_home_win_rate = sum(1 for m in last_5_home if m['goals_scored'] > m['goals_conceded']) / len(last_5_home)
                    else:
                        last_5_home_avg_goals = home_avg_goals
                        last_5_home_avg_conceded = home_avg_conceded
                        last_5_home_form = []
                        last_5_home_win_rate = 0.4
                    
                    # Son 5 deplasman maçı istatistikleri
                    if last_5_away:
                        last_5_away_goals = [m['goals_scored'] for m in last_5_away]
                        last_5_away_conceded = [m['goals_conceded'] for m in last_5_away]
                        last_5_away_avg_goals = sum(last_5_away_goals) / len(last_5_away_goals)
                        last_5_away_avg_conceded = sum(last_5_away_conceded) / len(last_5_away_conceded)
                        last_5_away_form = self._analyze_form(last_5_away)
                        last_5_away_win_rate = sum(1 for m in last_5_away if m['goals_scored'] > m['goals_conceded']) / len(last_5_away)
                    else:
                        last_5_away_avg_goals = away_avg_goals
                        last_5_away_avg_conceded = away_avg_conceded
                        last_5_away_form = []
                        last_5_away_win_rate = 0.3
                    
                    logger.info(f"Takım {team_id}: {len(recent_matches)} tamamlanmış maç işlendi")
                    logger.info(f"  - Son 5 ev maçı: {last_5_home_avg_goals:.2f} gol, {last_5_home_avg_conceded:.2f} yenen")
                    logger.info(f"  - Son 5 deplasman maçı: {last_5_away_avg_goals:.2f} gol, {last_5_away_avg_conceded:.2f} yenen")
                    
                    # Form analizi ekle
                    form_analysis = self._analyze_form(recent_matches[:10])  # Son 10 maçtan form
                    
                    # Country ve domestic league bilgisini al
                    country_name = ''
                    domestic_league_name = ''
                    domestic_league_id = None
                    
                    # CRITICAL: Extract domestic league from NON-UEFA matches (preserve league identity)
                    non_uefa_matches = [m for m in recent_matches if m.get('league_id') not in [3, 4, 683]]
                    if non_uefa_matches:
                        # En sık oynadığı ulusal ligi bul
                        league_counts = {}
                        for m in non_uefa_matches[:15]:  # Son 15 ulusal lig maçı
                            league_name = m.get('league', '')
                            league_id = m.get('league_id')
                            if league_name and league_id:
                                key = (league_name, league_id)
                                league_counts[key] = league_counts.get(key, 0) + 1
                        
                        if league_counts:
                            # En çok maç oynanan lig = domestic league
                            most_common_league = max(league_counts.items(), key=lambda x: x[1])
                            domestic_league_name = most_common_league[0][0]
                            domestic_league_id = most_common_league[0][1]
                            logger.info(f"Takım {team_id} domestic league: {domestic_league_name} (ID: {domestic_league_id})")
                    
                    # Fallback: Team ID → League ID mapping from config
                    if not domestic_league_id:
                        team_league_fallback = self.league_ids.get('team_league_fallback', {})
                        fallback_league_id = team_league_fallback.get(str(team_id))
                        if fallback_league_id:
                            domestic_league_id = fallback_league_id
                            # Get league name from league_ids
                            for name, lid in self.league_ids.get('league_names_to_ids', {}).items():
                                if lid == fallback_league_id:
                                    domestic_league_name = name
                                    break
                            logger.info(f"Takım {team_id} fallback league mapping: {domestic_league_name} (ID: {domestic_league_id})")
                    
                    try:
                        # Get teams API'sini çağır (country name için)
                        team_params = {
                            'action': 'get_teams',
                            'team_id': team_id,
                            'APIkey': api_key
                        }
                        team_response = requests.get(url, params=team_params, timeout=5)
                        if team_response.status_code == 200:
                            team_data_api = team_response.json()
                            if isinstance(team_data_api, list) and len(team_data_api) > 0:
                                country_name = team_data_api[0].get('team_country', '')
                                logger.info(f"Takım {team_id} ({team_name}) için ülke bulundu: {country_name}")
                    except Exception as e:
                        logger.warning(f"Takım ülke bilgisi alınamadı: {e}")
                    
                    return {
                        'team_id': team_id,
                        'team_name': team_name,
                        'country_name': country_name,  # Ülke bilgisi
                        'domestic_league_name': domestic_league_name,  # CRITICAL: Domestic league preserved
                        'domestic_league_id': domestic_league_id,  # CRITICAL: For cross-league adjustment
                        'recent_matches': recent_matches,
                        'form_analysis': form_analysis,
                        'recent_form': ''.join(form_analysis),  # W/L/D string'i
                        'matches_count': len(recent_matches),
                        'home_performance': {
                            'avg_goals': home_avg_goals,
                            'avg_conceded': home_avg_conceded,
                            # Son 5 ev maçı verileri
                            'last_5_avg_goals': last_5_home_avg_goals,
                            'last_5_avg_conceded': last_5_home_avg_conceded,
                            'last_5_form': ''.join(last_5_home_form),
                            'last_5_win_rate': last_5_home_win_rate,
                            'last_5_matches': len(last_5_home)
                        },
                        'away_performance': {
                            'avg_goals': away_avg_goals,
                            'avg_conceded': away_avg_conceded,
                            # Son 5 deplasman maçı verileri
                            'last_5_avg_goals': last_5_away_avg_goals,
                            'last_5_avg_conceded': last_5_away_avg_conceded,
                            'last_5_form': ''.join(last_5_away_form),
                            'last_5_win_rate': last_5_away_win_rate,
                            'last_5_matches': len(last_5_away)
                        },
                        # Takımın güncel ev/deplasman durumu için kullanılacak
                        'is_home': is_home,
                        'venue_specific_avg_goals': last_5_home_avg_goals if is_home else last_5_away_avg_goals,
                        'venue_specific_avg_conceded': last_5_home_avg_conceded if is_home else last_5_away_avg_conceded
                    }
            else:
                logger.warning(f"API'den veri alınamadı takım {team_id} için, yanıt kodu: {response.status_code}")
        except Exception as e:
            logger.error(f"API veri alımı başarısız takım {team_id} için: {e}")
        
        # API başarısız oldu - varsayılan değerler + fallback league mapping kullan
        logger.warning(f"Takım {team_id} için gerçek veri alınamadı, varsayılan değerler + fallback mapping kullanılacak")
        
        # CRITICAL FALLBACK: Use team-to-league mapping from config
        domestic_league_id = None
        domestic_league_name = ''
        team_league_fallback = self.league_ids.get('team_league_fallback', {})
        fallback_league_id = team_league_fallback.get(str(team_id))
        if fallback_league_id:
            domestic_league_id = fallback_league_id
            # Get league name from league_ids
            for name, lid in self.league_ids.get('league_names_to_ids', {}).items():
                if lid == fallback_league_id:
                    domestic_league_name = name
                    break
            logger.info(f"⚠️ FALLBACK: Takım {team_id} league mapping from config: {domestic_league_name} (ID: {domestic_league_id})")
        
        # Varsayılan takım verileri
        return {
            'team_id': team_id,
            'team_name': team_name,
            'country_name': '',  # Varsayılan boş ülke
            'domestic_league_name': domestic_league_name,  # CRITICAL: Fallback league
            'domestic_league_id': domestic_league_id,  # CRITICAL: Fallback league ID
            'recent_matches': [],
            'form_analysis': [],
            'recent_form': 'DDDDD',  # Varsayılan form
            'matches_count': 0,
            'home_performance': {
                'avg_goals': 1.3 if is_home else 1.0,
                'avg_conceded': 1.3
            },
            'away_performance': {
                'avg_goals': 1.0,
                'avg_conceded': 1.3
            },
            'form_score': 50,  # Orta düzey form
            'league_position': 10,  # Varsayılan pozisyon
            'goals_for_avg': 1.15,
            'goals_against_avg': 1.15,
            'xG': 1.2 if is_home else 1.0,
            'xGA': 1.2
        }
        

        
    def _process_poisson_results(self, matrix, lambda_home, lambda_away):
        """
        Poisson sonuçlarını işle
        """
        match_probs = self.poisson_model.get_match_probabilities(matrix)
        goal_probs = self.poisson_model.get_goals_probabilities(matrix)
        scores = self.poisson_model.get_exact_score_probabilities(matrix)
        
        # Dinamik güven hesaplama
        max_prob = max(match_probs['home_win'], match_probs['draw'], match_probs['away_win'])
        
        # Tahmin keskinliğine göre güven (0.4-0.9 arası)
        if max_prob > 60:  # Çok net favori
            confidence = 0.7 + (max_prob - 60) / 100  # 0.7-0.9
        elif max_prob > 45:  # Orta düzey favori
            confidence = 0.6 + (max_prob - 45) / 75  # 0.6-0.7
        else:  # Dengeli maç
            confidence = 0.5 + (max_prob - 33) / 60  # 0.5-0.6
        
        # Poisson modeli için temel güven
        confidence = min(0.85, max(0.5, confidence))
        
        return {
            'home_win': match_probs['home_win'],
            'draw': match_probs['draw'],
            'away_win': match_probs['away_win'],
            'over_2_5': goal_probs['over_2_5'],
            'under_2_5': goal_probs['under_2_5'],
            'btts_yes': goal_probs['both_teams_score_yes'],
            'btts_no': goal_probs['both_teams_score_no'],
            'expected_goals': {
                'home': lambda_home,
                'away': lambda_away
            },
            'score_probabilities': scores,
            'confidence': round(confidence * 100, 2)
        }
        
    def _process_dixon_coles_results(self, matrix, lambda_home, lambda_away):
        """
        Dixon-Coles sonuçlarını işle
        """
        match_probs = self.dixon_coles.get_match_probabilities(matrix)
        
        # Gol tahminleri için Poisson fonksiyonlarını kullan
        goal_probs = self.poisson_model.get_goals_probabilities(matrix)
        scores = self.poisson_model.get_exact_score_probabilities(matrix)
        
        # Dinamik güven hesaplama
        max_prob = max(match_probs['home_win'], match_probs['draw'], match_probs['away_win'])
        
        # Tahmin keskinliğine göre güven (0.4-0.9 arası)
        if max_prob > 60:  # Çok net favori
            confidence = 0.75 + (max_prob - 60) / 100  # 0.75-0.95
        elif max_prob > 45:  # Orta düzey favori
            confidence = 0.65 + (max_prob - 45) / 75  # 0.65-0.75
        else:  # Dengeli maç
            confidence = 0.55 + (max_prob - 33) / 60  # 0.55-0.65
        
        # Dixon-Coles modeli için temel güven
        confidence = min(0.88, max(0.5, confidence))
        
        return {
            'home_win': match_probs['home_win'],
            'draw': match_probs['draw'],
            'away_win': match_probs['away_win'],
            'over_2_5': goal_probs['over_2_5'],
            'under_2_5': goal_probs['under_2_5'],
            'btts_yes': goal_probs['both_teams_score_yes'],
            'btts_no': goal_probs['both_teams_score_no'],
            'expected_goals': {
                'home': lambda_home,
                'away': lambda_away
            },
            'score_probabilities': scores,
            'confidence': round(confidence * 100, 2)
        }
        
    def _process_monte_carlo_results(self, results):
        """
        Monte Carlo sonuçlarını işle
        """
        # Dinamik güven hesaplama
        max_prob = max(results['outcomes']['home_win'], results['outcomes']['draw'], results['outcomes']['away_win'])
        
        # Tahmin keskinliğine göre güven (0.4-0.9 arası)
        if max_prob > 60:  # Çok net favori
            confidence = 0.68 + (max_prob - 60) / 100  # 0.68-0.88
        elif max_prob > 45:  # Orta düzey favori
            confidence = 0.58 + (max_prob - 45) / 75  # 0.58-0.68
        else:  # Dengeli maç
            confidence = 0.48 + (max_prob - 33) / 60  # 0.48-0.58
        
        # Monte Carlo modeli için temel güven
        confidence = min(0.82, max(0.45, confidence))
        
        return {
            'home_win': results['outcomes']['home_win'],
            'draw': results['outcomes']['draw'],
            'away_win': results['outcomes']['away_win'],
            'over_2_5': results['over_under']['over_2_5'],
            'under_2_5': results['over_under']['under_2_5'],
            'btts_yes': results['btts']['yes'],
            'btts_no': results['btts']['no'],
            'expected_goals': {
                'home': results['avg_home_goals'],
                'away': results['avg_away_goals']
            },
            'score_probabilities': self._convert_mc_scores(results['scores']),
            'confidence': round(confidence * 100, 2)
        }
        
    def _convert_mc_scores(self, scores_dict):
        """
        Monte Carlo skor dict'ini listeye çevir
        """
        scores_list = []
        for score, prob in sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
            scores_list.append({
                'score': score,
                'probability': prob
            })
        return scores_list
        
    def _format_prediction(self, final_pred, context, home_name, away_name, home_id, away_id, home_data, away_data, h2h_data=None, home_team_analysis=None, away_team_analysis=None, team_comparison=None, form_comparison=None, enhanced_features=None, league_analysis=None, psychological_analysis=None):
        """
        Tahmin sonuçlarını frontend formatına dönüştür (Phase 3 enhanced)
        """
        # En olası skor
        most_likely_score = "1-1"
        most_likely_prob = 0.0
        
        if 'most_likely_scores' in final_pred and final_pred['most_likely_scores']:
            most_likely = final_pred['most_likely_scores'][0]
            most_likely_score = most_likely['score']
            most_likely_prob = most_likely['probability']
            
        # Form analizi
        home_form = self._analyze_form(home_data.get('recent_matches', [])[:5])
        away_form = self._analyze_form(away_data.get('recent_matches', [])[:5])
        
        return {
            "match_info": {
                "home_team": {
                    "id": home_id,
                    "name": home_name
                },
                "away_team": {
                    "id": away_id,
                    "name": away_name
                },
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "predictions": {
                "most_likely_outcome": final_pred['most_likely_outcome'],
                "home_win_probability": round(final_pred['home_win'], 1),
                "draw_probability": round(final_pred['draw'], 1),
                "away_win_probability": round(final_pred['away_win'], 1),
                "most_likely_score": most_likely_score,
                "most_likely_score_probability": round(most_likely_prob, 1),
                "expected_goals": {
                    "home": round(final_pred['expected_goals']['home'], 2),
                    "away": round(final_pred['expected_goals']['away'], 2)
                },
                "over_under": {
                    "over_2_5": round(final_pred['over_2_5'], 1),
                    "under_2_5": round(final_pred['under_2_5'], 1)
                },
                "both_teams_to_score": {
                    "yes": round(final_pred['btts_yes'], 1),
                    "no": round(final_pred['btts_no'], 1)
                },
                "exact_scores": final_pred.get('most_likely_scores', []),
                # Frontend için gerekli ekstra alanlar
                "betting_predictions": self._generate_betting_predictions(final_pred),
                "most_confident_bet": self._get_most_confident_bet(final_pred),
                "most_likely_bet": self._get_most_likely_bet(final_pred),
                # İlk yarı tahminleri (HT/FT)
                "half_time_predictions": final_pred.get('advanced_predictions', {}).get('halftime_goals', {}),
                "half_time_full_time": final_pred.get('advanced_predictions', {}).get('htft', {}),
                # Yeni tahmin türleri
                "advanced_predictions": final_pred.get('advanced_predictions', {})
            },
            "team_stats": {
                "home": {
                    "form": home_form,
                    "elo_rating": round(context.get('home_elo', 1500)),
                    "xg": round(context['home_xg'], 2),
                    "xga": round(context['home_xga'], 2)
                },
                "away": {
                    "form": away_form,
                    "elo_rating": round(context.get('away_elo', 1500)),
                    "xg": round(context['away_xg'], 2),
                    "xga": round(context['away_xga'], 2)
                }
            },
            "confidence": round(final_pred['confidence'], 2),
            "algorithm": "Ensemble (Poisson + Dixon-Coles + XGBoost + Monte Carlo + CRF + Neural Network)",
            "elo_difference": round(context['elo_diff']) if not math.isnan(context.get('elo_diff', 0)) else 0,
            "analysis": self._generate_analysis(final_pred, context, home_name, away_name),
            # Açıklanabilir AI
            "explanation": None,  # Daha sonra eklenecek
            # Model performans raporu
            "model_performance": self.model_evaluator.get_model_performance_report(),
            "team_data": {
                "home": {
                    "form": home_form[:5] if home_form else [],
                    "avg_goals_scored": round(home_data['home_performance']['avg_goals'], 1),
                    "avg_goals_conceded": round(home_data['home_performance']['avg_conceded'], 1),
                    "avg_goals_scored_away": round(home_data['away_performance']['avg_goals'], 1),
                    "recent_form": ''.join(home_form[:5]) if home_form else "WWDLW",
                    "strength": self._calculate_team_strength(home_data, home_form),
                    "motivation": self._calculate_team_motivation(home_data, home_form, context.get('home_elo', 1500)),
                    "fatigue": self._calculate_team_fatigue(home_data),
                    "h2h_record": home_data.get('h2h_record', {"wins": 2, "draws": 1, "losses": 2})
                },
                "away": {
                    "form": away_form[:5] if away_form else [],
                    "avg_goals_scored": round(away_data['away_performance']['avg_goals'], 1),
                    "avg_goals_conceded": round(away_data['away_performance']['avg_conceded'], 1), 
                    "avg_goals_scored_away": round(away_data['away_performance']['avg_goals'], 1),
                    "recent_form": ''.join(away_form[:5]) if away_form else "LWDWL",
                    "strength": self._calculate_team_strength(away_data, away_form),
                    "motivation": self._calculate_team_motivation(away_data, away_form, context.get('away_elo', 1500)),
                    "fatigue": self._calculate_team_fatigue(away_data),
                    "h2h_record": away_data.get('h2h_record', {"wins": 2, "draws": 1, "losses": 2})
                }
            },
            # H2H verileri eklendi
            "h2h_data": {
                "matches": h2h_data.get('response', {}).get('matches', []) if h2h_data and h2h_data.get('success') else []
            },
            # Dynamic Team Analyzer verileri
            "dynamic_analysis": {
                "home_team": home_team_analysis if home_team_analysis else None,
                "away_team": away_team_analysis if away_team_analysis else None,
                "comparison": team_comparison if team_comparison else None
            },
            # Phase 3: Advanced Analytics
            "form_trend_analysis": form_comparison if form_comparison else None,
            "feature_importance": enhanced_features.get('feature_importance', {}) if enhanced_features else {},
            "enhanced_features": enhanced_features if enhanced_features else None,
            # League Strength Analysis
            "league_analysis": league_analysis if league_analysis else None,
            # Psychological Analysis (Enhanced)
            "psychological_analysis": self._format_psychological_analysis(psychological_analysis) if psychological_analysis else None
        }
        
    def _calculate_team_strength(self, team_data, form):
        """
        Takım gücünü dinamik olarak hesapla (0-100 arası)
        """
        base_strength = 50
        
        # Form bazlı güç (son 5 maç)
        if form:
            wins = form[:5].count('W')
            draws = form[:5].count('D')
            form_points = (wins * 3 + draws) / 15  # Max 15 puan mümkün
            base_strength += form_points * 20  # Max +20 puan
        
        # Gol performansı
        home_perf = team_data.get('home_performance', {})
        away_perf = team_data.get('away_performance', {})
        avg_goals = (home_perf.get('avg_goals', 1.2) + away_perf.get('avg_goals', 1.0)) / 2
        avg_conceded = (home_perf.get('avg_conceded', 1.3) + away_perf.get('avg_conceded', 1.5)) / 2
        
        # Gol farkı bazlı güç
        goal_diff = avg_goals - avg_conceded
        base_strength += goal_diff * 10  # Gol farkı başına +/-10 puan
        
        # Elo rating etkisi
        elo = team_data.get('elo_rating', 1500)
        elo_factor = (elo - 1500) / 50  # Her 50 Elo puanı için +/-1 güç puanı
        base_strength += elo_factor
        
        # 0-100 arasında sınırla
        return max(0, min(100, round(base_strength)))
    
    def _calculate_team_motivation(self, team_data, form, elo_rating):
        """
        Takım motivasyonunu dinamik olarak hesapla (0-100 arası)
        """
        base_motivation = 50
        
        # Son form trendi (momentum)
        if form and len(form) >= 3:
            recent_wins = form[:3].count('W')
            if recent_wins >= 2:
                base_motivation += 15  # Güçlü momentum
            elif recent_wins == 0 and form[:3].count('L') >= 2:
                base_motivation -= 10  # Kötü momentum
        
        # Gol atma performansı
        recent_matches = team_data.get('recent_matches', [])
        if recent_matches:
            recent_goals = sum(m.get('goals_scored', 0) for m in recent_matches)
            if recent_goals > 10:  # Son 5 maçta 10+ gol
                base_motivation += 10
            elif recent_goals < 3:  # Son 5 maçta 3'ten az gol
                base_motivation -= 10
        
        # Rakip kalitesi (Elo bazlı)
        if elo_rating > 1600:
            base_motivation += 5  # Güçlü takım bonusu
        elif elo_rating < 1400:
            base_motivation -= 5  # Zayıf takım cezası
        
        # 0-100 arasında sınırla
        return max(0, min(100, round(base_motivation)))
    
    def _calculate_team_fatigue(self, team_data):
        """
        Takım yorgunluğunu dinamik olarak hesapla (0-100 arası, yüksek = daha yorgun)
        """
        base_fatigue = 20
        
        recent_matches = team_data.get('recent_matches', [])
        if not recent_matches:
            return base_fatigue
        
        # Son 7 gündeki maç sayısı
        from datetime import datetime, timedelta
        today = datetime.now()
        matches_in_week = 0
        
        for match in recent_matches:
            match_date_str = match.get('date', '')
            if match_date_str:
                try:
                    match_date = datetime.strptime(match_date_str, '%Y-%m-%d')
                    if (today - match_date).days <= 7:
                        matches_in_week += 1
                except:
                    continue
        
        # Her ekstra maç için +15 yorgunluk
        if matches_in_week > 1:
            base_fatigue += (matches_in_week - 1) * 15
        
        # Seyahat faktörü (son 5 maçta deplasman sayısı)
        away_matches = sum(1 for m in recent_matches if not m.get('is_home', True))
        base_fatigue += away_matches * 5  # Her deplasman maçı için +5 yorgunluk
        
        # 0-100 arasında sınırla
        return max(0, min(100, round(base_fatigue)))
    
    def _format_psychological_analysis(self, psychological_analysis):
        """
        Psikolojik analiz sonuçlarını frontend formatına dönüştür
        """
        if not psychological_analysis:
            return None
            
        try:
            return {
                "match_importance": {
                    "score": round(psychological_analysis['match_importance_score'], 1),
                    "is_critical_match": psychological_analysis['critical_match_analysis']['is_critical_match'],
                    "critical_types": psychological_analysis['critical_match_analysis']['critical_types']
                },
                "team_motivation": {
                    "home": {
                        "total_score": psychological_analysis['motivation_analysis']['home_team']['total_motivation'],
                        "level": psychological_analysis['motivation_analysis']['home_team']['motivation_level'],
                        "factors": psychological_analysis['motivation_analysis']['home_team']['motivation_factors']
                    },
                    "away": {
                        "total_score": psychological_analysis['motivation_analysis']['away_team']['total_motivation'],
                        "level": psychological_analysis['motivation_analysis']['away_team']['motivation_level'],
                        "factors": psychological_analysis['motivation_analysis']['away_team']['motivation_factors']
                    },
                    "differential": psychological_analysis['motivation_analysis']['motivation_differential']
                },
                "pressure_analysis": {
                    "home": {
                        "level": psychological_analysis['pressure_analysis']['home_team']['pressure_level'],
                        "category": psychological_analysis['pressure_analysis']['home_team']['pressure_category'],
                        "crowd_pressure": psychological_analysis['pressure_analysis']['home_team']['crowd_pressure']
                    },
                    "away": {
                        "level": psychological_analysis['pressure_analysis']['away_team']['pressure_level'],
                        "category": psychological_analysis['pressure_analysis']['away_team']['pressure_category']
                    },
                    "high_pressure_match": psychological_analysis['pressure_analysis']['high_pressure_match']
                },
                "momentum": {
                    "home": {
                        "confidence": psychological_analysis['momentum_analysis']['home_team']['confidence_level'],
                        "momentum_score": psychological_analysis['momentum_analysis']['home_team']['momentum_score'],
                        "mental_fatigue": psychological_analysis['momentum_analysis']['home_team']['mental_fatigue']
                    },
                    "away": {
                        "confidence": psychological_analysis['momentum_analysis']['away_team']['confidence_level'],
                        "momentum_score": psychological_analysis['momentum_analysis']['away_team']['momentum_score'],
                        "mental_fatigue": psychological_analysis['momentum_analysis']['away_team']['mental_fatigue']
                    },
                    "advantage": psychological_analysis['momentum_analysis']['momentum_advantage']
                },
                "psychological_advantage": psychological_analysis['psychological_advantage'],
                "derby_analysis": psychological_analysis['critical_match_analysis'].get('derby_analysis', {}),
                "summary": {
                    "dominant_factors": psychological_analysis['overall_assessment']['dominant_factors'],
                    "home_psychological_score": psychological_analysis['overall_assessment']['home_psychological_score'],
                    "away_psychological_score": psychological_analysis['overall_assessment']['away_psychological_score']
                }
            }
        except Exception as e:
            logger.warning(f"Psikolojik analiz formatlamada hata: {e}")
            return {
                "match_importance": {"score": 5.0, "is_critical_match": False, "critical_types": []},
                "team_motivation": {
                    "home": {"total_score": 50, "level": "neutral_motivated", "factors": {}},
                    "away": {"total_score": 50, "level": "neutral_motivated", "factors": {}},
                    "differential": 0
                },
                "pressure_analysis": {
                    "home": {"level": 30, "category": "low_pressure", "crowd_pressure": 10},
                    "away": {"level": 30, "category": "low_pressure"},
                    "high_pressure_match": False
                },
                "momentum": {
                    "home": {"confidence": 50, "momentum_score": 50, "mental_fatigue": 30},
                    "away": {"confidence": 50, "momentum_score": 50, "mental_fatigue": 30},
                    "advantage": "balanced_momentum"
                },
                "psychological_advantage": "balanced_psychological_state",
                "derby_analysis": {},
                "summary": {"dominant_factors": [], "home_psychological_score": 50, "away_psychological_score": 50}
            }
    
    def _analyze_form(self, matches):
        """
        Son maçların form analizini yap
        """
        if not matches:
            return []
            
        form = []
        for match in matches:
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            if goals_for > goals_against:
                form.append('W')
            elif goals_for == goals_against:
                form.append('D')
            else:
                form.append('L')
                
        return form
        
    def _generate_analysis(self, prediction, context, home_name, away_name):
        """
        Tahmin analizi metni oluştur
        """
        analysis = []
        
        # Favori analizi
        if prediction['most_likely_outcome'] == 'HOME_WIN':
            fav_team = home_name
            fav_prob = prediction['home_win']
        elif prediction['most_likely_outcome'] == 'AWAY_WIN':
            fav_team = away_name
            fav_prob = prediction['away_win']
        else:
            fav_team = None
            fav_prob = prediction['draw']
            
        if fav_team:
            analysis.append(f"{fav_team} maçın favorisi (%{fav_prob:.0f} kazanma şansı)")
        else:
            analysis.append(f"Dengeli bir maç bekleniyor (%{fav_prob:.0f} beraberlik olasılığı)")
            
        # Gol analizi
        total_goals = prediction['expected_goals']['home'] + prediction['expected_goals']['away']
        if total_goals > 2.5:
            analysis.append(f"Gollü bir maç bekleniyor (Ort. {total_goals:.1f} gol)")
        else:
            analysis.append(f"Düşük skorlu bir maç olabilir (Ort. {total_goals:.1f} gol)")
            
        # KG analizi
        if prediction['btts_yes'] > 60:
            analysis.append(f"Her iki takımın da gol atma ihtimali yüksek (%{prediction['btts_yes']:.0f})")
            
        # Elo analizi
        elo_diff = abs(context['elo_diff'])
        if elo_diff > 300:
            analysis.append("Takımlar arasında belirgin bir güç farkı var")
        elif elo_diff < 100:
            analysis.append("Takımlar güç olarak birbirine yakın")
            
        return " ".join(analysis)
        
    def _get_cached_prediction(self, cache_key):
        """
        Önbellekten tahmin al
        """
        if cache_key in self.cache_data:
            # 1 saatten eski önbellekleri yoksay
            cache_time = self.cache_data[cache_key].get('timestamp', 0)
            if time.time() - cache_time > 3600:
                return None
            
            return self.cache_data[cache_key]
                
        return None
        
    def _cache_prediction(self, cache_key, prediction):
        """
        Tahmini önbelleğe kaydet
        """
        try:
            # Timestamp ekle
            prediction['timestamp'] = time.time()
            
            # Önbelleğe ekle
            self.cache_data[cache_key] = prediction
            
            # Dosyaya kaydet
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Önbellek kayıt hatası: {e}")
    
    def get_async_predictions(self, match_ids):
        """
        Birden çok maç için asenkron tahmin
        """
        import asyncio
        from async_data_fetcher import run_async_workflow
        
        logger.info(f"{len(match_ids)} maç için asenkron tahmin başlatılıyor")
        
        # API anahtarını al
        from api_config import APIConfig
        api_config = APIConfig()
        api_key = api_config.get_api_key()
        
        # Asenkron workflow'u çalıştır
        results = run_async_workflow(
            match_ids, 
            api_key, 
            lambda match_data: self.predict_match(
                match_data['home_team_id'],
                match_data['away_team_id']
            )
        )
        
        return results
            
    def _get_fallback_prediction(self, home_id, away_id, home_name, away_name):
        """
        Hata durumunda basit tahmin döndür
        """
        return {
            "match_info": {
                "home_team": {"id": home_id, "name": home_name},
                "away_team": {"id": away_id, "name": away_name},
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "predictions": {
                "most_likely_outcome": "DRAW",
                "home_win_probability": 33.3,
                "draw_probability": 33.4,
                "away_win_probability": 33.3,
                "most_likely_score": "1-1",
                "most_likely_score_probability": 10.0,
                "expected_goals": {"home": 1.2, "away": 1.2},
                "over_under": {"over_2_5": 45.0, "under_2_5": 55.0},
                "both_teams_to_score": {"yes": 50.0, "no": 50.0}
            },
            "confidence": 0.5,
            "algorithm": "Fallback (Basit tahmin)",
            "error": True
        }
        
    def clear_cache(self):
        """
        Önbellek temizleme
        """
        try:
            self.cache_data = {}
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            logger.info("Önbellek temizlendi")
            return True
        except Exception as e:
            logger.error(f"Önbellek temizleme hatası: {e}")
            return False
            
    def _generate_betting_predictions(self, prediction):
        """
        Frontend için bahis tahminlerini oluştur
        """
        betting_preds = {}
        
        # Maç sonucu
        betting_preds['match_result'] = {
            'prediction': prediction['most_likely_outcome'],
            'probability': max(prediction['home_win'], prediction['draw'], prediction['away_win'])
        }
        
        # KG Var/Yok - her zaman yüksek olasılığı göster
        if prediction['btts_yes'] > prediction['btts_no']:
            betting_preds['both_teams_to_score'] = {
                'prediction': 'YES',
                'probability': prediction['btts_yes']
            }
        else:
            betting_preds['both_teams_to_score'] = {
                'prediction': 'NO',
                'probability': prediction['btts_no']
            }
        
        # 2.5 Üst/Alt - her zaman yüksek olasılığı göster
        if prediction['over_2_5'] > prediction['under_2_5']:
            betting_preds['over_2_5_goals'] = {
                'prediction': 'YES',
                'probability': prediction['over_2_5']
            }
        else:
            betting_preds['over_2_5_goals'] = {
                'prediction': 'NO',
                'probability': prediction['under_2_5']
            }
        
        # 3.5 Üst/Alt - her zaman yüksek olasılığı göster
        over_3_5 = prediction.get('over_3_5', prediction['over_2_5'] * 0.7)  # Tahmini değer
        under_3_5 = 100 - over_3_5
        if over_3_5 > under_3_5:
            betting_preds['over_3_5_goals'] = {
                'prediction': 'YES',
                'probability': over_3_5
            }
        else:
            betting_preds['over_3_5_goals'] = {
                'prediction': 'NO',
                'probability': under_3_5
            }
        
        # Kesin skor
        if prediction.get('most_likely_scores'):
            betting_preds['exact_score'] = {
                'prediction': prediction['most_likely_scores'][0]['score'],
                'probability': prediction['most_likely_scores'][0]['probability']
            }
        else:
            betting_preds['exact_score'] = {
                'prediction': '1-1',
                'probability': 10.0
            }
            
        return betting_preds
        
    def _get_most_confident_bet(self, prediction):
        """
        En yüksek olasılıklı bahis tahmini
        """
        all_bets = []
        
        # Maç sonucu
        all_bets.append({
            'market': 'match_result',
            'prediction': prediction['most_likely_outcome'],
            'probability': max(prediction['home_win'], prediction['draw'], prediction['away_win'])
        })
        
        # KG Var/Yok - her zaman yüksek olasılığı göster
        if prediction['btts_yes'] > prediction['btts_no']:
            btts_pred = 'YES'
            btts_prob = prediction['btts_yes']
        else:
            btts_pred = 'NO'
            btts_prob = prediction['btts_no']
            
        all_bets.append({
            'market': 'both_teams_to_score',
            'prediction': btts_pred,
            'probability': btts_prob
        })
        
        # 2.5 Üst/Alt - her zaman yüksek olasılığı göster
        if prediction['over_2_5'] > prediction['under_2_5']:
            over_pred = 'YES'
            over_prob = prediction['over_2_5']
        else:
            over_pred = 'NO'
            over_prob = prediction['under_2_5']
            
        all_bets.append({
            'market': 'over_2_5_goals',
            'prediction': over_pred,
            'probability': over_prob
        })
        
        # En yüksek olasılıklı olanı seç
        return max(all_bets, key=lambda x: x['probability'])
        
    def _get_most_likely_bet(self, prediction):
        """
        En olası bahis (frontend uyumluluk için)
        """
        confident = self._get_most_confident_bet(prediction)
        return f"{confident['market']}:{confident['prediction']}"
    
    def _get_team_league_from_api(self, team_id):
        """API'den takım detaylarını alarak ulusal ligi bul"""
        try:
            # API football get_teams metodunu kullan
            params = {
                'action': 'get_teams',
                'team_id': team_id,
                'APIkey': self.api_key
            }
            
            response = requests.get(
                'https://v3.football.api-sports.io/teams',
                params={'id': team_id},
                headers={'x-apisports-key': self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"API'den takım bilgisi geldi - team_id: {team_id}")
                if data and isinstance(data, list) and len(data) > 0:
                    team_info = data[0]
                    logger.info(f"Takım detayları: {team_info.get('team_name')} - {team_info.get('team_country')}")
                    # Takımın ülkesini al
                    team_country = team_info.get('team_country', '')
                    
                    # Şimdi bu ülkenin liglerini al
                    country_leagues = self._get_country_leagues(team_country, team_id)
                    
                    if country_leagues:
                        logger.info(f"Takım {team_id} ({team_info.get('team_name', '')}) için lig bulundu: {country_leagues}")
                        return country_leagues
                    
            return None
            
        except Exception as e:
            logger.error(f"API'den takım ligi alınırken hata: {e}")
            return None
    
    def _get_country_leagues(self, country_name, team_id):
        """Ülkenin liglerini al ve takımın hangi ligde olduğunu bul"""
        try:
            # Önce ülke ID'sini bul
            country_params = {
                'action': 'get_countries',
                'APIkey': self.api_key
            }
            
            response = requests.get(
                'https://v3.football.api-sports.io/countries',
                headers={'x-apisports-key': self.api_key},
                timeout=10
            )
            
            country_id = None
            if response.status_code == 200:
                countries = response.json()
                for country in countries:
                    if country.get('country_name', '').lower() == country_name.lower():
                        country_id = country.get('country_id')
                        break
            
            if not country_id:
                return None
            
            # Ülkenin liglerini al
            league_params = {
                'action': 'get_leagues',
                'country_id': country_id,
                'APIkey': self.api_key
            }
            
            response = requests.get(
                'https://v3.football.api-sports.io/leagues',
                headers={'x-apisports-key': self.api_key},
                params={'country': country_name},
                timeout=10
            )
            
            if response.status_code == 200:
                leagues = response.json()
                
                # ÖZEL DURUM: Ülke bazlı öncelik ligi
                # Bu ülkelerin birden fazla büyük ligi var - en güçlüsünü seç
                country_priority_leagues = {
                    'england': 'Premier League',
                    'spain': 'La Liga',
                    'germany': 'Bundesliga',
                    'italy': 'Serie A',
                    'france': 'Ligue 1',
                    'portugal': 'Primeira Liga',
                    'netherlands': 'Eredivisie',
                    'turkey': 'Süper Lig',
                    'türkiye': 'Süper Lig',
                    'belgium': 'First Division A',
                    'scotland': 'Scottish Premiership',
                    'austria': 'Austrian Bundesliga',
                    'switzerland': 'Swiss Super League',
                    'greece': 'Super League',
                    'denmark': 'Danish Superliga',
                    'norway': 'Eliteserien',
                    'sweden': 'Allsvenskan',
                }
                
                country_key = country_name.lower()
                if country_key in country_priority_leagues:
                    priority_league = country_priority_leagues[country_key]
                    # API'den gelen liglerde bu ligi ara
                    for league in leagues:
                        if priority_league.lower() in league.get('league_name', '').lower():
                            logger.info(f"Öncelikli lig bulundu: {league.get('league_name')} ({country_name})")
                            return league.get('league_name')
                
                # En üst seviye ulusal ligi bul (cup veya super cup olmayanlar)
                national_leagues = []
                for league in leagues:
                    league_name = league.get('league_name', '').lower()
                    if ('cup' not in league_name and 'super' not in league_name and 
                        'copa' not in league_name and 'women' not in league_name and
                        'u19' not in league_name and 'u21' not in league_name and
                        'championship' not in league_name):  # 2. ligleri atla
                        national_leagues.append(league.get('league_name'))
                
                # İlk ulusal ligi döndür (genelde en üst lig)
                if national_leagues:
                    return national_leagues[0]
                    
            return None
            
        except Exception as e:
            logger.error(f"Ülke ligleri alınırken hata: {e}")
            return None
    
    def _load_league_ids(self):
        """Load league ID mappings from config"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'league_ids.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                league_ids = json.load(f)
                logger.info(f"✅ League ID mappings loaded: {len(league_ids.get('league_strength_scores', {}))} leagues")
                
                # Add known team-to-league fallback mapping for when API data is missing
                league_ids['team_league_fallback'] = {
                    # Premier League teams (Liverpool, Manchester clubs, etc.)
                    '40': 152, '33': 152, '34': 152, '84': 152,
                    # La Liga teams (Real Madrid, Barcelona, etc.)
                    '76': 302, '77': 302,
                    # Süper Lig teams (Galatasaray, Fenerbahçe, etc.)
                    '192': 237, '193': 237, '609': 237,
                }
                
                return league_ids
        except Exception as e:
            logger.error(f"Failed to load league IDs: {e}")
            return {}
    
    def _get_league_id(self, league_name):
        """Convert league name to league ID"""
        if not league_name:
            return None
        
        # Try direct lookup
        league_name_map = self.league_ids.get('league_names_to_ids', {})
        if league_name in league_name_map:
            return league_name_map[league_name]
        
        # Try fuzzy match (lowercase, partial)
        league_name_lower = league_name.lower()
        for name, league_id in league_name_map.items():
            if league_name_lower in name.lower() or name.lower() in league_name_lower:
                return league_id
        
        return None
    
    def _is_uefa_competition(self, league_id):
        """Check if league ID is a UEFA competition"""
        if not league_id:
            return False
        uefa_comps = self.league_ids.get('uefa_competitions', {})
        return league_id in uefa_comps.values()
    
    def _get_league_strength_score(self, league_id):
        """Get league strength score by league ID"""
        if not league_id:
            return 50  # Default mid-tier
        
        scores = self.league_ids.get('league_strength_scores', {})
        return scores.get(str(league_id), 50)
    
    def _extract_league_info(self, team_data):
        """Takım verilerinden ULUSAL lig bilgisini çıkar (yedek metod)"""
        # Kupa ve uluslararası turnuva isimleri (bunları atlayacağız)
        cup_keywords = ['Cup', 'UEFA', 'Champions', 'Europa', 'Conference', 'Friendlies', 
                       'World Cup', 'Euro', 'Copa', 'International', 'Nations League',
                       'Kupa', 'Kupası', 'Shield', 'Trophy', 'Supercup', 'Super Cup',
                       'Qualification', 'Qualifying', 'Play-off', 'Playoff']
        
        # Önce son maçlardan ULUSAL lig bilgisi almayı dene
        recent_matches = team_data.get('recent_matches', [])
        if recent_matches:
            # Lig adlarını say ve en çok kullanılanı bul
            league_counts = {}
            
            for match in recent_matches[:20]:  # Son 20 maçı kontrol et
                # API'den farklı alanlarda gelebilir
                league = match.get('league', '') or match.get('league_name', '') or match.get('match_league', '')
                if league and league != 'Unknown' and league != '':
                    # Kupa maçı mı kontrol et
                    is_cup = any(keyword.lower() in league.lower() for keyword in cup_keywords)
                    if not is_cup:
                        # Ulusal lig sayacını artır
                        league_counts[league] = league_counts.get(league, 0) + 1
            
            # En çok oynanan ulusal ligi döndür
            if league_counts:
                most_common_league = max(league_counts, key=league_counts.get)
                logger.info(f"Takımın ulusal ligi bulundu: {most_common_league} ({league_counts[most_common_league]} maç)")
                return most_common_league
        
        # Takım bilgilerinde lig adı varsa ve kupa değilse
        if 'league_name' in team_data:
            league = team_data.get('league_name', '')
            if league:
                is_cup = any(keyword.lower() in league.lower() for keyword in cup_keywords)
                if not is_cup:
                    return league
        
        # H2H verisinde lig bilgisi
        if 'competition' in team_data:
            league = team_data['competition']
            is_cup = any(keyword.lower() in league.lower() for keyword in cup_keywords)
            if not is_cup:
                return league
        
        # API yanıtının ilk maçından lig bilgisi
        if 'all_matches' in team_data and team_data['all_matches']:
            for match in team_data['all_matches'][:10]:
                league = match.get('league_name', '') or match.get('league', '')
                if league:
                    is_cup = any(keyword.lower() in league.lower() for keyword in cup_keywords)
                    if not is_cup:
                        return league
        
        # Varsayılan
        logger.warning("Takımın ulusal ligi bulunamadı")
        return "Unknown League"
    
    def _prepare_venue_info(self, home_data, away_data, home_league):
        """Prepare venue information for venue performance optimizer"""
        try:
            # Extract venue information from available data
            # home_league can be string or dict, handle both cases
            if isinstance(home_league, dict):
                league_name = home_league.get('name', 'Unknown')
                league_id = home_league.get('id')
            elif isinstance(home_league, str):
                league_name = home_league
                league_id = None
            else:
                league_name = 'Unknown'
                league_id = None
            
            venue_info = {
                'name': home_data.get('venue_name') or f"{home_data.get('team_name', 'Unknown')} Home",
                'id': f"venue_{home_data.get('team_id', 'unknown')}",
                'city': home_data.get('city') or 'Unknown',
                'country': home_data.get('country_name') or 'Unknown',
                'league': league_name,
                'league_id': league_id,
                'capacity': home_data.get('stadium_capacity', 30000),
                'coordinates': self._get_venue_coordinates(home_data),
                'altitude': home_data.get('altitude', 100),
                'surface': home_data.get('surface_type', 'grass'),
                'roof_type': home_data.get('roof_type', 'open'),
                'atmosphere_rating': home_data.get('atmosphere_rating', 7.0)
            }
            
            return venue_info
            
        except Exception as e:
            logger.warning(f"Error preparing venue info: {e}")
            return {
                'name': 'Unknown Venue',
                'id': 'unknown',
                'city': 'Unknown',
                'country': 'Unknown',
                'capacity': 30000,
                'coordinates': (41.0, 29.0),  # Default to Istanbul
                'altitude': 100,
                'surface': 'grass'
            }
    
    def _get_venue_coordinates(self, team_data):
        """Get venue coordinates based on team data"""
        # This could be enhanced with a proper venue database
        country = team_data.get('country_name', '').lower()
        city = team_data.get('city', '').lower()
        
        # Default coordinates for major football countries/cities
        default_coordinates = {
            'england': (51.5074, -0.1278),    # London
            'spain': (40.4168, -3.7038),      # Madrid
            'italy': (41.9028, 12.4964),      # Rome
            'germany': (52.5200, 13.4050),    # Berlin
            'france': (48.8566, 2.3522),      # Paris
            'turkey': (41.0082, 28.9784),     # Istanbul
            'portugal': (38.7223, -9.1393),   # Lisbon
            'netherlands': (52.3676, 4.9041), # Amsterdam
        }
        
        # Try to get coordinates based on country
        for country_key, coords in default_coordinates.items():
            if country_key in country:
                return coords
        
        # Default to Istanbul if country not found
        return (41.0, 29.0)
    
    def _apply_venue_effects_to_xg(self, home_xg, away_xg, venue_analysis):
        """Apply venue effects to expected goals"""
        if not venue_analysis:
            return home_xg, away_xg
        
        try:
            # Get venue adjustment factors
            home_boost = venue_analysis['performance_predictions'].get('home_team_boost', 1.1)
            away_penalty = venue_analysis['performance_predictions'].get('away_team_penalty', 0.95)
            
            # Apply adjustments
            adjusted_home_xg = home_xg * home_boost
            adjusted_away_xg = away_xg * away_penalty
            
            logger.info(f"Venue xG adjustments applied - Home: {home_xg:.2f} -> {adjusted_home_xg:.2f}, "
                       f"Away: {away_xg:.2f} -> {adjusted_away_xg:.2f}")
            
            return adjusted_home_xg, adjusted_away_xg
            
        except Exception as e:
            logger.warning(f"Error applying venue effects to xG: {e}")
            return home_xg, away_xg
        
    
    def _extract_league_name(self, league_info):
        """Extract league name string from league info (can be dict or string)"""
        if isinstance(league_info, dict):
            return league_info.get('name', league_info.get('league_name', 'Unknown'))
        elif isinstance(league_info, str):
            return league_info
        else:
            return 'Unknown'
