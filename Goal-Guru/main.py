import logging
import os
import requests
import threading
import socket
import time
from datetime import datetime, timedelta
import pytz
from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask_caching import Cache
from match_prediction import MatchPredictor
from api_config import api_config
from database.connection import get_db_manager, db_error_handler
from database.init_db import verify_database, init_database
from error_handling import register_error_handlers, handle_errors
from performance.parallel_processor import ParallelProcessor, BatchPredictionManager
from performance.cache_manager import CacheManager, cached
# API v2 ve WebSocket özellikleri kaldırıldı
# from api.api_enhancement import create_enhanced_api_blueprint, setup_swagger_ui
# from realtime.websocket_server import create_websocket_server
# Create and load api_routes only after setting up the Flask app
# This avoids circular imports
api_v3_bp = None  # Will be set after app creation
# from model_validation import ModelValidator  # KALDIRILDI
# from dynamic_team_analyzer import DynamicTeamAnalyzer  # KALDIRILDI
# from team_performance_updater import TeamPerformanceUpdater  # KALDIRILDI
# from self_learning_predictor import SelfLearningPredictor  # KALDIRILDI

# Global değişkenler temizlendi - sıfırdan başlangıç
# team_analyzer = None  # KALDIRILDI
# self_learning_model = None  # KALDIRILDI  
# performance_updater = None  # KALDIRILDI

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Flask-Caching konfigürasyonu
cache_config = {
    "CACHE_TYPE": "SimpleCache",  # Basit bellek içi önbellek
    "CACHE_DEFAULT_TIMEOUT": 300,  # Varsayılan 5 dakika (300 saniye) önbellek süresi
    "CACHE_THRESHOLD": 500,        # Maksimum önbellek öğe sayısı
}
cache = Cache(app, config=cache_config)

# Initialize performance modules
parallel_processor = ParallelProcessor(max_workers=4, max_api_concurrent=10)
batch_manager = BatchPredictionManager(parallel_processor)
perf_cache_manager = CacheManager()

logger.info("Performance modules initialized")

# API Blueprint'leri kaydet - moved below
# api_v3_bp will be imported after app creation

# Tahmin modelini oluştur
predictor = MatchPredictor()

# Initialize database
db_manager = get_db_manager()

# Verify and initialize database on startup
if not verify_database():
    logger.info("Initializing database...")
    init_database()
else:
    logger.info("Database already initialized")

# Register error handlers
register_error_handlers(app)

# API Blueprint'i import et ve kaydet
try:
    from api_routes import api_v3_bp
    app.register_blueprint(api_v3_bp)
    logger.info("API v3 Blueprint başarıyla kaydedildi")
except Exception as e:
    logger.error(f"API Blueprint kaydedilemedi: {str(e)}")

# WebSocket ve API v2 dokümantasyon özellikleri kaldırıldı
# Kullanıcı talebi üzerine gereksiz oldukları için devre dışı bırakıldı
websocket_components = None

# Model doğrulama kaldırıldı - temiz başlangıç
# model_validator = ModelValidator(predictor)  # KALDIRILDI

def get_matches(selected_date=None):
    try:
        # Create timezone objects
        utc = pytz.UTC
        turkey_tz = pytz.timezone('Europe/Istanbul')

        if not selected_date:
            selected_date = datetime.now().strftime('%Y-%m-%d')

        matches = []
        api_key = api_config.get_api_key()

        # Get matches from new API-Football
        url = "https://v3.football.api-sports.io/fixtures"
        headers = {'x-apisports-key': api_key}
        params = {
            'date': selected_date,
            'timezone': 'Europe/Istanbul'
        }
        logger.info(f"Sending API request to {url} with params: {params}")

        logger.info(f"Fetching matches for date: {selected_date}")
        response = requests.get(url, headers=headers, params=params, timeout=30)
        logger.info(f"API Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"API response received. Type: {type(data)}")
            
            # Extract matches from response
            matches_list = data.get('response', []) if isinstance(data, dict) else []
            
            # Debug: Log first 5 matches
            if isinstance(matches_list, list) and len(matches_list) > 0:
                logger.info(f"Total matches from API: {len(matches_list)}")
                logger.info(f"First 3 matches for debugging:")
                for i, match in enumerate(matches_list[:3]):
                    league_name = match.get('league', {}).get('name', 'NO_LEAGUE')
                    home_name = match.get('teams', {}).get('home', {}).get('name', 'NO_HOME')
                    away_name = match.get('teams', {}).get('away', {}).get('name', 'NO_AWAY')
                    logger.info(f"  Match {i+1}: {league_name} -  {home_name} vs {away_name}")
                
                # Count leagues
                leagues_found = set()
                for match in matches_list:
                    league_id = match.get('league', {}).get('id', 'unknown')
                    league_name = match.get('league', {}).get('name', 'unknown')
                    leagues_found.add(f"{league_id}:{league_name}")
                logger.info(f"Unique leagues found in API response: {len(leagues_found)}")
                for league in list(leagues_found)[:5]:  # Show first 5
                    logger.info(f"  - {league}")
            elif matches_list == []:
                logger.warning(f"API returned empty data array. Using Key: {api_key[:5]}...{api_key[-3:]}")

            if isinstance(matches_list, list):
                logger.info(f"Processing {len(matches_list)} matches...")
                for match in matches_list:
                    match_obj = process_match(match, utc, turkey_tz)
                    if match_obj:
                        matches.append(match_obj)
                        logger.debug(f"Added match: {match_obj['competition']['name']} - {match_obj['homeTeam']['name']} vs {match_obj['awayTeam']['name']}")
            elif isinstance(data, dict) and 'errors' in data:
               logger.error(f"API returned error: {data.get('errors', 'Unknown error')}")

        # Group matches by league
        league_matches = {}
        for match in matches:
            league_id = match['competition']['id']
            league_name = match['competition']['name']

            if league_id not in league_matches:
                league_matches[league_id] = {
                    'name': league_name,
                    'matches': []
                }
            league_matches[league_id]['matches'].append(match)

        # Sort matches within each league
        for league_data in league_matches.values():
            league_data['matches'].sort(key=lambda x: (
                0 if x['is_live'] else (1 if x['status'] == 'FINISHED' else 2),
                x['turkish_time']
            ))

        # Format leagues for template
        formatted_leagues = []
        for league_id, league_data in league_matches.items():
            formatted_leagues.append({
                'id': league_id,
                'name': league_data['name'],
                'matches': league_data['matches'],
                'priority': get_league_priority(league_id)
            })

        # Sort leagues by priority (high to low) and then by name
        formatted_leagues.sort(key=lambda x: (-x['priority'], x['name']))

        logger.info(f"Total leagues found: {len(formatted_leagues)}")
        for league in formatted_leagues:
            logger.info(f"League: {league['name']} - {len(league['matches'])} matches")

        return {'leagues': formatted_leagues}

    except Exception as e:
        logger.error(f"Error fetching matches: {str(e)}")
        return {'leagues': []}

def get_league_priority(league_id):
    """Return priority for league sorting. Higher number means higher priority."""

    # Convert league_id to string for comparison
    league_id_str = str(league_id)

    # Top leagues with correct API-Football v3 IDs
    # Priority: UEFA > Top 5 Leagues > Turkish League > Other Notable Leagues
    favorite_leagues = {
        # UEFA Competitions (Highest Priority)
        "2": 100,     # UEFA Champions League
        "3": 99,      # UEFA Europa League
        "848": 98,    # UEFA Europa Conference League
        "531": 97,    # UEFA Super Cup
        
        # Top 5 European Leagues
        "39": 95,     # England - Premier League
        "140": 94,    # Spain - La Liga
        "135": 93,    # Italy - Serie A
        "78": 92,     # Germany - Bundesliga
        "61": 91,     # France - Ligue 1
        
        # Turkish Super Lig (High Priority for user)
        "203": 90,    # Turkey - Süper Lig
        
        # Other Top European Leagues
        "94": 85,     # Portugal - Primeira Liga
        "88": 84,     # Netherlands - Eredivisie
        "144": 83,    # Belgium - Pro League
        "179": 82,    # Scotland - Premiership
        
        # South American Top Leagues
        "71": 80,     # Brazil - Série A
        "128": 79,    # Argentina - Primera División
        
        # Other Notable Leagues
        "2": 75,      # FIFA World Cup
        "4": 74,      # UEFA Euro Championship
        "1": 73,      # World Cup Qualification
        
        # Turkish Cups
        "206": 70,    # Turkey - Türkiye Kupası
        "552": 69,    # Turkey - Super Cup
        
        # AFC Competitions  
        "17": 65,     # AFC Champions League
        "18": 64,     # AFC Cup
        
        # African Competitions
        "6": 60,      # Africa Cup of Nations
        "20": 59,     # CAF Champions League
    }

    # Direct ID check
    if league_id_str in favorite_leagues:
        return favorite_leagues[league_id_str]

    return 0

def process_match(match, utc, turkey_tz):
    try:
        # New API format has nested structure
        fixture = match.get('fixture', {})
        teams = match.get('teams', {})
        goals = match.get('goals', {})
        league = match.get('league', {})
        score = match.get('score', {})
        
        # Get team names from new format
        home_team = teams.get('home', {})
        away_team = teams.get('away', {})
        home_name = home_team.get('name', '')
        away_name = away_team.get('name', '')

        if not home_name or not away_name:
            return None

        # Get match date and time from fixture
        fixture_date = fixture.get('date', '')  # ISO format: 2025-12-23T20:00:00+03:00
        
        # Parse date and time
        match_date = ''
        turkish_time_str = "Belirlenmedi"
        try:
            if fixture_date:
                # Parse ISO datetime
                from datetime import datetime as dt
                if 'T' in fixture_date:
                    date_part = fixture_date.split('T')[0]
                    time_part = fixture_date.split('T')[1][:5]  # Get HH:MM
                    match_date = date_part
                    turkish_time_str = time_part
                else:
                    match_date = fixture_date
        except Exception as e:
            logger.error(f"Date parsing error: {e}")

        # Log for debugging
        logger.debug(f"Processing match: {home_name} vs {away_name} at {turkish_time_str}")

        # Get match status
        status_info = fixture.get('status', {})
        status_short = status_info.get('short', '')  # FT, NS, 1H, 2H, HT, etc.
        status_elapsed = status_info.get('elapsed', 0)
        
        home_score = goals.get('home')
        away_score = goals.get('away')
        is_live = False
        live_minute = ''

        # Determine match status
        if status_short == 'FT':  # Finished
            is_live = False
        elif status_short in ['1H', '2H', 'HT', 'ET', 'P', 'LIVE']:
            is_live = True
            live_minute = str(status_elapsed) if status_elapsed else ''
        elif status_short == 'NS':  # Not Started
            is_live = False
            home_score = None
            away_score = None

        return {
            'id': str(fixture.get('id', '')),
            'competition': {
                'id': str(league.get('id', '')),
                'name': league.get('name', '')
            },
            'utcDate': match_date,
            'status': 'LIVE' if is_live else ('FINISHED' if status_short == 'FT' else 'SCHEDULED'),
            'homeTeam': {
                'name': home_name,
                'id': str(home_team.get('id', ''))
            },
            'awayTeam': {
                'name': away_name,
                'id': str(away_team.get('id', ''))
            },
            'score': {
                'fullTime': {
                    'home': home_score if home_score is not None else 0,
                    'away': away_score if away_score is not None else 0
                },
                'halfTime': {
                    'home': score.get('halftime', {}).get('home', '-'),
                    'away': score.get('halftime', {}).get('away', '-')
                }
            },
            'turkish_time': turkish_time_str,
            'is_live': is_live,
            'live_minute': live_minute
        }

    except Exception as e:
        logger.error(f"Error processing match: {str(e)}")
        return None

@app.route('/')
@cache.cached(timeout=60, query_string=True)  # 1 dakika önbellek (daha kısa), query string parametrelerine duyarlı
def index():
    """
    Ana sayfa - Günün maçlarını listeler
    Cache ile performans artırılmıştır (1 dakika önbellek)
    query_string=True sayesinde farklı tarihler için farklı önbellek oluşturulur
    """
    selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    logger.info(f"Ana sayfa yükleniyor - Tarih: {selected_date}")
    cache.delete(f"view//{selected_date}")  # Eski cache'i temizle
    start_time = datetime.now()
    matches_data = get_matches(selected_date)
    elapsed_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Maç listesi yüklendi, süre: {elapsed_time:.2f} saniye, Toplam lig: {len(matches_data.get('leagues', []))}")
    return render_template('index.html', matches=matches_data, selected_date=selected_date)

@app.route('/api/team-stats/<team_id>')
def team_stats(team_id):
    try:
        # APIFootball API anahtarı
        api_key = api_config.get_api_key()

        # GÜNCEL VERİLER: Son 60 günlük maçları al (2025 verileri)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # Son 60 gün (güncel veriler)

        # Get team matches from new API-Football
        url = "https://v3.football.api-sports.io/fixtures"
        headers = {'x-apisports-key': api_key}
        params = {
            'team': team_id,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'timezone': 'Europe/Istanbul'
        }

        logger.debug(f"Fetching team stats for team_id: {team_id}")
        response = requests.get(url, params=params)
        logger.debug(f"API Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            matches = data.get('response', []) if isinstance(data, dict) else []
            logger.debug(f"Total matches found: {len(matches)}")

            # 2025 VERİLERİNİ FİLTRELE
            current_year = datetime.now().year
            filtered_matches = []
            for match in matches:
                fixture = match.get('fixture', {})
                match_date = fixture.get('date', '')
                if match_date and str(current_year) in match_date:
                    filtered_matches.append(match)
            
            logger.info(f"Takım {team_id}: Toplam {len(matches)} maçtan {len(filtered_matches)} tanesi 2025 verisi")
            
            # Maçları tarihe göre sırala (en yeniden en eskiye)
            filtered_matches.sort(key=lambda x: x.get('fixture', {}).get('date', ''), reverse=True)

            # Son 10 maçı al ve formatla (daha fazla güncel veri)
            last_matches = []
            for match in filtered_matches[:10]:  # Son 10 güncel maç
                fixture = match.get('fixture', {})
                teams = match.get('teams', {})
                goals = match.get('goals', {})
                
                match_date_iso = fixture.get('date', '')
                # Extract date from ISO format
                match_date = match_date_iso.split('T')[0] if 'T' in match_date_iso else match_date_iso
                try:
                    # Tarihi düzgün formata çevir
                    date_obj = datetime.strptime(match_date, '%Y-%m-%d')
                    formatted_date = date_obj.strftime('%d.%m.%Y')
                except ValueError:
                    formatted_date = match_date

                home_name = teams.get('home', {}).get('name', '')
                away_name = teams.get('away', {}).get('name', '')
                home_score = goals.get('home', 0)
                away_score = goals.get('away', 0)
                
                match_data = {
                    'date': formatted_date,
                    'match': f"{home_name} vs {away_name}",
                    'score': f"{home_score} - {away_score}"
                }
                last_matches.append(match_data)

            return jsonify(last_matches)

        return jsonify([])

    except Exception as e:
        logger.error(f"Error fetching team stats: {str(e)}")
        return jsonify([])


@app.route('/test_half_time_stats')
def test_half_time_stats():
    """Test sayfası - İlk yarı/ikinci yarı istatistiklerini test etmek için"""
    return render_template('half_time_stats_test.html')
    
def get_league_standings(league_id):
    """Get standings for a specific league"""
    try:
        logger.info(f"Attempting to fetch standings for league_id: {league_id}")

        api_key = os.environ.get('FOOTBALL_DATA_API_KEY')
        if not api_key:
            logger.error("FOOTBALL_DATA_API_KEY is not set")
            return None

        # Football-data.org API endpoint
        url = f"https://api.football-data.org/v4/competitions/{league_id}/standings"
        headers = {'X-Auth-Token': api_key}

        logger.info(f"Making API request to {url}")
        response = requests.get(url, headers=headers)

        # Yanıt başlıklarını kontrol et
        logger.info(f"API Response headers: {response.headers}")

        # Yanıt içeriğini kontrol et
        try:
            data = response.json()
            logger.info(f"API Response data: {data}")
        except Exception as e:
            logger.error(f"Error parsing JSON response: {e}")
            return None

        if response.status_code != 200:
            logger.error(f"API request failed with status code: {response.status_code}")
            logger.error(f"Error message: {data.get('message', 'No error message provided')}")
            return None

        if 'standings' not in data:
            logger.error("API response doesn't contain standings data")
            logger.error(f"Full response: {data}")
            return None

        standings = []
        for standing_type in data['standings']:
            if standing_type['type'] == 'TOTAL':  # Ana puan durumu
                for team in standing_type['table']:
                    team_data = {
                        'rank': team['position'],
                        'name': team['team']['name'],
                        'logo': team['team']['crest'],
                        'played': team['playedGames'],
                        'won': team['won'],
                        'draw': team['draw'],
                        'lost': team['lost'],
                        'goals_for': team['goalsFor'],
                        'goals_against': team['goalsAgainst'],
                        'goal_diff': team['goalDifference'],
                        'points': team['points']
                    }
                    standings.append(team_data)
                break

        if not standings:
            logger.error("No standings data was processed")
            return None

        logger.info(f"Successfully processed standings data. Found {len(standings)} teams.")
        return standings

    except Exception as e:
        logger.error(f"Error in get_league_standings: {str(e)}")
        logger.exception("Full traceback:")
        return None


@app.route('/api/predict', methods=['POST'])
def predict_match_post():
    """POST metodu ile maç tahmini yap"""
    try:
        # JSON verisi al
        data = request.json
        if not data:
            return jsonify({"error": "JSON verisi eksik"}), 400
        
        # Takım ID ve adları
        home_team_id = data.get('home_team_id')
        away_team_id = data.get('away_team_id')
        home_team_name = data.get('home_team_name', 'Ev Sahibi')
        away_team_name = data.get('away_team_name', 'Deplasman')
        force_update = data.get('force_update', False)
        
        # Takım ID'lerini doğrula
        if not home_team_id or not away_team_id:
            return jsonify({"error": "Takım ID'leri eksik"}), 400
            
        # Tahmin yap
        prediction = predictor.predict_match(
            home_team_id, 
            away_team_id, 
            home_team_name, 
            away_team_name, 
            force_update=force_update
        )
        
        return jsonify(prediction)
    except Exception as e:
        logger.error(f"Tahmin POST işlemi sırasında hata: {str(e)}", exc_info=True)
        return jsonify({"error": f"Tahmin yapılırken hata oluştu: {str(e)}"}), 500

@app.route('/api/predict-match/<home_team_id>/<away_team_id>')
def predict_match(home_team_id, away_team_id):
    """Belirli bir maç için tahmin yap"""
    try:
        # Takım adlarını alın
        home_team_name = request.args.get('home_name', 'Ev Sahibi')
        away_team_name = request.args.get('away_name', 'Deplasman')
        force_update = request.args.get('force_update', 'false').lower() == 'true'
        
        # Takım ID'lerini doğrula
        if not home_team_id or not away_team_id or not home_team_id.isdigit() or not away_team_id.isdigit():
            return jsonify({"error": "Geçersiz takım ID'leri"}), 400

        # Use performance cache manager
        if not force_update:
            cached_prediction = perf_cache_manager.get_cached_prediction(home_team_id, away_team_id)
            if cached_prediction:
                logger.info(f"Önbellekten tahmin alındı: {home_team_name} vs {away_team_name}")
                # Önbellekteki veriyi timestampli olarak işaretle
                cached_prediction['from_cache'] = True
                cached_prediction['cache_timestamp'] = datetime.now().timestamp()
                return jsonify(cached_prediction)
            
        # Eğer önbellekte değilse veya force_update ise yeni tahmin yap
        logger.info(f"Yeni tahmin yapılıyor. Force update: {force_update}, Takımlar: {home_team_name} vs {away_team_name}")
            
        try:
            # Tahmin yap
            prediction = predictor.predict_match(home_team_id, away_team_id, home_team_name, away_team_name, force_update)
            
            # Save prediction to PostgreSQL database
            if prediction and isinstance(prediction, dict) and not prediction.get('error'):
                try:
                    from database.dal import DAL
                    dal = DAL()
                    
                    # Save match and prediction to database
                    match_date = datetime.now()  # You might want to get actual match date from prediction
                    # Get league_id from prediction or use default (203 = Süper Lig)
                    league_id = prediction.get('league_id', 203)
                    if league_id is None:
                        league_id = 203  # Default to Süper Lig if still None
                    
                    match = dal.create_or_update_match({
                        'home_team_id': int(home_team_id),
                        'away_team_id': int(away_team_id),
                        'match_date': match_date,
                        'league_id': league_id,
                        'status': 'SCHEDULED',
                        'api_fixture_id': None  # We don't have fixture ID in this context
                    })
                    
                    if match and match.id and 'predictions' in prediction:
                        pred_data = prediction['predictions']
                        betting = pred_data.get('betting_predictions', {})
                        
                        # Save prediction to database
                        dal.save_prediction({
                            'match_id': match.id,
                            'predicted_winner': pred_data.get('predicted_winner'),
                            'home_win_probability': betting.get('match_result', {}).get('probabilities', {}).get('HOME_WIN', 0),
                            'draw_probability': betting.get('match_result', {}).get('probabilities', {}).get('DRAW', 0),
                            'away_win_probability': betting.get('match_result', {}).get('probabilities', {}).get('AWAY_WIN', 0),
                            'predicted_home_score': pred_data.get('expected_goals', {}).get('home', 0),
                            'predicted_away_score': pred_data.get('expected_goals', {}).get('away', 0),
                            'confidence_score': pred_data.get('confidence', 0.5),
                            'algorithm_weights': pred_data.get('ensemble_weights', {}),
                            'created_at': datetime.now()
                        })
                        logger.info(f"Prediction saved to PostgreSQL for match {home_team_name} vs {away_team_name}")
                except Exception as db_error:
                    logger.error(f"Failed to save prediction to database: {str(db_error)}")
                    # Continue even if database save fails
            
            # Yeni tahmini önbelleğe ekle (10 dakika süreyle)
            if prediction and (isinstance(prediction, dict) and not prediction.get('error')):
                prediction['from_cache'] = False
                prediction['cache_timestamp'] = datetime.now().timestamp()
                # Cache with performance cache manager
                perf_cache_manager.cache_prediction(home_team_id, away_team_id, prediction)

            if not prediction:
                return jsonify({"error": "Tahmin yapılamadı, takım verileri eksik olabilir", 
                               "match": f"{home_team_name} vs {away_team_name}"}), 400
                
            # Tahmin hata içeriyorsa
            if isinstance(prediction, dict) and "error" in prediction:
                return jsonify(prediction), 400

            # Maksimum yanıt boyutu kontrolü - büyük tahmin verilerinde hata olmasını önle
            import json
            response_size = len(json.dumps(prediction))
            
            if response_size > 1000000:  # 1MB'dan büyükse
                logger.warning(f"Çok büyük yanıt boyutu: {response_size} byte. Gereksiz detaylar kırpılıyor.")
                # Bazı gereksiz alanları kırp
                if 'home_team' in prediction and 'form' in prediction['home_team']:
                    # Form detaylarını azalt
                    prediction['home_team']['form'].pop('detailed_data', None)
                    prediction['home_team'].pop('form_periods', None)
                
                if 'away_team' in prediction and 'form' in prediction['away_team']:
                    # Form detaylarını azalt
                    prediction['away_team']['form'].pop('detailed_data', None)
                    prediction['away_team'].pop('form_periods', None)
                
                if 'predictions' in prediction and 'raw_metrics' in prediction['predictions']:
                    # Raw metrikleri kaldır
                    prediction['predictions'].pop('raw_metrics', None)

            return jsonify(prediction)
        except Exception as predict_error:
            logger.error(f"Tahmin işlemi sırasında hata: {str(predict_error)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Daha basit bir yanıt dön - veri boyutu nedenli hatalar için
            return jsonify({
                "error": "Tahmin işlemi sırasında teknik bir hata oluştu, lütfen daha sonra tekrar deneyin",
                "match": f"{home_team_name} vs {away_team_name}",
                "timestamp": datetime.now().timestamp()
            }), 500

    except Exception as e:
        logger.error(f"Tahmin yapılırken beklenmeyen hata: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Güvenli erişim - değişkenler tanımlanmamış veya None olabilir
        home_name = home_team_name if 'home_team_name' in locals() and home_team_name is not None else f"Takım {home_team_id}"
        away_name = away_team_name if 'away_team_name' in locals() and away_team_name is not None else f"Takım {away_team_id}"
        
        return jsonify({"error": "Sistem hatası. Lütfen daha sonra tekrar deneyin.", 
                        "match": f"{home_name} vs {away_name}"}), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_predictions_cache():
    """Tahmin önbelleğini temizle (hem dosya tabanlı önbelleği hem de Flask-Cache önbelleğini)"""
    try:
        # Predictor dosya tabanlı önbelleğini temizle
        success_file_cache = predictor.clear_cache()
        
        # Flask-Cache önbelleğini temizle
        with app.app_context():
            success_flask_cache = cache.clear()
        
        # Her iki önbelleğin de temizlenme durumunu değerlendir
        success = success_file_cache and success_flask_cache
        
        if success:
            logger.info("Hem dosya tabanlı önbellek hem de Flask-Cache önbelleği başarıyla temizlendi.")
            return jsonify({
                "success": True, 
                "message": "Tüm önbellekler temizlendi, yeni tahminler yapılabilir",
                "flask_cache_cleared": success_flask_cache,
                "file_cache_cleared": success_file_cache
            })
        else:
            logger.warning(f"Önbellek temizleme kısmen başarılı oldu. Dosya önbelleği: {success_file_cache}, Flask-Cache: {success_flask_cache}")
            return jsonify({
                "success": False, 
                "message": "Önbellek temizlenirken bazı sorunlar oluştu, ancak işlem devam edebilir", 
                "flask_cache_cleared": success_flask_cache,
                "file_cache_cleared": success_file_cache
            }), 200
    except Exception as e:
        error_msg = f"Önbellek temizlenirken beklenmeyen hata: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg, "success": False}), 500

@app.route('/api/batch-predict', methods=['POST'])
@handle_errors
def batch_predict():
    """Batch prediction endpoint for multiple matches"""
    try:
        data = request.get_json()
        if not data or 'matches' not in data:
            return jsonify({"error": "matches listesi gerekli"}), 400
            
        matches = data['matches']
        batch_id = f"batch_{int(time.time())}"
        
        # Create batch job
        batch_status = batch_manager.create_batch(batch_id, matches)
        
        # Process batch asynchronously
        batch_manager.process_batch_async(batch_id, predictor)
        
        return jsonify({
            "batch_id": batch_id,
            "status": "processing",
            "total_matches": len(matches),
            "message": "Tahminler işleniyor, /api/batch-status/{batch_id} ile durumu kontrol edebilirsiniz"
        })
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-status/<batch_id>')
@handle_errors
def batch_status(batch_id):
    """Get status of a batch prediction job"""
    status = batch_manager.get_batch_status(batch_id)
    return jsonify(status)

@app.route('/api/cache-stats')
@handle_errors
def cache_stats():
    """Get cache performance statistics"""
    stats = perf_cache_manager.get_cache_stats()
    return jsonify(stats)

@app.route('/api/warm-cache', methods=['POST'])
@handle_errors
def warm_cache():
    """Pre-warm cache with popular matches"""
    try:
        data = request.get_json()
        popular_matches = data.get('matches', [])
        
        # Run cache warming in background
        def warm_task():
            perf_cache_manager.warm_cache(predictor, popular_matches)
            
        thread = threading.Thread(target=warm_task)
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Cache warming started for {len(popular_matches)} matches"
        })
    except Exception as e:
        logger.error(f"Cache warming error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# AI İçgörüleri route'u
@app.route('/insights/<home_team_id>/<away_team_id>', methods=['GET'])
def match_insights(home_team_id, away_team_id):
    """Maç için AI içgörüleri ve doğal dil açıklamaları göster"""
    try:
        from match_insights import MatchInsightsGenerator
        insights_generator = MatchInsightsGenerator()
        
        # Takım verilerini al
        home_team_name = request.args.get('home_name', 'Ev Sahibi')
        away_team_name = request.args.get('away_name', 'Deplasman')
        
        # İçgörüleri oluştur
        insights = insights_generator.generate_match_insights(
            home_team_id, away_team_id, 
            additional_data={
                'home_team_name': home_team_name,
                'away_team_name': away_team_name
            }
        )
        
        # Eğer içgörüler başarıyla oluşturulursa şablonu render et
        if insights and 'error' not in insights:
            template_data = {
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team_name': home_team_name,
                'away_team_name': away_team_name,
                'insights': insights
            }
            return render_template('match_insights.html', **template_data)
        else:
            # Hata durumunda ana sayfaya yönlendir
            flash('İçgörüler oluşturulamadı. Lütfen daha sonra tekrar deneyin.', 'warning')
            return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"İçgörüler oluşturulurken hata: {str(e)}")
        flash('Bir hata oluştu. Lütfen daha sonra tekrar deneyin.', 'danger')
        return redirect(url_for('index'))

@app.route('/api/save-api-key', methods=['POST'])
def save_api_key():
    """API anahtarını kaydet"""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({'success': False, 'message': 'API anahtarı boş olamaz.'})
        
        # Test API key first
        is_valid, test_message = api_config.test_api_key(api_key)
        if not is_valid:
            return jsonify({'success': False, 'message': f'API anahtarı geçersiz: {test_message}'})
        
        # Save the API key
        if api_config.save_config(api_key):
            logger.info(f"API key updated successfully and propagated to all files")
            
            # Clear any cached data since we have a new API key
            cache.clear()
            
            # Reload modules that use the API key
            try:
                # Reload match_prediction module to use new API key
                import importlib
                import match_prediction
                importlib.reload(match_prediction)
                
                # Reload api_routes module
                import api_routes
                importlib.reload(api_routes)
                
                # Reload the global predictor instance with new API key
                global predictor
                predictor = match_prediction.MatchPredictor()
                
                logger.info("Modules reloaded with new API key")
            except Exception as reload_error:
                logger.warning(f"Module reload error (non-critical): {reload_error}")
            
            # Force refresh fixture data with new API key
            try:
                # Clear all caches to force fresh data load
                cache.clear()
                
                # Force reload API config globally 
                api_config.load_config()
                
                logger.info("All caches cleared, API config reloaded - fixture data will refresh with new API key")
            except Exception as cache_error:
                logger.warning(f"Cache clear error (non-critical): {cache_error}")
            
            return jsonify({
                'success': True, 
                'message': 'API anahtarı kaydedildi, sistem güncellendi ve fikstür verileri yenilenecek.'
            })
        else:
            return jsonify({'success': False, 'message': 'API anahtarı kaydedilemedi.'})
            
    except Exception as e:
        logger.error(f"Error saving API key: {e}")
        return jsonify({'success': False, 'message': 'Beklenmeyen bir hata oluştu.'})

@app.route('/api/test-api-key', methods=['POST'])
def test_api_key():
    """API anahtarını test et"""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({'success': False, 'message': 'API anahtarı boş olamaz.'})
        
        # Test the API key
        is_valid, test_message = api_config.test_api_key(api_key)
        
        if is_valid:
            # Try to get some additional info if possible
            return jsonify({
                'success': True,
                'message': test_message,
                'plan': 'Bilinmiyor'  # API'den plan bilgisi almaya çalışabiliriz
            })
        else:
            return jsonify({'success': False, 'message': test_message})
            
    except Exception as e:
        logger.error(f"Error testing API key: {e}")
        return jsonify({'success': False, 'message': 'Test edilirken hata oluştu.'})

@app.route('/api/get-current-api-status')
def get_current_api_status():
    """Mevcut API durumunu kontrol et"""
    try:
        current_key = api_config.get_api_key()
        is_valid, test_message = api_config.test_api_key(current_key)
        
        return jsonify({
            'success': True,
            'api_key_valid': is_valid,
            'message': test_message,
            'has_custom_key': current_key != api_config.default_api_key
        })
        
    except Exception as e:
        logger.error(f"Error checking API status: {e}")
        return jsonify({'success': False, 'message': 'API durumu kontrol edilemedi.'})

def find_available_port(preferred_ports=None):
    """
    Kullanılabilir bir port bul
    
    Args:
        preferred_ports: Tercih edilen portların listesi, önce bunlar denenecek
        
    Returns:
        int: Kullanılabilir port numarası
    """
    import socket
    
    # Hiç tercih edilen port belirtilmemişse varsayılan listeyi kullan
    if preferred_ports is None:
        # Sırasıyla denenecek portlar - Replit için 5000 öncelikli
        preferred_ports = [5000, 80, 8080, 3000, 8000, 8888, 9000]
    
    # Önce çevre değişkeninden PORT değerini kontrol et
    env_port = os.environ.get('PORT')
    if env_port:
        try:
            env_port = int(env_port)
            if env_port not in preferred_ports:
                # Çevre değişkeni varsa onu listenin başına ekle
                preferred_ports.insert(0, env_port)
        except ValueError:
            logger.warning(f"Çevre değişkenindeki PORT değeri ({env_port}) geçerli bir sayı değil, yok sayılıyor")
    
    # Her bir portu dene ve kullanılabilir olanı bul
    for port in preferred_ports:
        try:
            # Port müsait mi kontrol et
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result != 0:  # Port açık değilse (bağlantı başarısız oldu)
                logger.info(f"Port {port} kullanılabilir, bu port kullanılacak")
                return port
            else:
                logger.warning(f"Port {port} zaten kullanımda, başka port deneniyor")
        except Exception as e:
            logger.warning(f"Port {port} kontrolü sırasında hata: {str(e)}")
    
    # Hiçbir tercih edilen port kullanılamıyorsa, rastgele bir port ata
    logger.warning("Tercih edilen portların hiçbiri kullanılamıyor, rastgele bir port atanacak")
    return 0  # 0 verilirse, sistem otomatik olarak kullanılabilir bir port atar

# Basit test route'u
@app.route('/api/v3/test')
def api_test():
    return jsonify({"status": "ok", "message": "API v3 is working"})

# API endpoint'lerini doğrudan tanımlayalım (global scope'ta)
@app.route('/api/v3/fixtures/team/<int:team_id>', methods=['GET'])
def get_team_stats_api(team_id):
    """
    Takımın detaylı istatistiklerini döndüren API endpoint
    Popup takım istatistikleri için kullanılır
    """
    try:
        # Takımın son maçlarını al
        from api_config import APIConfig
        api_config = APIConfig()
        api_key = api_config.get_api_key()
        
        if not api_key:
            logger.warning("API anahtarı bulunamadı")
            return jsonify([])
            
        url = "https://v3.football.api-sports.io/fixtures"
        headers = {'x-apisports-key': api_key}
        
        # Son 120 günlük maçları al
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)
        
        params = {
            'team': team_id,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'timezone': 'Europe/Istanbul'
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return jsonify([])
            
        data = response.json()
        matches = data.get('response', []) if isinstance(data, dict) else []
        if not isinstance(matches, list):
            return jsonify([])
            
        # Maçları tarihe göre sırala (en yeni önce)
        sorted_matches = sorted(matches, key=lambda x: x.get('fixture', {}).get('date', ''), reverse=True)
        
        # Son 10 maçı formatla ve döndür
        formatted_matches = []
        for match in sorted_matches[:10]:  # Son 10 maç
            fixture = match.get('fixture', {})
            teams = match.get('teams', {})
            goals = match.get('goals', {})
            
            fixture_date = fixture.get('date', '')
            try:
                # Tarihi daha okunabilir formata dönüştür
                if 'T' in fixture_date:
                    date_str = fixture_date.split('T')[0]
                    match_date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    formatted_date = match_date_obj.strftime('%d %b %Y')
                else:
                    formatted_date = fixture_date
            except Exception:
                formatted_date = fixture_date
                
            home_team = teams.get('home', {}).get('name', '')
            away_team = teams.get('away', {}).get('name', '')
            home_score = goals.get('home', '')
            away_score = goals.get('away', '')
            
            formatted_match = {
                'date': formatted_date,
                'match': f"{home_team} vs {away_team}",
                'score': f"{home_score} - {away_score}"
            }
            formatted_matches.append(formatted_match)
            
        return jsonify(formatted_matches)
        
    except Exception as e:
        logger.error(f"Takım istatistikleri alınırken hata: {str(e)}")
        return jsonify([])

# Flask uygulamasını başlat
if __name__ == '__main__':
    # Uygulama 127.0.0.1:8080'de çalışacak
    logger.info("Goal-Guru uygulaması başlatılıyor: http://127.0.0.1:8080/")
    app.run(host='127.0.0.1', port=8080, debug=True)
