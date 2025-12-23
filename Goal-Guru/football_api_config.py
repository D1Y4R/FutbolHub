import os

# API Football Yapılandırması (PRIMARY)
API_FOOTBALL_CONFIG = {
    "base_url": "https://v3.football.api-sports.io",
    "api_key": os.environ.get('API_FOOTBALL_KEY', '2f0c06f149e51424f4c9be24eb70cb8f'),
    "timezone": "Europe/Istanbul",
    "header_name": "x-apisports-key"
}

# Legacy API konfigürasyonu (BACKUP - football-data.org)
FOOTBALL_DATA_CONFIG = {
    "base_url": "https://api.football-data.org/v4",
    "api_key": os.environ.get('FOOTBALL_DATA_API_KEY', '668dd03e0aea41b58fce760cdf4eddc8'),
    "leagues": {
        "UCL": 3,         # UEFA Şampiyonlar Ligi
        "UEL": 4,         # UEFA Avrupa Ligi
        "UECL": 683,      # UEFA Konferans Ligi
        "LL": 302,        # La Liga
        "PL": 152,        # Premier League
        "SA": 207,        # Serie A
        "BL": 175,        # Bundesliga
        "L1": 168,        # Ligue 1
        "SL": 322,        # Süper Lig
        "PPL": 266       # Primeira Liga
    },
    "max_future_days": 365
}

# Backward compatibility
FOOTBALL_API_CONFIG = FOOTBALL_DATA_CONFIG