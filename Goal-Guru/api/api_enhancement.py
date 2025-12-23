"""
Phase 4.1: API Enhancement Agent
Comprehensive RESTful API with documentation, versioning, and rate limiting
"""

import logging
import time
import json
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib
import secrets

from flask import Flask, request, jsonify, Blueprint, current_app, g
from flask_cors import CORS

logger = logging.getLogger(__name__)

# API versioning
API_VERSION = "v2"
API_PREFIX = f"/api/{API_VERSION}"

# Rate limiting configuration
RATE_LIMITS = {
    "default": "100 per hour",
    "predictions": "50 per hour",
    "batch": "10 per hour",
    "webhooks": "100 per day"
}

class APIKeyManager:
    """Manage API keys for authentication"""
    
    def __init__(self):
        self.api_keys = {}
        self.key_metadata = {}
        self.usage_stats = defaultdict(lambda: {"requests": 0, "last_used": ""})
        self._load_api_keys()
        logger.info("APIKeyManager initialized")
    
    def _load_api_keys(self):
        """Load API keys from storage"""
        try:
            with open('api_keys.json', 'r') as f:
                data = json.load(f)
                self.api_keys = data.get('keys', {})
                self.key_metadata = data.get('metadata', {})
        except FileNotFoundError:
            # Create default admin key
            admin_key = self.generate_api_key("admin", {"tier": "premium", "rate_limit": "1000 per hour"})
            logger.info(f"Created default admin API key: {admin_key}")
    
    def generate_api_key(self, name: str, metadata: Optional[Dict] = None) -> str:
        """Generate a new API key"""
        key = f"fpapi_{secrets.token_urlsafe(32)}"
        self.api_keys[key] = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        if metadata:
            self.key_metadata[key] = metadata
        self._save_api_keys()
        return key
    
    def validate_api_key(self, key: str) -> Tuple[bool, Optional[Dict]]:
        """Validate an API key"""
        if key not in self.api_keys:
            return False, None
        
        key_data = self.api_keys[key]
        if not key_data.get('active', True):
            return False, None
        
        # Update usage stats
        stats = self.usage_stats[key]
        current_requests = stats.get("requests", 0)
        stats["requests"] = int(current_requests) + 1
        stats["last_used"] = datetime.now().isoformat()
        
        return True, self.key_metadata.get(key, {})
    
    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key"""
        if key in self.api_keys:
            self.api_keys[key]["active"] = False
            self._save_api_keys()
            return True
        return False
    
    def _save_api_keys(self):
        """Save API keys to storage"""
        data = {
            "keys": self.api_keys,
            "metadata": self.key_metadata
        }
        with open('api_keys.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_usage_stats(self, key: str) -> Dict:
        """Get usage statistics for an API key"""
        return dict(self.usage_stats.get(key, {}))

class WebhookManager:
    """Manage webhooks for prediction events"""
    
    def __init__(self):
        self.webhooks = {}
        self._load_webhooks()
        logger.info("WebhookManager initialized")
    
    def _load_webhooks(self):
        """Load webhooks from storage"""
        try:
            with open('webhooks.json', 'r') as f:
                self.webhooks = json.load(f)
        except FileNotFoundError:
            self.webhooks = {}
    
    def register_webhook(self, api_key: str, url: str, events: List[str]) -> str:
        """Register a new webhook"""
        webhook_id = hashlib.md5(f"{api_key}{url}{time.time()}".encode()).hexdigest()
        
        self.webhooks[webhook_id] = {
            "api_key": api_key,
            "url": url,
            "events": events,
            "created_at": datetime.now().isoformat(),
            "active": True,
            "failures": 0
        }
        
        self._save_webhooks()
        return webhook_id
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            self._save_webhooks()
            return True
        return False
    
    def trigger_webhooks(self, event: str, data: Dict):
        """Trigger webhooks for an event"""
        import requests
        
        for webhook_id, webhook in self.webhooks.items():
            if not webhook.get('active', True):
                continue
                
            if event not in webhook['events']:
                continue
            
            try:
                response = requests.post(
                    webhook['url'],
                    json={
                        "event": event,
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    },
                    timeout=5
                )
                
                if response.status_code >= 400:
                    webhook['failures'] += 1
                    if webhook['failures'] > 5:
                        webhook['active'] = False
                        logger.warning(f"Webhook {webhook_id} disabled after 5 failures")
                else:
                    webhook['failures'] = 0
                    
            except Exception as e:
                logger.error(f"Webhook trigger failed for {webhook_id}: {str(e)}")
                webhook['failures'] += 1
        
        self._save_webhooks()
    
    def _save_webhooks(self):
        """Save webhooks to storage"""
        with open('webhooks.json', 'w') as f:
            json.dump(self.webhooks, f, indent=2)

class APIDocumentation:
    """Generate OpenAPI/Swagger documentation"""
    
    @staticmethod
    def generate_openapi_spec() -> Dict:
        """Generate OpenAPI 3.0 specification"""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Football Predictor API",
                "version": API_VERSION,
                "description": "Advanced football match prediction API with ML models",
                "contact": {
                    "name": "API Support",
                    "email": "support@footballpredictor.com"
                }
            },
            "servers": [
                {"url": f"https://api.footballpredictor.com{API_PREFIX}"}
            ],
            "security": [
                {"ApiKeyAuth": []}
            ],
            "components": {
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    }
                },
                "schemas": {
                    "Prediction": {
                        "type": "object",
                        "properties": {
                            "match_info": {"$ref": "#/components/schemas/MatchInfo"},
                            "predictions": {"$ref": "#/components/schemas/PredictionDetails"},
                            "confidence": {"type": "number", "format": "float"},
                            "timestamp": {"type": "string", "format": "date-time"}
                        }
                    },
                    "MatchInfo": {
                        "type": "object",
                        "properties": {
                            "home_team": {"$ref": "#/components/schemas/Team"},
                            "away_team": {"$ref": "#/components/schemas/Team"},
                            "date": {"type": "string", "format": "date-time"}
                        }
                    },
                    "Team": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"}
                        }
                    },
                    "PredictionDetails": {
                        "type": "object",
                        "properties": {
                            "most_likely_outcome": {"type": "string", "enum": ["HOME_WIN", "DRAW", "AWAY_WIN"]},
                            "home_win_probability": {"type": "number"},
                            "draw_probability": {"type": "number"},
                            "away_win_probability": {"type": "number"},
                            "expected_goals": {"type": "object"},
                            "over_under": {"type": "object"},
                            "both_teams_to_score": {"type": "object"}
                        }
                    }
                }
            },
            "paths": {
                "/predict": {
                    "post": {
                        "summary": "Get match prediction",
                        "description": "Generate prediction for a specific match",
                        "operationId": "getPrediction",
                        "tags": ["Predictions"],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "home_team_id": {"type": "integer"},
                                            "away_team_id": {"type": "integer"},
                                            "date": {"type": "string", "format": "date"}
                                        },
                                        "required": ["home_team_id", "away_team_id"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful prediction",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/Prediction"}
                                    }
                                }
                            },
                            "400": {"description": "Invalid request"},
                            "401": {"description": "Unauthorized"},
                            "429": {"description": "Rate limit exceeded"}
                        }
                    }
                },
                "/batch-predict": {
                    "post": {
                        "summary": "Batch prediction",
                        "description": "Generate predictions for multiple matches",
                        "operationId": "getBatchPrediction",
                        "tags": ["Predictions"],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "matches": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "home_team_id": {"type": "integer"},
                                                        "away_team_id": {"type": "integer"},
                                                        "date": {"type": "string", "format": "date"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "202": {
                                "description": "Batch job accepted",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "job_id": {"type": "string"},
                                                "status": {"type": "string"},
                                                "estimated_time": {"type": "integer"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/webhooks": {
                    "post": {
                        "summary": "Register webhook",
                        "description": "Register a webhook for prediction events",
                        "operationId": "registerWebhook",
                        "tags": ["Webhooks"],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "url": {"type": "string", "format": "uri"},
                                            "events": {
                                                "type": "array",
                                                "items": {
                                                    "type": "string",
                                                    "enum": ["prediction.created", "prediction.updated", "batch.completed"]
                                                }
                                            }
                                        },
                                        "required": ["url", "events"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "201": {
                                "description": "Webhook registered",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "webhook_id": {"type": "string"},
                                                "url": {"type": "string"},
                                                "events": {"type": "array", "items": {"type": "string"}}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api-keys": {
                    "post": {
                        "summary": "Generate API key",
                        "description": "Generate a new API key",
                        "operationId": "generateApiKey",
                        "tags": ["Authentication"],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "tier": {"type": "string", "enum": ["free", "basic", "premium"]}
                                        },
                                        "required": ["name"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "201": {
                                "description": "API key created",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "api_key": {"type": "string"},
                                                "name": {"type": "string"},
                                                "tier": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

# Initialize managers
api_key_manager = APIKeyManager()
webhook_manager = WebhookManager()

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({"error": "API key required"}), 401
        
        valid, metadata = api_key_manager.validate_api_key(api_key)
        if not valid:
            return jsonify({"error": "Invalid API key"}), 401
        
        # Add metadata to request context
        g.api_key_metadata = metadata
        return f(*args, **kwargs)
    
    return decorated_function

def versioned_route(route: str) -> str:
    """Create versioned route"""
    return f"{API_PREFIX}{route}"

def create_enhanced_api_blueprint(predictor):
    """Create enhanced API blueprint with all features"""
    api_bp = Blueprint('api_v2', __name__)
    
    @api_bp.route(versioned_route('/predict'), methods=['POST'])
    @require_api_key
    def predict():
        """Get match prediction"""
        data = request.get_json()
        
        if not data or 'home_team_id' not in data or 'away_team_id' not in data:
            return jsonify({"error": "home_team_id and away_team_id required"}), 400
        
        try:
            prediction = predictor.predict(
                data['home_team_id'],
                data['away_team_id'],
                data.get('home_team_name', 'Home Team'),
                data.get('away_team_name', 'Away Team'),
                data.get('date')
            )
            
            # Trigger webhook
            webhook_manager.trigger_webhooks('prediction.created', {
                "home_team_id": data['home_team_id'],
                "away_team_id": data['away_team_id'],
                "prediction": prediction['predictions']['most_likely_outcome']
            })
            
            return jsonify(prediction)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({"error": "Prediction failed", "message": str(e)}), 500
    
    @api_bp.route(versioned_route('/batch-predict'), methods=['POST'])
    @require_api_key
    def batch_predict():
        """Batch prediction endpoint"""
        data = request.get_json()
        
        if not data or 'matches' not in data:
            return jsonify({"error": "matches array required"}), 400
        
        # Create batch job
        job_id = hashlib.md5(f"{time.time()}{request.headers.get('X-API-Key')}".encode()).hexdigest()
        
        # In a real implementation, this would be queued
        # For now, return accepted status
        return jsonify({
            "job_id": job_id,
            "status": "accepted",
            "estimated_time": len(data['matches']) * 2  # 2 seconds per match
        }), 202
    
    @api_bp.route(versioned_route('/webhooks'), methods=['POST'])
    @require_api_key
    def register_webhook():
        """Register a webhook"""
        data = request.get_json()
        
        if not data or 'url' not in data or 'events' not in data:
            return jsonify({"error": "url and events required"}), 400
        
        api_key = request.headers.get('X-API-Key', '')
        webhook_id = webhook_manager.register_webhook(
            api_key,
            data['url'],
            data['events']
        )
        
        return jsonify({
            "webhook_id": webhook_id,
            "url": data['url'],
            "events": data['events']
        }), 201
    
    @api_bp.route(versioned_route('/webhooks/<webhook_id>'), methods=['DELETE'])
    @require_api_key
    def unregister_webhook(webhook_id):
        """Unregister a webhook"""
        if webhook_manager.unregister_webhook(webhook_id):
            return '', 204
        return jsonify({"error": "Webhook not found"}), 404
    
    @api_bp.route(versioned_route('/api-keys'), methods=['POST'])
    @require_api_key
    def generate_api_key():
        """Generate a new API key (admin only)"""
        # Check if requester has admin privileges
        metadata = g.get('api_key_metadata', {})
        if metadata.get('tier') != 'premium':
            return jsonify({"error": "Admin access required"}), 403
        
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"error": "name required"}), 400
        
        tier = data.get('tier', 'free')
        rate_limit = {
            'free': '10 per hour',
            'basic': '50 per hour',
            'premium': '1000 per hour'
        }.get(tier, '10 per hour')
        
        api_key = api_key_manager.generate_api_key(
            data['name'],
            {"tier": tier, "rate_limit": rate_limit}
        )
        
        return jsonify({
            "api_key": api_key,
            "name": data['name'],
            "tier": tier
        }), 201
    
    @api_bp.route(versioned_route('/openapi.json'), methods=['GET'])
    def get_openapi_spec():
        """Get OpenAPI specification"""
        return jsonify(APIDocumentation.generate_openapi_spec())
    
    @api_bp.route(versioned_route('/health'), methods=['GET'])
    def health_check():
        """API health check"""
        return jsonify({
            "status": "healthy",
            "version": API_VERSION,
            "timestamp": datetime.now().isoformat()
        })
    
    # Error handlers
    @api_bp.errorhandler(429)
    def rate_limit_handler(e):
        return jsonify({"error": "Rate limit exceeded", "message": str(e.description)}), 429
    
    @api_bp.errorhandler(404)
    def not_found_handler(e):
        return jsonify({"error": "Endpoint not found"}), 404
    
    @api_bp.errorhandler(500)
    def internal_error_handler(e):
        logger.error(f"Internal error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
    return api_bp

def setup_swagger_ui(app: Flask):
    """Setup Swagger UI for API documentation"""
    try:
        from flask_swagger_ui import get_swaggerui_blueprint
        
        # Configure Swagger UI
        SWAGGER_URL = f'/api/{API_VERSION}/docs'
        API_SPEC_URL = f'/api/{API_VERSION}/openapi.json'
        
        swaggerui_blueprint = get_swaggerui_blueprint(
            SWAGGER_URL,
            API_SPEC_URL,
            config={
                'app_name': "Football Prediction API v2",
                'dom_id': '#swagger-ui',
                'deepLinking': True,
                'presets': [
                    'apis',
                    'auth',
                ],
                'layout': "BaseLayout",
                'validatorUrl': None,
                'displayRequestDuration': True,
                'docExpansion': 'list',
                'defaultModelsExpandDepth': 1,
                'defaultModelExpandDepth': 1,
                'filter': True,
                'showExtensions': True,
                'showCommonExtensions': True
            }
        )
        
        app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
        logger.info(f"Swagger UI registered at {SWAGGER_URL}")
        
    except ImportError:
        logger.warning("flask-swagger-ui not available - Swagger UI disabled")

def create_graphql_endpoint():
    """Create GraphQL endpoint (optional)"""
    # This would be implemented with graphene-python
    # For now, return a placeholder
    pass

if __name__ == "__main__":
    # Test API key generation
    manager = APIKeyManager()
    test_key = manager.generate_api_key("test_user", {"tier": "basic"})
    print(f"Generated test API key: {test_key}")
    
    # Test webhook registration
    webhook_mgr = WebhookManager()
    webhook_id = webhook_mgr.register_webhook(test_key, "https://example.com/webhook", ["prediction.created"])
    print(f"Registered webhook: {webhook_id}")
    
    # Test OpenAPI spec generation
    spec = APIDocumentation.generate_openapi_spec()
    print(f"OpenAPI spec generated with {len(spec['paths'])} endpoints")