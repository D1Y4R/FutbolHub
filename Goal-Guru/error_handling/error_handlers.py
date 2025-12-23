"""
Centralized error handling framework for Football Prediction Hub
Phase 1.2 - Error Handling Framework Implementation
"""

import logging
import traceback
from functools import wraps
from flask import jsonify, request
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ErrorLogger:
    """Centralized error logging with context"""
    
    @staticmethod
    def log_error(error_type, error_message, context=None, stack_trace=None):
        """Log error with full context"""
        error_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': error_type,
            'error_message': str(error_message),
            'request_url': request.url if request else None,
            'request_method': request.method if request else None,
            'user_agent': request.headers.get('User-Agent') if request else None,
            'context': context or {},
            'stack_trace': stack_trace
        }
        
        # Log to file
        logger.error(f"Error: {json.dumps(error_data, indent=2)}")
        
        # Store in database for analysis (future enhancement)
        # db_manager.save_error_log(error_data)
        
        return error_data

class APIError(Exception):
    """Custom API error class"""
    def __init__(self, message, status_code=500, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}

class ValidationError(APIError):
    """Validation error class"""
    def __init__(self, message, field=None):
        super().__init__(message, status_code=400)
        if field:
            self.payload['field'] = field

class AuthenticationError(APIError):
    """Authentication error class"""
    def __init__(self, message="Authentication required"):
        super().__init__(message, status_code=401)

class RateLimitError(APIError):
    """Rate limit error class"""
    def __init__(self, message="Rate limit exceeded"):
        super().__init__(message, status_code=429)

class ExternalAPIError(APIError):
    """External API error class"""
    def __init__(self, message, api_name=None):
        super().__init__(message, status_code=503)
        if api_name:
            self.payload['api'] = api_name

def handle_errors(func):
    """Decorator to handle all errors consistently"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            # Handle custom API errors
            ErrorLogger.log_error(
                error_type=e.__class__.__name__,
                error_message=e.message,
                context=e.payload
            )
            return jsonify({
                'error': True,
                'message': e.message,
                **e.payload
            }), e.status_code
            
        except ValueError as e:
            # Handle validation errors
            ErrorLogger.log_error(
                error_type='ValueError',
                error_message=str(e)
            )
            return jsonify({
                'error': True,
                'message': 'Invalid input data',
                'details': str(e)
            }), 400
            
        except KeyError as e:
            # Handle missing data errors
            ErrorLogger.log_error(
                error_type='KeyError',
                error_message=f"Missing required field: {str(e)}"
            )
            return jsonify({
                'error': True,
                'message': 'Missing required data',
                'field': str(e)
            }), 400
            
        except Exception as e:
            # Handle unexpected errors
            stack_trace = traceback.format_exc()
            ErrorLogger.log_error(
                error_type=e.__class__.__name__,
                error_message=str(e),
                stack_trace=stack_trace
            )
            
            # Don't expose internal errors in production
            return jsonify({
                'error': True,
                'message': 'An unexpected error occurred',
                'error_id': datetime.utcnow().timestamp()  # For support reference
            }), 500
            
    return wrapper

def validate_request_data(required_fields, data=None):
    """Validate request data has required fields"""
    if data is None:
        data = request.get_json() or {}
    
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            field=missing_fields[0] if len(missing_fields) == 1 else None
        )
    
    return data

def handle_external_api_error(api_name, error):
    """Handle external API errors consistently"""
    error_message = str(error)
    
    if 'rate limit' in error_message.lower():
        raise RateLimitError(f"{api_name} rate limit exceeded")
    elif 'unauthorized' in error_message.lower() or '401' in error_message:
        raise AuthenticationError(f"{api_name} authentication failed")
    elif 'timeout' in error_message.lower():
        raise ExternalAPIError(f"{api_name} request timeout", api_name)
    else:
        raise ExternalAPIError(f"{api_name} error: {error_message}", api_name)

# Flask error handlers
def register_error_handlers(app):
    """Register Flask error handlers"""
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': True,
            'message': 'Resource not found'
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'error': True,
            'message': 'Method not allowed'
        }), 405
    
    @app.errorhandler(500)
    def internal_error(error):
        ErrorLogger.log_error(
            error_type='InternalServerError',
            error_message=str(error),
            stack_trace=traceback.format_exc()
        )
        return jsonify({
            'error': True,
            'message': 'Internal server error',
            'error_id': datetime.utcnow().timestamp()
        }), 500
    
    @app.errorhandler(APIError)
    def handle_api_error(error):
        return jsonify({
            'error': True,
            'message': error.message,
            **error.payload
        }), error.status_code