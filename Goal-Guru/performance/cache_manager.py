"""
Advanced Caching System for Football Prediction Hub
Phase 2.2 - Caching Optimization Implementation
"""

import json
import time
import hashlib
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import logging
import pickle
from functools import wraps
import threading

logger = logging.getLogger(__name__)

class InMemoryCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self):
        self.cache = {}
        self.lock = threading.RLock()
        self._start_cleanup_thread()
        
    def _start_cleanup_thread(self):
        """Start background thread to clean expired entries"""
        def cleanup():
            while True:
                time.sleep(60)  # Clean every minute
                self._cleanup_expired()
                
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()
        
    def _cleanup_expired(self):
        """Remove expired entries from cache"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, expiry) in self.cache.items()
                if expiry and expiry < current_time
            ]
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
                
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if expiry and expiry < time.time():
                    del self.cache[key]
                    return None
                return value
            return None
            
    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache with optional TTL"""
        with self.lock:
            expiry = time.time() + ttl if ttl else None
            self.cache[key] = (value, expiry)
            
    def delete(self, key: str):
        """Delete key from cache"""
        with self.lock:
            self.cache.pop(key, None)
            
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'keys': list(self.cache.keys())[:10]  # First 10 keys only
            }


class RedisCache:
    """Redis cache implementation (simulated with file storage for Replit)"""
    
    def __init__(self, cache_dir='cache_data'):
        self.cache_dir = cache_dir
        self.memory_cache = InMemoryCache()  # L1 cache
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_file_path(self, key: str) -> str:
        """Get file path for a cache key"""
        import os
        # Hash the key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (checks memory first, then disk)"""
        # Check L1 cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
            
        # Check L2 cache (disk)
        file_path = self._get_file_path(key)
        try:
            import os
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    if data['expiry'] and data['expiry'] < time.time():
                        os.remove(file_path)
                        return None
                    
                    # Store in L1 cache for faster access
                    remaining_ttl = int(data['expiry'] - time.time()) if data['expiry'] else None
                    self.memory_cache.set(key, data['value'], remaining_ttl)
                    
                    return data['value']
        except Exception as e:
            logger.error(f"Error reading cache file {file_path}: {str(e)}")
            
        return None
        
    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache with optional TTL"""
        # Store in L1 cache
        self.memory_cache.set(key, value, ttl)
        
        # Store in L2 cache (disk)
        file_path = self._get_file_path(key)
        try:
            data = {
                'value': value,
                'expiry': time.time() + ttl if ttl else None,
                'created_at': time.time()
            }
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error writing cache file {file_path}: {str(e)}")
            
    def delete(self, key: str):
        """Delete key from cache"""
        # Delete from L1 cache
        self.memory_cache.delete(key)
        
        # Delete from L2 cache
        file_path = self._get_file_path(key)
        try:
            import os
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting cache file {file_path}: {str(e)}")
            
    def clear(self):
        """Clear all cache"""
        # Clear L1 cache
        self.memory_cache.clear()
        
        # Clear L2 cache
        import os
        import glob
        try:
            pattern = os.path.join(self.cache_dir, "*.cache")
            for file_path in glob.glob(pattern):
                os.remove(file_path)
            logger.info("Cleared all cache files")
        except Exception as e:
            logger.error(f"Error clearing cache files: {str(e)}")


class CacheManager:
    """Manages caching strategy for the application"""
    
    def __init__(self, cache_backend: Optional[RedisCache] = None):
        self.cache = cache_backend or RedisCache()
        self.hit_count = 0
        self.miss_count = 0
        self.cache_config = {
            'prediction': 600,  # 10 minutes
            'team_stats': 3600,  # 1 hour
            'league_standings': 3600,  # 1 hour
            'api_response': 300,  # 5 minutes
            'match_list': 300  # 5 minutes
        }
        
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a unique cache key"""
        key_parts = [prefix]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        return ":".join(key_parts)
        
    def cache_prediction(self, home_id: str, away_id: str, prediction: Dict, force_cache: bool = False):
        """Cache a match prediction"""
        key = self._generate_cache_key('prediction', home_id, away_id)
        ttl = self.cache_config['prediction']
        
        # Add cache metadata
        cached_data = {
            'data': prediction,
            'cached_at': datetime.now().isoformat(),
            'ttl': ttl
        }
        
        self.cache.set(key, cached_data, ttl)
        logger.info(f"Cached prediction for {home_id} vs {away_id}")
        
    def get_cached_prediction(self, home_id: str, away_id: str) -> Optional[Dict]:
        """Get cached prediction if available"""
        key = self._generate_cache_key('prediction', home_id, away_id)
        cached = self.cache.get(key)
        
        if cached:
            self.hit_count += 1
            logger.info(f"Cache hit for prediction {home_id} vs {away_id}")
            return cached['data']
        else:
            self.miss_count += 1
            return None
            
    def cache_api_response(self, url: str, response: Dict, ttl: int = None):
        """Cache API response"""
        key = self._generate_cache_key('api', url)
        ttl = ttl or self.cache_config['api_response']
        self.cache.set(key, response, ttl)
        
    def get_cached_api_response(self, url: str) -> Optional[Dict]:
        """Get cached API response"""
        key = self._generate_cache_key('api', url)
        return self.cache.get(key)
        
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        # For file-based cache, this would need to scan all files
        # For now, we'll just log the intention
        logger.info(f"Cache invalidation requested for pattern: {pattern}")
        
    def warm_cache(self, predictor, popular_matches: List[Dict]):
        """Pre-warm cache with popular matches"""
        logger.info(f"Warming cache with {len(popular_matches)} matches")
        
        for match in popular_matches:
            try:
                # Check if already cached
                existing = self.get_cached_prediction(match['home_id'], match['away_id'])
                if not existing:
                    # Generate and cache prediction
                    prediction = predictor.predict_match(
                        match['home_id'], 
                        match['away_id'],
                        match['home_name'],
                        match['away_name']
                    )
                    self.cache_prediction(match['home_id'], match['away_id'], prediction)
            except Exception as e:
                logger.error(f"Error warming cache for match {match}: {str(e)}")
                
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': f"{hit_rate:.2f}%",
            'total_requests': total_requests,
            'backend_stats': self.cache.memory_cache.get_stats()
        }
        
    def clear_expired(self):
        """Clear expired cache entries"""
        self.cache.memory_cache._cleanup_expired()
        
        
def cached(cache_manager: CacheManager, cache_type: str = 'generic', ttl: int = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key = cache_manager._generate_cache_key(
                f"{cache_type}:{func.__name__}",
                *args,
                **kwargs
            )
            
            # Check cache
            cached_result = cache_manager.cache.get(key)
            if cached_result is not None:
                cache_manager.hit_count += 1
                return cached_result
                
            # Cache miss - execute function
            cache_manager.miss_count += 1
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_ttl = ttl or cache_manager.cache_config.get(cache_type, 300)
            cache_manager.cache.set(key, result, cache_ttl)
            
            return result
        return wrapper
    return decorator