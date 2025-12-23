"""
Phase 4.2: Real-time Updates Agent
WebSocket server for live updates, real-time predictions, and event-driven architecture
"""

import logging
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import threading
import queue

# For WebSocket support in Flask
try:
    from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
    from flask import request
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    logging.warning("flask-socketio not available - WebSocket features will be limited")
    # Define dummy functions for when socketio is not available
    def emit(*args, **kwargs): pass
    def join_room(*args, **kwargs): pass
    def leave_room(*args, **kwargs): pass
    request = None

logger = logging.getLogger(__name__)

# Global socketio instance (will be initialized later)
socketio = None

class EventManager:
    """Manage real-time events and subscriptions"""
    
    def __init__(self):
        self.subscribers = defaultdict(set)
        self.event_queue = queue.Queue()
        self.event_history = []
        self.max_history = 1000
        logger.info("EventManager initialized")
    
    def subscribe(self, event_type: str, subscriber_id: str):
        """Subscribe to an event type"""
        self.subscribers[event_type].add(subscriber_id)
        logger.info(f"Subscriber {subscriber_id} subscribed to {event_type}")
    
    def unsubscribe(self, event_type: str, subscriber_id: str):
        """Unsubscribe from an event type"""
        if subscriber_id in self.subscribers[event_type]:
            self.subscribers[event_type].remove(subscriber_id)
            logger.info(f"Subscriber {subscriber_id} unsubscribed from {event_type}")
    
    def publish(self, event_type: str, data: Dict):
        """Publish an event"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "subscribers": list(self.subscribers[event_type])
        }
        
        self.event_queue.put(event)
        self.event_history.append(event)
        
        # Keep history size limited
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        logger.info(f"Event published: {event_type} to {len(event['subscribers'])} subscribers")
        return event
    
    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get recent events"""
        events = self.event_history
        if event_type:
            events = [e for e in events if e['type'] == event_type]
        return events[-limit:]

class LiveDataFetcher:
    """Fetch live match data and scores"""
    
    def __init__(self, api_config):
        self.api_config = api_config
        self.live_matches = {}
        self.update_interval = 30  # seconds
        self.running = False
        self._thread = None
        logger.info("LiveDataFetcher initialized")
    
    def start(self):
        """Start fetching live data"""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._fetch_loop, daemon=True)
            self._thread.start()
            logger.info("Live data fetching started")
    
    def stop(self):
        """Stop fetching live data"""
        self.running = False
        if self._thread:
            self._thread.join()
        logger.info("Live data fetching stopped")
    
    def _fetch_loop(self):
        """Main fetch loop"""
        while self.running:
            try:
                self._fetch_live_matches()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in live data fetch loop: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _fetch_live_matches(self):
        """Fetch live match data from API"""
        import requests
        
        try:
            api_key = self.api_config.get_api_key()
            if not api_key:
                logger.warning("No API key available for live data")
                return
            
            url = "https://v3.football.api-sports.io/fixtures"
            headers = {'x-apisports-key': api_key}
            params = {
                'live': 'all',
                'timezone': 'Europe/Istanbul'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                matches = data.get('response', []) if isinstance(data, dict) else []
                if isinstance(matches, list):
                    self._process_live_matches(matches)
                    logger.info(f"Fetched {len(matches)} live matches")
        except Exception as e:
            logger.error(f"Failed to fetch live matches: {str(e)}")
    
    def _process_live_matches(self, matches: List[Dict]):
        """Process live match updates"""
        updated_matches = []
        
        for match in matches:
            match_id = match.get('match_id')
            if not match_id:
                continue
            
            # Check if this is an update
            old_data = self.live_matches.get(match_id)
            if old_data:
                # Check for score changes
                if (old_data.get('match_hometeam_score') != match.get('match_hometeam_score') or
                    old_data.get('match_awayteam_score') != match.get('match_awayteam_score')):
                    updated_matches.append({
                        'match_id': match_id,
                        'type': 'goal',
                        'home_score': match.get('match_hometeam_score'),
                        'away_score': match.get('match_awayteam_score'),
                        'match': match
                    })
            
            self.live_matches[match_id] = match
        
        return updated_matches
    
    def get_live_match(self, match_id: str) -> Optional[Dict]:
        """Get a specific live match"""
        return self.live_matches.get(match_id)
    
    def get_all_live_matches(self) -> Dict[str, Dict]:
        """Get all live matches"""
        return self.live_matches.copy()

class PredictionUpdater:
    """Update predictions in real-time based on live data"""
    
    def __init__(self, predictor, event_manager):
        self.predictor = predictor
        self.event_manager = event_manager
        self.update_queue = queue.Queue()
        self.running = False
        self._thread = None
        logger.info("PredictionUpdater initialized")
    
    def start(self):
        """Start prediction updater"""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()
            logger.info("Prediction updater started")
    
    def stop(self):
        """Stop prediction updater"""
        self.running = False
        if self._thread:
            self._thread.join()
        logger.info("Prediction updater stopped")
    
    def queue_update(self, match_id: str, match_data: Dict):
        """Queue a match for prediction update"""
        self.update_queue.put((match_id, match_data))
    
    def _update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                # Get update from queue with timeout
                match_id, match_data = self.update_queue.get(timeout=1)
                self._update_prediction(match_id, match_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in prediction update loop: {str(e)}")
    
    def _update_prediction(self, match_id: str, match_data: Dict):
        """Update prediction for a match"""
        try:
            # Extract team IDs and names
            home_team_id = match_data.get('match_hometeam_id')
            away_team_id = match_data.get('match_awayteam_id')
            home_team_name = match_data.get('match_hometeam_name')
            away_team_name = match_data.get('match_awayteam_name')
            
            if not all([home_team_id, away_team_id]):
                logger.warning(f"Missing team IDs for match {match_id}")
                return
            
            # Generate updated prediction
            prediction = self.predictor.predict(
                home_team_id,
                away_team_id,
                home_team_name or "Home Team",
                away_team_name or "Away Team"
            )
            
            # Add live context
            prediction['live_context'] = {
                'current_score': {
                    'home': int(match_data.get('match_hometeam_score', 0)),
                    'away': int(match_data.get('match_awayteam_score', 0))
                },
                'match_status': match_data.get('match_status'),
                'match_time': match_data.get('match_time')
            }
            
            # Publish update event
            self.event_manager.publish('prediction.updated', {
                'match_id': match_id,
                'prediction': prediction,
                'reason': 'live_update'
            })
            
        except Exception as e:
            logger.error(f"Failed to update prediction for match {match_id}: {str(e)}")

class NotificationManager:
    """Manage push notifications for real-time events"""
    
    def __init__(self):
        self.notification_preferences = defaultdict(dict)
        self.notification_queue = queue.Queue()
        logger.info("NotificationManager initialized")
    
    def set_preferences(self, user_id: str, preferences: Dict):
        """Set notification preferences for a user"""
        self.notification_preferences[user_id] = preferences
        logger.info(f"Updated notification preferences for user {user_id}")
    
    def should_notify(self, user_id: str, event_type: str) -> bool:
        """Check if user should be notified for an event type"""
        prefs = self.notification_preferences.get(user_id, {})
        return prefs.get(event_type, False)
    
    def queue_notification(self, user_id: str, notification: Dict):
        """Queue a notification"""
        self.notification_queue.put({
            'user_id': user_id,
            'notification': notification,
            'timestamp': datetime.now().isoformat()
        })
    
    def send_notification(self, user_id: str, title: str, body: str, data: Optional[Dict] = None):
        """Send a push notification"""
        notification = {
            'title': title,
            'body': body,
            'data': data or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.queue_notification(user_id, notification)
        logger.info(f"Notification queued for user {user_id}: {title}")

class CollaborationManager:
    """Manage real-time collaboration features"""
    
    def __init__(self):
        self.active_sessions = defaultdict(set)
        self.session_data = {}
        self.chat_history = defaultdict(list)
        logger.info("CollaborationManager initialized")
    
    def join_session(self, session_id: str, user_id: str):
        """Join a collaboration session"""
        self.active_sessions[session_id].add(user_id)
        logger.info(f"User {user_id} joined session {session_id}")
    
    def leave_session(self, session_id: str, user_id: str):
        """Leave a collaboration session"""
        if user_id in self.active_sessions[session_id]:
            self.active_sessions[session_id].remove(user_id)
            logger.info(f"User {user_id} left session {session_id}")
    
    def update_session_data(self, session_id: str, data: Dict):
        """Update shared session data"""
        self.session_data[session_id] = data
        return list(self.active_sessions[session_id])
    
    def add_chat_message(self, session_id: str, user_id: str, message: str):
        """Add a chat message to session"""
        chat_entry = {
            'user_id': user_id,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.chat_history[session_id].append(chat_entry)
        
        # Keep chat history limited
        if len(self.chat_history[session_id]) > 100:
            self.chat_history[session_id].pop(0)
        
        return chat_entry

def create_websocket_server(app, predictor, api_config):
    """Create and configure WebSocket server"""
    global socketio
    
    if not SOCKETIO_AVAILABLE:
        logger.warning("SocketIO not available - WebSocket features disabled")
        return None
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Initialize managers
    event_manager = EventManager()
    live_fetcher = LiveDataFetcher(api_config)
    prediction_updater = PredictionUpdater(predictor, event_manager)
    notification_manager = NotificationManager()
    collaboration_manager = CollaborationManager()
    
    # Start background services
    live_fetcher.start()
    prediction_updater.start()
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'status': 'connected', 'sid': request.sid})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info(f"Client disconnected: {request.sid}")
        # Clean up any subscriptions
        for event_type, subscribers in event_manager.subscribers.items():
            if request.sid in subscribers:
                event_manager.unsubscribe(event_type, request.sid)
    
    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Handle event subscription"""
        event_type = data.get('event_type')
        if event_type:
            event_manager.subscribe(event_type, request.sid)
            emit('subscribed', {'event_type': event_type})
            
            # Send recent events
            recent_events = event_manager.get_recent_events(event_type, limit=5)
            emit('recent_events', {'events': recent_events})
    
    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        """Handle event unsubscription"""
        event_type = data.get('event_type')
        if event_type:
            event_manager.unsubscribe(event_type, request.sid)
            emit('unsubscribed', {'event_type': event_type})
    
    @socketio.on('get_live_matches')
    def handle_get_live_matches():
        """Get all live matches"""
        matches = live_fetcher.get_all_live_matches()
        emit('live_matches', {'matches': matches})
    
    @socketio.on('get_live_match')
    def handle_get_live_match(data):
        """Get specific live match"""
        match_id = data.get('match_id')
        if match_id:
            match = live_fetcher.get_live_match(match_id)
            emit('live_match', {'match_id': match_id, 'match': match})
    
    @socketio.on('join_room')
    def handle_join_room(data):
        """Join a room for real-time updates"""
        room = data.get('room')
        if room:
            join_room(room)
            emit('joined_room', {'room': room})
            logger.info(f"Client {request.sid} joined room {room}")
    
    @socketio.on('leave_room')
    def handle_leave_room(data):
        """Leave a room"""
        room = data.get('room')
        if room:
            leave_room(room)
            emit('left_room', {'room': room})
            logger.info(f"Client {request.sid} left room {room}")
    
    @socketio.on('set_notifications')
    def handle_set_notifications(data):
        """Set notification preferences"""
        preferences = data.get('preferences', {})
        notification_manager.set_preferences(request.sid, preferences)
        emit('notifications_set', {'preferences': preferences})
    
    @socketio.on('join_collaboration')
    def handle_join_collaboration(data):
        """Join collaboration session"""
        session_id = data.get('session_id')
        user_id = data.get('user_id', request.sid)
        
        if session_id:
            collaboration_manager.join_session(session_id, user_id)
            join_room(f"collab_{session_id}")
            
            # Send current session data
            session_data = collaboration_manager.session_data.get(session_id, {})
            chat_history = collaboration_manager.chat_history.get(session_id, [])
            
            emit('collaboration_joined', {
                'session_id': session_id,
                'session_data': session_data,
                'chat_history': chat_history[-20:],  # Last 20 messages
                'participants': list(collaboration_manager.active_sessions[session_id])
            })
    
    @socketio.on('collaboration_update')
    def handle_collaboration_update(data):
        """Handle collaboration data update"""
        session_id = data.get('session_id')
        update_data = data.get('data')
        
        if session_id and update_data:
            participants = collaboration_manager.update_session_data(session_id, update_data)
            
            # Broadcast to all participants
            socketio.emit('collaboration_data_updated', {
                'session_id': session_id,
                'data': update_data,
                'updated_by': request.sid
            }, to=f"collab_{session_id}")
    
    @socketio.on('send_chat')
    def handle_send_chat(data):
        """Handle chat message"""
        session_id = data.get('session_id')
        message = data.get('message')
        user_id = data.get('user_id', request.sid)
        
        if session_id and message:
            chat_entry = collaboration_manager.add_chat_message(session_id, user_id, message)
            
            # Broadcast to all participants
            socketio.emit('chat_message', {
                'session_id': session_id,
                'chat': chat_entry
            }, to=f"collab_{session_id}")
    
    # Background task to broadcast events
    def broadcast_events():
        """Broadcast queued events to subscribers"""
        while True:
            try:
                event = event_manager.event_queue.get(timeout=1)
                
                # Broadcast to subscribers
                for subscriber_id in event['subscribers']:
                    socketio.emit('event', event, to=subscriber_id)
                
                # Special handling for certain events
                if event['type'] == 'goal':
                    # Send goal notification
                    match_data = event['data'].get('match', {})
                    title = "GOAL!"
                    body = f"{match_data.get('match_hometeam_name')} {match_data.get('match_hometeam_score')} - {match_data.get('match_awayteam_score')} {match_data.get('match_awayteam_name')}"
                    
                    for subscriber_id in event['subscribers']:
                        if notification_manager.should_notify(subscriber_id, 'goals'):
                            notification_manager.send_notification(subscriber_id, title, body, event['data'])
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error broadcasting events: {str(e)}")
    
    # Start event broadcaster
    event_thread = threading.Thread(target=broadcast_events, daemon=True)
    event_thread.start()
    
    # Return objects for external access
    return {
        'socketio': socketio,
        'event_manager': event_manager,
        'live_fetcher': live_fetcher,
        'prediction_updater': prediction_updater,
        'notification_manager': notification_manager,
        'collaboration_manager': collaboration_manager
    }

# WebSocket client example
WEBSOCKET_CLIENT_EXAMPLE = """
// Example WebSocket client code
const socket = io('http://localhost:5000');

// Connection events
socket.on('connect', () => {
    console.log('Connected to WebSocket server');
    
    // Subscribe to live scores
    socket.emit('subscribe', { event_type: 'live_scores' });
    
    // Join a specific match room
    socket.emit('join_room', { room: 'match_12345' });
    
    // Set notification preferences
    socket.emit('set_notifications', {
        preferences: {
            goals: true,
            predictions: true,
            match_start: true
        }
    });
});

// Handle events
socket.on('event', (event) => {
    console.log('Received event:', event);
    
    if (event.type === 'goal') {
        // Handle goal event
        showGoalNotification(event.data);
    } else if (event.type === 'prediction.updated') {
        // Handle prediction update
        updatePrediction(event.data);
    }
});

// Handle live matches
socket.on('live_matches', (data) => {
    console.log('Live matches:', data.matches);
    updateLiveMatchList(data.matches);
});

// Collaboration example
socket.emit('join_collaboration', {
    session_id: 'match_analysis_12345',
    user_id: 'user123'
});

socket.on('collaboration_data_updated', (data) => {
    console.log('Collaboration update:', data);
    updateSharedAnalysis(data.data);
});

socket.emit('send_chat', {
    session_id: 'match_analysis_12345',
    message: 'Great prediction on this match!'
});
"""

if __name__ == "__main__":
    # Test event manager
    em = EventManager()
    em.subscribe('test_event', 'user1')
    em.subscribe('test_event', 'user2')
    
    event = em.publish('test_event', {'message': 'Hello World'})
    print(f"Published event: {event}")
    
    recent = em.get_recent_events('test_event')
    print(f"Recent events: {recent}")
    
    print("\n" + "="*50 + "\n")
    print("WebSocket Client Example:")
    print(WEBSOCKET_CLIENT_EXAMPLE)