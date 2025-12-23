/**
 * WebSocket Client for Real-time Updates
 * Connects to the Football Predictor WebSocket server for live updates
 */

class FootballWebSocketClient {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.subscriptions = new Set();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.eventHandlers = {};
    }

    /**
     * Connect to WebSocket server
     */
    connect() {
        try {
            // Use the same host as the current page
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/socket.io/`;
            
            console.log('Connecting to WebSocket server:', wsUrl);
            
            // Initialize Socket.IO connection
            this.socket = io(wsUrl, {
                reconnection: true,
                reconnectionAttempts: this.maxReconnectAttempts,
                reconnectionDelay: this.reconnectDelay,
                transports: ['websocket', 'polling']
            });

            // Set up event handlers
            this.setupEventHandlers();
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
        }
    }

    /**
     * Set up WebSocket event handlers
     */
    setupEventHandlers() {
        // Connection events
        this.socket.on('connect', () => {
            this.connected = true;
            this.reconnectAttempts = 0;
            console.log('WebSocket connected!', this.socket.id);
            
            // Re-subscribe to previous subscriptions
            this.subscriptions.forEach(eventType => {
                this.subscribe(eventType);
            });
            
            this.emit('connection.status', { connected: true });
        });

        this.socket.on('disconnect', () => {
            this.connected = false;
            console.log('WebSocket disconnected');
            this.emit('connection.status', { connected: false });
        });

        this.socket.on('connect_error', (error) => {
            console.error('WebSocket connection error:', error);
            this.reconnectAttempts++;
        });

        // Custom events
        this.socket.on('event', (data) => {
            console.log('Received event:', data);
            this.handleEvent(data);
        });

        this.socket.on('prediction.updated', (data) => {
            console.log('Prediction updated:', data);
            this.emit('prediction.updated', data);
        });

        this.socket.on('match.live_update', (data) => {
            console.log('Live match update:', data);
            this.emit('match.live_update', data);
        });

        this.socket.on('goal', (data) => {
            console.log('GOAL!', data);
            this.emit('goal', data);
            this.showGoalNotification(data);
        });

        this.socket.on('recent_events', (data) => {
            console.log('Recent events:', data);
            if (data.events) {
                data.events.forEach(event => this.handleEvent(event));
            }
        });
    }

    /**
     * Subscribe to an event type
     */
    subscribe(eventType) {
        if (!this.connected) {
            console.warn('Not connected to WebSocket server');
            return;
        }

        this.socket.emit('subscribe', { event_type: eventType });
        this.subscriptions.add(eventType);
        console.log(`Subscribed to ${eventType}`);
    }

    /**
     * Unsubscribe from an event type
     */
    unsubscribe(eventType) {
        if (!this.connected) return;

        this.socket.emit('unsubscribe', { event_type: eventType });
        this.subscriptions.delete(eventType);
        console.log(`Unsubscribed from ${eventType}`);
    }

    /**
     * Subscribe to match updates
     */
    subscribeToMatch(matchId) {
        this.subscribe(`match.${matchId}`);
        this.socket.emit('match_subscribe', { 
            match_id: matchId,
            events: ['score', 'prediction', 'stats']
        });
    }

    /**
     * Register an event handler
     */
    on(eventType, handler) {
        if (!this.eventHandlers[eventType]) {
            this.eventHandlers[eventType] = [];
        }
        this.eventHandlers[eventType].push(handler);
    }

    /**
     * Remove an event handler
     */
    off(eventType, handler) {
        if (this.eventHandlers[eventType]) {
            this.eventHandlers[eventType] = this.eventHandlers[eventType].filter(h => h !== handler);
        }
    }

    /**
     * Emit an event to handlers
     */
    emit(eventType, data) {
        if (this.eventHandlers[eventType]) {
            this.eventHandlers[eventType].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${eventType}:`, error);
                }
            });
        }
    }

    /**
     * Handle incoming events
     */
    handleEvent(event) {
        const { type, data } = event;
        this.emit(type, data);
    }

    /**
     * Show goal notification
     */
    showGoalNotification(data) {
        const match = data.match || {};
        const title = 'GOL!';
        const body = `${match.match_hometeam_name} ${match.match_hometeam_score} - ${match.match_awayteam_score} ${match.match_awayteam_name}`;
        
        // Check if browser supports notifications
        if ('Notification' in window) {
            if (Notification.permission === 'granted') {
                new Notification(title, { body, icon: '/static/icon.png' });
            } else if (Notification.permission !== 'denied') {
                Notification.requestPermission().then(permission => {
                    if (permission === 'granted') {
                        new Notification(title, { body, icon: '/static/icon.png' });
                    }
                });
            }
        }
        
        // Also show in-page notification
        this.showInPageNotification(title, body, 'goal');
    }

    /**
     * Show in-page notification
     */
    showInPageNotification(title, message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `websocket-notification websocket-notification-${type}`;
        notification.innerHTML = `
            <div class="notification-title">${title}</div>
            <div class="notification-message">${message}</div>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => notification.classList.add('show'), 10);
        
        // Remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }

    /**
     * Join collaboration session
     */
    joinCollaboration(sessionId, userId) {
        if (!this.connected) return;
        
        this.socket.emit('join_collaboration', {
            session_id: sessionId,
            user_id: userId
        });
    }

    /**
     * Send chat message
     */
    sendChatMessage(sessionId, message) {
        if (!this.connected) return;
        
        this.socket.emit('send_chat', {
            session_id: sessionId,
            message: message
        });
    }

    /**
     * Update collaboration data
     */
    updateCollaborationData(sessionId, data) {
        if (!this.connected) return;
        
        this.socket.emit('collaboration_update', {
            session_id: sessionId,
            data: data
        });
    }

    /**
     * Set notification preferences
     */
    setNotificationPreferences(preferences) {
        if (!this.connected) return;
        
        this.socket.emit('set_notifications', {
            preferences: preferences
        });
    }

    /**
     * Disconnect from server
     */
    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.connected = false;
            this.subscriptions.clear();
        }
    }
}

// Create global instance
window.footballWebSocket = new FootballWebSocketClient();

// Auto-connect when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Connect to WebSocket server
    window.footballWebSocket.connect();
    
    // Subscribe to general events
    window.footballWebSocket.on('connection.status', (data) => {
        const statusEl = document.getElementById('websocket-status');
        if (statusEl) {
            statusEl.textContent = data.connected ? 'Bağlı' : 'Bağlantı Kesildi';
            statusEl.className = data.connected ? 'connected' : 'disconnected';
        }
    });
    
    // Subscribe to prediction updates
    window.footballWebSocket.on('prediction.updated', (data) => {
        console.log('Tahmin güncellendi:', data);
        // Update UI with new prediction
        const matchEl = document.querySelector(`[data-match-id="${data.match_id}"]`);
        if (matchEl) {
            // Refresh prediction display
            matchEl.querySelector('.refresh-btn')?.click();
        }
    });
    
    // Subscribe to live updates
    window.footballWebSocket.on('match.live_update', (data) => {
        console.log('Canlı skor güncellemesi:', data);
        // Update match score in UI
        updateMatchScore(data);
    });
    
    // Subscribe to goal events
    window.footballWebSocket.on('goal', (data) => {
        console.log('GOL bildirimi:', data);
        // Special goal animation
        showGoalAnimation(data);
    });
});

// Helper functions for UI updates
function updateMatchScore(data) {
    const matchEl = document.querySelector(`[data-match-id="${data.match_id}"]`);
    if (matchEl) {
        const scoreEl = matchEl.querySelector('.match-score');
        if (scoreEl) {
            scoreEl.textContent = `${data.home_score} - ${data.away_score}`;
            scoreEl.classList.add('score-updated');
            setTimeout(() => scoreEl.classList.remove('score-updated'), 2000);
        }
    }
}

function showGoalAnimation(data) {
    // Create goal overlay
    const overlay = document.createElement('div');
    overlay.className = 'goal-overlay';
    overlay.innerHTML = `
        <div class="goal-animation">
            <div class="goal-text">GOL!</div>
            <div class="goal-details">
                ${data.team_name} - ${data.player_name || 'Bilinmeyen'} ${data.minute}'
            </div>
        </div>
    `;
    
    document.body.appendChild(overlay);
    
    // Remove after animation
    setTimeout(() => overlay.remove(), 3000);
}