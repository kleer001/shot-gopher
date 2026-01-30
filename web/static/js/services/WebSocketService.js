/**
 * WebSocketService - Real-time communication
 *
 * Manages WebSocket connection for real-time progress updates.
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Ping/pong keep-alive
 * - Connection lifecycle management
 * - Event-driven architecture
 *
 * Follows Single Responsibility Principle by only handling WebSocket communication.
 */

import { WEBSOCKET } from '../config/constants.js';

export class WebSocketService extends EventTarget {
    constructor() {
        super();
        this.ws = null;
        this.projectId = null;
        this.pingInterval = null;
        this.reconnectAttempts = 0;
        this.reconnectTimeout = null;
        this.shouldReconnect = false;
    }

    /**
     * Connect to WebSocket
     * @param {string} projectId - Project ID to connect to
     */
    connect(projectId) {
        this.projectId = projectId;
        this.shouldReconnect = true;
        this._connect();
    }

    /**
     * Internal connection logic
     * @private
     */
    _connect() {
        // Clean up existing connection without resetting shouldReconnect
        const shouldReconnect = this.shouldReconnect;
        this.disconnect();
        this.shouldReconnect = shouldReconnect;

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.projectId}`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => this._handleOpen();
            this.ws.onmessage = (event) => this._handleMessage(event);
            this.ws.onclose = () => this._handleClose();
            this.ws.onerror = (error) => this._handleError(error);
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this._scheduleReconnect();
        }
    }

    /**
     * Handle WebSocket open
     * @private
     */
    _handleOpen() {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;

        // Dispatch connected event
        this.dispatchEvent(new CustomEvent('connected'));

        // Start ping interval
        this._startPing();
    }

    /**
     * Handle incoming message
     * @private
     */
    _handleMessage(event) {
        try {
            const data = JSON.parse(event.data);

            // Ignore ping/pong messages
            if (data.type === 'ping' || data.type === 'pong') {
                return;
            }

            // Dispatch message event
            this.dispatchEvent(new CustomEvent('message', {
                detail: data,
            }));
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }

    /**
     * Handle WebSocket close
     * @private
     */
    _handleClose() {
        console.log('WebSocket disconnected');

        // Stop ping interval
        this._stopPing();

        // Dispatch disconnected event
        this.dispatchEvent(new CustomEvent('disconnected'));

        // Attempt reconnection if needed
        if (this.shouldReconnect) {
            this._scheduleReconnect();
        }
    }

    /**
     * Handle WebSocket error
     * @private
     */
    _handleError(error) {
        console.error('WebSocket error:', error);
        this.dispatchEvent(new CustomEvent('error', {
            detail: error,
        }));
    }

    /**
     * Schedule reconnection with exponential backoff
     * @private
     */
    _scheduleReconnect() {
        if (this.reconnectAttempts >= WEBSOCKET.MAX_RECONNECT_ATTEMPTS) {
            console.error('Max reconnection attempts reached');
            this.dispatchEvent(new CustomEvent('maxReconnectAttemptsReached'));
            return;
        }

        // Exponential backoff: 2s, 4s, 8s, 16s, 32s
        const delay = WEBSOCKET.RECONNECT_DELAY * Math.pow(2, this.reconnectAttempts);
        this.reconnectAttempts++;

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        this.reconnectTimeout = setTimeout(() => {
            this._connect();
        }, delay);
    }

    /**
     * Start ping interval to keep connection alive
     * @private
     */
    _startPing() {
        this._stopPing(); // Clear any existing interval

        this.pingInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            } else {
                this._stopPing();
            }
        }, WEBSOCKET.PING_INTERVAL);
    }

    /**
     * Stop ping interval
     * @private
     */
    _stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    /**
     * Disconnect WebSocket
     */
    disconnect() {
        this.shouldReconnect = false;

        // Clear reconnect timeout
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }

        // Stop ping
        this._stopPing();

        // Close WebSocket
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.projectId = null;
        this.reconnectAttempts = 0;
    }

    /**
     * Check if connected
     */
    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }

    /**
     * Send message
     * @param {Object} data - Data to send
     */
    send(data) {
        if (this.isConnected()) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.warn('Cannot send message: WebSocket not connected');
        }
    }
}

// Export singleton instance
export const wsService = new WebSocketService();
