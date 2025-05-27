// websocket.js - Fixed version
class WebSocketManager {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;

        this.initializeSocket();
    }

    initializeSocket() {
        console.log('🔌 Initializing SocketIO connection...');

        // Initialize socket connection
        this.socket = io({
            transports: ['websocket', 'polling'],
            timeout: 20000,
            forceNew: true
        });

        // Connection events
        this.socket.on('connect', () => {
            console.log('✅ SocketIO connected:', this.socket.id);
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus(true);
        });

        this.socket.on('disconnect', (reason) => {
            console.log('❌ SocketIO disconnected:', reason);
            this.isConnected = false;
            this.updateConnectionStatus(false);

            // Clear camera feeds
            this.clearCameraFeeds();
        });

        this.socket.on('connect_error', (error) => {
            console.error('🔥 SocketIO connection error:', error);
            this.handleReconnect();
        });

        // Frame data handler - THIS IS THE KEY FIX
        this.socket.on('frames', (data) => {
            console.log('📸 Received frame data:', {
                success: data.success,
                hasLeft: !!data.left,
                hasRight: !!data.right,
                hasDisparity: !!data.disparity,
                timestamp: data.timestamp
            });

            if (data.success) {
                this.updateCameraFeeds(data);
            } else {
                console.error('❌ Frame data error:', data.message);
                this.clearCameraFeeds();
            }
        });

        // Error handler
        this.socket.on('error', (error) => {
            console.error('🚨 SocketIO error:', error);
        });
    }

    updateCameraFeeds(frameData) {
        try {
            // Update DASHBOARD tab cameras
            if (frameData.left) {
                const leftImg = document.getElementById('left-camera');
                if (leftImg) {
                    leftImg.src = `data:image/jpeg;base64,${frameData.left}`;
                    console.log('✅ Updated Dashboard left camera');
                }

                // Update status
                const leftStatus = document.getElementById('left-camera-status');
                if (leftStatus) {
                    leftStatus.textContent = 'Connected';
                    leftStatus.className = 'text-success';
                }
            }

            if (frameData.right) {
                const rightImg = document.getElementById('right-camera');
                if (rightImg) {
                    rightImg.src = `data:image/jpeg;base64,${frameData.right}`;
                    console.log('✅ Updated Dashboard right camera');
                }

                // Update status
                const rightStatus = document.getElementById('right-camera-status');
                if (rightStatus) {
                    rightStatus.textContent = 'Connected';
                    rightStatus.className = 'text-success';
                }
            }

            // Update CALIBRATION tab cameras (different IDs)
            if (frameData.left) {
                const calibLeftImg = document.getElementById('calib-left-camera');
                if (calibLeftImg) {
                    calibLeftImg.src = `data:image/jpeg;base64,${frameData.left}`;
                    console.log('✅ Updated Calibration left camera');
                }
            }

            if (frameData.right) {
                const calibRightImg = document.getElementById('calib-right-camera');
                if (calibRightImg) {
                    calibRightImg.src = `data:image/jpeg;base64,${frameData.right}`;
                    console.log('✅ Updated Calibration right camera');
                }
            }

            // Update disparity if available (matches your existing HTML ID)
            if (frameData.disparity) {
                const disparityImg = document.getElementById('disparity-map');
                if (disparityImg) {
                    disparityImg.src = `data:image/jpeg;base64,${frameData.disparity}`;
                    console.log('✅ Updated disparity map');
                }
            }

            // Update FPS if available (matches your existing HTML ID)
            if (frameData.fps) {
                const fpsDisplay = document.getElementById('stream-fps');
                if (fpsDisplay) {
                    fpsDisplay.textContent = `${frameData.fps}`;
                }
            }

        } catch (error) {
            console.error('❌ Error updating camera feeds:', error);
        }
    }

    clearCameraFeeds() {
        console.log('🧹 Clearing camera feeds');

        // Reset DASHBOARD tab to placeholder images
        const leftImg = document.getElementById('left-camera');
        if (leftImg) {
            leftImg.src = '/static/img/placeholder-camera.jpg';
        }

        const rightImg = document.getElementById('right-camera');
        if (rightImg) {
            rightImg.src = '/static/img/placeholder-camera.jpg';
        }

        const disparityImg = document.getElementById('disparity-map');
        if (disparityImg) {
            disparityImg.src = '/static/img/placeholder-disparity.jpg';
        }

        // Reset CALIBRATION tab to placeholder images
        const calibLeftImg = document.getElementById('calib-left-camera');
        if (calibLeftImg) {
            calibLeftImg.src = '/static/img/placeholder-camera.jpg';
        }

        const calibRightImg = document.getElementById('calib-right-camera');
        if (calibRightImg) {
            calibRightImg.src = '/static/img/placeholder-camera.jpg';
        }

        // Update status indicators
        const leftStatus = document.getElementById('left-camera-status');
        if (leftStatus) {
            leftStatus.textContent = 'Not connected';
            leftStatus.className = 'text-danger';
        }

        const rightStatus = document.getElementById('right-camera-status');
        if (rightStatus) {
            rightStatus.textContent = 'Not connected';
            rightStatus.className = 'text-danger';
        }

        const fpsDisplay = document.getElementById('stream-fps');
        if (fpsDisplay) {
            fpsDisplay.textContent = '0';
        }
    }

    updateConnectionStatus(connected) {
        // Update the main connection status indicator in title bar
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');

        if (statusIndicator && statusText) {
            if (connected) {
                statusIndicator.className = 'status-indicator status-online';
                statusText.textContent = 'Connected';
            } else {
                statusIndicator.className = 'status-indicator status-offline';
                statusText.textContent = 'Disconnected';
            }
        }
    }

    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`🔄 Reconnecting... Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);

            setTimeout(() => {
                this.socket.connect();
            }, this.reconnectDelay);
        } else {
            console.error('💀 Max reconnection attempts reached');
        }
    }

    startStream() {
        console.log('▶️ Starting video stream...');

        fetch('/api/stream/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('✅ Stream started successfully');

                // Update button states
                const startBtn = document.getElementById('start-stream-btn');
                const stopBtn = document.getElementById('stop-stream-btn');
                const captureBtn = document.getElementById('capture-frame-btn');

                if (startBtn) startBtn.disabled = true;
                if (stopBtn) stopBtn.disabled = false;
                if (captureBtn) captureBtn.disabled = false;

            } else {
                console.error('❌ Failed to start stream:', data.message);
                alert(`Failed to start stream: ${data.message}`);
            }
        })
        .catch(error => {
            console.error('🚨 Error starting stream:', error);
            alert(`Error starting stream: ${error.message}`);
        });
    }

    stopStream() {
        console.log('⏹️ Stopping video stream...');

        fetch('/api/stream/stop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log('Stream stop response:', data);
            this.clearCameraFeeds();

            // Update button states
            const startBtn = document.getElementById('start-stream-btn');
            const stopBtn = document.getElementById('stop-stream-btn');
            const captureBtn = document.getElementById('capture-frame-btn');

            if (startBtn) startBtn.disabled = false;
            if (stopBtn) stopBtn.disabled = true;
            if (captureBtn) captureBtn.disabled = true;
        })
        .catch(error => {
            console.error('Error stopping stream:', error);
        });
    }
}

// Initialize WebSocket manager when page loads
let webSocketManager;

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM loaded, initializing WebSocket manager...');
    webSocketManager = new WebSocketManager();

    // Connect the existing buttons to WebSocket functions
    const startBtn = document.getElementById('start-stream-btn');
    const stopBtn = document.getElementById('stop-stream-btn');

    if (startBtn) {
        startBtn.addEventListener('click', () => {
            console.log('▶️ Start button clicked');
            webSocketManager.startStream();
            startBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            console.log('⏹️ Stop button clicked');
            webSocketManager.stopStream();
            stopBtn.disabled = true;
            if (startBtn) startBtn.disabled = false;
        });
    }

    // Auto-start stream when page loads
    setTimeout(() => {
        if (startBtn && !startBtn.disabled) {
            startBtn.click();
        }
    }, 1000);
});

// Make it globally available
window.webSocketManager = webSocketManager;