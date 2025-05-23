// websocket.js - WebSocket connection and event handling

// Socket.IO connection instance
let socket;
let isConnected = false;
let eventHandlers = {};

// Initialize WebSocket connection
function connectWebSocket() {
    // Get the current host and port
    const host = window.location.hostname;
    const port = window.location.port;

    // Create Socket.IO connection with reconnection options
    socket = io(`http://${host}:${port}`, {
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000
    });

    // Socket.IO event handlers
    setupSocketEvents();
}

// Clean up socket event handlers
function cleanupSocketEvents() {
    if (socket) {
        for (const event in eventHandlers) {
            if (eventHandlers.hasOwnProperty(event)) {
                socket.off(event, eventHandlers[event]);
            }
        }
        eventHandlers = {};
    }
}

// Set up socket event listeners
function setupSocketEvents() {
    // Clean up any existing handlers first
    cleanupSocketEvents();

    // Connect event
    eventHandlers.connect = function() {
        isConnected = true;
        updateConnectionStatus(true);
        addToActivityLog('Connected to server');
    };
    socket.on('connect', eventHandlers.connect);

    // Disconnect event
    eventHandlers.disconnect = function() {
        isConnected = false;
        updateConnectionStatus(false);
        addToActivityLog('Disconnected from server');
    };
    socket.on('disconnect', eventHandlers.disconnect);

    // Reconnect attempt event
    eventHandlers.reconnect_attempt = function(attemptNumber) {
        console.log(`Reconnection attempt: ${attemptNumber}`);
        updateConnectionStatus(false, `Reconnecting (${attemptNumber})...`);
    };
    socket.on('reconnect_attempt', eventHandlers.reconnect_attempt);

    // Reconnect event
    eventHandlers.reconnect = function() {
        console.log('Reconnected to server');
        updateConnectionStatus(true);

        // Synchronize state after reconnection
        syncStateAfterReconnect();
    };
    socket.on('reconnect', eventHandlers.reconnect);

    // Reconnect failed event
    eventHandlers.reconnect_failed = function() {
        console.log('Failed to reconnect');
        updateConnectionStatus(false, 'Reconnection failed');
        showToast('Error', 'Failed to reconnect to server', 'danger');
    };
    socket.on('reconnect_failed', eventHandlers.reconnect_failed);

    // Frames event
    eventHandlers.frames = function(data) {
        updateCameraFeeds(data);
    };
    socket.on('frames', eventHandlers.frames);

    // Error event
    eventHandlers.error = function(data) {
        showToast('Error', data.message, 'danger');
        addToActivityLog('Error: ' + data.message);
    };
    socket.on('error', eventHandlers.error);

    // Status event
    eventHandlers.status = function(data) {
        addToActivityLog(data.message);
    };
    socket.on('status', eventHandlers.status);

    // Calibration status event
    eventHandlers.calibration_status = function(data) {
        if (typeof updateCalibrationStatus === 'function') {
            updateCalibrationStatus(data);
        }
    };
    socket.on('calibration_status', eventHandlers.calibration_status);

    // Calibration complete event
    eventHandlers.calibration_complete = function(data) {
        if (typeof handleCalibrationComplete === 'function') {
            handleCalibrationComplete(data);
        }
    };
    socket.on('calibration_complete', eventHandlers.calibration_complete);
}

// Update connection status indicator
function updateConnectionStatus(isConnected, message = null) {
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');

    if (!statusIndicator || !statusText) return;

    if (isConnected) {
        statusIndicator.classList.remove('status-offline');
        statusIndicator.classList.add('status-online');
        statusText.textContent = 'Connected';
    } else {
        statusIndicator.classList.remove('status-online');
        statusIndicator.classList.add('status-offline');
        statusText.textContent = message || 'Disconnected';
    }
}

// Update camera feeds with received frame data
function updateCameraFeeds(data) {
    // Update camera feeds
    if (data.left) {
        document.querySelectorAll('[id$="left-camera"]').forEach(element => {
            element.src = 'data:image/jpeg;base64,' + data.left;
        });
    }

    if (data.right) {
        document.querySelectorAll('[id$="right-camera"]').forEach(element => {
            element.src = 'data:image/jpeg;base64,' + data.right;
        });
    }

    if (data.disparity) {
        const disparityMap = document.getElementById('disparity-map');
        if (disparityMap) {
            disparityMap.src = 'data:image/jpeg;base64,' + data.disparity;
        }
    }

    // Update FPS counter if provided
    if (data.fps !== undefined) {
        const fpsElement = document.getElementById('stream-fps');
        if (fpsElement) {
            fpsElement.textContent = data.fps;
        }
    }

    // Update camera status indicators
    const leftCameraStatus = document.getElementById('left-camera-status');
    const rightCameraStatus = document.getElementById('right-camera-status');

    if (leftCameraStatus) {
        leftCameraStatus.textContent = data.left ? 'Connected' : 'Not connected';
        leftCameraStatus.className = data.left ? 'text-success' : 'text-danger';
    }

    if (rightCameraStatus) {
        rightCameraStatus.textContent = data.right ? 'Connected' : 'Not connected';
        rightCameraStatus.className = data.right ? 'text-success' : 'text-danger';
    }
}

// Synchronize state after reconnection
function syncStateAfterReconnect() {
    // Check if streaming was active
    const streamBtn = document.getElementById('start-stream-btn');
    const stopBtn = document.getElementById('stop-stream-btn');

    if (streamBtn && streamBtn.disabled && stopBtn && !stopBtn.disabled) {
        // We were streaming, restart it
        fetch('/api/stream/start', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    showToast('Warning', 'Failed to restart stream after reconnection', 'warning');
                }
            })
            .catch(error => {
                console.error('Error restarting stream:', error);
            });
    }

    // Check current tab to refresh its data
    const activeTabId = document.querySelector('.nav-link.active')?.id;

    if (activeTabId === 'dashboard-tab') {
        // Check calibration status for dashboard
        if (typeof checkCalibrationStatus === 'function') {
            checkCalibrationStatus();
        }
    } else if (activeTabId === 'settings-tab') {
        // Reload settings
        if (typeof loadCurrentSettings === 'function') {
            window.settingsLoaded = false;
            loadCurrentSettings();
        }
    } else if (activeTabId === 'calibration-tab') {
        // Check calibration status
        fetch('/api/calibrate/status');
    }
}

// Clean up when window/document is unloaded
window.addEventListener('beforeunload', function() {
    cleanupSocketEvents();
});

// Trigger connection when document is ready
document.addEventListener('DOMContentLoaded', connectWebSocket);