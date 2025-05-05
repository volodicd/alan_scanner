// websocket.js - WebSocket connection and event handling

// Socket.IO connection instance
let socket;
let isConnected = false;

// Initialize WebSocket connection
function connectWebSocket() {
    // Get the current host and port
    const host = window.location.hostname;
    const port = window.location.port;

    // Create Socket.IO connection
    socket = io(`http://${host}:${port}`);

    // Socket.IO event handlers
    setupSocketEvents();
}

// Set up socket event listeners
function setupSocketEvents() {
    socket.on('connect', function() {
        isConnected = true;
        updateConnectionStatus(true);
        addToActivityLog('Connected to server');
    });

    socket.on('disconnect', function() {
        isConnected = false;
        updateConnectionStatus(false);
        addToActivityLog('Disconnected from server');
    });

    socket.on('frames', function(data) {
        updateCameraFeeds(data);
    });

    socket.on('error', function(data) {
        showToast('Error', data.message, 'danger');
        addToActivityLog('Error: ' + data.message);
    });

    socket.on('status', function(data) {
        addToActivityLog(data.message);
    });

    socket.on('calibration_status', function(data) {
        if (typeof updateCalibrationStatus === 'function') {
            updateCalibrationStatus(data);
        }
    });

    socket.on('calibration_capture', function(data) {
        handleCalibrationCapture(data);
    });
}

// Update connection status indicator
function updateConnectionStatus(isConnected) {
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
        statusText.textContent = 'Disconnected';
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

    // Extract FPS from the frame if available (displayed on the image)
    // This is just a placeholder - actual FPS would be extracted differently
    const streamFps = document.getElementById('stream-fps');
    if (streamFps) {
        streamFps.textContent = 'Streaming';
    }
}

// Handle calibration capture events
function handleCalibrationCapture(data) {
    if (data.success) {
        // Handle auto-capture event from server
        if (data.auto_captured) {
            const message = `Auto-captured frame pair ${data.pair_count}/${data.needed_pairs}`;
            addToActivityLog(message);

            if (typeof addToCalibrationLog === 'function') {
                addToCalibrationLog(message);
                addToCalibrationLog(`Saved to ${data.left_path} and ${data.right_path}`);
            }

            // Update list and progress if functions exist
            if (typeof addCalibrationPair === 'function') {
                addCalibrationPair(data.timestamp, data.left_path, data.right_path);
            }

            // Update progress bar
            updateCalibrationProgress(data);
        }
    }
}

// Update calibration progress indicators
function updateCalibrationProgress(data) {
    const progressBar = document.getElementById('captured-progress');
    const pairsCount = document.getElementById('captured-pairs-count');

    if (!progressBar || !pairsCount) return;

    if (data.pair_count !== undefined && data.needed_pairs !== undefined) {
        pairsCount.textContent = data.pair_count + '/' + data.needed_pairs;

        const progress = Math.min((data.pair_count / data.needed_pairs) * 100, 100);
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);

        if (data.pair_count >= data.needed_pairs) {
            progressBar.classList.add('bg-success');

            if (typeof addToCalibrationLog === 'function') {
                addToCalibrationLog(`You now have enough image pairs. Ready to process calibration.`);
            }
        }
    }
}

// Trigger connection when document is ready
document.addEventListener('DOMContentLoaded', connectWebSocket);