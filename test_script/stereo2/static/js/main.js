// main.js - Core functionality for the application

// Global variables
let isStreaming = false;
let currentMode = 'idle';

// Initialize the page when DOM content is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Setup tab event listeners for proper initialization
    setupTabListeners();

    // Initial status check
    fetchStatus();

    // Create placeholder images if needed
    checkPlaceholderImages();
});

// Set up tab change event listeners
function setupTabListeners() {
    // Code tab special handling (initialize editor when shown)
    document.querySelector('#code-tab').addEventListener('shown.bs.tab', function(event) {
        if (typeof initializeCodeEditor === 'function') {
            initializeCodeEditor();
        }
    });

    // Initialize bootstrap tabs
    const tabElements = document.querySelectorAll('#mainTab [data-bs-toggle="tab"]');
    tabElements.forEach(tabEl => {
        new bootstrap.Tab(tabEl);
    });
}

// Fetch current system status
function fetchStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            isStreaming = data.is_streaming;
            currentMode = data.current_mode;

            // Update UI based on status
            document.getElementById('current-mode').textContent = currentMode.charAt(0).toUpperCase() + currentMode.slice(1);
            document.getElementById('start-stream-btn').disabled = isStreaming;
            document.getElementById('stop-stream-btn').disabled = !isStreaming;
            document.getElementById('capture-frame-btn').disabled = !isStreaming;

            // Additional status updates
            if (isStreaming) {
                document.getElementById('left-camera-status').textContent = 'Connected';
                document.getElementById('right-camera-status').textContent = 'Connected';
            } else {
                document.getElementById('left-camera-status').textContent = 'Not connected';
                document.getElementById('right-camera-status').textContent = 'Not connected';
            }

            // Check if calibration exists
            if (data.config && data.config.calibration_exists) {
                document.getElementById('calibration-status').textContent = 'Calibrated';
            } else {
                document.getElementById('calibration-status').textContent = 'Not calibrated';
            }
        })
        .catch(error => {
            console.error('Error fetching status:', error);
            addToActivityLog('Error fetching status: ' + error.message);
        });
}

// Add message to activity log
function addToActivityLog(message) {
    const activityLog = document.getElementById('recent-activity');
    if (!activityLog) return; // Early return if element not found

    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.innerHTML = `<span class="text-muted">[${timestamp}]</span> ${message}`;

    // Remove 'No recent activity' message if it exists
    const noActivityMsg = activityLog.querySelector('.text-muted');
    if (noActivityMsg && noActivityMsg.textContent === 'No recent activity') {
        activityLog.removeChild(noActivityMsg);
    }

    // Add new log entry
    activityLog.appendChild(logEntry);

    // Auto-scroll to bottom
    activityLog.scrollTop = activityLog.scrollHeight;
}

// Show toast notification
function showToast(title, message, type = 'info') {
    const toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) return; // Early return if container not found

    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast show bg-${type} text-white`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');

    const toastHeader = document.createElement('div');
    toastHeader.className = 'toast-header bg-dark text-white';

    const toastTitle = document.createElement('strong');
    toastTitle.className = 'me-auto';
    toastTitle.textContent = title;

    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'btn-close btn-close-white';
    closeButton.setAttribute('data-bs-dismiss', 'toast');
    closeButton.setAttribute('aria-label', 'Close');

    toastHeader.appendChild(toastTitle);
    toastHeader.appendChild(closeButton);

    const toastBody = document.createElement('div');
    toastBody.className = 'toast-body';
    toastBody.textContent = message;

    toast.appendChild(toastHeader);
    toast.appendChild(toastBody);

    // Add toast to container
    toastContainer.appendChild(toast);

    // Set timeout to remove toast after 5 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toastContainer.removeChild(toast);
        }, 500);
    }, 5000);

    // Add event listener to close button
    closeButton.addEventListener('click', () => {
        toast.classList.remove('show');
        setTimeout(() => {
            toastContainer.removeChild(toast);
        }, 500);
    });
}

// Create placeholder images if needed
function checkPlaceholderImages() {
    // Paths to check
    const placeholderImgUrl = "/static/placeholder-camera.jpg";
    const placeholderDispUrl = "/static/placeholder-disparity.jpg";

    // Check if camera placeholder exists
    fetch(placeholderImgUrl)
        .catch(() => {
            console.log("Creating placeholder images...");
            createPlaceholderImages();
        });
}

// Create placeholder images for cameras and disparity map
function createPlaceholderImages() {
    // Create a placeholder camera image
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');

    // Fill background
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Add text
    ctx.font = '24px Arial';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.fillText('No Camera Feed', canvas.width / 2, canvas.height / 2 - 12);
    ctx.fillText('Start Stream to View', canvas.width / 2, canvas.height / 2 + 24);

    // Add camera icon
    ctx.font = '64px FontAwesome';
    ctx.fillText('\uf030', canvas.width / 2, canvas.height / 2 - 64);

    // Convert to blob and create object URL
    canvas.toBlob(function(blob) {
        const url = URL.createObjectURL(blob);

        // Update all camera placeholder images
        document.querySelectorAll('[id$="-camera"]').forEach(img => {
            img.src = url;
        });

        // Create disparity placeholder (different color)
        const dispCanvas = document.createElement('canvas');
        dispCanvas.width = 640;
        dispCanvas.height = 480;
        const dispCtx = dispCanvas.getContext('2d');

        // Fill background
        dispCtx.fillStyle = '#000';
        dispCtx.fillRect(0, 0, dispCanvas.width, dispCanvas.height);

        // Add text
        dispCtx.font = '24px Arial';
        dispCtx.fillStyle = '#fff';
        dispCtx.textAlign = 'center';
        dispCtx.fillText('No Disparity Map', dispCanvas.width / 2, dispCanvas.height / 2 - 12);
        dispCtx.fillText('Calibration Required', dispCanvas.width / 2, dispCanvas.height / 2 + 24);

        // Add icon
        dispCtx.font = '64px FontAwesome';
        dispCtx.fillText('\uf07e', dispCanvas.width / 2, dispCanvas.height / 2 - 64);

        dispCanvas.toBlob(function(dispBlob) {
            const dispUrl = URL.createObjectURL(dispBlob);
            document.getElementById('disparity-map').src = dispUrl;
        });
    });
}