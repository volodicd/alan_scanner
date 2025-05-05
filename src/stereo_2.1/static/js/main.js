// main.js - Core functionality for the application
let isStreaming = false;
let currentMode = 'idle';

// Add message to activity log (referenced in multiple files)
function addToActivityLog(message) {
    const activityLog = document.getElementById('recent-activity');
    if (!activityLog) return;

    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.innerHTML = `<span class="text-muted">[${timestamp}]</span> ${message}`;

    // Remove 'No recent activity' message if it exists
    const noActivityMsg = activityLog.querySelector('.text-muted');
    if (noActivityMsg && noActivityMsg.textContent === 'No recent activity') {
        activityLog.removeChild(noActivityMsg);
    }

    activityLog.appendChild(logEntry);
    activityLog.scrollTop = activityLog.scrollHeight;
}

// Show toast notification
function showToast(title, message, type = 'info') {
    // Toast creation logic here
}

// Document ready setup
document.addEventListener('DOMContentLoaded', function() {
    // Setup tabs & initial status
});