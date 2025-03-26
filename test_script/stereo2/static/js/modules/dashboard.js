// dashboard.js
function startStream() {
    fetch('/api/stream/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ mode: 'test' }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            isStreaming = true;
            document.getElementById('start-stream-btn').disabled = true;
            document.getElementById('stop-stream-btn').disabled = false;
            document.getElementById('capture-frame-btn').disabled = false;

            addToActivityLog('Stream started in test mode');
        } else {
            showToast('Error', data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error starting stream:', error);
        showToast('Error', 'Failed to start stream', 'danger');
    });
}

function stopStream() {
    fetch('/api/stream/stop', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            isStreaming = false;
            document.getElementById('start-stream-btn').disabled = false;
            document.getElementById('stop-stream-btn').disabled = true;
            document.getElementById('capture-frame-btn').disabled = true;

            addToActivityLog('Stream stopped');
            document.getElementById('stream-fps').textContent = '0';
        } else {
            showToast('Error', data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error stopping stream:', error);
        showToast('Error', 'Failed to stop stream', 'danger');
    });
}

function captureFrame() {
    fetch('/api/capture', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addToActivityLog(`Frame captured: ${data.timestamp}`);
            showToast('Success', 'Frame captured successfully', 'success');
        } else {
            showToast('Error', data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error capturing frame:', error);
        showToast('Error', 'Failed to capture frame', 'danger');
    });
}

function addToActivityLog(message) {
    const activityLog = document.getElementById('recent-activity');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.innerHTML = `<span class="text-muted">[${timestamp}]</span> ${message}`;

    // Remove 'No recent activity' message if it exists
    const noActivityMsg = activityLog.querySelector('.text-muted');
    if (noActivityMsg && noActivityMsg.textContent === 'No recent activity') {
        activityLog.removeChild(noActivityMsg);
    }

    activityLog.appendChild(logEntry);

    // Auto-scroll to bottom
    activityLog.scrollTop = activityLog.scrollHeight;
}

function initDashboard() {
    document.getElementById('start-stream-btn').addEventListener('click', startStream);
    document.getElementById('stop-stream-btn').addEventListener('click', stopStream);
    document.getElementById('capture-frame-btn').addEventListener('click', captureFrame);
}

document.addEventListener('DOMContentLoaded', initDashboard);