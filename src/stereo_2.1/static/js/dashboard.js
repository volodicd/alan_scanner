// dashboard.js
function startStream() {
    fetch('/api/stream/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('start-stream-btn').disabled = true;
            document.getElementById('stop-stream-btn').disabled = false;
            document.getElementById('capture-frame-btn').disabled = false;

            addToActivityLog('Stream started');
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

            // Update calibration status if we also have disparity map
            if (data.disparity_path) {
                document.getElementById('calibration-status').textContent = 'Calibrated';
                document.getElementById('calibration-status').className = 'text-success';
            }
        } else {
            showToast('Error', data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error capturing frame:', error);
        showToast('Error', 'Failed to capture frame', 'danger');
    });
}

function checkCalibrationStatus() {
    fetch('/api/calibrate/status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const calibrationStatus = document.getElementById('calibration-status');
                if (data.has_calibration) {
                    calibrationStatus.textContent = 'Calibrated';
                    calibrationStatus.className = 'text-success';

                    // If we have calibration info, show it
                    if (data.calibration_info && data.calibration_info.date) {
                        calibrationStatus.textContent = `Calibrated (${data.calibration_info.date})`;
                    }
                } else {
                    calibrationStatus.textContent = 'Not calibrated';
                    calibrationStatus.className = 'text-danger';
                }
            }
        })
        .catch(error => {
            console.error('Error checking calibration status:', error);
        });
}

function initDashboard() {
    document.getElementById('start-stream-btn').addEventListener('click', startStream);
    document.getElementById('stop-stream-btn').addEventListener('click', stopStream);
    document.getElementById('capture-frame-btn').addEventListener('click', captureFrame);

    // Check calibration status
    checkCalibrationStatus();
}

document.addEventListener('DOMContentLoaded', initDashboard);