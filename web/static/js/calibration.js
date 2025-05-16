// calibration.js
function updateCalibrationStatus(data) {
    // Update pattern detection indicators
    const leftIndicator = document.getElementById('left-pattern-indicator');
    const rightIndicator = document.getElementById('right-pattern-indicator');
    const bothIndicator = document.getElementById('both-pattern-indicator');

    if (data.left_found) {
        leftIndicator.className = 'fas fa-check-circle text-success';
    } else {
        leftIndicator.className = 'fas fa-times-circle text-danger';
    }

    if (data.right_found) {
        rightIndicator.className = 'fas fa-check-circle text-success';
    } else {
        rightIndicator.className = 'fas fa-times-circle text-danger';
    }

    if (data.both_found) {
        bothIndicator.className = 'fas fa-check-circle text-success';
    } else {
        bothIndicator.className = 'fas fa-times-circle text-danger';
    }

    // Update calibration progress if available
    if (data.pairs_captured !== undefined && data.pairs_recommended !== undefined) {
        document.getElementById('captured-pairs-count').textContent =
            data.pairs_captured + '/' + data.pairs_recommended;

        const progress = Math.min((data.pairs_captured / data.pairs_needed) * 100, 100);
        const progressBar = document.getElementById('captured-progress');
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);

        if (data.pairs_captured >= data.pairs_needed) {
            progressBar.classList.add('bg-success');
        }
    }
}

function handleCalibrationComplete(data) {
    if (data.success) {
        // Update calibration result display
        const resultDiv = document.getElementById('calibration-result');
        resultDiv.classList.remove('d-none');

        document.getElementById('rms-error').textContent = data.rms_error.toFixed(6);
        document.getElementById('calibration-date').textContent = data.date;

        // Update global calibration status
        document.getElementById('calibration-status').textContent = 'Calibrated';
        document.getElementById('calibration-status').className = 'text-success';

        // Enable buttons
        document.getElementById('start-calibration-btn').disabled = false;

        addToCalibrationLog(`Calibration completed with RMS error: ${data.rms_error.toFixed(6)}`);
        showToast('Success', 'Calibration processed successfully', 'success');
    } else {
        addToCalibrationLog(`ERROR: ${data.message || 'Unknown error'}`);
        showToast('Error', data.message || 'Calibration failed', 'danger');

        // Re-enable start button
        document.getElementById('start-calibration-btn').disabled = false;
    }
}

function addToCalibrationLog(message) {
    const logsContainer = document.getElementById('calibration-logs');
    if (!logsContainer) return;

    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.innerHTML = `<span class="text-muted">[${timestamp}]</span> ${message}`;

    // Remove "No logs yet" message if present
    const noLogsMsg = logsContainer.querySelector('.text-muted');
    if (noLogsMsg && noLogsMsg.textContent === 'No calibration logs yet') {
        logsContainer.removeChild(noLogsMsg);
    }

    // Add the log entry
    logsContainer.appendChild(logEntry);

    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;

    // Also add to main activity log
    addToActivityLog(message);
}

function startCalibration() {
    // Get checkerboard parameters
    const cols = parseInt(document.getElementById('checkerboard-cols').value);
    const rows = parseInt(document.getElementById('checkerboard-rows').value);
    const squareSize = parseFloat(document.getElementById('square-size').value);
    const autoCapture = document.getElementById('auto-capture-toggle').checked;
    const stabilityThreshold = parseFloat(document.getElementById('stability-threshold').value);

    // Validate inputs
    if (isNaN(cols) || isNaN(rows) || isNaN(squareSize) || cols < 3 || rows < 3) {
        showToast('Error', 'Invalid checkerboard parameters', 'danger');
        return;
    }

    // Disable button to prevent multiple starts
    document.getElementById('start-calibration-btn').disabled = true;

    // Clear previous logs
    const logsContainer = document.getElementById('calibration-logs');
    logsContainer.innerHTML = '<div class="text-muted">Starting calibration...</div>';

    fetch('/api/calibrate/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            auto_capture: autoCapture,
            stability_seconds: stabilityThreshold,
            checkerboard_size: [cols, rows],
            square_size: squareSize
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addToActivityLog('Calibration started');
            addToCalibrationLog('Calibration started');

            if (data.auto_capture) {
                addToCalibrationLog(`Auto-capture enabled (${data.stability_seconds}s threshold)`);
            } else {
                addToCalibrationLog('Manual capture mode');
            }

            // If there are existing pairs, log that
            if (data.existing_pairs > 0) {
                addToCalibrationLog(`Found ${data.existing_pairs} existing calibration pairs`);
            }
        } else {
            showToast('Error', data.message, 'danger');
            addToCalibrationLog(`Error: ${data.message}`);
            document.getElementById('start-calibration-btn').disabled = false;
        }
    })
    .catch(error => {
        console.error('Error starting calibration:', error);
        showToast('Error', 'Failed to start calibration', 'danger');
        addToCalibrationLog(`Error: ${error.message}`);
        document.getElementById('start-calibration-btn').disabled = false;
    });
}

function updateCalibrationParams() {
    const cols = parseInt(document.getElementById('checkerboard-cols').value);
    const rows = parseInt(document.getElementById('checkerboard-rows').value);
    const squareSize = parseFloat(document.getElementById('square-size').value);
    const autoCapture = document.getElementById('auto-capture-toggle').checked;
    const stabilityThreshold = parseFloat(document.getElementById('stability-threshold').value);

    // Validate inputs
    if (isNaN(cols) || isNaN(rows) || isNaN(squareSize) || cols < 3 || rows < 3) {
        showToast('Error', 'Invalid checkerboard parameters', 'danger');
        return;
    }

    fetch('/api/calibrate/settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            checkerboard_size: [cols, rows],
            square_size: squareSize,
            auto_capture: autoCapture,
            stability_seconds: stabilityThreshold
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Success', 'Calibration parameters updated', 'success');
            addToCalibrationLog('Calibration parameters updated');
        } else {
            showToast('Error', data.message, 'danger');
            addToCalibrationLog(`Error updating parameters: ${data.message}`);
        }
    })
    .catch(error => {
        console.error('Error updating calibration parameters:', error);
        showToast('Error', 'Failed to update parameters', 'danger');
        addToCalibrationLog(`Error updating parameters: ${error.message}`);
    });
}

function testDetection() {
    // Update pattern indicators
    const leftIndicator = document.getElementById('left-pattern-indicator');
    const rightIndicator = document.getElementById('right-pattern-indicator');
    const bothIndicator = document.getElementById('both-pattern-indicator');

    // Set to loading state
    leftIndicator.className = 'fas fa-spinner fa-spin text-warning';
    rightIndicator.className = 'fas fa-spinner fa-spin text-warning';
    bothIndicator.className = 'fas fa-spinner fa-spin text-warning';

    fetch('/api/calibrate/detect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update pattern indicators
            if (data.left_found) {
                leftIndicator.className = 'fas fa-check-circle text-success';
            } else {
                leftIndicator.className = 'fas fa-times-circle text-danger';
            }

            if (data.right_found) {
                rightIndicator.className = 'fas fa-check-circle text-success';
            } else {
                rightIndicator.className = 'fas fa-times-circle text-danger';
            }

            if (data.both_found) {
                bothIndicator.className = 'fas fa-check-circle text-success';
            } else {
                bothIndicator.className = 'fas fa-times-circle text-danger';
            }

            // Update images
            if (data.left_image) {
                document.getElementById('calib-left-camera').src = 'data:image/jpeg;base64,' + data.left_image;
            }

            if (data.right_image) {
                document.getElementById('calib-right-camera').src = 'data:image/jpeg;base64,' + data.right_image;
            }

            // Add log message
            if (data.both_found) {
                addToCalibrationLog('Checkerboard detected in both cameras!');
            } else {
                addToCalibrationLog('Checkerboard not detected in both cameras.');
            }

        } else {
            // Handle error
            leftIndicator.className = 'fas fa-times-circle text-danger';
            rightIndicator.className = 'fas fa-times-circle text-danger';
            bothIndicator.className = 'fas fa-times-circle text-danger';

            showToast('Error', data.message, 'danger');
            addToCalibrationLog(`Error: ${data.message}`);
        }
    })
    .catch(error => {
        console.error('Error testing detection:', error);

        // Reset indicators
        leftIndicator.className = 'fas fa-times-circle text-danger';
        rightIndicator.className = 'fas fa-times-circle text-danger';
        bothIndicator.className = 'fas fa-times-circle text-danger';

        showToast('Error', 'Failed to test detection', 'danger');
        addToCalibrationLog(`Error: ${error.message}`);
    });
}

// Update stability slider feedback
function initStabilitySlider() {
    document.getElementById('stability-threshold').addEventListener('input', function() {
        document.getElementById('stability-value').textContent = this.value + 's';
    });
}

// Initialize calibration tab
function initCalibration() {
    document.getElementById('start-calibration-btn').addEventListener('click', startCalibration);
    document.getElementById('update-calib-params-btn').addEventListener('click', updateCalibrationParams);
    document.getElementById('test-detection-btn').addEventListener('click', testDetection);

    initStabilitySlider();

    // Check calibration status when tab is shown
    document.getElementById('calibration-tab').addEventListener('shown.bs.tab', function() {
        fetch('/api/calibrate/status')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.has_calibration) {
                    // Update calibration result display
                    const resultDiv = document.getElementById('calibration-result');
                    resultDiv.classList.remove('d-none');

                    if (data.calibration_info) {
                        document.getElementById('rms-error').textContent =
                            data.calibration_info.rms_error ? data.calibration_info.rms_error.toFixed(6) : '-';
                        document.getElementById('calibration-date').textContent =
                            data.calibration_info.date || '-';
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching calibration status:', error);
            });
    });
}

// Add the initialization to the document ready event
document.addEventListener('DOMContentLoaded', initCalibration);