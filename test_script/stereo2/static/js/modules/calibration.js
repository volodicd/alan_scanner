// calibration.js
function updateCalibrationStatus(data) {
    // Update pattern detection indicators
    const leftIndicator = document.getElementById('left-pattern-indicator');
    const rightIndicator = document.getElementById('right-pattern-indicator');
    const bothIndicator = document.getElementById('both-pattern-indicator');
    const stabilityContainer = document.getElementById('stability-container');
    const stabilityTime = document.getElementById('stability-time');

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
        document.getElementById('capture-calibration-btn').disabled = false;

        // Show stability info if available
        if (data.stable_seconds !== undefined) {
            stabilityContainer.style.display = 'flex';
            stabilityTime.textContent = data.stable_seconds + 's';

            // If close to threshold, change color to indicate imminent capture
            const threshold = parseFloat(document.getElementById('stability-threshold').value);
            if (data.auto_capture && data.stable_seconds > threshold * 0.7) {
                stabilityTime.className = 'text-warning'; // approaching threshold
            }
            if (data.auto_capture && data.stable_seconds >= threshold) {
                stabilityTime.className = 'text-success'; // at/above threshold
            }

            // Add entry to calibration logs when stability changes
            addToCalibrationLog(`Checkerboard stable for ${data.stable_seconds}s`);
        }
    } else {
        bothIndicator.className = 'fas fa-times-circle text-danger';
        document.getElementById('capture-calibration-btn').disabled = true;
        stabilityContainer.style.display = 'none';
    }

    // Update auto-capture UI based on server state
    if (data.auto_capture !== undefined) {
        document.getElementById('auto-capture-toggle').checked = data.auto_capture;
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

function addToCalibrationLog(message) {
    const logsContainer = document.getElementById('calibration-logs');
    const timestamp = new Date().toLocaleTimeString();

    // Remove "No logs yet" message if present
    const noLogsMsg = logsContainer.querySelector('.text-muted');
    if (noLogsMsg && noLogsMsg.textContent === 'No calibration logs yet') {
        logsContainer.removeChild(noLogsMsg);
    }

    // Add the log entry
    const logEntry = document.createElement('div');
    logEntry.innerHTML = `<span class="text-muted">[${timestamp}]</span> ${message}`;
    logsContainer.appendChild(logEntry);

    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

function startCalibration() {
    // Get auto-capture settings
    const autoCapture = document.getElementById('auto-capture-toggle').checked;
    const stabilityThreshold = parseFloat(document.getElementById('stability-threshold').value);

    fetch('/api/calibrate/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            auto_capture: autoCapture,
            stability_seconds: stabilityThreshold
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addToActivityLog('Calibration mode started');
            addToCalibrationLog('Calibration mode started');

            if (data.auto_capture) {
                addToCalibrationLog(`Auto-capture enabled (${data.stability_seconds}s threshold)`);
            } else {
                addToCalibrationLog('Manual capture mode');
            }

            // If there are existing pairs, log that
            if (data.existing_pairs > 0) {
                addToCalibrationLog(`Found ${data.existing_pairs} existing calibration pairs`);
            }

            document.getElementById('start-calibration-btn').disabled = true;
            document.getElementById('process-calibration-btn').disabled = false;

            // Check current calibration parameters and update UI
            updateCalibrationParamsUI();
        } else {
            showToast('Error', data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error starting calibration:', error);
        showToast('Error', 'Failed to start calibration', 'danger');
    });
}

function updateCalibrationParamsUI() {
    fetch('/api/config')
        .then(response => response.json())
        .then(data => {
            if (data.calibration_checkerboard_size) {
                document.getElementById('checkerboard-cols').value = data.calibration_checkerboard_size[0];
                document.getElementById('checkerboard-rows').value = data.calibration_checkerboard_size[1];
            }

            if (data.calibration_square_size) {
                document.getElementById('square-size').value = data.calibration_square_size;
            }
        })
        .catch(error => {
            console.error('Error fetching config:', error);
        });
}

function updateCalibrationParams() {
    const cols = parseInt(document.getElementById('checkerboard-cols').value);
    const rows = parseInt(document.getElementById('checkerboard-rows').value);
    const squareSize = parseFloat(document.getElementById('square-size').value);

    fetch('/api/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            calibration_checkerboard_size: [cols, rows],
            calibration_square_size: squareSize
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Success', 'Calibration parameters updated', 'success');
        } else {
            showToast('Error', data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error updating calibration parameters:', error);
        showToast('Error', 'Failed to update parameters', 'danger');
    });
}

function captureCalibrationFrame() {
    // Disable button to prevent multiple clicks
    document.getElementById('capture-calibration-btn').disabled = true;
    addToCalibrationLog('Manually capturing calibration frame...');

    fetch('/api/calibrate/capture', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update calibration pairs list
            addCalibrationPair(data.timestamp, data.left_path, data.right_path);

            // Update progress
            const pairsList = document.getElementById('calibration-pairs-list');
            const pairsCount = pairsList.querySelectorAll('div:not(.text-muted)').length;

            document.getElementById('captured-pairs-count').textContent = pairsCount;

            const minPairs = parseInt(document.getElementById('min-pairs').value);
            const progress = Math.min((pairsCount / minPairs) * 100, 100);

            const progressBar = document.getElementById('captured-progress');
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);

            if (pairsCount >= minPairs) {
                progressBar.classList.add('bg-success');
            }

            const captureMessage = `Calibration frame pair captured: ${data.timestamp}`;
            addToActivityLog(captureMessage);
            addToCalibrationLog(captureMessage);

            // Log file paths
            addToCalibrationLog(`Saved images to ${data.left_path} and ${data.right_path}`);

            showToast('Success', 'Calibration pair captured', 'success');

            // If we have enough pairs, suggest processing
            if (pairsCount >= minPairs) {
                addToCalibrationLog(`You now have ${pairsCount} image pairs (${minPairs} required). Ready to process calibration.`);
            }
        } else {
            addToCalibrationLog(`ERROR: ${data.message}`);
            showToast('Error', data.message, 'danger');
        }

        // Re-enable button if pattern is still visible
        const bothFound = document.getElementById('both-pattern-indicator').className.includes('success');
        document.getElementById('capture-calibration-btn').disabled = !bothFound;
    })
    .catch(error => {
        console.error('Error capturing calibration frame:', error);
        addToCalibrationLog(`ERROR: Failed to capture frame: ${error.message}`);
        showToast('Error', 'Failed to capture calibration frame', 'danger');
        document.getElementById('capture-calibration-btn').disabled = false;
    });
}

function addCalibrationPair(timestamp, leftPath, rightPath) {
    const pairsList = document.getElementById('calibration-pairs-list');

    // Remove 'No calibration pairs' message if it exists
    const noMsg = pairsList.querySelector('.text-muted');
    if (noMsg && noMsg.textContent === 'No calibration pairs captured yet') {
        pairsList.removeChild(noMsg);
    }

    const pairEntry = document.createElement('div');
    pairEntry.className = 'mb-1';
    pairEntry.innerHTML = `<span class="text-info">[${timestamp}]</span> Calibration pair captured`;

    pairsList.appendChild(pairEntry);
    pairsList.scrollTop = pairsList.scrollHeight;
}

function updateAutoCapture() {
    const autoCapture = document.getElementById('auto-capture-toggle').checked;
    const stabilityThreshold = parseFloat(document.getElementById('stability-threshold').value);

    fetch('/api/calibrate/settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            auto_capture: autoCapture,
            stability_seconds: stabilityThreshold
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const message = `Auto-capture ${autoCapture ? 'enabled' : 'disabled'}` +
                          (autoCapture ? ` (${stabilityThreshold}s threshold)` : '');

            addToActivityLog(message);
            addToCalibrationLog(message);
            showToast('Success', 'Auto-capture settings updated', 'success');
        } else {
            showToast('Error', data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error updating auto-capture settings:', error);
        showToast('Error', 'Failed to update auto-capture settings', 'danger');
    });
}

function processCalibration() {
    // Show processing indicator
    showToast('Info', 'Processing calibration. This may take a while...', 'info');
    addToCalibrationLog('Starting calibration processing...');

    // Disable process button to prevent multiple clicks
    document.getElementById('process-calibration-btn').disabled = true;

    fetch('/api/calibrate/process', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update calibration result display
            const resultDiv = document.getElementById('calibration-result');
            resultDiv.classList.remove('d-none');

            document.getElementById('rms-error').textContent = data.rms_error.toFixed(6);
            document.getElementById('calibration-date').textContent = new Date().toLocaleString();

            // Update global calibration status
            document.getElementById('calibration-status').textContent = 'Calibrated';

            const successMessage = `Calibration completed with RMS error: ${data.rms_error.toFixed(6)}`;
            addToActivityLog(successMessage);
            addToCalibrationLog(successMessage);
            addToCalibrationLog(`Calibration files saved to ${data.calibration_file}`);
            if (data.backup_file) {
                addToCalibrationLog(`Backup saved to ${data.backup_file}`);
            }

            showToast('Success', 'Calibration processed successfully', 'success');

            // Re-enable calibration start button
            document.getElementById('start-calibration-btn').disabled = false;
        } else {
            addToCalibrationLog(`ERROR: ${data.message}`);
            showToast('Error', data.message, 'danger');
            // Re-enable process button
            document.getElementById('process-calibration-btn').disabled = false;
        }
    })
    .catch(error => {
        console.error('Error processing calibration:', error);
        addToCalibrationLog(`ERROR: Failed to process calibration: ${error.message}`);
        showToast('Error', 'Failed to process calibration', 'danger');
        // Re-enable process button
        document.getElementById('process-calibration-btn').disabled = false;
    });
}

// Initialize stability slider feedback
function initStabilitySlider() {
    document.getElementById('stability-threshold').addEventListener('input', function() {
        document.getElementById('stability-value').textContent = this.value + 's';
    });
}

// Init function to be called when the page loads
function initCalibration() {
    document.getElementById('start-calibration-btn').addEventListener('click', startCalibration);
    document.getElementById('capture-calibration-btn').addEventListener('click', captureCalibrationFrame);
    document.getElementById('process-calibration-btn').addEventListener('click', processCalibration);
    document.getElementById('update-calib-params-btn').addEventListener('click', updateCalibrationParams);
    document.getElementById('update-auto-capture-btn').addEventListener('click', updateAutoCapture);

    initStabilitySlider();
}

// Add the initialization to the document ready event
document.addEventListener('DOMContentLoaded', initCalibration);