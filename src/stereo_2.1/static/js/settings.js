// settings.js
let settingsLoaded = false; // Flag to prevent loading settings multiple times

// Function to fetch and populate current settings
function loadCurrentSettings() {
    if (settingsLoaded) return;

    // Show loading indicators
    const settingsInputs = document.querySelectorAll('#settings input, #settings select');
    settingsInputs.forEach(input => {
        input.classList.add('loading');
    });

    fetch('/api/config')
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                showToast('Error', 'Failed to load settings', 'danger');
                return;
            }

            const config = data.config;

            // Camera settings
            if (config.left_cam_idx !== undefined) {
                document.getElementById('left-camera-index').value = config.left_cam_idx;
            }
            if (config.right_cam_idx !== undefined) {
                document.getElementById('right-camera-index').value = config.right_cam_idx;
            }
            if (config.width !== undefined) {
                document.getElementById('camera-width').value = config.width;
            }
            if (config.height !== undefined) {
                document.getElementById('camera-height').value = config.height;
            }

            // Disparity settings
            if (config.sgbm_params) {
                if (config.sgbm_params.window_size !== undefined) {
                    document.getElementById('window-size').value = config.sgbm_params.window_size;
                }
                if (config.sgbm_params.num_disp !== undefined) {
                    document.getElementById('num-disparities').value = config.sgbm_params.num_disp;
                }
                if (config.sgbm_params.uniqueness_ratio !== undefined) {
                    const uniquenessRatio = document.getElementById('uniqueness-ratio');
                    uniquenessRatio.value = config.sgbm_params.uniqueness_ratio;
                    document.getElementById('uniqueness-ratio-value').textContent = uniquenessRatio.value;
                }
                if (config.sgbm_params.speckle_window_size !== undefined) {
                    const speckleWindowSize = document.getElementById('speckle-window-size');
                    speckleWindowSize.value = config.sgbm_params.speckle_window_size;
                    document.getElementById('speckle-window-size-value').textContent = speckleWindowSize.value;
                }
            }

            // Enable inputs
            settingsInputs.forEach(input => {
                input.classList.remove('loading');
            });

            // Set the loaded flag
            settingsLoaded = true;

            addToActivityLog('Settings loaded');
        })
        .catch(error => {
            console.error('Error loading settings:', error);
            showToast('Error', 'Failed to load settings', 'danger');

            // Enable inputs despite error
            settingsInputs.forEach(input => {
                input.classList.remove('loading');
            });
        });
}

function updateCameraSettings() {
    const leftCamIdx = parseInt(document.getElementById('left-camera-index').value);
    const rightCamIdx = parseInt(document.getElementById('right-camera-index').value);
    const width = parseInt(document.getElementById('camera-width').value);
    const height = parseInt(document.getElementById('camera-height').value);

    // Validate settings
    if (leftCamIdx === rightCamIdx) {
        showToast('Error', 'Left and right camera indices must be different', 'danger');
        return;
    }

    if (width % 8 !== 0 || height % 8 !== 0) {
        showToast('Warning', 'Resolution width and height should be multiples of 8 for best results', 'warning');
    }

    // Show saving indicator
    const saveBtn = document.getElementById('update-camera-settings-btn');
    const originalText = saveBtn.innerHTML;
    saveBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Saving...';
    saveBtn.disabled = true;

    fetch('/api/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            left_cam_idx: leftCamIdx,
            right_cam_idx: rightCamIdx,
            width: width,
            height: height
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Success', 'Camera settings updated', 'success');
            addToActivityLog('Camera settings updated');
        } else {
            showToast('Error', data.message, 'danger');
        }

        // Restore button
        saveBtn.innerHTML = originalText;
        saveBtn.disabled = false;
    })
    .catch(error => {
        console.error('Error updating camera settings:', error);
        showToast('Error', 'Failed to update camera settings', 'danger');

        // Restore button
        saveBtn.innerHTML = originalText;
        saveBtn.disabled = false;
    });
}

function updateDisparitySettings() {
    const windowSize = parseInt(document.getElementById('window-size').value);
    const numDisparities = parseInt(document.getElementById('num-disparities').value);
    const uniquenessRatio = parseInt(document.getElementById('uniqueness-ratio').value);
    const speckleWindowSize = parseInt(document.getElementById('speckle-window-size').value);
    const speckleRange = 32; // Fixed value, could be made adjustable

    // Validate
    if (numDisparities % 16 !== 0) {
        showToast('Error', 'Number of disparities must be a multiple of 16', 'danger');
        return;
    }

    // Show saving indicator
    const saveBtn = document.getElementById('update-disparity-settings-btn');
    const originalText = saveBtn.innerHTML;
    saveBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Saving...';
    saveBtn.disabled = true;

    fetch('/api/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            sgbm_params: {
                window_size: windowSize,
                num_disp: numDisparities,
                min_disp: 0,
                uniqueness_ratio: uniquenessRatio,
                speckle_window_size: speckleWindowSize,
                speckle_range: speckleRange
            }
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Success', 'Disparity settings updated', 'success');
            addToActivityLog('Disparity settings updated');
        } else {
            showToast('Error', data.message, 'danger');
        }

        // Restore button
        saveBtn.innerHTML = originalText;
        saveBtn.disabled = false;
    })
    .catch(error => {
        console.error('Error updating disparity settings:', error);
        showToast('Error', 'Failed to update disparity settings', 'danger');

        // Restore button
        saveBtn.innerHTML = originalText;
        saveBtn.disabled = false;
    });
}

function loadSystemInfo() {
    const systemInfoContainer = document.getElementById('system-info');
    if (!systemInfoContainer) return;

    systemInfoContainer.innerHTML = `
        <div class="d-flex justify-content-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
    `;

    fetch('/api/system/info')
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                systemInfoContainer.innerHTML = '<div class="alert alert-danger">Failed to load system information</div>';
                return;
            }

            // Format system info
            let html = '';

            // Platform info
            html += `
                <div class="system-info-item">
                    <div class="fw-bold mb-1">Platform</div>
                    <div>${data.system_info.platform}</div>
                </div>
                <div class="system-info-item">
                    <div class="fw-bold mb-1">Processor</div>
                    <div>${data.system_info.processor}</div>
                </div>
                <div class="system-info-item">
                    <div class="fw-bold mb-1">Software</div>
                    <div>Python ${data.system_info.python_version}</div>
                    <div>OpenCV ${data.system_info.opencv_version}</div>
                </div>
                <div class="system-info-item">
                    <div class="fw-bold mb-1">Calibration Status</div>
                    <div>${data.has_calibration ? '<span class="text-success">Calibrated</span>' : '<span class="text-danger">Not Calibrated</span>'}</div>
                </div>
            `;

            systemInfoContainer.innerHTML = html;

            // Also update the available cameras list
            updateAvailableCameras(data.available_cameras);
        })
        .catch(error => {
            console.error('Error loading system info:', error);
            systemInfoContainer.innerHTML = '<div class="alert alert-danger">Failed to load system information</div>';
        });
}

function updateAvailableCameras(cameras) {
    const camerasContainer = document.getElementById('available-cameras');
    if (!camerasContainer) return;

    if (!cameras || cameras.length === 0) {
        camerasContainer.innerHTML = '<div class="alert alert-warning">No cameras detected</div>';
        return;
    }

    let html = '';
    cameras.forEach(camera => {
        html += `
            <div class="camera-item">
                <div class="fw-bold">Camera ${camera.index}</div>
                <div>Resolution: ${camera.resolution}</div>
                <div>FPS: ${camera.fps.toFixed(1)}</div>
            </div>
        `;
    });

    camerasContainer.innerHTML = html;
}

function detectCameras() {
    const camerasContainer = document.getElementById('available-cameras');
    if (!camerasContainer) return;

    // Show loading
    camerasContainer.innerHTML = `
        <div class="d-flex justify-content-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
    `;

    // Reload system info which includes camera detection
    loadSystemInfo();
}

// Event handlers for range input displays
function setupRangeInputDisplays() {
    // Uniqueness ratio display
    document.getElementById('uniqueness-ratio').addEventListener('input', function() {
        document.getElementById('uniqueness-ratio-value').textContent = this.value;
    });

    // Speckle window size display
    document.getElementById('speckle-window-size').addEventListener('input', function() {
        document.getElementById('speckle-window-size-value').textContent = this.value;
    });
}

// Init function to be called when the page loads
function initSettings() {
    // Setup range input displays
    setupRangeInputDisplays();

    // Add button event listeners
    document.getElementById('update-camera-settings-btn').addEventListener('click', updateCameraSettings);
    document.getElementById('update-disparity-settings-btn').addEventListener('click', updateDisparitySettings);
    document.getElementById('refresh-system-info-btn').addEventListener('click', loadSystemInfo);
    document.getElementById('detect-cameras-btn').addEventListener('click', detectCameras);

    // Load settings when tab is shown
    document.getElementById('settings-tab').addEventListener('shown.bs.tab', function() {
        loadCurrentSettings();
        loadSystemInfo();
    });
}

// Add the initialization to the document ready event
document.addEventListener('DOMContentLoaded', initSettings);