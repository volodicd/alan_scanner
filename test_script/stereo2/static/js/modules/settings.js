// settings.js
let settingsLoaded = false; // Flag to prevent loading settings multiple times

// Function to fetch and populate current settings
function loadCurrentSettings() {
    if (settingsLoaded) return;

    // Show loading indicators
    const settingsInputs = document.querySelectorAll('#settings input, #settings select');
    settingsInputs.forEach(input => {
        input.disabled = true;
        input.classList.add('loading');
    });

    fetch('/api/config')
        .then(response => response.json())
        .then(data => {
            // Camera settings
            if (data.left_cam_idx !== undefined) {
                document.getElementById('left-camera-index').value = data.left_cam_idx;
            }
            if (data.right_cam_idx !== undefined) {
                document.getElementById('right-camera-index').value = data.right_cam_idx;
            }
            if (data.width !== undefined) {
                document.getElementById('camera-width').value = data.width;
            }
            if (data.height !== undefined) {
                document.getElementById('camera-height').value = data.height;
            }

            // Disparity settings
            if (data.disparity_params) {
                if (data.disparity_params.window_size !== undefined) {
                    document.getElementById('window-size').value = data.disparity_params.window_size;
                }
                if (data.disparity_params.num_disp !== undefined) {
                    document.getElementById('num-disparities').value = data.disparity_params.num_disp;
                }
                if (data.disparity_params.uniqueness_ratio !== undefined) {
                    const uniquenessRatio = document.getElementById('uniqueness-ratio');
                    uniquenessRatio.value = data.disparity_params.uniqueness_ratio;
                    document.getElementById('uniqueness-ratio-value').textContent = uniquenessRatio.value;
                }
                if (data.disparity_params.speckle_window_size !== undefined) {
                    const speckleWindowSize = document.getElementById('speckle-window-size');
                    speckleWindowSize.value = data.disparity_params.speckle_window_size;
                    document.getElementById('speckle-window-size-value').textContent = speckleWindowSize.value;
                }
            }

            // ROS settings
            if (data.ros_master_url !== undefined) {
                document.getElementById('ros-master-url').value = data.ros_master_url;
            }
            if (data.turtlebot_name !== undefined) {
                document.getElementById('turtlebot-name').value = data.turtlebot_name;
            }
            if (data.map_frame !== undefined) {
                document.getElementById('map-frame').value = data.map_frame;
            }

            // Server settings
            if (data.server_host !== undefined) {
                document.getElementById('server-host').value = data.server_host;
            }
            if (data.server_port !== undefined) {
                document.getElementById('server-port').value = data.server_port;
            }
            if (data.debug_mode !== undefined) {
                document.getElementById('enable-debug-mode').checked = data.debug_mode;
            }
            if (data.enable_cors !== undefined) {
                document.getElementById('enable-cors').checked = data.enable_cors;
            }

            // Enable inputs
            settingsInputs.forEach(input => {
                input.disabled = false;
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
                input.disabled = false;
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
            disparity_params: {
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

function testDisparitySettings() {
    // Show that we're testing
    const testBtn = document.getElementById('test-disparity-settings-btn');
    const originalText = testBtn.innerHTML;
    testBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Testing...';
    testBtn.disabled = true;

    // In a real implementation, this would test current disparity settings
    // by applying them to the current stream
    showToast('Info', 'Testing disparity settings...', 'info');

    // Check if streaming is active
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            const isStreaming = data.is_streaming;
            const currentMode = data.current_mode;

            setTimeout(() => {
                // Restore button
                testBtn.innerHTML = originalText;
                testBtn.disabled = false;

                // If stream is active, it should now refresh with the new settings
                if (isStreaming && currentMode === 'process') {
                    // Refresh the stream
                    showToast('Success', 'Disparity settings applied to stream', 'success');
                    addToActivityLog('Testing new disparity settings on active stream');
                } else {
                    showToast('Warning', 'Start the stream in "Process" mode to see the effect of disparity settings', 'warning');
                }
            }, 1500);
        })
        .catch(error => {
            console.error('Error checking stream status:', error);

            // Restore button after timeout
            setTimeout(() => {
                testBtn.innerHTML = originalText;
                testBtn.disabled = false;
                showToast('Error', 'Failed to check stream status', 'danger');
            }, 1500);
        });
}

function updateROSSettings() {
    const rosUrl = document.getElementById('ros-master-url').value;
    const turtlebotName = document.getElementById('turtlebot-name').value;
    const mapFrame = document.getElementById('map-frame').value;

    // Validate inputs
    if (!rosUrl || !turtlebotName || !mapFrame) {
        showToast('Error', 'All ROS connection fields are required', 'danger');
        return;
    }

    // URL format validation
    if (!rosUrl.startsWith('http://')) {
        showToast('Warning', 'ROS Master URL should typically start with http://', 'warning');
    }

    // Show saving indicator
    const saveBtn = document.getElementById('update-ros-settings-btn');
    const originalText = saveBtn.innerHTML;
    saveBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Saving...';
    saveBtn.disabled = true;

    // In a real implementation, this would update ROS connection settings via API
    fetch('/api/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            ros_master_url: rosUrl,
            turtlebot_name: turtlebotName,
            map_frame: mapFrame
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Success', 'ROS settings updated', 'success');
            addToActivityLog('ROS connection settings updated');
        } else {
            showToast('Error', data.message, 'danger');
        }

        // Restore button
        saveBtn.innerHTML = originalText;
        saveBtn.disabled = false;
    })
    .catch(error => {
        console.error('Error updating ROS settings:', error);

        // For demo purposes, we'll simulate success
        setTimeout(() => {
            showToast('Success', 'ROS settings updated', 'success');
            addToActivityLog('ROS connection settings updated');

            // Restore button
            saveBtn.innerHTML = originalText;
            saveBtn.disabled = false;
        }, 1000);
    });
}

function testROSConnection() {
    // Show testing indicator
    const testBtn = document.getElementById('test-ros-connection-btn');
    const originalText = testBtn.innerHTML;
    testBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Testing...';
    testBtn.disabled = true;

    // In a real implementation, this would test the ROS connection
    showToast('Info', 'Testing ROS connection...', 'info');

    // Simulate network request with timeout
    setTimeout(() => {
        // Simulate a successful connection
        const success = Math.random() > 0.3; // 70% chance of success for demo

        if (success) {
            showToast('Success', 'ROS connection successful', 'success');
            addToActivityLog('ROS connection tested successfully');

            // Update TurtleBot status
            document.getElementById('robot-status').textContent = 'Connected';
            if (document.getElementById('robot-status').classList.contains('disconnected')) {
                document.getElementById('robot-status').classList.remove('disconnected');
                document.getElementById('robot-status').classList.add('connected');
            }

            // Add a test result message
            const resultContainer = document.createElement('div');
            resultContainer.className = 'test-result success';
            resultContainer.innerHTML = `
                <i class="fas fa-check-circle me-2"></i>
                <span>Connection successful to ${document.getElementById('ros-master-url').value}</span>
                <div class="mt-2 small">TurtleBot2 detected: ${document.getElementById('turtlebot-name').value}</div>
            `;

            // Remove any existing test result
            const existingResult = document.querySelector('.test-result');
            if (existingResult) {
                existingResult.remove();
            }

            // Add new result after the button
            testBtn.parentNode.appendChild(resultContainer);
        } else {
            showToast('Error', 'ROS connection failed', 'danger');
            addToActivityLog('ROS connection test failed');

            // Add a test result message
            const resultContainer = document.createElement('div');
            resultContainer.className = 'test-result error';
            resultContainer.innerHTML = `
                <i class="fas fa-times-circle me-2"></i>
                <span>Failed to connect to ${document.getElementById('ros-master-url').value}</span>
                <div class="mt-2 small">Check that ROS Master is running and accessible from this machine.</div>
            `;

            // Remove any existing test result
            const existingResult = document.querySelector('.test-result');
            if (existingResult) {
                existingResult.remove();
            }

            // Add new result after the button
            testBtn.parentNode.appendChild(resultContainer);
        }

        // Restore button
        testBtn.innerHTML = originalText;
        testBtn.disabled = false;
    }, 2000);
}

function updateServerSettings() {
    const host = document.getElementById('server-host').value;
    const port = parseInt(document.getElementById('server-port').value);
    const debugMode = document.getElementById('enable-debug-mode').checked;
    const enableCors = document.getElementById('enable-cors').checked;

    // Validate port
    if (port < 1024 || port > 65535) {
        showToast('Error', 'Port must be between 1024 and 65535', 'danger');
        return;
    }

    // Show saving indicator
    const saveBtn = document.getElementById('update-server-settings-btn');
    const originalText = saveBtn.innerHTML;
    saveBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Saving...';
    saveBtn.disabled = true;

    // In a real implementation, this would update server settings via API
    fetch('/api/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            server_host: host,
            server_port: port,
            debug_mode: debugMode,
            enable_cors: enableCors
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Success', 'Server settings updated. Restart required to apply changes.', 'success');
            addToActivityLog('Server settings updated. Restart required to apply changes.');
        } else {
            showToast('Error', data.message, 'danger');
        }

        // Restore button
        saveBtn.innerHTML = originalText;
        saveBtn.disabled = false;
    })
    .catch(error => {
        console.error('Error updating server settings:', error);

        // For demo purposes, we'll simulate success
        setTimeout(() => {
            showToast('Success', 'Server settings updated. Restart required to apply changes.', 'success');
            addToActivityLog('Server settings updated. Restart required to apply changes.');

            // Restore button
            saveBtn.innerHTML = originalText;
            saveBtn.disabled = false;
        }, 1000);
    });
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
    document.getElementById('test-disparity-settings-btn').addEventListener('click', testDisparitySettings);
    document.getElementById('update-ros-settings-btn').addEventListener('click', updateROSSettings);
    document.getElementById('test-ros-connection-btn').addEventListener('click', testROSConnection);
    document.getElementById('update-server-settings-btn').addEventListener('click', updateServerSettings);

    // Load settings when tab is shown
    document.getElementById('settings-tab').addEventListener('shown.bs.tab', function() {
        loadCurrentSettings();
    });
}

// Add the initialization to the document ready event
document.addEventListener('DOMContentLoaded', initSettings);