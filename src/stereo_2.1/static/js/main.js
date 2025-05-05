// main.js - Core functionality for the application

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
    const toastContainer = document.querySelector('.toast-container');

    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast`; // Bootstrap toast class
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');

    // Toast header color based on type
    let headerClass = 'bg-info';
    if (type === 'success') headerClass = 'bg-success';
    if (type === 'warning') headerClass = 'bg-warning';
    if (type === 'danger') headerClass = 'bg-danger';

    // Create toast content
    toast.innerHTML = `
        <div class="toast-header ${headerClass} text-white">
            <strong class="me-auto">${title}</strong>
            <small>${new Date().toLocaleTimeString()}</small>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;

    // Add toast to container
    toastContainer.appendChild(toast);

    // Initialize and show the toast
    const bsToast = new bootstrap.Toast(toast, {
        autohide: true,
        delay: 5000
    });
    bsToast.show();

    // Add event listener to remove toast after it's hidden
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });

    // Log message to activity
    addToActivityLog(message);
}

// Create placeholder directories
async function createPlaceholderDirectories() {
    try {
        // Create necessary directories
        const directories = ['img', 'css/modules', 'js/modules'];

        for (const dir of directories) {
            try {
                // Try to create directory (this will fail in the browser, but it's just a precaution)
                if (window.fs && window.fs.mkdir) {
                    await window.fs.mkdir(`static/${dir}`, { recursive: true });
                }
            } catch (e) {
                console.warn(`Failed to create directory static/${dir}:`, e);
            }
        }
    } catch (e) {
        console.warn("Failed to create placeholder directories:", e);
    }
}

// Wait for DOM content to be loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));

});


document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible') {
        // Page became visible - refresh current tab's data
        const activeTabId = document.querySelector('.nav-link.active').id;

        if (activeTabId === 'dashboard-tab') {
            // Check calibration status for dashboard
            if (typeof checkCalibrationStatus === 'function') {
                checkCalibrationStatus();
            }
        } else if (activeTabId === 'settings-tab') {
            // Reload settings
            settingsLoaded = false;
            if (typeof loadCurrentSettings === 'function') {
                loadCurrentSettings();
            }
        } else if (activeTabId === 'calibration-tab') {
            // Check calibration status
            fetch('/api/calibrate/status');
        }
    }
});