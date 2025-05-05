// logs.js
let currentLogFilters = {
    type: 'app',
    level: '',
    limit: 100,
    search: ''
};

function refreshLogs(filters = null) {
    // Show loading state
    const logsContainer = document.getElementById('full-logs');
    logsContainer.innerHTML = `
        <div class="logs-loading">
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Loading logs...
        </div>
    `;

    // Use provided filters or current filters
    const activeFilters = filters || currentLogFilters;

    // Update current filters if new ones were provided
    if (filters) {
        currentLogFilters = {...filters};

        // Update UI to match current filters
        document.getElementById('log-type-select').value = currentLogFilters.type;
        document.getElementById('log-level-select').value = currentLogFilters.level;
        document.getElementById('log-limit-select').value = currentLogFilters.limit.toString();
        document.getElementById('log-search-input').value = currentLogFilters.search;
    }

    // Build query parameters
    const queryParams = new URLSearchParams({
        type: activeFilters.type,
        level: activeFilters.level,
        limit: activeFilters.limit,
        search: activeFilters.search
    }).toString();

    fetch(`/api/logs?${queryParams}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateLogsDisplay(data);
            } else {
                showToast('Error', data.message, 'danger');
                logsContainer.innerHTML = `
                    <div class="logs-empty">
                        <i class="fas fa-exclamation-circle text-danger"></i>
                        <p>Error loading logs: ${data.message}</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error fetching logs:', error);
            showToast('Error', 'Failed to fetch logs', 'danger');
            logsContainer.innerHTML = `
                <div class="logs-empty">
                    <i class="fas fa-exclamation-circle text-danger"></i>
                    <p>Error loading logs: ${error.message}</p>
                </div>
            `;
        });
}

function updateLogsDisplay(data) {
    const logsContainer = document.getElementById('full-logs');

    // Clear current content
    logsContainer.innerHTML = '';

    if (data.logs.length === 0) {
        logsContainer.innerHTML = `
            <div class="logs-empty">
                <i class="fas fa-search text-secondary"></i>
                <p>No logs matching the current filters</p>
            </div>
        `;
        return;
    }

    // Add log information header
    const infoElement = document.createElement('div');
    infoElement.className = 'log-info mb-2 text-secondary';
    infoElement.innerHTML = `
        <small>
            Showing ${data.logs.length} of ${data.total_lines} logs from ${data.log_file} (${data.log_size_kb} KB)
        </small>
    `;
    logsContainer.appendChild(infoElement);

    // Process and display each log
    data.logs.forEach(log => {
        const logElement = document.createElement('div');
        logElement.className = 'log-entry';

        // Process structured logs
        if (log.timestamp && log.level) {
            // This is a parsed structured log
            const levelClass = `log-level-${log.level}`;

            logElement.innerHTML = `
                <span class="log-timestamp">[${log.timestamp}]</span>
                <span class="log-module">${log.module}</span>
                <span class="${levelClass}">${log.level}</span>
                <span class="log-message">${formatLogMessage(log.message)}</span>
            `;
        } else if (log.raw) {
            // This is a raw log line
            logElement.innerHTML = `<span class="log-raw">${formatLogMessage(log.raw)}</span>`;
        } else {
            // Simple text log
            logElement.innerHTML = formatLogMessage(log);
        }

        logsContainer.appendChild(logElement);
    });

    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

function formatLogMessage(message) {
    // Escape HTML entities for safety
    let safeMessage = String(message)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');

    // Highlight search term if present
    if (currentLogFilters.search && currentLogFilters.search.trim() !== '') {
        const searchTerm = currentLogFilters.search.trim();
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        safeMessage = safeMessage.replace(regex, '<span class="log-highlight">$1</span>');
    }

    return safeMessage;
}

function clearLogs() {
    const logsContainer = document.getElementById('full-logs');
    logsContainer.innerHTML = '<div class="text-muted">Logs cleared from view</div>';
    showToast('Info', 'Logs view cleared', 'info');
}

function downloadLogs() {
    // Show loading toast
    showToast('Info', 'Preparing logs for download...', 'info');

    // Get current filters
    const queryParams = new URLSearchParams({
        type: currentLogFilters.type,
        level: currentLogFilters.level,
        limit: 0, // Set to 0 to get all logs for download
        search: currentLogFilters.search
    }).toString();

    fetch(`/api/logs?${queryParams}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Create a blob with the logs
                const logType = currentLogFilters.type;
                const logLevel = currentLogFilters.level || 'all';
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

                let logsText;
                if (Array.isArray(data.logs) && data.logs[0] && data.logs[0].raw) {
                    // If logs are structured objects with raw property
                    logsText = data.logs.map(log => log.raw).join('\n');
                } else {
                    // Otherwise just join the logs
                    logsText = data.logs.join('\n');
                }

                const blob = new Blob([logsText], { type: 'text/plain' });
                const filename = `${logType}_${logLevel}_logs_${timestamp}.txt`;

                // Create a download link
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;

                // Trigger download
                document.body.appendChild(a);
                a.click();

                // Clean up
                setTimeout(() => {
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }, 0);

                showToast('Success', 'Logs downloaded successfully', 'success');
            } else {
                showToast('Error', data.message, 'danger');
            }
        })
        .catch(error => {
            console.error('Error downloading logs:', error);
            showToast('Error', 'Failed to download logs', 'danger');
        });
}

function applyLogFilters() {
    // Get filter values from UI
    currentLogFilters = {
        type: document.getElementById('log-type-select').value,
        level: document.getElementById('log-level-select').value,
        limit: parseInt(document.getElementById('log-limit-select').value),
        search: document.getElementById('log-search-input').value.trim()
    };

    // Refresh with new filters
    refreshLogs(currentLogFilters);
}

// Init function to be called when the page loads
function initLogs() {
    console.log("Initializing logs module");

    // Add event listeners for log controls
    document.getElementById('refresh-logs-btn').addEventListener('click', () => refreshLogs());
    document.getElementById('clear-logs-btn').addEventListener('click', clearLogs);
    document.getElementById('download-logs-btn').addEventListener('click', downloadLogs);

    // Add event listeners for filters
    document.getElementById('log-type-select').addEventListener('change', applyLogFilters);
    document.getElementById('log-level-select').addEventListener('change', applyLogFilters);
    document.getElementById('log-limit-select').addEventListener('change', applyLogFilters);
    document.getElementById('search-logs-btn').addEventListener('click', applyLogFilters);

    // Add event listener for search input (on Enter key)
    document.getElementById('log-search-input').addEventListener('keyup', function(event) {
        if (event.key === 'Enter') {
            applyLogFilters();
        }
    });

    // Load logs when tab is shown
    document.getElementById('logs-tab').addEventListener('shown.bs.tab', function() {
        console.log("Logs tab shown, refreshing logs");
        refreshLogs();
    });

    // Initial logs load if we're starting on the logs tab
    if (document.getElementById('logs-tab').classList.contains('active')) {
        console.log("Logs tab active on page load, loading logs");
        refreshLogs();
    }
}

// Add the initialization to the document ready event
document.addEventListener('DOMContentLoaded', initLogs);