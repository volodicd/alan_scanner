// code_editor.js
let editor = null; // Monaco editor instance
let originalCode = ''; // Original code content for comparison
let selectedCodeVersion = null; // Currently selected version for rollback
let editorInitialized = false; // Flag to prevent duplicate initialization

function initializeCodeEditor() {
    console.log("initializeCodeEditor called");

    // If editor already exists and is attached to DOM, don't create a new one
    if (editor) {
        console.log("Editor already exists, not reinitializing");
        return;
    }

    // Show loading indicator
    document.getElementById('code-editor').innerHTML = '<div class="d-flex justify-content-center align-items-center h-100 text-secondary">' +
        '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>' +
        'Loading editor...</div>';

    console.log('Loading Monaco editor...');

    // Load the editor
    require(['vs/editor/editor.main'], function() {
        console.log('Monaco editor loaded, fetching code content');
        // Fetch the current code
        fetchCurrentCode();
    });
}

function fetchCurrentCode() {
    console.log('Fetching current code from API');
    fetch('/api/code/current')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('Code fetched successfully');
                // Save original code
                originalCode = data.code;

                // Create editor - only if it doesn't already exist
                if (!editor) {
                    createEditor(data.code);
                } else {
                    console.log('Updating existing editor value');
                    editor.setValue(data.code);
                }

                // Update code info
                updateCodeInfo(data);

                // Fetch code versions for history panel
                fetchCodeVersions();

                // Set initialization flag
                editorInitialized = true;
            } else {
                console.error('Error from API:', data.message);
                showToast('Error', data.message, 'danger');
                document.getElementById('code-editor').innerHTML =
                    '<div class="alert alert-danger">Failed to load code: ' + data.message + '</div>';
            }
        })
        .catch(error => {
            console.error('Error fetching code:', error);
            showToast('Error', 'Failed to fetch code', 'danger');
            document.getElementById('code-editor').innerHTML =
                '<div class="alert alert-danger">Failed to load code: ' + error.message + '</div>';
        });
}

function createEditor(code) {
    console.log('Creating Monaco editor instance');

    // Safety check - if the element already has an editor attached, don't create a new one
    const editorElement = document.getElementById('code-editor');
    if (editorElement.classList.contains('monaco-initialized')) {
        console.warn('Element already has an editor initialized');
        return;
    }

    try {
        // Clear any previous content
        editorElement.innerHTML = '';

        // Register Python language for syntax highlighting
        monaco.languages.register({ id: 'python' });

        // Define editor options
        const options = {
            value: code,
            language: 'python',
            theme: 'vs-dark',
            automaticLayout: true,
            fontSize: 14,
            lineNumbers: 'on',
            scrollBeyondLastLine: false,
            minimap: {
                enabled: true
            },
            scrollbar: {
                verticalScrollbarSize: 10,
                horizontalScrollbarSize: 10
            }
        };

        // Create the editor
        editor = monaco.editor.create(editorElement, options);
        console.log('Editor created successfully');

        // Mark the element as having an initialized editor
        editorElement.classList.add('monaco-initialized');

        // Add change listener to update status
        editor.onDidChangeModelContent(() => {
            updateCodeStatus();
        });

        // Add keyboard shortcut for save (Ctrl+S or Cmd+S)
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, function() {
            saveCode();
        });
    } catch (err) {
        console.error('Error creating editor:', err);
        editorElement.innerHTML =
            '<div class="alert alert-danger">Error creating editor: ' + err.message + '</div>';
    }
}

function destroyEditor() {
    // Properly dispose the editor if it exists
    if (editor) {
        console.log('Disposing editor instance');
        editor.dispose();
        editor = null;

        // Remove the initialized class
        const editorElement = document.getElementById('code-editor');
        if (editorElement) {
            editorElement.classList.remove('monaco-initialized');
        }
    }
}

function updateCodeInfo(data) {
    // Calculate code size
    const sizeInBytes = new Blob([data.code]).size;
    let sizeText;

    if (sizeInBytes > 1024 * 1024) {
        sizeText = (sizeInBytes / (1024 * 1024)).toFixed(2) + ' MB';
    } else if (sizeInBytes > 1024) {
        sizeText = (sizeInBytes / 1024).toFixed(2) + ' KB';
    } else {
        sizeText = sizeInBytes + ' bytes';
    }

    document.getElementById('code-size').textContent = sizeText;

    // Set last modified time (using current time as a fallback)
    const lastModified = data.last_modified || new Date().toLocaleString();
    document.getElementById('code-last-modified').textContent = lastModified;

    // Set initial status
    document.getElementById('code-status').textContent = 'Loaded';
    document.getElementById('code-status').className = 'badge bg-secondary';
}

function updateCodeStatus() {
    const statusBadge = document.getElementById('code-status');

    // Compare current code with original
    if (editor && originalCode !== editor.getValue()) {
        statusBadge.textContent = 'Modified';
        statusBadge.className = 'badge bg-warning';
    } else {
        statusBadge.textContent = 'Unchanged';
        statusBadge.className = 'badge bg-secondary';
    }
}

function fetchCodeVersions() {
    fetch('/api/code/versions')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateCodeVersionsList(data.versions);
            } else {
                showToast('Error', data.message, 'danger');
            }
        })
        .catch(error => {
            console.error('Error fetching code versions:', error);
            showToast('Error', 'Failed to fetch code versions', 'danger');
        });
}

function updateCodeVersionsList(versions) {
    const listContainer = document.getElementById('code-versions-list');

    // Clear current list
    listContainer.innerHTML = '';

    if (!versions || versions.length === 0) {
        listContainer.innerHTML = '<div class="text-muted">No version history yet</div>';
        document.getElementById('rollback-code-btn').disabled = true;
        return;
    }

    // Enable rollback button (will be enabled when a version is selected)
    document.getElementById('rollback-code-btn').disabled = true;

    // Add each version to the list
    versions.forEach(version => {
        const versionItem = document.createElement('div');
        versionItem.className = 'version-item';
        versionItem.setAttribute('data-version', version.timestamp);
        versionItem.innerHTML = `
            <div class="form-check">
                <input class="form-check-input code-version-radio" type="radio"
                       name="code-version" value="${version.timestamp}"
                       id="version-${version.timestamp}">
                <label class="form-check-label" for="version-${version.timestamp}">
                    <span class="text-info">${version.datetime}</span>
                </label>
            </div>
        `;

        listContainer.appendChild(versionItem);

        // Add click handler for the whole version item
        versionItem.addEventListener('click', function() {
            // Find the radio button inside this item and check it
            const radio = this.querySelector('input[type="radio"]');
            radio.checked = true;

            // Update selected version
            selectedCodeVersion = radio.value;

            // Update UI to show selection
            document.querySelectorAll('.version-item').forEach(item => {
                item.classList.remove('selected');
            });
            this.classList.add('selected');

            // Enable rollback button
            document.getElementById('rollback-code-btn').disabled = false;
        });
    });

    // Add change event listeners to radio buttons
    const radios = document.querySelectorAll('.code-version-radio');
    radios.forEach(radio => {
        radio.addEventListener('change', function() {
            selectedCodeVersion = this.value;
            document.getElementById('rollback-code-btn').disabled = false;
        });
    });
}

function saveCode() {
    if (!editor) {
        showToast('Error', 'Code editor not initialized', 'danger');
        return;
    }

    const code = editor.getValue();

    // Show saving indicator
    const statusBadge = document.getElementById('code-status');
    statusBadge.textContent = 'Saving...';
    statusBadge.className = 'badge bg-info';

    fetch('/api/code/update', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Success', 'Code saved successfully', 'success');
            addToActivityLog('Code updated');

            // Update original code
            originalCode = code;

            // Update status
            statusBadge.textContent = 'Saved';
            statusBadge.className = 'badge bg-success';

            // Update code last modified time
            document.getElementById('code-last-modified').textContent = new Date().toLocaleString();

            // Update version list
            updateCodeVersionsList(data.backup_versions);

            // After a delay, return to "Unchanged" status
            setTimeout(() => {
                if (originalCode === editor.getValue()) {
                    statusBadge.textContent = 'Unchanged';
                    statusBadge.className = 'badge bg-secondary';
                }
            }, 3000);
        } else {
            showToast('Error', data.message, 'danger');

            // Update status to show error
            statusBadge.textContent = 'Error';
            statusBadge.className = 'badge bg-danger';
        }
    })
    .catch(error => {
        console.error('Error saving code:', error);
        showToast('Error', 'Failed to save code', 'danger');

        // Update status to show error
        statusBadge.textContent = 'Error';
        statusBadge.className = 'badge bg-danger';
    });
}

function revertCode() {
    if (!editor) {
        showToast('Error', 'Code editor not initialized', 'danger');
        return;
    }

    showToast('Info', 'Reverting changes...', 'info');

    // Revert to original code
    editor.setValue(originalCode);

    // Update status
    document.getElementById('code-status').textContent = 'Unchanged';
    document.getElementById('code-status').className = 'badge bg-secondary';
}

function rollbackCode() {
    if (!selectedCodeVersion) {
        showToast('Error', 'No version selected', 'warning');
        return;
    }

    // Show loading state
    const statusBadge = document.getElementById('code-status');
    statusBadge.textContent = 'Rolling back...';
    statusBadge.className = 'badge bg-info';

    fetch('/api/code/rollback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ version: selectedCodeVersion }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Success', `Rolled back to version ${selectedCodeVersion}`, 'success');
            addToActivityLog(`Code rolled back to version ${selectedCodeVersion}`);

            // Reload the editor content
            fetchCurrentCode();
        } else {
            showToast('Error', data.message, 'danger');

            // Update status to show error
            statusBadge.textContent = 'Error';
            statusBadge.className = 'badge bg-danger';
        }
    })
    .catch(error => {
        console.error('Error rolling back code:', error);
        showToast('Error', 'Failed to roll back code', 'danger');

        // Update status to show error
        statusBadge.textContent = 'Error';
        statusBadge.className = 'badge bg-danger';
    });
}

// Handle tab navigation - clean up editor when leaving the tab
function handleTabChange() {
    // Add event to detect when leaving the code tab
    document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(button => {
        button.addEventListener('shown.bs.tab', function(event) {
            // If we're navigating away from code tab
            if (event.relatedTarget && event.relatedTarget.id === 'code-tab') {
                // Clean up existing editor when leaving code tab
                destroyEditor();
            }
        });
    });
}

// Init function to be called when the page loads
function initCodeEditor() {
    console.log('Initializing code editor module');

    // Add event listeners for the code editor tab
    document.getElementById('code-tab').addEventListener('shown.bs.tab', function() {
        console.log('Code editor tab shown, initializing editor');
        initializeCodeEditor();
    });

    document.getElementById('save-code-btn').addEventListener('click', saveCode);
    document.getElementById('revert-code-btn').addEventListener('click', revertCode);
    document.getElementById('rollback-code-btn').addEventListener('click', rollbackCode);

    // Handle tab navigation cleanup
    handleTabChange();
}

// Add the initialization to the document ready event
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, setting up code editor');
    initCodeEditor();
});