// mapping.js
// Three.js scene variables
let scene, camera, renderer, controls, pointCloud;
let pointsGeometry, pointsMaterial;
let axesHelper;

function startMapping() {
    fetch('/api/mapping/start', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addToActivityLog('Mapping mode started');
            document.getElementById('start-mapping-btn').disabled = true;
            document.getElementById('capture-point-cloud-btn').disabled = false;
            document.getElementById('save-map-btn').disabled = false;

            // Initialize point cloud viewer
            initPointCloudViewer();
        } else {
            showToast('Error', data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error starting mapping:', error);
        showToast('Error', 'Failed to start mapping', 'danger');
    });
}

function capturePointCloud() {
    fetch('/api/capture', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addToActivityLog(`Point cloud captured: ${data.timestamp}`);
            showToast('Success', 'Point cloud captured successfully', 'success');

            // Update point cloud list
            refreshPointCloudList();

            // If we have a pointcloud path, update the 3D view
            if (data.pointcloud_path) {
                // This would load the new point cloud into the viewer
                // For a real implementation, we'd need more complex code to load the .npy file
                updatePointCloudStats();
            }
        } else {
            showToast('Error', data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error capturing point cloud:', error);
        showToast('Error', 'Failed to capture point cloud', 'danger');
    });
}

function saveCompleteMap() {
    // This would save the complete merged map
    showToast('Info', 'Saving complete map...', 'info');

    // Placeholder - in a real implementation, we'd call an API endpoint
    setTimeout(() => {
        showToast('Success', 'Complete map saved successfully', 'success');
        addToActivityLog('Complete 3D map saved');
    }, 2000);
}

function refreshPointCloudList() {
    fetch('/api/pointcloud/list')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updatePointCloudList(data.pointclouds);
            } else {
                showToast('Error', data.message, 'danger');
            }
        })
        .catch(error => {
            console.error('Error fetching point cloud list:', error);
            showToast('Error', 'Failed to fetch point cloud list', 'danger');
        });
}

function updatePointCloudList(pointclouds) {
    const listContainer = document.getElementById('point-cloud-list');

    // Clear current list
    listContainer.innerHTML = '';

    if (pointclouds.length === 0) {
        listContainer.innerHTML = '<div class="text-muted">No point clouds saved yet</div>';
        return;
    }

    // Add each point cloud to the list
    pointclouds.forEach(pc => {
        const pcItem = document.createElement('div');
        pcItem.className = 'mb-2';
        pcItem.innerHTML = `
            <div class="form-check">
                <input class="form-check-input point-cloud-checkbox" type="checkbox"
                       value="${pc.path}" id="pc-${pc.timestamp}">
                <label class="form-check-label" for="pc-${pc.timestamp}">
                    <span class="text-info">${pc.datetime}</span> - ${pc.filename}
                </label>
            </div>
        `;

        listContainer.appendChild(pcItem);
    });

    // Update merge button state
    updateMergeButtonState();

    // Add change event listeners to checkboxes
    const checkboxes = document.querySelectorAll('.point-cloud-checkbox');
    checkboxes.forEach(cb => {
        cb.addEventListener('change', updateMergeButtonState);
    });
}

function updateMergeButtonState() {
    const checkboxes = document.querySelectorAll('.point-cloud-checkbox:checked');
    const mergeButton = document.getElementById('merge-selected-btn');

    mergeButton.disabled = checkboxes.length < 2;
}

function mergeSelectedPointClouds() {
    const checkboxes = document.querySelectorAll('.point-cloud-checkbox:checked');
    const selected = Array.from(checkboxes).map(cb => cb.value);

    if (selected.length < 2) {
        showToast('Error', 'Select at least two point clouds to merge', 'danger');
        return;
    }

    showToast('Info', 'Merging point clouds...', 'info');

    // Placeholder - in a real implementation, we'd call an API endpoint
    setTimeout(() => {
        showToast('Success', 'Point clouds merged successfully', 'success');
        addToActivityLog(`Merged ${selected.length} point clouds`);
        updatePointCloudStats();
    }, 2000);
}

function updatePointCloudStats() {
    // Placeholder values - in a real implementation, these would come from the backend
    document.getElementById('total-points').textContent = '1,256,789';
    document.getElementById('frame-count').textContent = '12';
    document.getElementById('map-size').textContent = '48.2 MB';
    document.getElementById('map-density').textContent = '325 points/m³';
}

function initPointCloudViewer() {
    // Get the container
    const container = document.getElementById('point-cloud-view');

    // Show loading message
    container.innerHTML = '<div class="loading">Initializing 3D viewer...</div>';

    try {
        // Create scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111111);

        // Create camera
        camera = new THREE.PerspectiveCamera(
            75, container.clientWidth / container.clientHeight, 0.1, 1000
        );
        camera.position.z = 5;

        // Create renderer
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.innerHTML = '';
        container.appendChild(renderer.domElement);

        // Add orbit controls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;

        // Add axes helper (initially hidden)
        axesHelper = new THREE.AxesHelper(5);
        axesHelper.visible = document.getElementById('show-axis').checked;
        scene.add(axesHelper);

        // Create a demo point cloud
        createDemoPointCloud();

        // Set up resize handling
        window.addEventListener('resize', onWindowResize);

        // Start animation loop
        animate();

        // Set up UI controls
        setupPointCloudControls();

        showToast('Success', '3D viewer initialized', 'success');
    } catch (error) {
        console.error('Error initializing point cloud viewer:', error);
        container.innerHTML = '<div class="loading">Failed to initialize 3D viewer. Try using a browser with WebGL support.</div>';
        showToast('Error', 'Failed to initialize 3D viewer', 'danger');
    }
}

function createDemoPointCloud() {
    // Create a demo point cloud with random points
    const numPoints = 5000;
    pointsGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(numPoints * 3);
    const colors = new Float32Array(numPoints * 3);

    for (let i = 0; i < numPoints; i++) {
        // Position
        positions[i * 3] = (Math.random() - 0.5) * 5;  // x
        positions[i * 3 + 1] = (Math.random() - 0.5) * 5;  // y
        positions[i * 3 + 2] = (Math.random() - 0.5) * 5;  // z

        // Color based on depth (z)
        const depth = (positions[i * 3 + 2] + 2.5) / 5;  // normalize to 0-1
        colors[i * 3] = 1 - depth;  // r
        colors[i * 3 + 1] = depth;  // g
        colors[i * 3 + 2] = depth * 0.5;  // b
    }

    pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointsGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    pointsMaterial = new THREE.PointsMaterial({
        size: 0.1,
        vertexColors: true
    });

    pointCloud = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(pointCloud);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById('point-cloud-view');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function setupPointCloudControls() {
    // Point size control
    document.getElementById('point-size').addEventListener('input', function() {
        if (pointsMaterial) {
            pointsMaterial.size = parseFloat(this.value) * 0.05;
        }
    });

    // Color scheme control
    document.getElementById('color-scheme').addEventListener('change', function() {
        if (pointsGeometry && pointsGeometry.attributes.color && pointsGeometry.attributes.position) {
            const scheme = this.value;
            const positions = pointsGeometry.attributes.position.array;
            const colors = pointsGeometry.attributes.color.array;
            const numPoints = positions.length / 3;

            for (let i = 0; i < numPoints; i++) {
                const x = positions[i * 3];
                const y = positions[i * 3 + 1];
                const z = positions[i * 3 + 2];

                if (scheme === 'depth') {
                    // Color by depth (z)
                    const depth = (z + 2.5) / 5;  // normalize to 0-1
                    colors[i * 3] = 1 - depth;  // r
                    colors[i * 3 + 1] = depth;  // g
                    colors[i * 3 + 2] = depth * 0.5;  // b
                } else if (scheme === 'height') {
                    // Color by height (y)
                    const height = (y + 2.5) / 5;  // normalize to 0-1
                    colors[i * 3] = 0;  // r
                    colors[i * 3 + 1] = height;  // g
                    colors[i * 3 + 2] = 1 - height;  // b
                } else if (scheme === 'rgb') {
                    // Relative RGB based on position
                    colors[i * 3] = (x + 2.5) / 5;  // r - normalize x to 0-1
                    colors[i * 3 + 1] = (y + 2.5) / 5;  // g - normalize y to 0-1
                    colors[i * 3 + 2] = (z + 2.5) / 5;  // b - normalize z to 0-1
                }
            }

            pointsGeometry.attributes.color.needsUpdate = true;
        }
    });

    // Show/hide axes
    document.getElementById('show-axis').addEventListener('change', function() {
        if (axesHelper) {
            axesHelper.visible = this.checked;
        }
    });
}

// Init function to be called when the page loads
function initMapping() {
    document.getElementById('start-mapping-btn').addEventListener('click', startMapping);
    document.getElementById('capture-point-cloud-btn').addEventListener('click', capturePointCloud);
    document.getElementById('save-map-btn').addEventListener('click', saveCompleteMap);
    document.getElementById('refresh-point-cloud-list-btn').addEventListener('click', refreshPointCloudList);
    document.getElementById('merge-selected-btn').addEventListener('click', mergeSelectedPointClouds);

    // Initialize point cloud list
    refreshPointCloudList();
}

// Add the initialization to the document ready event
document.addEventListener('DOMContentLoaded', initMapping);