// navigation.js
// Three.js variables for navigation map
let navScene, navCamera, navRenderer, navControls;
let navMapMesh, robotMarker, goalMarker;
let currentGoalPosition = { x: 0, y: 0, z: 0 };
let isNavigationActive = false;

function startNavigation() {
    // Set loading state
    document.getElementById('navigation-map-view').innerHTML =
        '<div class="d-flex justify-content-center align-items-center h-100 text-white">' +
        '<span class="spinner-border me-2" role="status" aria-hidden="true"></span>' +
        'Loading navigation mode...</div>';

    // Enable navigation controls
    document.getElementById('send-goal-btn').disabled = false;
    document.getElementById('cancel-goal-btn').disabled = false;

    // Enable manual controls
    document.getElementById('move-forward-btn').disabled = false;
    document.getElementById('move-backward-btn').disabled = false;
    document.getElementById('turn-left-btn').disabled = false;
    document.getElementById('turn-right-btn').disabled = false;
    document.getElementById('stop-btn').disabled = false;
    document.getElementById('home-btn').disabled = false;

    // Update robot status
    document.getElementById('robot-status').textContent = 'Connected';
    document.getElementById('robot-status').className = 'connected';

    // Initialize navigation map (in a real implementation, this would load the actual map)
    initNavigationMap();

    addToActivityLog('Navigation mode started');
    addNavigationHistoryEntry('Navigation mode initialized');

    showToast('Success', 'Navigation mode activated', 'success');

    // Start position updates (simulated for demo)
    startPositionUpdates();

    isNavigationActive = true;
}

function sendGoal() {
    if (!isNavigationActive) {
        showToast('Error', 'Navigation mode not active', 'danger');
        return;
    }

    showToast('Info', 'Sending navigation goal to TurtleBot...', 'info');

    // In a real implementation, this would send the actual goal to the robot
    // For demo, we'll use the current goal position set by clicking on the map
    setTimeout(() => {
        const goalText = `X: ${currentGoalPosition.x.toFixed(2)}, Y: ${currentGoalPosition.y.toFixed(2)}`;
        document.getElementById('current-goal').textContent = goalText;

        showToast('Success', 'Navigation goal sent', 'success');
        addToActivityLog('Navigation goal sent: ' + goalText);

        // Add entry to navigation history
        addNavigationHistoryEntry('Goal sent: ' + goalText);

        // Start simulated navigation to goal
        simulateNavigationToGoal();
    }, 1000);
}

function cancelGoal() {
    showToast('Info', 'Canceling navigation goal...', 'info');

    // In a real implementation, this would cancel the current goal
    setTimeout(() => {
        document.getElementById('current-goal').textContent = 'None';
        showToast('Success', 'Navigation goal canceled', 'success');
        addToActivityLog('Navigation goal canceled');

        // Add entry to navigation history
        addNavigationHistoryEntry('Goal canceled');
    }, 1000);
}

function controlRobot(command) {
    if (!isNavigationActive) {
        showToast('Error', 'Navigation mode not active', 'danger');
        return;
    }

    let commandText = '';
    let duration = 1.0; // Default movement duration in seconds

    switch (command) {
        case 'forward':
            commandText = 'Moving forward';
            // Simulate moving robot forward
            moveRobotForward(duration);
            break;
        case 'backward':
            commandText = 'Moving backward';
            // Simulate moving robot backward
            moveRobotBackward(duration);
            break;
        case 'left':
            commandText = 'Turning left';
            // Simulate turning robot left
            turnRobotLeft(duration);
            break;
        case 'right':
            commandText = 'Turning right';
            // Simulate turning robot right
            turnRobotRight(duration);
            break;
        case 'stop':
            commandText = 'Stopping';
            // Stop all robot movement
            break;
        case 'home':
            commandText = 'Returning to home position';
            // Simulate returning to home
            returnRobotToHome();
            break;
    }

    showToast('Command', commandText, 'info');
    addToActivityLog(`Manual control: ${commandText}`);

    // Add entry to navigation history
    addNavigationHistoryEntry(`Manual: ${commandText}`);
}

function addNavigationHistoryEntry(message) {
    const historyContainer = document.getElementById('navigation-history');

    // Remove 'No navigation history' message if it exists
    const noHistoryMsg = historyContainer.querySelector('.text-muted');
    if (noHistoryMsg && noHistoryMsg.textContent === 'No navigation history yet') {
        historyContainer.removeChild(noHistoryMsg);
    }

    const timestamp = new Date().toLocaleTimeString();
    const historyEntry = document.createElement('div');
    historyEntry.innerHTML = `<span class="text-muted">[${timestamp}]</span> ${message}`;

    historyContainer.appendChild(historyEntry);
    historyContainer.scrollTop = historyContainer.scrollHeight;
}

function initNavigationMap() {
    const container = document.getElementById('navigation-map-view');

    try {
        // Create scene
        navScene = new THREE.Scene();
        navScene.background = new THREE.Color(0x111111);

        // Create camera
        navCamera = new THREE.PerspectiveCamera(
            75, container.clientWidth / container.clientHeight, 0.1, 1000
        );
        navCamera.position.set(0, 8, 0);  // Top-down view
        navCamera.lookAt(0, 0, 0);

        // Create renderer
        navRenderer = new THREE.WebGLRenderer({ antialias: true });
        navRenderer.setSize(container.clientWidth, container.clientHeight);
        container.innerHTML = '';
        container.appendChild(navRenderer.domElement);

        // Add orbit controls but limit to top-down view
        navControls = new THREE.OrbitControls(navCamera, navRenderer.domElement);
        navControls.enableDamping = true;
        navControls.dampingFactor = 0.25;
        navControls.maxPolarAngle = Math.PI / 2.5;  // Limit rotation to maintain top-down-ish view

        // Create a simple floor grid
        const gridHelper = new THREE.GridHelper(10, 10);
        navScene.add(gridHelper);

        // Create a simple demo room map - in a real application, this would be generated from point clouds
        createDemoRoomMap();

        // Add robot marker
        createRobotMarker();

        // Add event listener for goal selection
        navRenderer.domElement.addEventListener('click', onMapClick);

        // Set up resize handling
        window.addEventListener('resize', onNavWindowResize);

        // Start animation loop
        animateNavMap();

    } catch (error) {
        console.error('Error initializing navigation map:', error);
        container.innerHTML = '<div class="d-flex justify-content-center align-items-center h-100 text-white">' +
            'Failed to initialize 3D map. Try using a browser with WebGL support.</div>';
        showToast('Error', 'Failed to initialize navigation map', 'danger');
    }
}

function createDemoRoomMap() {
    // Create a simple rectangular room with walls for demo
    const roomGeometry = new THREE.BoxGeometry(8, 0.1, 8);  // Floor
    const roomMaterial = new THREE.MeshBasicMaterial({
        color: 0x333333,
        wireframe: false,
        transparent: true,
        opacity: 0.7
    });
    navMapMesh = new THREE.Mesh(roomGeometry, roomMaterial);
    navMapMesh.position.y = -0.05;  // Place just below 0
    navScene.add(navMapMesh);

    // Add walls
    const wallMaterial = new THREE.MeshBasicMaterial({
        color: 0x555555,
        transparent: true,
        opacity: 0.5
    });

    // North wall
    const northWall = new THREE.Mesh(
        new THREE.BoxGeometry(8, 0.5, 0.1),
        wallMaterial
    );
    northWall.position.set(0, 0.25, -4);
    navScene.add(northWall);

    // South wall
    const southWall = new THREE.Mesh(
        new THREE.BoxGeometry(8, 0.5, 0.1),
        wallMaterial
    );
    southWall.position.set(0, 0.25, 4);
    navScene.add(southWall);

    // East wall
    const eastWall = new THREE.Mesh(
        new THREE.BoxGeometry(0.1, 0.5, 8),
        wallMaterial
    );
    eastWall.position.set(4, 0.25, 0);
    navScene.add(eastWall);

    // West wall
    const westWall = new THREE.Mesh(
        new THREE.BoxGeometry(0.1, 0.5, 8),
        wallMaterial
    );
    westWall.position.set(-4, 0.25, 0);
    navScene.add(westWall);

    // Add some simple furniture obstacles
    const obstacleMaterial = new THREE.MeshBasicMaterial({ color: 0x8B4513 });

    // Table
    const table = new THREE.Mesh(
        new THREE.BoxGeometry(1.5, 0.5, 1),
        obstacleMaterial
    );
    table.position.set(2, 0.25, 2);
    navScene.add(table);

    // Cabinet
    const cabinet = new THREE.Mesh(
        new THREE.BoxGeometry(0.8, 0.6, 2),
        obstacleMaterial
    );
    cabinet.position.set(-3, 0.3, -3);
    navScene.add(cabinet);
}

function createRobotMarker() {
    // Create a cone to represent the robot and its orientation
    const robotGeometry = new THREE.ConeGeometry(0.3, 0.5, 8);
    const robotMaterial = new THREE.MeshBasicMaterial({ color: 0x00aaff });

    robotMarker = new THREE.Mesh(robotGeometry, robotMaterial);
    robotMarker.rotation.x = Math.PI / 2;  // Point forward (along Z axis)
    robotMarker.position.set(0, 0.25, 0);  // Center of the map
    navScene.add(robotMarker);
}

function onMapClick(event) {
    if (!isNavigationActive) return;

    // Calculate mouse position in normalized device coordinates (-1 to +1)
    const rect = navRenderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
    );

    // Create raycaster
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, navCamera);

    // Check for intersections with the floor/map
    const intersects = raycaster.intersectObject(navMapMesh);

    if (intersects.length > 0) {
        // Get the point where the ray intersects the floor
        const point = intersects[0].point;

        // Update goal position
        currentGoalPosition = { x: point.x, y: point.y, z: point.z };

        // Add or update goal marker
        if (goalMarker) {
            navScene.remove(goalMarker);
        }

        const goalGeometry = new THREE.CylinderGeometry(0.3, 0.3, 0.1, 16);
        const goalMaterial = new THREE.MeshBasicMaterial({ color: 0xffaa00 });
        goalMarker = new THREE.Mesh(goalGeometry, goalMaterial);
        goalMarker.position.set(point.x, 0.05, point.z);
        navScene.add(goalMarker);

        // Update UI
        addNavigationHistoryEntry(`Goal selected: X: ${point.x.toFixed(2)}, Z: ${point.z.toFixed(2)}`);

        // Enable the send goal button
        document.getElementById('send-goal-btn').disabled = false;
    }
}

function animateNavMap() {
    if (!isNavigationActive) return;

    requestAnimationFrame(animateNavMap);
    navControls.update();
    navRenderer.render(navScene, navCamera);
}

function onNavWindowResize() {
    const container = document.getElementById('navigation-map-view');
    navCamera.aspect = container.clientWidth / container.clientHeight;
    navCamera.updateProjectionMatrix();
    navRenderer.setSize(container.clientWidth, container.clientHeight);
}

// Simulated robot movement functions
function startPositionUpdates() {
    // Update the position display every second (simulated)
    setInterval(() => {
        if (!isNavigationActive) return;

        // Get current position from the robotMarker
        const pos = robotMarker.position;
        const euler = new THREE.Euler().setFromQuaternion(robotMarker.quaternion);

        // Convert to degrees for display
        const yaw = THREE.MathUtils.radToDeg(euler.y).toFixed(1);

        // Update displays
        document.getElementById('robot-position').textContent =
            `X: ${pos.x.toFixed(2)}, Y: ${pos.y.toFixed(2)}, Z: ${pos.z.toFixed(2)}`;
        document.getElementById('robot-orientation').textContent =
            `Yaw: ${yaw}°`;
        document.getElementById('robot-battery').textContent = '85%';
    }, 500);
}

function moveRobotForward(duration) {
    if (!robotMarker) return;

    // Extract direction from robot's rotation
    const direction = new THREE.Vector3(0, 0, 1);
    direction.applyQuaternion(robotMarker.quaternion);
    direction.normalize();

    // Move in that direction
    const targetPosition = robotMarker.position.clone().add(direction.multiplyScalar(1));

    // Simple animation
    const startPosition = robotMarker.position.clone();
    const startTime = Date.now();

    function animate() {
        const now = Date.now();
        const elapsed = (now - startTime) / 1000; // seconds

        if (elapsed < duration) {
            const t = elapsed / duration; // 0 to 1
            robotMarker.position.lerpVectors(startPosition, targetPosition, t);
            requestAnimationFrame(animate);
        } else {
            robotMarker.position.copy(targetPosition);
        }
    }

    animate();
}

function moveRobotBackward(duration) {
    if (!robotMarker) return;

    // Extract direction from robot's rotation
    const direction = new THREE.Vector3(0, 0, -1);
    direction.applyQuaternion(robotMarker.quaternion);
    direction.normalize();

    // Move in that direction
    const targetPosition = robotMarker.position.clone().add(direction.multiplyScalar(1));

    // Simple animation
    const startPosition = robotMarker.position.clone();
    const startTime = Date.now();

    function animate() {
        const now = Date.now();
        const elapsed = (now - startTime) / 1000; // seconds

        if (elapsed < duration) {
            const t = elapsed / duration; // 0 to 1
            robotMarker.position.lerpVectors(startPosition, targetPosition, t);
            requestAnimationFrame(animate);
        } else {
            robotMarker.position.copy(targetPosition);
        }
    }

    animate();
}

function turnRobotLeft(duration) {
    if (!robotMarker) return;

    // Get current rotation
    const startRotation = robotMarker.rotation.y;
    const targetRotation = startRotation + Math.PI/2; // 90 degrees

    const startTime = Date.now();

    function animate() {
        const now = Date.now();
        const elapsed = (now - startTime) / 1000; // seconds

        if (elapsed < duration) {
            const t = elapsed / duration; // 0 to 1
            robotMarker.rotation.y = startRotation + t * (targetRotation - startRotation);
            requestAnimationFrame(animate);
        } else {
            robotMarker.rotation.y = targetRotation;
        }
    }

    animate();
}

function turnRobotRight(duration) {
    if (!robotMarker) return;

    // Get current rotation
    const startRotation = robotMarker.rotation.y;
    const targetRotation = startRotation - Math.PI/2; // 90 degrees

    const startTime = Date.now();

    function animate() {
        const now = Date.now();
        const elapsed = (now - startTime) / 1000; // seconds

        if (elapsed < duration) {
            const t = elapsed / duration; // 0 to 1
            robotMarker.rotation.y = startRotation + t * (targetRotation - startRotation);
            requestAnimationFrame(animate);
        } else {
            robotMarker.rotation.y = targetRotation;
        }
    }

    animate();
}

function returnRobotToHome() {
    if (!robotMarker) return;

    // Home position is (0, 0.25, 0)
    const targetPosition = new THREE.Vector3(0, 0.25, 0);
    const startPosition = robotMarker.position.clone();
    const duration = 2.0; // seconds

    const startTime = Date.now();

    function animate() {
        const now = Date.now();
        const elapsed = (now - startTime) / 1000; // seconds

        if (elapsed < duration) {
            const t = elapsed / duration; // 0 to 1
            robotMarker.position.lerpVectors(startPosition, targetPosition, t);
            requestAnimationFrame(animate);
        } else {
            robotMarker.position.copy(targetPosition);
            robotMarker.rotation.y = 0; // Reset rotation
        }
    }

    animate();

    addNavigationHistoryEntry("Robot returned to home position");
}

function simulateNavigationToGoal() {
    if (!robotMarker || !goalMarker) return;

    // Calculate path from current position to goal
    const startPosition = robotMarker.position.clone();
    const targetPosition = goalMarker.position.clone();
    targetPosition.y = 0.25; // Match robot's height

    // We'll do a simple direct path for demo
    const duration = 3.0; // seconds
    const startTime = Date.now();

    // Also calculate the angle to face the target
    const direction = new THREE.Vector3().subVectors(targetPosition, startPosition);
    const angle = Math.atan2(direction.x, direction.z);

    // First rotate towards the goal
    const startRotation = robotMarker.rotation.y;
    const rotationDuration = 1.0; // seconds

    function rotateTowardsGoal() {
        const now = Date.now();
        const elapsed = (now - startTime) / 1000; // seconds

        if (elapsed < rotationDuration) {
            const t = elapsed / rotationDuration; // 0 to 1
            robotMarker.rotation.y = startRotation + t * (angle - startRotation);
            requestAnimationFrame(rotateTowardsGoal);
        } else {
            robotMarker.rotation.y = angle;
            moveTowardsGoal();
        }
    }

    function moveTowardsGoal() {
        const moveStartTime = Date.now();

        function animate() {
            const now = Date.now();
            const elapsed = (now - moveStartTime) / 1000; // seconds

            if (elapsed < duration) {
                const t = elapsed / duration; // 0 to 1
                robotMarker.position.lerpVectors(startPosition, targetPosition, t);
                requestAnimationFrame(animate);
            } else {
                robotMarker.position.copy(targetPosition);
                goalReached();
            }
        }

        animate();
    }

    function goalReached() {
        // Update UI
        addNavigationHistoryEntry("Goal reached!");
        showToast('Success', 'Navigation goal reached', 'success');
        document.getElementById('current-goal').textContent = 'None';
    }

    // Start the sequence
    rotateTowardsGoal();
}

// Init function to be called when the page loads
function initNavigation() {
    document.getElementById('start-navigation-btn').addEventListener('click', startNavigation);
    document.getElementById('send-goal-btn').addEventListener('click', sendGoal);
    document.getElementById('cancel-goal-btn').addEventListener('click', cancelGoal);

    // Manual control buttons
    document.getElementById('move-forward-btn').addEventListener('click', () => controlRobot('forward'));
    document.getElementById('move-backward-btn').addEventListener('click', () => controlRobot('backward'));
    document.getElementById('turn-left-btn').addEventListener('click', () => controlRobot('left'));
    document.getElementById('turn-right-btn').addEventListener('click', () => controlRobot('right'));
    document.getElementById('stop-btn').addEventListener('click', () => controlRobot('stop'));
    document.getElementById('home-btn').addEventListener('click', () => controlRobot('home'));
}

// Add the initialization to the document ready event
document.addEventListener('DOMContentLoaded', initNavigation);