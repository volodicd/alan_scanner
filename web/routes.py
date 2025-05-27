import logging
import os
import threading
import time
from flask import Blueprint, render_template, jsonify, request
from client import VisionClient

web = Blueprint('web', __name__)
logger = logging.getLogger(__name__)

vision_client = VisionClient(base_url=os.environ.get('VISION_SERVICE_URL', 'http://localhost:5050'))

streaming_active = False
streaming_thread = None
streaming_lock = threading.Lock()


@web.route('/')
def index():
    return render_template('index.html')


import logging
import os
import threading
import time
from flask import Blueprint, render_template, jsonify, request
from client import VisionClient

web = Blueprint('web', __name__)
logger = logging.getLogger(__name__)

vision_client = VisionClient(base_url=os.environ.get('VISION_SERVICE_URL', 'http://localhost:5050'))

streaming_active = False
streaming_thread = None
streaming_lock = threading.Lock()


@web.route('/')
def index():
    return render_template('index.html')


@web.route('/api/stream/start', methods=['POST'])
def start_stream():
    """Start stream thread to push frames via socketio"""
    global streaming_active, streaming_thread

    with streaming_lock:
        if streaming_active:
            return jsonify({'success': False, 'message': 'Stream already running'})

        # STEP 0: Stop any running vision processing first
        logger.info("Stopping any existing vision processing...")
        vision_client.stop_vision()
        
        # STEP 1: Initialize vision system
        logger.info("Initializing vision system...")
        init_response = vision_client.initialize_vision()
        if not init_response.get('success', False):
            logger.error(f"Vision initialization failed: {init_response.get('message')}")
            return jsonify({
                'success': False,
                'message': f"Vision initialization failed: {init_response.get('message')}"
            }), 500

        # STEP 2: Start vision processing
        logger.info("Starting vision processing...")
        start_response = vision_client.start_vision()
        if not start_response.get('success', False):
            logger.warning(f"Vision processing start warning: {start_response.get('message')}")
            # Continue anyway, as this might not be critical

        # STEP 3: Wait a moment for cameras to initialize
        time.sleep(2)

        # STEP 4: Wait for vision processing to stabilize, then test frames
        logger.info("Waiting for vision processing to stabilize...")
        time.sleep(3)
        
        logger.info("Testing frame retrieval...")
        test_response = vision_client.get_frames()
        logger.info(f"Frame test response: {test_response}")
        
        if not test_response.get('success', False):
            logger.error(f"Frame test failed: {test_response.get('message')}")
            return jsonify({
                'success': False,
                'message': f"Cannot get frames from vision service: {test_response.get('message')}"
            }), 500

        # STEP 5: Start streaming thread
        streaming_active = True
        streaming_thread = threading.Thread(target=stream_frames)
        streaming_thread.daemon = True
        streaming_thread.start()

        logger.info("Streaming started successfully")
        return jsonify({'success': True, 'message': 'Stream started successfully'})



@web.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    """Stop streaming thread"""
    global streaming_active

    with streaming_lock:
        if not streaming_active:
            return jsonify({'success': False, 'message': 'Stream not running'})

        streaming_active = False

        # Do not stop vision service here - it may be used by turtle bot
        # DO NOT DELTE STEREO INSTANCE HERE, WILL LEAD TO THE SITUATION WHERE WEB IS OFF, TURTLE IS ON, STEREO IS CRASHED

        return jsonify({'success': True})


# API proxy routes - these forward requests to the vision service
@web.route('/api/vision/initialize', methods=['POST'])
def initialize_vision():
    return jsonify(vision_client.initialize_vision(**request.json))


@web.route('/api/vision/start', methods=['POST'])
def start_vision():
    return jsonify(vision_client.start_vision())


@web.route('/api/vision/stop', methods=['POST'])
def stop_vision():
    return jsonify(vision_client.stop_vision())


@web.route('/api/capture', methods=['POST'])
def capture_frame():
    return jsonify(vision_client.capture_frame())


@web.route('/api/calibrate', methods=['POST'])
def run_calibration():
    return jsonify(vision_client.run_calibration(**request.json))


@web.route('/api/calibrate/start', methods=['POST'])
def start_calibration():
    """Start the calibration process"""
    try:
        # First, stop vision processing if it's running
        vision_client.stop_vision()

        # Wait a brief moment for processing to fully stop
        time.sleep(1)

        # Extract calibration parameters from request
        data = request.json or {}
        checkerboard_size = data.get('checkerboard_size', [7, 6])
        square_size = data.get('square_size', 0.025)
        num_samples = data.get('num_samples', 20)

        # Start calibration
        result = vision_client.run_calibration(
            checkerboard_size=checkerboard_size,
            square_size=square_size,
            num_samples=num_samples
        )

        if isinstance(result, dict) and 'success' in result:
            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid response from vision service'
            })

    except Exception as e:
        logger.error(f"Error starting calibration: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error starting calibration: {str(e)}"
        })


@web.route('/api/calibrate/status', methods=['GET'])
def get_calibration_status():
    return jsonify(vision_client.get_calibration_status())


@web.route('/api/calibrate/detect', methods=['POST'])
def detect_checkerboard():
    return jsonify(vision_client.detect_checkerboard(**request.json))


@web.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        return jsonify(vision_client.get_config())
    else:
        return jsonify(vision_client.update_config(request.json))


@web.route('/api/system/info', methods=['GET'])
def get_system_info():
    return jsonify(vision_client.get_system_info())


# Function to run in streaming thread
def stream_frames():
    """Thread that streams frames via SocketIO"""
    global streaming_active
    frames_count = 0
    last_fps_update = time.time()

    # Import socketio instance (circular import prevention)
    from app import socketio

    while True:
        with streaming_lock:
            if not streaming_active:
                break

        try:
            # Get frames from vision service
            response = vision_client.get_frames()

            if not response.get('success', False):
                logger.warning(f"Stream failed to get frames: {response}")
                socketio.emit('error', {'message': 'Failed to get frames'})
                time.sleep(1)
                continue

            # Calculate FPS
            frames_count += 1
            current_time = time.time()

            if current_time - last_fps_update >= 1.0:
                fps = frames_count
                frames_count = 0
                last_fps_update = current_time
                response['fps'] = fps

            # Emit frames to clients
            socketio.emit('frames', response)

            # Control frame rate
            time.sleep(0.03)  # ~30 FPS target

        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            socketio.emit('error', {'message': f"Streaming error: {str(e)}"})
            time.sleep(1)

    logger.info("Streaming stopped")