from flask import Blueprint, render_template, jsonify, request
from flask_socketio import emit
import logging
import threading
import time
import os
from client import VisionClient

web = Blueprint('web', __name__)
logger = logging.getLogger(__name__)

# Create client instance
vision_client = VisionClient(base_url=os.environ.get('VISION_SERVICE_URL', 'http://localhost:5050'))

# Streaming state
streaming_active = False
streaming_thread = None
streaming_lock = threading.Lock()


@web.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@web.route('/api/stream/start', methods=['POST'])
def start_stream():
    """Start stream thread to push frames via SocketIO"""
    global streaming_active, streaming_thread

    with streaming_lock:
        if streaming_active:
            return jsonify({'success': False, 'message': 'Stream already running'})

        # Start vision service if not running
        response = vision_client.start_vision()
        if not response.get('success', False):
            return jsonify(response), 500

        streaming_active = True
        streaming_thread = threading.Thread(target=stream_frames)
        streaming_thread.daemon = True
        streaming_thread.start()

        return jsonify({'success': True})


@web.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    """Stop streaming thread"""
    global streaming_active

    with streaming_lock:
        if not streaming_active:
            return jsonify({'success': False, 'message': 'Stream not running'})

        streaming_active = False

        # Do not stop vision service here - it may be used by other clients

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