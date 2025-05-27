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

# Store socketio instance reference (will be set from app.py)
socketio_instance = None


def set_socketio_instance(socketio):
    """Called from app.py to set the socketio instance"""
    global socketio_instance
    socketio_instance = socketio
    logger.info("SocketIO instance set for routes")


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

        # Check if socketio instance is available
        if not socketio_instance:
            return jsonify({'success': False, 'message': 'SocketIO not initialized'}), 500

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
        logger.info(f"Frame test response success: {test_response.get('success', False)}")

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
        logger.info("Stream stop requested")

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
    consecutive_errors = 0
    max_consecutive_errors = 10

    logger.info("=== STREAMING THREAD STARTED ===")

    # Check if socketio instance is available
    if not socketio_instance:
        logger.error("❌ SocketIO instance not available in streaming thread!")
        return

    logger.info("✅ SocketIO instance available, starting frame streaming")

    while True:
        with streaming_lock:
            if not streaming_active:
                logger.info("🛑 Streaming thread stopping - streaming_active is False")
                break

        try:
            # Get frames from vision service
            response = vision_client.get_frames()

            # DEBUG: Log response details every 30 frames (once per second at 30fps)
            if frames_count % 30 == 0:
                logger.info(f"📸 Frame #{frames_count}, success: {response.get('success', False)}")
                if response.get('success', False):
                    logger.info(
                        f"   Left: {len(response.get('left', '')) > 0}, Right: {len(response.get('right', '')) > 0}")

            if not response.get('success', False):
                consecutive_errors += 1
                logger.warning(
                    f"⚠️ Stream failed to get frames (error #{consecutive_errors}): {response.get('message', 'Unknown error')}")

                # Emit error to clients
                try:
                    socketio_instance.emit('error', {'message': 'Failed to get frames from vision service'})
                    logger.debug("✅ Error emitted to clients")
                except Exception as emit_error:
                    logger.error(f"❌ Failed to emit error: {emit_error}")

                # If too many consecutive errors, longer delay
                if consecutive_errors > max_consecutive_errors:
                    logger.error(f"❌ Too many consecutive errors ({consecutive_errors}), longer delay")
                    time.sleep(5)
                else:
                    time.sleep(1)
                continue

            # Reset error counter on success
            consecutive_errors = 0

            # Calculate FPS
            frames_count += 1
            current_time = time.time()

            if current_time - last_fps_update >= 1.0:
                fps = frames_count
                frames_count = 0
                last_fps_update = current_time
                response['fps'] = fps
                logger.debug(f"📊 Current FPS: {fps}")

            # Emit frames to clients
            try:
                socketio_instance.emit('frames', response)

                # Log successful emission every 30 frames
                if frames_count % 30 == 0:
                    logger.debug(f"📡 Frames emitted successfully (#{frames_count})")

            except Exception as emit_error:
                logger.error(f"❌ Failed to emit frames: {emit_error}")

            # Control frame rate
            time.sleep(0.03)  # ~30 FPS target

        except Exception as e:
            consecutive_errors += 1
            logger.error(f"💥 Stream error #{consecutive_errors}: {str(e)}")

            try:
                socketio_instance.emit('error', {'message': f"Streaming error: {str(e)}"})
            except Exception as emit_error:
                logger.error(f"❌ Failed to emit stream error: {emit_error}")

            time.sleep(1)

    logger.info("🏁 Streaming thread stopped")