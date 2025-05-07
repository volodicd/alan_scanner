# routes/camera_routes.py
from flask import jsonify
import cv2
import base64
import threading
import time
import os
from datetime import datetime
from collections import deque

from config import (
    current_config, logger
)
from utils import emit

# Global thread for streaming
stream_thread = None
stream_thread_active = False  # Track if the thread should continue running

# Thread safety lock
thread_lock = threading.Lock()

# For FPS calculation
frame_times = deque(maxlen=30)  # Store last 30 frame timestamps
last_fps_update = 0
current_fps = 0


def register_camera_routes(app):
    """Register camera-related routes."""

    @app.route('/api/stream/start', methods=['POST'])
    def start_stream():
        global is_streaming, stream_thread, stream_thread_active, stereo_vision

        with thread_lock:
            # Check if thread is still running from previous session
            if stream_thread is not None and stream_thread.is_alive():
                # Stop previous thread
                is_streaming = False
                stream_thread_active = False
                # Wait for thread to finish
                stream_thread.join(timeout=2.0)

            if is_streaming:
                return jsonify({'success': False, 'message': 'Stream is already running'})

            # Initialize stereo vision if needed
            if stereo_vision is None:
                from stereo_vision import StereoVision
                stereo_vision = StereoVision(
                    left_cam_idx=current_config["left_cam_idx"],
                    right_cam_idx=current_config["right_cam_idx"],
                    width=current_config["width"],
                    height=current_config["height"]
                )

            # Start streaming thread
            is_streaming = True
            stream_thread_active = True
            stream_thread = threading.Thread(target=process_stream)
            stream_thread.daemon = True
            stream_thread.start()

        return jsonify({'success': True})

    @app.route('/api/stream/stop', methods=['POST'])
    def stop_stream():
        global is_streaming, stream_thread, stream_thread_active

        with thread_lock:
            if not is_streaming:
                return jsonify({'success': False, 'message': 'Stream is not running'})

            # Set flags to stop streaming
            is_streaming = False
            stream_thread_active = False

            # Wait for thread to finish
            if stream_thread is not None and stream_thread.is_alive():
                stream_thread.join(timeout=5.0)
                if stream_thread.is_alive():
                    logger.warning("Stream thread did not terminate within timeout")

        return jsonify({'success': True})

    @app.route('/api/capture', methods=['POST'])
    def handle_capture_frame():
        if stereo_vision is None:
            return jsonify({'success': False, 'message': 'Stereo vision not initialized'}), 400

        try:
            left_frame, right_frame = stereo_vision.capture_frames()
            if left_frame is None or right_frame is None:
                return jsonify({'success': False, 'message': 'Failed to capture frames'}), 500

            # Get the current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Make sure directory exists
            os.makedirs('static/captures', exist_ok=True)

            # Save raw captures
            left_filepath = f"static/captures/left_{timestamp}.jpg"
            right_filepath = f"static/captures/right_{timestamp}.jpg"
            cv2.imwrite(left_filepath, left_frame)
            cv2.imwrite(right_filepath, right_frame)

            # Process and save disparity if calibration is loaded
            disparity_filepath = None
            has_calibration = False

            try:
                has_calibration = stereo_vision.load_calibration()
                if has_calibration:
                    left_rect, right_rect = stereo_vision.get_rectified_images(left_frame, right_frame)
                    _, disp_color = stereo_vision.compute_disparity_map(left_rect, right_rect)
                    disparity_filepath = f"static/captures/disparity_{timestamp}.jpg"
                    cv2.imwrite(disparity_filepath, disp_color)
            except Exception as e:
                logger.error(f"Error processing disparity: {str(e)}")
                # Continue with returning the raw frames

            return jsonify({
                'success': True,
                'timestamp': timestamp,
                'left_path': left_filepath,
                'right_path': right_filepath,
                'disparity_path': disparity_filepath,
                'has_calibration': has_calibration
            })

        except Exception as e:
            logger.error(f"Failed to capture frame: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500


def calculate_fps():
    """Calculate the current FPS based on stored frame times."""
    global frame_times, current_fps

    if len(frame_times) < 2:
        return 0.0

    # Calculate time difference between oldest and newest frame
    time_diff = frame_times[-1] - frame_times[0]
    if time_diff <= 0:
        return 0.0

    # Calculate FPS based on number of frames and time elapsed
    return (len(frame_times) - 1) / time_diff


def process_stream():
    """Process and stream frames from the cameras."""
    global is_streaming, stereo_vision, stream_thread_active, frame_times, current_fps, last_fps_update

    if stereo_vision is None:
        logger.error("Stereo vision not initialized")
        emit('error', {'message': 'Stereo vision not initialized'})
        with thread_lock:
            is_streaming = False
            stream_thread_active = False
        return

    # Open cameras
    try:
        stereo_vision.open_cameras()
    except Exception as e:
        logger.error(f"Failed to open cameras: {str(e)}")
        emit('error', {'message': f"Failed to open cameras: {str(e)}"})
        with thread_lock:
            is_streaming = False
            stream_thread_active = False
        return

    # Check if calibration is available
    has_calibration = False
    try:
        has_calibration = stereo_vision.load_calibration()
        if has_calibration:
            logger.info("Calibration loaded successfully")
        else:
            logger.info("No calibration data found, streaming raw frames")
    except Exception as e:
        logger.error(f"Error loading calibration: {str(e)}")
        emit('error', {'message': f"Calibration error: {str(e)}"})

    # Target frame rate control
    target_fps = 30
    frame_time = 1.0 / target_fps
    last_frame_time = time.time()

    # Main streaming loop
    try:
        while is_streaming and stream_thread_active:
            # Maintain frame rate
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

            try:
                # Record this frame's time for FPS calculation
                current_time = time.time()
                frame_times.append(current_time)

                # Update FPS calculation every second
                if current_time - last_fps_update >= 1.0:
                    current_fps = calculate_fps()
                    last_fps_update = current_time

                # Capture frames
                left_frame, right_frame = stereo_vision.capture_frames()

                if left_frame is None or right_frame is None:
                    logger.warning("Failed to capture frames")
                    emit('error', {'message': "Frame capture failed. Check camera connections."})
                    time.sleep(0.5)
                    continue

                # Process frames if calibration is available
                frames_to_send = {}
                if has_calibration:
                    try:
                        # Rectify images
                        left_rect, right_rect = stereo_vision.get_rectified_images(left_frame, right_frame)

                        # Compute disparity map
                        _, disp_color = stereo_vision.compute_disparity_map(left_rect, right_rect)

                        frames_to_send = {
                            'left': left_rect,
                            'right': right_rect,
                            'disparity': disp_color
                        }
                    except Exception as e:
                        logger.error(f"Error processing stereo images: {str(e)}")
                        # Fall back to raw frames if rectification fails
                        frames_to_send = {
                            'left': left_frame,
                            'right': right_frame
                        }
                else:
                    # Just use raw frames
                    frames_to_send = {
                        'left': left_frame,
                        'right': right_frame
                    }

                # Convert frames to JPEG for streaming
                encoded_frames = {}
                for key, frame in frames_to_send.items():
                    try:
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        encoded_frames[key] = base64.b64encode(buffer).decode('utf-8')
                    except Exception as e:
                        logger.error(f"Error encoding {key} frame: {str(e)}")

                # Send frames via websocket if we have any
                if encoded_frames:
                    # Add FPS information
                    data_to_send = encoded_frames.copy()
                    data_to_send['fps'] = round(current_fps, 1)
                    emit('frames', data_to_send)

                # Update last frame time
                last_frame_time = current_time

            except Exception as e:
                logger.error(f"Error in streaming thread: {str(e)}")
                emit('error', {'message': f"Streaming error: {str(e)}"})
                time.sleep(0.5)

    finally:
        # Clean up when streaming stops
        try:
            stereo_vision.close_cameras()
        except Exception as e:
            logger.error(f"Error closing cameras: {str(e)}")

        # Reset flags
        with thread_lock:
            is_streaming = False
            stream_thread_active = False
        logger.info("Streaming stopped")