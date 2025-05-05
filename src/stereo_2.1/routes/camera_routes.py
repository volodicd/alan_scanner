# routes/camera_routes.py
from flask import request, jsonify
import cv2
import base64
import threading
import time
import os
from datetime import datetime

from config import (
    stereo_vision, is_streaming, current_config, logger
)
from utils import emit

# Global thread for streaming
stream_thread = None
stream_thread_active = False  # Track if the thread should continue running


def register_camera_routes(app):
    """Register camera-related routes."""

    @app.route('/api/stream/start', methods=['POST'])
    def start_stream():
        global is_streaming, stream_thread, stream_thread_active, stereo_vision

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

        if not is_streaming:
            return jsonify({'success': False, 'message': 'Stream is not running'})

        # Set flags to stop streaming
        is_streaming = False
        stream_thread_active = False

        # Wait for thread to finish
        if stream_thread is not None and stream_thread.is_alive():
            stream_thread.join(timeout=2.0)

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


def process_stream():
    """Process and stream frames from the cameras."""
    global is_streaming, stereo_vision, stream_thread_active

    if stereo_vision is None:
        logger.error("Stereo vision not initialized")
        emit('error', {'message': 'Stereo vision not initialized'})
        is_streaming = False
        stream_thread_active = False
        return

    # Open cameras
    try:
        stereo_vision.open_cameras()
    except Exception as e:
        logger.error(f"Failed to open cameras: {str(e)}")
        emit('error', {'message': f"Failed to open cameras: {str(e)}"})
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

    # Main streaming loop
    try:
        while is_streaming and stream_thread_active:
            try:
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
                    emit('frames', encoded_frames)

                # Small delay to reduce CPU usage
                time.sleep(0.05)

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
        is_streaming = False
        stream_thread_active = False
        logger.info("Streaming stopped")