# routes/calibration_routes.py
from flask import request, jsonify
import threading
import os
import cv2
import base64
import numpy as np
from stereo_vision import StereoVision
from config import current_config, logger
from utils import emit

# Thread safety
# So, basically it is a lock for a global vars, so they would be locked during the calibration
thread_lock = threading.Lock()
calibration_in_progress = False


def register_calibration_routes(app):
    """Register calibration-related routes."""

    @app.route('/api/calibrate/start', methods=['POST'])
    def start_calibration():
        global stereo_vision, is_streaming, calibration_state, calibration_in_progress

        with thread_lock:
            # Checking for double calibration
            if calibration_in_progress:
                return jsonify({'success': False, 'message': 'Calibration already in progress'}), 409
            calibration_in_progress = True

        # Process request parameters
        try:
            data = request.json or {}

            # Set auto-capture mode if provided
            if 'auto_capture' in data:
                current_config["auto_capture"] = bool(data['auto_capture'])
                logger.info("Auto-capture mode set to: %s", "enabled" if current_config["auto_capture"] else "disabled")

            # Set stability threshold if provided
            # Wont recomend to set less than 1 second, even manually. Will lead to very poor calibration. 0.5 s only for testing.
            if 'stability_seconds' in data:
                stability = float(data['stability_seconds'])
                if stability < 0.5:
                    stability = 0.5
                    logger.warning("Stability threshold too low, setting to minimum 0.5 seconds")
                elif stability > 10.0:
                    stability = 10.0
                    logger.warning("Stability threshold too high, setting to maximum 10.0 seconds")
                current_config["stability_seconds"] = stability
                logger.info("Stability threshold set to %.1f seconds", stability)

            # Is is also possible to set other methods of callibration here
            if 'checkerboard_size' in data and isinstance(data['checkerboard_size'], list) and len(
                    data['checkerboard_size']) == 2:
                current_config["calibration_checkerboard_size"] = tuple(data['checkerboard_size'])
                logger.info("Checkerboard size set to %dx%d",
                            current_config["calibration_checkerboard_size"][0],
                            current_config["calibration_checkerboard_size"][1])

            # Not really important(from practical exp)
            if 'square_size' in data:
                try:
                    square_size = float(data['square_size'])
                    if 0.001 <= square_size <= 0.5:  # Reasonable range in meters
                        current_config["calibration_square_size"] = square_size
                        logger.info("Square size set to %.3f meters", square_size)
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            logger.warning("Error processing calibration parameters: %s", str(e))

        # Lock for calibration state
        with thread_lock:
            global is_streaming
            was_streaming = is_streaming
            if is_streaming:
                is_streaming = False

        # Init camera if stream is off
        if stereo_vision is None:
            stereo_vision = StereoVision(
                left_cam_idx=current_config["left_cam_idx"],
                right_cam_idx=current_config["right_cam_idx"],
                width=current_config["width"],
                height=current_config["height"]
            )
        reset_calibration_state()

        # Start calibration process in a separate thread
        try:
            logger.info("Starting calibration mode with auto-capture=%s, stability=%.1fs",
                        current_config["auto_capture"], current_config["stability_seconds"])

            # Start calibration in a separate thread
            calib_thread = threading.Thread(
                target=run_calibration,
                args=(current_config["calibration_checkerboard_size"],
                      current_config["calibration_square_size"],
                      current_config["auto_capture"],
                      current_config["stability_seconds"])
            )
            calib_thread.daemon = True
            calib_thread.start()

            # Send detailed response
            return jsonify({
                'success': True,
                'auto_capture': current_config["auto_capture"],
                'stability_seconds': current_config["stability_seconds"],
                'existing_pairs': calibration_state["captured_pairs"],
                'min_pairs_needed': calibration_state["min_pairs_needed"],
                'recommended_pairs': calibration_state["recommended_pairs"]
            })

        except Exception as e:
            logger.error("Failed to start calibration: %s", str(e))
            with thread_lock:
                calibration_in_progress = False
            return jsonify({'success': False, 'message': str(e)}), 500

    @app.route('/api/calibrate/settings', methods=['GET', 'POST'])
    def calibration_settings():
        """Get or update calibration settings."""

        if request.method == 'GET':
            # Return current calibration settings
            return jsonify({
                'success': True,
                'auto_capture': current_config.get('auto_capture', True),
                'stability_seconds': current_config.get('stability_seconds', 3.0),
                'checkerboard_size': current_config.get('calibration_checkerboard_size', (7, 6)),
                'square_size': current_config.get('calibration_square_size', 0.025)
            })
        elif request.method == 'POST':
            try:
                data = request.json or {}

                # Update auto-capture setting
                if 'auto_capture' in data:
                    current_config['auto_capture'] = bool(data['auto_capture'])
                    logger.info("Auto-capture mode %s", "enabled" if current_config['auto_capture'] else "disabled")

                # Update stability threshold
                if 'stability_seconds' in data:
                    stability = float(data['stability_seconds'])
                    # Validate stability threshold
                    if stability < 0.5:
                        stability = 0.5
                        logger.warning("Stability threshold too low, setting to minimum 0.5 seconds")
                    elif stability > 10.0:
                        stability = 10.0
                        logger.warning("Stability threshold too high, setting to maximum 10.0 seconds")
                    current_config['stability_seconds'] = stability
                    logger.info("Stability threshold set to %.1f seconds", stability)

                # Update checkerboard size if provided
                if 'checkerboard_size' in data:
                    size = data['checkerboard_size']
                    if isinstance(size, list) and len(size) == 2:
                        current_config['calibration_checkerboard_size'] = tuple(size)
                        logger.info("Checkerboard size set to %dx%d", size[0], size[1])

                # Update square size if provided
                if 'square_size' in data:
                    square_size = float(data['square_size'])
                    if 0.001 <= square_size <= 0.5:  # Reasonable range in meters
                        current_config['calibration_square_size'] = square_size
                        logger.info("Checkerboard square size set to %.3f meters", square_size)

                return jsonify({
                    'success': True,
                    'message': 'Calibration settings updated',
                    'current_settings': {
                        'auto_capture': current_config.get('auto_capture', True),
                        'stability_seconds': current_config.get('stability_seconds', 3.0),
                        'checkerboard_size': current_config.get('calibration_checkerboard_size', (7, 6)),
                        'square_size': current_config.get('calibration_square_size', 0.025)
                    }
                })

            except Exception as e:
                logger.error("Failed to update calibration settings: %s", str(e))
                return jsonify({'success': False, 'message': str(e)}), 400

    @app.route('/api/calibrate/status', methods=['GET'])
    def get_calibration_status():
        """Get current calibration status."""

        # Check if calibration file exists
        has_calibration = os.path.exists('stereo_calibration.npy')

        # Get information about calibration file if it exists
        calibration_info = {}
        if has_calibration:
            try:
                # Get calibration information from file
                calibration_data = np.load('stereo_calibration.npy', allow_pickle=True).item()
                calibration_info = {
                    'date': calibration_data.get('calibration_date', 'Unknown'),
                    'frame_count': calibration_data.get('frame_count', 0),
                    'rms_error': calibration_data.get('rms_error', 0.0),
                    'image_size': calibration_data.get('image_size', [0, 0])
                }
            except Exception as e:
                logger.error(f"Error loading calibration info: {str(e)}")
                calibration_info = {'error': str(e)}

        return jsonify({
            'success': True,
            'has_calibration': has_calibration,
            'calibration_info': calibration_info,
            'current_state': calibration_state
        })

    @app.route('/api/calibrate/detect', methods=['POST'])
    def detect_checkerboard():
        """Test checkerboard detection with current camera images."""
        if stereo_vision is None:
            return jsonify({'success': False, 'message': 'Stereo vision not initialized'}), 400

        try:
            # Capture frames
            left_frame, right_frame = stereo_vision.capture_frames()
            if left_frame is None or right_frame is None:
                return jsonify({'success': False, 'message': 'Failed to capture frames'}), 500

            # Convert to grayscale
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

            # Apply preprocessing
            left_gray = cv2.GaussianBlur(left_gray, (5, 5), 0)
            right_gray = cv2.GaussianBlur(right_gray, (5, 5), 0)

            # Check for checkerboard
            cb_size = current_config["calibration_checkerboard_size"]
            pattern_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

            found_left, left_corners = cv2.findChessboardCorners(
                left_gray, cb_size, pattern_flags)
            found_right, right_corners = cv2.findChessboardCorners(
                right_gray, cb_size, pattern_flags)

            # If found, draw on copies
            left_display = left_frame.copy()
            right_display = right_frame.copy()

            if found_left:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                left_corners = cv2.cornerSubPix(
                    left_gray, left_corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(
                    left_display, cb_size, left_corners, found_left)

            if found_right:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                right_corners = cv2.cornerSubPix(
                    right_gray, right_corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(
                    right_display, cb_size, right_corners, found_right)

            # Convert to base64 for response
            _, left_buffer = cv2.imencode('.jpg', left_display, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _, right_buffer = cv2.imencode('.jpg', right_display, [cv2.IMWRITE_JPEG_QUALITY, 80])

            left_b64 = base64.b64encode(left_buffer).decode('utf-8')
            right_b64 = base64.b64encode(right_buffer).decode('utf-8')

            return jsonify({
                'success': True,
                'left_found': found_left,
                'right_found': found_right,
                'both_found': found_left and found_right,
                'left_image': left_b64,
                'right_image': right_b64,
                'checkerboard_size': cb_size
            })

        except Exception as e:
            logger.error(f"Error in checkerboard detection: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500


def reset_calibration_state():
    """Reset the calibration state to default values."""
    with thread_lock:
        calibration_state.update({
            "is_stable": False,
            "stable_since": 0,
            "last_capture_time": 0,
            "captured_pairs": 0,
            "min_pairs_needed": 10,
            "recommended_pairs": 20
        })


def run_calibration(checkerboard_size, square_size, auto_capture, stability_seconds):
    """Run the calibration process in a separate thread."""
    global calibration_in_progress

    if stereo_vision is None:
        emit('error', {'message': 'Stereo vision not initialized'})
        with thread_lock:
            calibration_in_progress = False
        return

    logger.info("Starting calibration process")
    emit('status', {'message': 'Starting calibration process'})

    try:
        success = stereo_vision.calibrate_cameras(
            checkerboard_size=checkerboard_size,
            square_size=square_size,
            auto_capture=auto_capture,
            stability_seconds=stability_seconds
        )

        if success:
            logger.info("Calibration completed successfully")
            emit('status', {'message': 'Calibration completed successfully'})

            # Get calibration information
            calibration_data = np.load('stereo_calibration.npy', allow_pickle=True).item()

            emit('calibration_complete', {
                'success': True,
                'date': calibration_data.get('calibration_date', 'Unknown'),
                'frame_count': calibration_data.get('frame_count', 0),
                'rms_error': calibration_data.get('rms_error', 0.0)
            })
        else:
            logger.error("Calibration failed")
            emit('error', {'message': 'Calibration failed'})
            emit('calibration_complete', {'success': False})

    except Exception as e:
        logger.error(f"Error during calibration: {str(e)}")
        emit('error', {'message': f'Error during calibration: {str(e)}'})
        emit('calibration_complete', {'success': False, 'error': str(e)})
    finally:
        with thread_lock:
            calibration_in_progress = False