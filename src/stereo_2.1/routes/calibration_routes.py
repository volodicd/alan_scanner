# routes/calibration_routes.py
from flask import request, jsonify
import threading
import logging

from config import (
    current_config, is_streaming, calibration_state,
    stream_thread, current_mode, logger
)
from services.calibration_service import (
    backup_current_code, reset_calibration_state,
    capture_calibration_frame, process_calibration
)
from services.stream_service import init_stereo_vision, process_stream


def register_calibration_routes(app, socketio):
    """Register calibration-related routes."""

    @app.route('/api/calibrate/start', methods=['POST'])
    def start_calibration():
        global is_streaming, current_mode, calibration_state, current_config

        # Process request parameters
        try:
            data = request.json or {}

            # Set auto-capture mode if provided
            if 'auto_capture' in data:
                current_config["auto_capture"] = bool(data['auto_capture'])
                logger.info("Auto-capture mode set to: %s", "enabled" if current_config["auto_capture"] else "disabled")

            # Set stability threshold if provided
            if 'stability_seconds' in data:
                stability = float(data['stability_seconds'])
                # Validate stability threshold
                if stability < 0.5:
                    stability = 0.5
                    logger.warning("Stability threshold too low, setting to minimum 0.5 seconds")
                elif stability > 10.0:
                    stability = 10.0
                    logger.warning("Stability threshold too high, setting to maximum 10.0 seconds")
                current_config["stability_seconds"] = stability
                logger.info("Stability threshold set to %.1f seconds", stability)
        except Exception as e:
            logger.warning("Error processing calibration parameters: %s", str(e))

        # Stop any existing stream
        if is_streaming:
            is_streaming = False
            if stream_thread is not None and stream_thread.is_alive():
                stream_thread.join(timeout=2.0)

        # Initialize stereo vision if needed
        if stereo_vision is None:
            if not init_stereo_vision():
                logger.error("Failed to initialize stereo vision")
                return jsonify({'success': False, 'message': 'Failed to initialize stereo vision'}), 500

        # Reset calibration state
        reset_calibration_state()

        # Start calibration process
        try:
            # Backup current calibration if it exists
            backup_current_code()
            logger.info("Starting calibration mode with auto-capture=%s, stability=%.1fs",
                        current_config["auto_capture"], current_config["stability_seconds"])

            # Start calibration mode stream for visual feedback
            current_mode = "calibration"
            is_streaming = True
            new_stream_thread = threading.Thread(target=process_stream)
            new_stream_thread.daemon = True
            new_stream_thread.start()

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
                'checkerboard_size': current_config.get('calibration_checkerboard_size', (8, 6)),
                'square_size': current_config.get('calibration_square_size', 0.015)
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
                        'checkerboard_size': current_config.get('calibration_checkerboard_size', (8, 6)),
                        'square_size': current_config.get('calibration_square_size', 0.015)
                    }
                })

            except Exception as e:
                logger.error("Failed to update calibration settings: %s", str(e))
                return jsonify({'success': False, 'message': str(e)}), 400

    @app.route('/api/calibrate/capture', methods=['POST'])
    def handle_capture_calibration_frame():
        if not is_streaming or current_mode != "calibration":
            return jsonify({'success': False, 'message': 'Not in calibration mode'}), 400

        success, message, left_filepath, right_filepath = capture_calibration_frame()

        if not success:
            return jsonify({
                'success': False,
                'message': message,
                'left_found': left_filepath,  # In this case, these represent found status
                'right_found': right_filepath
            }), 400 if message == "Checkerboard not detected in both frames" else 500

        return jsonify({
            'success': True,
            'timestamp': message,
            'left_path': left_filepath,
            'right_path': right_filepath
        })

    @app.route('/api/calibrate/process', methods=['POST'])
    def handle_process_calibration():
        global stereo_vision, current_config, is_streaming

        # Stop streaming during calibration processing
        was_streaming = is_streaming
        if is_streaming:
            is_streaming = False
            if stream_thread is not None and stream_thread.is_alive():
                stream_thread.join(timeout=2.0)

        # Process calibration
        result = process_calibration()

        # Restart streaming if it was active before
        if was_streaming:
            is_streaming = True
            new_stream_thread = threading.Thread(target=process_stream)
            new_stream_thread.daemon = True
            new_stream_thread.start()

        if not result['success']:
            return jsonify(result), 400 if 'enough' in result['message'] else 500

        return jsonify(result)