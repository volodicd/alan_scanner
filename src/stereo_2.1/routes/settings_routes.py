# routes/settings_routes.py
from flask import request, jsonify
import os
import platform
import cv2

from app_context import app_ctx
import logging

logger = logging.getLogger(__name__)


def register_settings_routes(app):
    """Register settings-related routes."""

    @app.route('/api/config', methods=['GET', 'POST'])
    def handle_config():
        """Get or update configuration settings."""
        if request.method == 'GET':
            # Return current configuration
            with app_ctx.lock:
                return jsonify({
                    'success': True,
                    'config': app_ctx.config
                })

        elif request.method == 'POST':
            try:
                new_config = request.json
                if not new_config:
                    return jsonify({'success': False, 'message': 'No configuration data provided'}), 400

                with app_ctx.lock:
                    # Validate camera indices
                    if 'left_cam_idx' in new_config and 'right_cam_idx' in new_config:
                        if new_config['left_cam_idx'] == new_config['right_cam_idx']:
                            return jsonify({
                                'success': False,
                                'message': 'Left and right camera indices must be different'
                            }), 400

                    # Update SGBM parameters if provided
                    if 'sgbm_params' in new_config and isinstance(new_config['sgbm_params'], dict):
                        for key, value in new_config['sgbm_params'].items():
                            if key in app_ctx.config['sgbm_params']:
                                app_ctx.config['sgbm_params'][key] = value

                        # If stereo_vision is initialized, update its parameters too
                        if app_ctx.stereo_vision is not None:
                            app_ctx.stereo_vision.set_sgbm_params(app_ctx.config['sgbm_params'])

                        # Remove from new_config since we've handled it
                        del new_config['sgbm_params']

                    # Update other config items
                    for key, value in new_config.items():
                        if key in app_ctx.config:
                            app_ctx.config[key] = value

                # Log update
                logger.info("Configuration updated")

                return jsonify({
                    'success': True,
                    'message': 'Configuration updated',
                    'config': app_ctx.config
                })

            except Exception as e:
                logger.error(f"Error updating config: {str(e)}")
                return jsonify({'success': False, 'message': str(e)}), 500

    @app.route('/api/system/info', methods=['GET'])
    def get_system_info():
        """Get system information including camera details."""

        # Get system info
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'opencv_version': cv2.__version__
        }

        # Get available cameras
        available_cameras = []
        # Check first 5 indices
        for i in range(5):
            cap = None
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # Get camera details
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        available_cameras.append({
                            'index': i,
                            'resolution': f"{width}x{height}",
                            'fps': fps
                        })
            except Exception as e:
                logger.error(f"Error checking camera {i}: {str(e)}")
            finally:
                if cap is not None:
                    cap.release()

        # Check if calibration exists
        has_calibration = os.path.exists('stereo_calibration.npy')

        # Get current config
        with app_ctx.lock:
            current_config = app_ctx.config.copy()

        return jsonify({
            'success': True,
            'system_info': system_info,
            'available_cameras': available_cameras,
            'has_calibration': has_calibration,
            'current_config': current_config
        })