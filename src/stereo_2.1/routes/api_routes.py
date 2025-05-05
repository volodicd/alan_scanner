# routes/api_routes.py
from flask import request, jsonify
import threading
from config import (
    current_config, is_streaming, current_mode, stream_thread,
    stereo_vision, logger
)
from services.stream_service import init_stereo_vision, process_stream, capture_frame


def register_api_routes(app):
    """Register general API routes."""

    @app.route('/api/status', methods=['GET'])
    def get_status():
        return jsonify({
            'is_streaming': is_streaming,
            'current_mode': current_mode,
            'config': current_config
        })

    @app.route('/api/config', methods=['GET', 'POST'])
    def handle_config():
        if request.method == 'GET':
            return jsonify(current_config)

        elif request.method == 'POST':
            try:
                new_config = request.json
                # Validate config (add more validation as needed)
                if 'left_cam_idx' in new_config and 'right_cam_idx' in new_config:
                    if new_config['left_cam_idx'] == new_config['right_cam_idx']:
                        return jsonify(
                            {'success': False, 'message': 'Left and right camera indices must be different'}), 400

                # Update config
                for key, value in new_config.items():
                    if key in current_config:
                        if isinstance(current_config[key], dict) and isinstance(value, dict):
                            # For nested dictionaries like disparity_params
                            for sub_key, sub_value in value.items():
                                if sub_key in current_config[key]:
                                    current_config[key][sub_key] = sub_value
                        else:
                            current_config[key] = value

                return jsonify({'success': True, 'config': current_config})

            except Exception as e:
                logger.error(f"Error updating config: {str(e)}")
                return jsonify({'success': False, 'message': str(e)}), 400

    @app.route('/api/stream/start', methods=['POST'])
    def start_stream():
        global is_streaming, stream_thread, current_mode

        if is_streaming:
            return jsonify({'success': False, 'message': 'Stream is already running'})

        # Get mode from request
        data = request.json
        requested_mode = data.get('mode', 'test')

        if requested_mode not in ['test', 'calibration', 'process', 'mapping']:
            return jsonify({'success': False, 'message': 'Invalid mode'}), 400

        # Initialize stereo vision if needed
        if stereo_vision is None:
            if not init_stereo_vision():
                return jsonify({'success': False, 'message': 'Failed to initialize stereo vision'}), 500

        # Set current mode
        current_mode = requested_mode

        # Start streaming thread
        is_streaming = True
        stream_thread = threading.Thread(target=process_stream)
        stream_thread.daemon = True
        stream_thread.start()

        return jsonify({'success': True, 'mode': current_mode})

    @app.route('/api/stream/stop', methods=['POST'])
    def stop_stream():
        global is_streaming

        if not is_streaming:
            return jsonify({'success': False, 'message': 'Stream is not running'})

        is_streaming = False
        # Wait for thread to finish
        if stream_thread is not None and stream_thread.is_alive():
            stream_thread.join(timeout=2.0)

        return jsonify({'success': True})

    @app.route('/api/capture', methods=['POST'])
    def handle_capture_frame():
        if not is_streaming:
            return jsonify({'success': False, 'message': 'Stream is not running'}), 400

        success, message, left_filepath, right_filepath, disparity_filepath, pointcloud_filepath = capture_frame()

        if not success:
            return jsonify({'success': False, 'message': message}), 500

        return jsonify({
            'success': True,
            'timestamp': message,  # In this case, message is the timestamp
            'left_path': left_filepath,
            'right_path': right_filepath,
            'disparity_path': disparity_filepath,
            'pointcloud_path': pointcloud_filepath
        })