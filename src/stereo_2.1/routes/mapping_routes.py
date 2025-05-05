# routes/mapping_routes.py
from flask import request, jsonify
import os
import threading
from datetime import datetime

from config import (
    is_streaming, current_mode, stereo_vision, stream_thread, logger
)
from services.stream_service import init_stereo_vision, process_stream


def register_mapping_routes(app):
    """Register mapping-related routes."""

    @app.route('/api/mapping/start', methods=['POST'])
    def start_mapping():
        """Start the 3D mapping process."""
        global is_streaming, current_mode, stream_thread

        # Check if calibration is available
        try:
            if not os.path.exists('stereo_calibration.npy'):
                return jsonify(
                    {'success': False, 'message': 'Calibration file not found. Please calibrate cameras first.'}), 400

            # Stop any existing stream
            if is_streaming:
                is_streaming = False
                if stream_thread is not None and stream_thread.is_alive():
                    stream_thread.join(timeout=2.0)

            # Initialize stereo vision if needed
            if stereo_vision is None:
                if not init_stereo_vision():
                    return jsonify({'success': False, 'message': 'Failed to initialize stereo vision'}), 500

            # Load calibration
            if not stereo_vision.load_calibration():
                return jsonify({'success': False, 'message': 'Failed to load calibration'}), 500

            # Start mapping stream
            current_mode = "mapping"
            is_streaming = True
            stream_thread = threading.Thread(target=process_stream)
            stream_thread.daemon = True
            stream_thread.start()

            return jsonify({'success': True, 'message': 'Mapping started'})

        except Exception as e:
            logger.error(f"Failed to start mapping: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500

    @app.route('/api/pointcloud/list', methods=['GET'])
    def list_pointclouds():
        """List all saved point clouds."""

        try:
            # Get all point cloud files
            pointcloud_files = []
            for filename in os.listdir('static/maps'):
                if filename.startswith('pointcloud_') and filename.endswith('.npy'):
                    timestamp = filename.replace('pointcloud_', '').replace('.npy', '')
                    pointcloud_files.append({
                        'filename': filename,
                        'path': f'static/maps/{filename}',
                        'timestamp': timestamp,
                        'datetime': datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                    })

            return jsonify({
                'success': True,
                'pointclouds': sorted(pointcloud_files, key=lambda x: x['timestamp'], reverse=True)
            })

        except Exception as e:
            logger.error(f"Failed to list point clouds: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500