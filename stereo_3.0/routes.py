import logging
import platform

import cv2
from flask import Blueprint, request, jsonify

from controller import VisionController

api = Blueprint('api', __name__, url_prefix='/api')
vision_controller = VisionController()

logger = logging.getLogger(__name__)


@api.route('/vision/initialize', methods=['POST'])
def initialize_vision():
    """Initialize vision system with specified parameters"""
    try:
        data = request.json or {}
        left_cam_idx = int(data.get('left_cam_idx', 0))
        right_cam_idx = int(data.get('right_cam_idx', 1))
        width = int(data.get('width', 640))
        height = int(data.get('height', 480))

        success = vision_controller.initialize(
            left_cam_idx=left_cam_idx,
            right_cam_idx=right_cam_idx,
            width=width,
            height=height
        )

        return jsonify({
            'success': success,
            'message': 'Vision system initialized' if success else 'Failed to initialize vision system'
        })
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@api.route('/vision/start', methods=['POST'])
def start_vision():
    """Start the vision processing thread"""
    success = vision_controller.start_processing()
    return jsonify({
        'success': success,
        'message': 'Vision processing started' if success else 'Vision processing already running or not initialized'
    })


@api.route('/vision/stop', methods=['POST'])
def stop_vision():
    """Stop the vision processing thread"""
    success = vision_controller.stop_processing()
    return jsonify({
        'success': success,
        'message': 'Vision processing stopped' if success else 'Vision processing not running'
    })


@api.route('/vision/data', methods=['GET'])
def get_vision_data():
    """Get current vision analysis data"""
    return jsonify(vision_controller.get_vision_data())


@api.route('/turtlebot/vision', methods=['GET'])
def get_turtlebot_data():
    """TurtleBot-specific endpoint with required format"""
    data = vision_controller.get_vision_data()
    # Return only the exact fields needed by TurtleBot
    return jsonify({
        'is_object': data['is_object'], # send bool if object is close to the turtlebot
        'distance_object': data['distance_object'], # distance to the main object
        'objs': data['objs'], # list of 16 parts of the picture with the avarage distance to each of that 16 parts
        'final': data['final'] # probably will be deleted, was planned for turetlebot service to get know if the map is finished
    })


@api.route('/frames', methods=['GET'])
def get_frames():
    """Get current camera frames as base64 encoded JPEG images"""
    quality = int(request.args.get('quality', 80))
    full_size = request.args.get('full_size', 'true').lower() == 'true'

    frames = vision_controller.get_encoded_frames(quality=quality, full_size=full_size)
    if not frames:
        return jsonify({'success': False, 'message': 'No frames available'}), 404

    return jsonify({
        'success': True,
        **frames
    })


@api.route('/capture', methods=['POST'])
def capture_frame():
    """Capture and save current frames"""
    result = vision_controller.capture_and_save()

    if not result:
        return jsonify({'success': False, 'message': 'Failed to capture frames'}), 500

    return jsonify({
        'success': True,
        **result
    })


@api.route('/calibrate', methods=['POST'])
def calibrate():
    """Run the calibration process"""
    data = request.json or {}

    # Parse calibration parameters
    try:
        checkerboard_size = tuple(data.get('checkerboard_size', (7, 6)))
        square_size = float(data.get('square_size', 0.025))
        num_samples = int(data.get('num_samples', 20))
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'message': f'Invalid parameter: {str(e)}'}), 400

    # Run calibration
    success, result = vision_controller.run_calibration(
        checkerboard_size=checkerboard_size,
        square_size=square_size,
        num_samples=num_samples
    )

    if success:
        return jsonify({
            'success': True,
            'message': 'Calibration completed successfully',
            'calibration_info': result
        })
    else:
        return jsonify({
            'success': False,
            'message': result or 'Calibration failed'
        }), 500


@api.route('/calibrate/status', methods=['GET'])
def get_calibration_status():
    """Get current calibration status"""
    status = vision_controller.get_calibration_status()
    return jsonify({
        'success': True,
        **status
    })


@api.route('/calibrate/detect', methods=['POST'])
def detect_checkerboard():
    """Test checkerboard detection with current camera feed"""
    data = request.json or {}

    # Parse parameters
    try:
        checkerboard_size = tuple(data.get('checkerboard_size', (7, 6)))
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'message': f'Invalid parameter: {str(e)}'}), 400

    # Run detection
    success, message, result, error = vision_controller.detect_checkerboard(
        checkerboard_size=checkerboard_size
    )

    if success:
        return jsonify({
            'success': True,
            **result
        })
    else:
        return jsonify({
            'success': False,
            'message': message,
            'error': error
        }), 500


@api.route('/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration"""
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'config': vision_controller.get_config()
        })

    elif request.method == 'POST':
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'No configuration data provided'}), 400

        success, message = vision_controller.update_config(data)

        if success:
            return jsonify({
                'success': True,
                'message': message,
                'config': vision_controller.get_config()
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 400
    return None


@api.route('/system/info', methods=['GET'])
def get_system_info():
    """Get system information including available cameras"""
    # Get system info
    system_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'opencv_version': cv2.__version__
    }

    # Get available cameras
    available_cameras = []
    for i in range(5):  # Check first 5 indices
        cap = None
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
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

    # Get calibration status
    calibration_status = vision_controller.get_calibration_status()

    # Get current config
    current_config = vision_controller.get_config()

    return jsonify({
        'success': True,
        'system_info': system_info,
        'available_cameras': available_cameras,
        'has_calibration': calibration_status['has_calibration'],
        'current_config': current_config
    })