# app.py
import os
import socket

import cv2
import numpy as np
import time
import json
import threading
import base64
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import logging
import logging.handlers
from werkzeug.serving import run_simple
import torch
import os.path

# Import the StereoVision class from your existing code
from stereo_vision import StereoVision


# Configure logging with more detailed format and DEBUG level
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format='%(ascti±§me)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        # Rotating file handler to prevent log files from growing too large
        logging.handlers.RotatingFileHandler(
            "stereo_vision_app.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set specific module log levels
logging.getLogger('werkzeug').setLevel(logging.WARNING)  # Reduce Flask debug logs
logging.getLogger('engineio').setLevel(logging.WARNING)  # Reduce SocketIO logs
logging.getLogger('socketio').setLevel(logging.WARNING)  # Reduce SocketIO logs

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'stereo_vision_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading') # temporary eventlet, should be threading

# Create data directories if they don't exist
os.makedirs('static/captures', exist_ok=True)
os.makedirs('static/maps', exist_ok=True)
os.makedirs('static/logs', exist_ok=True)
os.makedirs('static/calibration', exist_ok=True)
os.makedirs('static/code_backups', exist_ok=True)

# Global variables
stereo_vision = None
stream_thread = None
is_streaming = False
dl_model = None
dl_model_loaded = False
dl_enabled = False
current_mode = "idle"  # idle, test, calibration, mapping
current_config = {
    "left_cam_idx": 0,
    "right_cam_idx": 1,
    "width": 640,
    "height": 480,
    "calibration_checkerboard_size": (8, 6),
    "calibration_square_size": 0.015,
    "auto_capture": True,           # Enable auto-capture by default
    "stability_seconds": 3.0,       # Stability threshold for auto-capture
    "disparity_params": {
        "window_size": 11,
        "min_disp": 0,
        "num_disp": 112,
        "uniqueness_ratio": 15,
        "speckle_window_size": 100,
        "speckle_range": 32
    },
    "disparity_method": "dl",  # 'sgbm' or 'dl'
    "dl_model_name": "raft_stereo",
    "dl_params": {
        "max_disp": 256,
        "mixed_precision": True,
        "downscale_factor": 1.0  # For performance tuning
    }
}

# Calibration state tracking
calibration_state = {
    "detected_count": 0,
    "is_stable": False,
    "stable_since": 0,
    "last_capture_time": 0,
    "captured_pairs": 0,
    "min_pairs_needed": 10,
    "recommended_pairs": 20,
    "checkerboard_history": []  # To track detection stability
}

# Keep a history of code changes for rollback
code_versions = []


# Function to backup current code
def backup_current_code():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        with open('stereo_vision.py', 'r') as f:
            code = f.read()

        backup_path = f'static/code_backups/stereo_vision_{timestamp}.py'
        with open(backup_path, 'w') as f:
            f.write(code)

        code_versions.append({
            'timestamp': timestamp,
            'filename': backup_path,
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Keep only the last 20 versions
        if len(code_versions) > 20:
            old_version = code_versions.pop(0)
            if os.path.exists(old_version['filename']):
                os.remove(old_version['filename'])

        return True
    except Exception as e:
        logger.error(f"Failed to backup code: {str(e)}")
        return False


# Initialize stereo vision system
def init_stereo_vision():
    global stereo_vision
    try:
        stereo_vision = StereoVision(
            left_cam_idx=current_config["left_cam_idx"],
            right_cam_idx=current_config["right_cam_idx"],
            width=current_config["width"],
            height=current_config["height"]
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize stereo vision: {str(e)}")
        return False


def init_dl_model():
    global dl_model, dl_model_loaded

    try:
        from models import load_model, get_device_info

        # Log device info
        device_info = get_device_info()
        logger.info("Initializing deep learning model with device information:")
        if device_info['cuda_available']:
            logger.info(f"CUDA available with {device_info['cuda_device_count']} devices")
        elif device_info['mps_available']:
            logger.info("Apple MPS (Metal Performance Shaders) available on M1 Mac")
        else:
            logger.info("Running on CPU only - this may be slow for deep learning inference")

        # Set model_name to raft_stereo if using that option
        model_name = 'raft_stereo' if current_config["disparity_method"] == "dl" else current_config['dl_model_name']
        current_config['dl_model_name'] = model_name
        
        # Check if model weights exist
        weights_dir = 'models/weights'
        weights_path = os.path.join(weights_dir, f"{model_name}.pth")

        if not os.path.exists(weights_path):
            logger.warning(f"Model weights not found at {weights_path}")
            
            # Check if alternative names might exist
            alt_weights = None
            if model_name == 'raft_stereo':
                for alt_name in ['raftstereo-middlebury.pth', 'raftstereo-sceneflow.pth']:
                    alt_path = os.path.join(weights_dir, alt_name)
                    if os.path.exists(alt_path):
                        logger.info(f"Found alternative weights at {alt_path}")
                        weights_path = alt_path
                        alt_weights = True
                        break
            
            # If no alternatives found, download weights
            if not alt_weights:
                from models.utils import download_model_weights
                logger.info("Attempting to download model weights...")
                weights_path = download_model_weights(
                    model_name=model_name,
                    save_dir=weights_dir
                )

        # Load model
        logger.info(f"Loading {model_name} model with weights from {weights_path}...")
        dl_model = load_model(
            model_name=model_name,
            weights_path=weights_path,
            max_disp=current_config['dl_params']['max_disp']
        )

        if dl_model is None:
            logger.error(f"Failed to load {model_name} model")
            return False

        dl_model_loaded = True
        logger.info(f"Deep learning model {model_name} loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize deep learning model: {str(e)}")
        dl_model_loaded = False
        return False


# Stream processing function for different modes
def process_stream():
    global is_streaming, current_mode, stereo_vision, dl_model, dl_model_loaded, dl_enabled

    logger.info(f"Starting stream in {current_mode} mode")

    # Try to open cameras
    try:
        stereo_vision.open_cameras()
    except Exception as e:
        logger.error(f"Failed to open cameras: {str(e)}")
        socketio.emit('error', {'message': f"Failed to open cameras: {str(e)}"})
        is_streaming = False
        return

    # Initialize rectification maps if we have calibration
    mapL1, mapL2, mapR1, mapR2 = None, None, None, None

    if current_mode in ["mapping", "process"]:
        try:
            # Load calibration data
            if not stereo_vision.load_calibration():
                logger.error("Calibration loading failed - load_calibration returned False")
                socketio.emit('error', {'message': "Calibration not loaded. Please calibrate cameras first."})
                # Continue anyway for basic video stream
                mapL1, mapL2, mapR1, mapR2 = None, None, None, None
            else:
                # Verify calibration data is valid
                if (not isinstance(stereo_vision.camera_matrix_left, np.ndarray) or
                        not isinstance(stereo_vision.dist_coeffs_left, np.ndarray) or
                        not isinstance(stereo_vision.R1, np.ndarray) or
                        not isinstance(stereo_vision.P1, np.ndarray)):
                    logger.error("Invalid calibration data - matrices are not numpy arrays")
                    logger.debug("camera_matrix_left type: %s", type(stereo_vision.camera_matrix_left))
                    logger.debug("dist_coeffs_left type: %s", type(stereo_vision.dist_coeffs_left))
                    logger.debug("R1 type: %s", type(stereo_vision.R1))
                    logger.debug("P1 type: %s", type(stereo_vision.P1))
                    socketio.emit('error', {'message': "Invalid calibration data. Please recalibrate cameras."})
                    mapL1, mapL2, mapR1, mapR2 = None, None, None, None
                else:
                    # Build rectification maps
                    logger.info("Building rectification maps with image size: %dx%d",
                                current_config["width"], current_config["height"])
                    mapL1, mapL2 = cv2.initUndistortRectifyMap(
                        stereo_vision.camera_matrix_left, stereo_vision.dist_coeffs_left,
                        stereo_vision.R1, stereo_vision.P1,
                        (current_config["width"], current_config["height"]), cv2.CV_32FC1
                    )
                    mapR1, mapR2 = cv2.initUndistortRectifyMap(
                        stereo_vision.camera_matrix_right, stereo_vision.dist_coeffs_right,
                        stereo_vision.R2, stereo_vision.P2,
                        (current_config["width"], current_config["height"]), cv2.CV_32FC1
                    )
                    logger.info("Rectification maps built successfully")
        except Exception as e:
            logger.error("Failed to initialize rectification maps: %s", str(e))
            logger.exception("Stack trace:")  # This will log the stack trace
            socketio.emit('error', {'message': f"Failed to initialize calibration: {str(e)}"})
            # Continue anyway for basic video stream
            mapL1, mapL2, mapR1, mapR2 = None, None, None, None

    frame_counter = 0
    last_fps_time = time.time()
    fps = 0

    # Main streaming loop
    while is_streaming:
        try:
            # Capture frames
            ret_left, left_frame = stereo_vision.left_cam.read()
            ret_right, right_frame = stereo_vision.right_cam.read()

            if not ret_left or not ret_right:
                logger.warning("Failed to capture frames")
                socketio.emit('error', {'message': "Frame capture failed. Check camera connections."})
                time.sleep(0.5)
                continue

            # Calculate FPS
            frame_counter += 1
            current_time = time.time()
            if (current_time - last_fps_time) > 1.0:
                fps = frame_counter / (current_time - last_fps_time)
                frame_counter = 0
                last_fps_time = current_time

            # Process frames based on mode
            if current_mode == "test":
                # Basic test mode - just display the raw frames
                frames_to_send = {
                    'left': left_frame,
                    'right': right_frame
                }

            elif current_mode == "calibration":
                # Calibration mode - look for checkerboard
                # [Your existing calibration code - no changes needed]
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                # Apply preprocessing to improve detection
                left_gray = cv2.GaussianBlur(left_gray, (5, 5), 0)
                right_gray = cv2.GaussianBlur(right_gray, (5, 5), 0)

                # Copy frames for display
                left_display = left_frame.copy()
                right_display = right_frame.copy()

                # Try to find checkerboard
                pattern_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                cb_size = current_config["calibration_checkerboard_size"]

                logger.debug("Attempting to find checkerboard in both frames...")
                start_time = time.time()
                found_left, left_corners = cv2.findChessboardCorners(
                    left_gray, cb_size, pattern_flags)
                found_right, right_corners = cv2.findChessboardCorners(
                    right_gray, cb_size, pattern_flags)
                detection_time = time.time() - start_time

                # Try with enhanced contrast if not found in both cameras
                if not (found_left and found_right):
                    logger.debug("Standard detection not successful, trying with enhanced contrast...")
                    # Enhanced contrast with CLAHE
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    left_gray_enhanced = clahe.apply(left_gray)
                    right_gray_enhanced = clahe.apply(right_gray)

                    # Add fast check flag
                    enhanced_flags = pattern_flags | cv2.CALIB_CB_FAST_CHECK

                    if not found_left:
                        found_left, left_corners = cv2.findChessboardCorners(
                            left_gray_enhanced, cb_size, enhanced_flags)

                    if not found_right:
                        found_right, right_corners = cv2.findChessboardCorners(
                            right_gray_enhanced, cb_size, enhanced_flags)

                # Draw corners if found
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                if found_left:
                    left_corners = cv2.cornerSubPix(
                        left_gray, left_corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(
                        left_display, cb_size, left_corners, found_left)

                if found_right:
                    right_corners = cv2.cornerSubPix(
                        right_gray, right_corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(
                        right_display, cb_size, right_corners, found_right)

                # Create status display overlay
                status_overlay = np.zeros((150, 400, 3), dtype=np.uint8)
                status_text = "Detecting Checkerboard"

                # Track detection for auto-capture
                current_time = time.time()
                both_found = found_left and found_right

                # Update the calibration state
                if both_found:
                    calibration_state["detected_count"] += 1

                    # Start or continue stability tracking
                    if not calibration_state["is_stable"]:
                        calibration_state["stable_since"] = current_time
                        calibration_state["is_stable"] = True
                        logger.debug("Checkerboard detected and stable tracking started")

                    # Calculate stability duration
                    stability_duration = current_time - calibration_state["stable_since"]
                    status_text = f"DETECTED - Stable for {stability_duration:.1f}s"

                    # Add visual feedback for stability
                    stability_pct = min(100, (stability_duration / current_config["stability_seconds"]) * 100)
                    progress_width = int(380 * (stability_pct / 100))
                    cv2.rectangle(status_overlay, (10, 70), (390, 90), (0, 0, 60), -1)
                    cv2.rectangle(status_overlay, (10, 70), (10 + progress_width, 90), (0, 165, 255), -1)

                    # Auto-capture if stable for required duration and auto-capture is enabled
                    min_capture_interval = 1.0  # Minimum seconds between captures
                    time_since_last_capture = current_time - calibration_state["last_capture_time"]

                    if (current_config["auto_capture"] and
                            stability_duration >= current_config["stability_seconds"] and
                            time_since_last_capture > min_capture_interval):
                        # Auto-capture this frame pair
                        logger.info("Auto-capturing calibration frame after %.1f seconds of stability",
                                    stability_duration)

                        # Save the calibration frame pair
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        left_filepath = f"static/calibration/left_{timestamp}.jpg"
                        right_filepath = f"static/calibration/right_{timestamp}.jpg"

                        cv2.imwrite(left_filepath, left_frame)
                        cv2.imwrite(right_filepath, right_frame)

                        # Update capture state
                        calibration_state["captured_pairs"] += 1
                        calibration_state["last_capture_time"] = current_time
                        calibration_state["is_stable"] = False  # Reset stability to prevent rapid captures

                        # Add visual feedback for auto-capture
                        cv2.putText(status_overlay, "AUTO-CAPTURED!", (80, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                        # Send capture success info via Socket.IO
                        socketio.emit('calibration_capture', {
                            'success': True,
                            'timestamp': timestamp,
                            'left_path': left_filepath,
                            'right_path': right_filepath,
                            'auto_captured': True,
                            'pair_count': calibration_state["captured_pairs"],
                            'needed_pairs': calibration_state["min_pairs_needed"]
                        })

                        logger.debug("Auto-captured image pair saved to %s and %s",
                                     left_filepath, right_filepath)
                else:
                    # Reset stability tracking
                    calibration_state["is_stable"] = False
                    status_text = "NOT DETECTED"

                # Draw the status overlay
                cv2.putText(status_overlay, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if both_found else (0, 0, 255), 2)

                # Add capture progress
                progress_text = f"Captured: {calibration_state['captured_pairs']}/{calibration_state['recommended_pairs']}"
                cv2.putText(status_overlay, progress_text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Add auto-capture status
                auto_capture_status = f"Auto-capture: {'ON' if current_config['auto_capture'] else 'OFF'}"
                cv2.putText(status_overlay, auto_capture_status, (250, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if current_config['auto_capture'] else (100, 100, 100), 1)

                # Send calibration status via Socket.IO with detailed info
                socketio.emit('calibration_status', {
                    'left_found': found_left,
                    'right_found': found_right,
                    'both_found': both_found,
                    'stable_seconds': round(stability_duration, 1) if both_found and calibration_state[
                        "is_stable"] else 0,
                    'auto_capture': current_config["auto_capture"],
                    'stability_threshold': current_config["stability_seconds"],
                    'pairs_captured': calibration_state["captured_pairs"],
                    'pairs_needed': calibration_state["min_pairs_needed"],
                    'pairs_recommended': calibration_state["recommended_pairs"]
                })

                # Add the status overlay to the frames
                frames_to_send = {
                    'left': left_display,
                    'right': right_display,
                    'status': status_overlay
                }

            elif current_mode in ["process", "mapping"]:
                # Use calibration and compute disparity
                if mapL1 is not None and mapR1 is not None:
                    # Rectify images
                    left_rect = cv2.remap(left_frame, mapL1, mapL2, cv2.INTER_LINEAR)
                    right_rect = cv2.remap(right_frame, mapR1, mapR2, cv2.INTER_LINEAR)

                    # Optional: draw horizontal lines for debugging
                    left_with_lines = left_rect.copy()
                    right_with_lines = right_rect.copy()

                    step = 40
                    for i in range(0, left_rect.shape[0], step):
                        cv2.line(left_with_lines, (0, i), (left_rect.shape[1], i), (0, 255, 0), 1)
                        cv2.line(right_with_lines, (0, i), (right_rect.shape[1], i), (0, 255, 0), 1)

                    # Use selected disparity method
                    if current_config["disparity_method"] == "dl" and dl_model_loaded and dl_enabled:
                        try:
                            # Process with deep learning model
                            disparity_start = time.time()

                            # Prepare inputs according to DL parameters
                            dl_params = current_config["dl_params"]
                            downscale_factor = dl_params["downscale_factor"]

                            # Downscale if needed for faster processing
                            if downscale_factor != 1.0:
                                h, w = left_rect.shape[:2]
                                new_h, new_w = int(h * downscale_factor), int(w * downscale_factor)
                                left_rect_scaled = cv2.resize(left_rect, (new_w, new_h))
                                right_rect_scaled = cv2.resize(right_rect, (new_w, new_h))
                            else:
                                left_rect_scaled = left_rect
                                right_rect_scaled = right_rect

                            # Run DL inference
                            disparity = dl_model.inference(left_rect_scaled, right_rect_scaled)

                            # Resize back to original if needed
                            if downscale_factor != 1.0:
                                disparity = cv2.resize(disparity, (left_rect.shape[1], left_rect.shape[0]))
                                # Scale disparity values accordingly
                                disparity = disparity * (1.0 / downscale_factor)

                            disparity_time = time.time() - disparity_start

                            # Log performance
                            logger.debug(f"DL disparity computation took {disparity_time:.3f} seconds")

                            # Normalize for visualization
                            disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255,
                                                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

                            # Add a label to show we're using DL method
                            cv2.putText(disp_color, "Method: CREStereo", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            # Compute confidence map if requested
                            if current_config.get("show_confidence", False):
                                try:
                                    from models.utils import compute_confidence_map
                                    confidence = compute_confidence_map(disparity, left_rect, right_rect)

                                    # Visualize confidence map
                                    confidence_color = cv2.applyColorMap(
                                        (confidence * 255).astype(np.uint8),
                                        cv2.COLORMAP_TURBO)
                                except Exception as e:
                                    logger.error(f"Error computing confidence map: {str(e)}")
                                    confidence_color = None
                            else:
                                confidence_color = None

                        except Exception as e:
                            logger.error(f"Error in DL disparity computation: {str(e)}")
                            logger.error("Falling back to SGBM method")

                            # Fall back to SGBM method
                            disparity_params = current_config["disparity_params"]
                            stereo = cv2.StereoSGBM_create(
                                minDisparity=disparity_params["min_disp"],
                                numDisparities=disparity_params["num_disp"],
                                blockSize=disparity_params["window_size"],
                                P1=8 * 3 * disparity_params["window_size"] ** 2,
                                P2=32 * 3 * disparity_params["window_size"] ** 2,
                                disp12MaxDiff=1,
                                uniquenessRatio=disparity_params["uniqueness_ratio"],
                                speckleWindowSize=disparity_params["speckle_window_size"],
                                speckleRange=disparity_params["speckle_range"],
                                preFilterCap=63,
                                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                            )

                            # Convert to grayscale for disparity calculation
                            left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
                            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

                            # Compute the disparity map
                            disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

                            # Normalize for visualization
                            disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255,
                                                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

                            # Add a label to show we're using SGBM fallback
                            cv2.putText(disp_color, "Method: SGBM (fallback)", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            confidence_color = None
                    else:
                        # Use SGBM method
                        disparity_start = time.time()
                        disparity_params = current_config["disparity_params"]

                        stereo = cv2.StereoSGBM_create(
                            minDisparity=disparity_params["min_disp"],
                            numDisparities=disparity_params["num_disp"],
                            blockSize=disparity_params["window_size"],
                            P1=8 * 3 * disparity_params["window_size"] ** 2,
                            P2=32 * 3 * disparity_params["window_size"] ** 2,
                            disp12MaxDiff=1,
                            uniquenessRatio=disparity_params["uniqueness_ratio"],
                            speckleWindowSize=disparity_params["speckle_window_size"],
                            speckleRange=disparity_params["speckle_range"],
                            preFilterCap=63,
                            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                        )

                        # Convert to grayscale for disparity calculation
                        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
                        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

                        # Compute the disparity map
                        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
                        disparity_time = time.time() - disparity_start

                        # Log performance
                        logger.debug(f"SGBM disparity computation took {disparity_time:.3f} seconds")

                        # Normalize for visualization
                        disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255,
                                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

                        # Add a label to show we're using SGBM method
                        cv2.putText(disp_color, "Method: SGBM", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        # Compute confidence map if requested
                        if current_config.get("show_confidence", False):
                            try:
                                # Use stereo_vision's confidence map function for SGBM
                                confidence = stereo_vision.compute_confidence_map(disparity, left_rect, right_rect)

                                # Visualize confidence map
                                confidence_color = cv2.applyColorMap(
                                    (confidence * 255).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)
                            except Exception as e:
                                logger.error(f"Error computing confidence map: {str(e)}")
                                confidence_color = None
                        else:
                            confidence_color = None

                    # Prepare frames to send
                    frames_to_send = {
                        'left': left_with_lines,
                        'right': right_with_lines,
                        'disparity': disp_color
                    }

                    # Add confidence map if available
                    if confidence_color is not None:
                        frames_to_send['confidence'] = confidence_color

                    # If in mapping mode, can add point cloud generation here
                    if current_mode == "mapping":
                        # Additional mapping-specific processing would go here
                        pass

                else:
                    # No calibration data, fall back to basic view
                    frames_to_send = {
                        'left': left_frame,
                        'right': right_frame
                    }

            # Add FPS display to all frames
            for key, frame in frames_to_send.items():
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert frames to JPEG for streaming
            encoded_frames = {}
            for key, frame in frames_to_send.items():
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                encoded_frames[key] = base64.b64encode(buffer).decode('utf-8')

            # Send frames via websocket
            socketio.emit('frames', encoded_frames)

            # Small delay to reduce CPU usage
            time.sleep(0.01)

        except Exception as e:
            logger.error(f"Error in streaming thread: {str(e)}")
            socketio.emit('error', {'message': f"Streaming error: {str(e)}"})
            time.sleep(0.5)

    # Clean up when streaming stops
    try:
        stereo_vision.left_cam.release()
        stereo_vision.right_cam.release()
    except:
        pass

    logger.info("Streaming stopped")

# Route for main page
@app.route('/')
def index():
    return render_template('index.html')


# API Routes
@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'is_streaming': is_streaming,
        'current_mode': current_mode,
        'config': current_config
    })


@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    global current_config

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


@app.route('/api/disparity/method', methods=['GET', 'POST'])
def disparity_method():
    """Get or set the disparity computation method."""
    global current_config, dl_model, dl_model_loaded, dl_enabled

    if request.method == 'GET':
        return jsonify({
            'success': True,
            'current_method': current_config['disparity_method'],
            'available_methods': ['sgbm', 'dl'],
            'dl_available': dl_model_loaded,
            'dl_enabled': dl_enabled,
            'dl_model_name': current_config['dl_model_name'] if dl_model_loaded else None
        })

    elif request.method == 'POST':
        try:
            data = request.json
            method = data.get('method')

            if method not in ['sgbm', 'dl']:
                return jsonify({
                    'success': False,
                    'message': f"Invalid method: {method}. Use 'sgbm' or 'dl'."
                }), 400

            # Check if DL is available when requested
            if method == 'dl':
                if not dl_model_loaded:
                    current_config['dl_model_name'] = 'raft_stereo'
                    if not init_dl_model():
                        return jsonify({
                            'success': False,
                            'message': "Deep learning model initialization failed. Using SGBM instead."
                        }), 500

                # Enable DL processing
                dl_enabled = True

            # Update configuration
            current_config['disparity_method'] = method

            return jsonify({
                'success': True,
                'message': f"Disparity method set to {method}",
                'current_method': method,
                'dl_enabled': dl_enabled
            })

        except Exception as e:
            logger.error(f"Error setting disparity method: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/disparity/dl_params', methods=['GET', 'POST'])
def disparity_dl_params():
    """Get or set the deep learning disparity parameters."""
    global current_config

    if request.method == 'GET':
        return jsonify({
            'success': True,
            'dl_params': current_config['dl_params'],
            'dl_model_name': current_config['dl_model_name']
        })

    elif request.method == 'POST':
        try:
            data = request.json

            # Validate parameters
            if 'max_disp' in data:
                max_disp = int(data['max_disp'])
                if max_disp < 64 or max_disp > 512:
                    return jsonify({
                        'success': False,
                        'message': "max_disp must be between 64 and 512"
                    }), 400
                current_config['dl_params']['max_disp'] = max_disp

            if 'mixed_precision' in data:
                current_config['dl_params']['mixed_precision'] = bool(data['mixed_precision'])

            if 'downscale_factor' in data:
                factor = float(data['downscale_factor'])
                if factor <= 0 or factor > 1.0:
                    return jsonify({
                        'success': False,
                        'message': "downscale_factor must be between 0 and 1.0"
                    }), 400
                current_config['dl_params']['downscale_factor'] = factor

            # Model needs to be reinitialized if max_disp changes
            if 'max_disp' in data and dl_model_loaded:
                logger.info("max_disp changed - model will be reinitialized on next use")
                # We'll reinitialize on next use rather than immediately

            return jsonify({
                'success': True,
                'message': "Deep learning parameters updated",
                'dl_params': current_config['dl_params']
            })

        except Exception as e:
            logger.error(f"Error setting deep learning parameters: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500



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
    calibration_state = {
        "detected_count": 0,
        "is_stable": False,
        "stable_since": 0,
        "last_capture_time": 0,
        "captured_pairs": 0,
        "min_pairs_needed": 10,
        "recommended_pairs": 20,
        "checkerboard_history": []
    }

    # Check if we have existing calibration files to count
    try:
        calibration_files = os.listdir('static/calibration')
        left_images = [f for f in calibration_files if f.startswith('left_') and f.endswith('.jpg')]
        right_images = [f for f in calibration_files if f.startswith('right_') and f.endswith('.jpg')]

        # Match timestamps to count valid pairs
        pairs = []
        for left in left_images:
            left_ts = left.replace('left_', '').replace('.jpg', '')
            matching_right = f'right_{left_ts}.jpg'
            if matching_right in right_images:
                pairs.append((left, matching_right))

        # Update calibration state with existing pairs
        calibration_state["captured_pairs"] = len(pairs)
        logger.info("Found %d existing calibration image pairs", len(pairs))
    except Exception as e:
        logger.warning("Error checking existing calibration files: %s", str(e))

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
    global current_config

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
def capture_calibration_frame():
    global stereo_vision, current_config, calibration_state

    if not is_streaming or current_mode != "calibration":
        return jsonify({'success': False, 'message': 'Not in calibration mode'}), 400

    try:
        # Get raw frames
        ret_left, left_frame = stereo_vision.left_cam.read()
        ret_right, right_frame = stereo_vision.right_cam.read()

        if not ret_left or not ret_right:
            return jsonify({'success': False, 'message': 'Failed to capture frames'}), 500

        # Process frames to find checkerboard
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        pattern_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        cb_size = current_config["calibration_checkerboard_size"]

        found_left, left_corners = cv2.findChessboardCorners(
            left_gray, cb_size, pattern_flags)
        found_right, right_corners = cv2.findChessboardCorners(
            right_gray, cb_size, pattern_flags)

        if not (found_left and found_right):
            return jsonify({
                'success': False,
                'message': 'Checkerboard not detected in both frames',
                'left_found': found_left,
                'right_found': found_right
            }), 400

        # Save the calibration frame pair
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        left_filepath = f"static/calibration/left_{timestamp}.jpg"
        right_filepath = f"static/calibration/right_{timestamp}.jpg"

        cv2.imwrite(left_filepath, left_frame)
        cv2.imwrite(right_filepath, right_frame)

        return jsonify({
            'success': True,
            'timestamp': timestamp,
            'left_path': left_filepath,
            'right_path': right_filepath
        })

    except Exception as e:
        logger.error(f"Failed to capture calibration frame: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/calibrate/process', methods=['POST'])
def process_calibration():
    global stereo_vision, current_config, is_streaming

    # Stop streaming during calibration processing
    was_streaming = is_streaming
    if is_streaming:
        is_streaming = False
        if stream_thread is not None and stream_thread.is_alive():
            stream_thread.join(timeout=2.0)

    try:
        # Get list of all calibration image pairs
        calibration_files = os.listdir('static/calibration')
        left_images = sorted([f for f in calibration_files if f.startswith('left_')])
        right_images = sorted([f for f in calibration_files if f.startswith('right_')])

        if len(left_images) < 10 or len(right_images) < 10:
            return jsonify({
                'success': False,
                'message': f'Not enough calibration images. Need at least 10 pairs, have {len(left_images)} left and {len(right_images)} right.'
            }), 400

        # Match timestamps to ensure we're using proper pairs
        pairs = []
        for left in left_images:
            left_ts = left.replace('left_', '').replace('.jpg', '')
            matching_right = f'right_{left_ts}.jpg'
            if matching_right in right_images:
                pairs.append((
                    os.path.join('static/calibration', left),
                    os.path.join('static/calibration', matching_right)
                ))

        if len(pairs) < 10:
            return jsonify({
                'success': False,
                'message': f'Not enough matched calibration image pairs. Need at least 10, have {len(pairs)}.'
            }), 400

        # Initialize stereo vision if needed
        if stereo_vision is None:
            if not init_stereo_vision():
                return jsonify({'success': False, 'message': 'Failed to initialize stereo vision'}), 500

        # Run calibration with the collected images
        logger.info(f"Starting calibration with {len(pairs)} image pairs")
        socketio.emit('status', {'message': f"Processing calibration with {len(pairs)} image pairs..."})

        # Set up calibration parameters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cb_size = current_config["calibration_checkerboard_size"]
        square_size = current_config["calibration_square_size"]

        # Prepare 3D object points
        objp = np.zeros((cb_size[0] * cb_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:cb_size[0], 0:cb_size[1]].T.reshape(-1, 2)
        objp *= square_size  # Scale to real-world units

        # Arrays to store points
        objpoints = []
        left_imgpoints = []
        right_imgpoints = []

        # Process each image pair
        successful_pairs = 0
        for left_path, right_path in pairs:
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)

            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            pattern_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            found_left, left_corners = cv2.findChessboardCorners(
                left_gray, cb_size, pattern_flags)
            found_right, right_corners = cv2.findChessboardCorners(
                right_gray, cb_size, pattern_flags)

            if found_left and found_right:
                # Refine corner locations
                left_corners = cv2.cornerSubPix(
                    left_gray, left_corners, (11, 11), (-1, -1), criteria)
                right_corners = cv2.cornerSubPix(
                    right_gray, right_corners, (11, 11), (-1, -1), criteria)

                # Store points
                objpoints.append(objp)
                left_imgpoints.append(left_corners)
                right_imgpoints.append(right_corners)
                successful_pairs += 1

                # Update status
                socketio.emit('status',
                              {'message': f"Processed {successful_pairs}/{len(pairs)} image pairs successfully"})

        if successful_pairs < 10:
            return jsonify({
                'success': False,
                'message': f'Not enough successful calibration pairs. Need at least 10, processed {successful_pairs} successfully.'
            }), 400

        # Run stereo calibration
        logger.info(f"Running stereo calibration with {successful_pairs} successful pairs")
        socketio.emit('status', {'message': "Running stereo calibration..."})

        image_size = left_gray.shape[::-1]  # (width, height)

        ret, camera_matrix_left, dist_coeffs_left, \
            camera_matrix_right, dist_coeffs_right, \
            R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            left_imgpoints,
            right_imgpoints,
            None,  # no initial left camera matrix
            None,  # no initial left distortion
            None,  # no initial right camera matrix
            None,  # no initial right distortion
            image_size,
            criteria=criteria,
            flags=0  # let OpenCV refine intrinsics
        )

        logger.info(f"Calibration complete with RMS error: {ret}")
        socketio.emit('status', {'message': f"Calibration complete with RMS error: {ret}"})

        # Stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        # Save calibration parameters to file
        calibration_data = {
            'camera_matrix_left': camera_matrix_left.tolist(),
            'dist_coeffs_left': dist_coeffs_left.tolist(),
            'camera_matrix_right': camera_matrix_right.tolist(),
            'dist_coeffs_right': dist_coeffs_right.tolist(),
            'R': R.tolist(),
            'T': T.tolist(),
            'Q': Q.tolist(),
            'R1': R1.tolist(),
            'R2': R2.tolist(),
            'P1': P1.tolist(),
            'P2': P2.tolist(),
            'image_size': image_size,
            'calibration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'rms_error': float(ret)
        }

        # Save calibration to both main and backup files with validation
        try:
            calibration_file = 'stereo_calibration.npy'

            # Save the main calibration file
            logger.info("Saving main calibration file to %s", calibration_file)
            np.save(calibration_file, calibration_data)

            # Verify the file was saved
            if not os.path.exists(calibration_file):
                logger.error("Failed to save calibration file! File doesn't exist after saving")
                raise IOError("Failed to save calibration file")

            file_size = os.path.getsize(calibration_file)
            if file_size < 1000:  # Minimum expected file size
                logger.error("Calibration file suspiciously small (%d bytes), may be corrupted", file_size)
                socketio.emit('status', {'message': "Warning: Calibration file may be incomplete"})
            else:
                logger.info("Calibration file saved successfully (%d bytes)", file_size)

            # Create and save a timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f'static/calibration/stereo_calibration_{timestamp}.npy'
            logger.info("Saving backup calibration file to %s", backup_file)
            np.save(backup_file, calibration_data)

            # Verify the backup file
            if not os.path.exists(backup_file):
                logger.error("Failed to save backup calibration file!")
                socketio.emit('status', {'message': "Warning: Backup calibration file failed to save"})
            else:
                logger.info("Backup calibration file saved successfully")

            # Clean up old backup files if there are too many (keep the 10 most recent)
            try:
                backup_files = [f for f in os.listdir('static/calibration')
                              if f.startswith('stereo_calibration_') and f.endswith('.npy')]
                if len(backup_files) > 10:
                    backup_files.sort()  # Sort by timestamp (oldest first)
                    files_to_remove = backup_files[:-10]  # Remove all but the 10 newest
                    for old_file in files_to_remove:
                        old_path = os.path.join('static/calibration', old_file)
                        os.remove(old_path)
                        logger.debug("Removed old calibration backup: %s", old_path)
                    logger.info("Cleaned up %d old calibration backups", len(files_to_remove))
            except Exception as e:
                logger.warning("Error cleaning up old calibration backups: %s", str(e))

        except Exception as e:
            logger.error("Failed to save calibration files: %s", str(e))
            socketio.emit('status', {'message': f"Error saving calibration: {str(e)}"})
            raise

        # Restart streaming if it was active before
        if was_streaming:
            is_streaming = True
            new_stream_thread = threading.Thread(target=process_stream)
            new_stream_thread.daemon = True
            new_stream_thread.start()

        return jsonify({
            'success': True,
            'rms_error': float(ret),
            'calibration_file': calibration_file,
            'backup_file': backup_file,
            'pairs_processed': successful_pairs
        })

    except Exception as e:
        logger.error(f"Failed to process calibration: {str(e)}")

        # Restart streaming if it was active before
        if was_streaming:
            is_streaming = True
            new_stream_thread = threading.Thread(target=process_stream)
            new_stream_thread.daemon = True
            new_stream_thread.start()

        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/capture', methods=['POST'])
def capture_frame():
    global stereo_vision

    if not is_streaming:
        return jsonify({'success': False, 'message': 'Stream is not running'}), 400

    try:
        # Capture raw frames
        ret_left, left_frame = stereo_vision.left_cam.read()
        ret_right, right_frame = stereo_vision.right_cam.read()

        if not ret_left or not ret_right:
            return jsonify({'success': False, 'message': 'Failed to capture frames'}), 500

        # Save the captured frames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        left_filepath = f"static/captures/left_{timestamp}.jpg"
        right_filepath = f"static/captures/right_{timestamp}.jpg"

        cv2.imwrite(left_filepath, left_frame)
        cv2.imwrite(right_filepath, right_frame)

        # If in process or mapping mode, save disparity map too
        disparity_filepath = None
        pointcloud_filepath = None

        if current_mode in ["process", "mapping"] and stereo_vision.Q is not None:
            # Get rectification maps
            mapL1, mapL2 = cv2.initUndistortRectifyMap(
                stereo_vision.camera_matrix_left, stereo_vision.dist_coeffs_left,
                stereo_vision.R1, stereo_vision.P1,
                (current_config["width"], current_config["height"]), cv2.CV_32FC1
            )
            mapR1, mapR2 = cv2.initUndistortRectifyMap(
                stereo_vision.camera_matrix_right, stereo_vision.dist_coeffs_right,
                stereo_vision.R2, stereo_vision.P2,
                (current_config["width"], current_config["height"]), cv2.CV_32FC1
            )

            # Rectify images
            left_rect = cv2.remap(left_frame, mapL1, mapL2, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_frame, mapR1, mapR2, cv2.INTER_LINEAR)

            # Compute disparity
            disparity, disp_color = stereo_vision.compute_disparity_map(left_rect, right_rect)

            disparity_filepath = f"static/captures/disparity_{timestamp}.jpg"
            cv2.imwrite(disparity_filepath, disp_color)

            # Generate and save point cloud for mapping mode
            if current_mode == "mapping":
                # Convert disparity to 3D points
                points = cv2.reprojectImageTo3D(disparity, stereo_vision.Q)

                # Save point cloud (simple format for now, can use PCL or other formats)
                pointcloud_filepath = f"static/maps/pointcloud_{timestamp}.npy"
                np.save(pointcloud_filepath, points)

        return jsonify({
            'success': True,
            'timestamp': timestamp,
            'left_path': left_filepath,
            'right_path': right_filepath,
            'disparity_path': disparity_filepath,
            'pointcloud_path': pointcloud_filepath
        })

    except Exception as e:
        logger.error(f"Failed to capture frame: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/code/update', methods=['POST'])
def update_code():
    """Update the stereo_vision.py file with new code."""

    try:
        # Get the new code from the request
        new_code = request.json.get('code')
        if not new_code:
            return jsonify({'success': False, 'message': 'No code provided'}), 400

        # Backup current code
        if not backup_current_code():
            return jsonify({'success': False, 'message': 'Failed to backup current code'}), 500

        # Write new code to file
        with open('stereo_vision.py', 'w') as f:
            f.write(new_code)

        logger.info("Code updated successfully")
        return jsonify({
            'success': True,
            'message': 'Code updated successfully',
            'backup_versions': code_versions
        })

    except Exception as e:
        logger.error(f"Failed to update code: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/code/rollback', methods=['POST'])
def rollback_code():
    """Rollback to a previous version of the code."""

    try:
        # Get the version to rollback to
        version = request.json.get('version')
        if not version:
            return jsonify({'success': False, 'message': 'No version specified'}), 400

        # Find the version in the history
        version_found = False
        for v in code_versions:
            if v['timestamp'] == version:
                version_found = True
                # Read the old code
                with open(v['filename'], 'r') as f:
                    old_code = f.read()

                # Backup current code first
                backup_current_code()

                # Write old code to the main file
                with open('stereo_vision.py', 'w') as f:
                    f.write(old_code)

                break

        if not version_found:
            return jsonify({'success': False, 'message': 'Version not found'}), 404

        logger.info(f"Code rolled back to version {version}")
        return jsonify({
            'success': True,
            'message': f'Code rolled back to version {version}',
            'backup_versions': code_versions
        })

    except Exception as e:
        logger.error(f"Failed to rollback code: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/code/versions', methods=['GET'])
def get_code_versions():
    """Get the available code versions for rollback."""
    global code_versions

    try:
        # Check if code_versions is empty and if backups exist
        if not code_versions:
            # Try to load backup versions from directory
            backup_dir = 'static/code_backups'
            if os.path.exists(backup_dir):
                backup_files = [f for f in os.listdir(backup_dir) if f.startswith('stereo_vision_') and f.endswith('.py')]

                # Sort by timestamp (newest first)
                backup_files.sort(reverse=True)

                # Create version entries
                for filename in backup_files:
                    # Extract timestamp from filename
                    timestamp = filename.replace('stereo_vision_', '').replace('.py', '')
                    try:
                        # Create datetime object for better display
                        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")

                        # Add to code_versions list
                        code_versions.append({
                            'timestamp': timestamp,
                            'filename': os.path.join(backup_dir, filename),
                            'datetime': formatted_date
                        })
                    except ValueError:
                        # Skip if timestamp format is invalid
                        logger.warning(f"Invalid timestamp format in backup file: {filename}")
                        continue

        return jsonify({
            'success': True,
            'versions': code_versions
        })
    except Exception as e:
        logger.error(f"Failed to get code versions: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/code/current', methods=['GET'])
def get_current_code():
    """Get the current stereo_vision.py code."""

    try:
        # Check if file exists first
        if not os.path.exists('stereo_vision.py'):
            logger.error("stereo_vision.py file not found")
            return jsonify({'success': False, 'message': 'stereo_vision.py file not found'}), 404

        # Get file modified time
        last_modified = datetime.fromtimestamp(os.path.getmtime('stereo_vision.py')).strftime("%Y-%m-%d %H:%M:%S")

        with open('stereo_vision.py', 'r') as f:
            code = f.read()

        return jsonify({
            'success': True,
            'code': code,
            'last_modified': last_modified
        })

    except Exception as e:
        logger.error(f"Failed to get current code: {str(e)}")
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


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get the application logs with filtering options."""

    try:
        # Parse query parameters for filtering
        log_type = request.args.get('type', 'app')  # 'app' (default) or 'stereo'
        level = request.args.get('level', '').upper()  # Filter by log level
        limit = int(request.args.get('limit', 100))  # Number of lines to return
        search = request.args.get('search', '')  # Text to search for

        # Cap the limit to prevent excessive responses
        if limit > 500:
            limit = 500

        # Determine which log file to read
        log_file = "stereo_vision_app.log" if log_type == 'app' else "stereo_vision.log"

        if not os.path.exists(log_file):
            return jsonify({
                'success': False,
                'message': f"Log file {log_file} not found",
                'available_logs': [f for f in os.listdir('.') if f.endswith('.log')]
            }), 404

        # Read the log file
        log_size = os.path.getsize(log_file)
        logger.debug("Reading log file %s (%.1f KB)", log_file, log_size/1024)

        all_logs = []
        with open(log_file, 'r') as f:
            all_logs = f.readlines()

        # Apply filters
        filtered_logs = []
        for line in all_logs:
            # Filter by level if specified
            if level and level not in line.upper():
                continue

            # Filter by search text if specified
            if search and search.lower() not in line.lower():
                continue

            filtered_logs.append(line)

        # Get the limited number of lines (from the end)
        if limit > 0:
            filtered_logs = filtered_logs[-limit:]

        # Create parsed logs with structured information
        parsed_logs = []
        for line in filtered_logs:
            try:
                # Parse log line into components (rough approximation)
                parts = line.split(' - ', 3)
                if len(parts) >= 4:
                    timestamp = parts[0]
                    module = parts[1]
                    level = parts[2]
                    message = parts[3].strip()

                    parsed_logs.append({
                        'timestamp': timestamp,
                        'module': module,
                        'level': level,
                        'message': message,
                        'raw': line.strip()
                    })
                else:
                    # If we can't parse it, just include the raw line
                    parsed_logs.append({
                        'raw': line.strip()
                    })
            except Exception:
                # If parsing fails, include the raw line
                parsed_logs.append({
                    'raw': line.strip()
                })

        # Log the number of lines returned
        logger.debug("Returning %d log lines (filtered from %d total lines)",
                    len(parsed_logs), len(all_logs))

        return jsonify({
            'success': True,
            'logs': parsed_logs,
            'total_lines': len(all_logs),
            'returned_lines': len(parsed_logs),
            'log_file': log_file,
            'log_size_kb': round(log_size/1024, 1),
            'filters': {
                'type': log_type,
                'level': level,
                'limit': limit,
                'search': search
            }
        })

    except Exception as e:
        logger.error("Failed to get logs: %s", str(e))
        return jsonify({'success': False, 'message': str(e)}), 500


# Outdates, will be deleted
# @app.route('/templates/index.html')
# def serve_template():
#     return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)



# Run the application
if __name__ == '__main__':
    # Make sure the stereo_vision.py file exists
    if not os.path.exists('stereo_vision.py'):
        logger.error("stereo_vision.py not found")
        print("Error: stereo_vision.py not found. Please create it first.")
        exit(1)

    # Start the Flask-SocketIO server
    hostname = socket.gethostname()
#    ip = socket.gethostbyname(hostname)
    logger.info("Starting Stereo Vision Web Interface")
    
   # logger.info(f"Running on http://{ip}:8080")
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
