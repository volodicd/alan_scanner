# services/stream_service.py
import cv2
import numpy as np
import time
import base64
from datetime import datetime
import threading
from utils import emit
from config import (
    stereo_vision, is_streaming, current_mode, current_config,
    dl_model, dl_model_loaded, dl_enabled, logger
)


def init_stereo_vision():
    """Initialize the stereo vision system."""
    global stereo_vision

    try:
        # Import the StereoVision class
        from stereo_vision import StereoVision

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


def process_stream():
    """Main streaming function that handles different processing modes."""
    global is_streaming, current_mode, stereo_vision

    logger.info(f"Starting stream in {current_mode} mode")

    # Try to open cameras
    try:
        stereo_vision.open_cameras()
    except Exception as e:
        logger.error(f"Failed to open cameras: {str(e)}")
        emit('error', {'message': f"Failed to open cameras: {str(e)}"})
        is_streaming = False
        return

    # Initialize rectification maps if we have calibration
    mapL1, mapL2, mapR1, mapR2 = None, None, None, None

    if current_mode in ["mapping", "process"]:
        try:
            # Load calibration data
            if not stereo_vision.load_calibration():
                logger.error("Calibration loading failed - load_calibration returned False")
                emit('error', {'message': "Calibration not loaded. Please calibrate cameras first."})
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
                    emit('error', {'message': "Invalid calibration data. Please recalibrate cameras."})
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
            emit('error', {'message': f"Failed to initialize calibration: {str(e)}"})
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
                emit('error', {'message': "Frame capture failed. Check camera connections."})
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
                frames_to_send = process_calibration_frame(left_frame, right_frame, fps)

            elif current_mode in ["process", "mapping"]:
                # Use calibration and compute disparity
                frames_to_send = process_disparity_frame(left_frame, right_frame, mapL1, mapL2, mapR1, mapR2, fps)

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
            emit('frames', encoded_frames)

            # Small delay to reduce CPU usage
            time.sleep(0.01)

        except Exception as e:
            logger.error(f"Error in streaming thread: {str(e)}")
            emit('error', {'message': f"Streaming error: {str(e)}"})
            time.sleep(0.5)

    # Clean up when streaming stops
    try:
        stereo_vision.left_cam.release()
        stereo_vision.right_cam.release()
    except:
        pass

    logger.info("Streaming stopped")


def process_calibration_frame(left_frame, right_frame, fps):
    """Process frames for calibration mode."""
    from config import calibration_state

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
            emit('calibration_capture', {
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
    emit('calibration_status', {
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

    # Return the frames to send
    return {
        'left': left_display,
        'right': right_display,
        'status': status_overlay
    }


def process_disparity_frame(left_frame, right_frame, mapL1, mapL2, mapR1, mapR2, fps):
    """Process frames for disparity computation (process or mapping mode)."""
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
                disp_color, confidence_color = compute_sgbm_disparity(left_rect, right_rect, "SGBM (fallback)")

        else:
            # Use SGBM method
            disp_color, confidence_color = compute_sgbm_disparity(left_rect, right_rect, "SGBM")

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

    return frames_to_send


def compute_sgbm_disparity(left_rect, right_rect, method_label):
    """Compute disparity using SGBM method."""
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
    cv2.putText(disp_color, f"Method: {method_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Compute confidence map if requested
    confidence_color = None
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

    return disp_color, confidence_color


def capture_frame():
    """Capture and save the current frame pair."""
    if not is_streaming or stereo_vision is None:
        return False, "Stream is not running or stereo vision not initialized", None, None, None, None

    try:
        # Capture raw frames
        ret_left, left_frame = stereo_vision.left_cam.read()
        ret_right, right_frame = stereo_vision.right_cam.read()

        if not ret_left or not ret_right:
            return False, "Failed to capture frames", None, None, None, None

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

        return True, timestamp, left_filepath, right_filepath, disparity_filepath, pointcloud_filepath

    except Exception as e:
        logger.error(f"Failed to capture frame: {str(e)}")
        return False, str(e), None, None, None, None