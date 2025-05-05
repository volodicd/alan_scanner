# services/calibration_service.py
import os
import cv2
import numpy as np
from datetime import datetime
import time
import logging
from utils import emit

from config import calibration_state, current_config, logger, stereo_vision


def backup_current_code():
    from services.code_service import backup_current_code as bcc
    return bcc()


def capture_calibration_frame():
    """Capture a pair of frames for calibration."""
    if stereo_vision is None:
        return False, "Stereo vision not initialized", None, None

    try:
        # Get raw frames
        ret_left, left_frame = stereo_vision.left_cam.read()
        ret_right, right_frame = stereo_vision.right_cam.read()

        if not ret_left or not ret_right:
            return False, "Failed to capture frames", None, None

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
            return False, "Checkerboard not detected in both frames", found_left, found_right

        # Save the calibration frame pair
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        left_filepath = f"static/calibration/left_{timestamp}.jpg"
        right_filepath = f"static/calibration/right_{timestamp}.jpg"

        cv2.imwrite(left_filepath, left_frame)
        cv2.imwrite(right_filepath, right_frame)

        # Update the calibration state
        calibration_state["captured_pairs"] += 1
        calibration_state["last_capture_time"] = time.time()

        return True, timestamp, left_filepath, right_filepath

    except Exception as e:
        logger.error(f"Failed to capture calibration frame: {str(e)}")
        return False, str(e), None, None


def process_calibration():
    """Process collected calibration images to generate calibration data."""
    try:
        # Get list of all calibration image pairs
        calibration_files = os.listdir('static/calibration')
        left_images = sorted([f for f in calibration_files if f.startswith('left_')])
        right_images = sorted([f for f in calibration_files if f.startswith('right_')])

        if len(left_images) < 10 or len(right_images) < 10:
            return {
                'success': False,
                'message': f'Not enough calibration images. Need at least 10 pairs, have {len(left_images)} left and {len(right_images)} right.'
            }

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
            return {
                'success': False,
                'message': f'Not enough matched calibration image pairs. Need at least 10, have {len(pairs)}.'
            }

        # Initialize stereo vision if needed
        if stereo_vision is None:
            from services.stream_service import init_stereo_vision
            if not init_stereo_vision():
                return {'success': False, 'message': 'Failed to initialize stereo vision'}

        # Run calibration with the collected images
        logger.info(f"Starting calibration with {len(pairs)} image pairs")
        emit('status', {'message': f"Processing calibration with {len(pairs)} image pairs..."})

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
                emit('status',
                     {'message': f"Processed {successful_pairs}/{len(pairs)} image pairs successfully"})

        if successful_pairs < 10:
            return {
                'success': False,
                'message': f'Not enough successful calibration pairs. Need at least 10, processed {successful_pairs} successfully.'
            }

        # Run stereo calibration
        logger.info(f"Running stereo calibration with {successful_pairs} successful pairs")
        emit('status', {'message': "Running stereo calibration..."})

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
        emit('status', {'message': f"Calibration complete with RMS error: {ret}"})

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

        # Save calibration to both main and backup files
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
                emit('status', {'message': "Warning: Calibration file may be incomplete"})
            else:
                logger.info("Calibration file saved successfully (%d bytes)", file_size)

            # Create and save a timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f'static/calibration/stereo_calibration_{timestamp}.npy'
            logger.info("Saving backup calibration file to %s", backup_file)
            np.save(backup_file, calibration_data)

            # Clean up old backup files if there are too many
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

            return {
                'success': True,
                'rms_error': float(ret),
                'calibration_file': calibration_file,
                'backup_file': backup_file,
                'pairs_processed': successful_pairs
            }

        except Exception as e:
            logger.error("Failed to save calibration files: %s", str(e))
            emit('status', {'message': f"Error saving calibration: {str(e)}"})
            raise

    except Exception as e:
        logger.error(f"Failed to process calibration: {str(e)}")
        return {'success': False, 'message': str(e)}


def reset_calibration_state():
    """Reset the calibration state to default values."""
    calibration_state.update({
        "detected_count": 0,
        "is_stable": False,
        "stable_since": 0,
        "last_capture_time": 0,
        "captured_pairs": 0,
        "min_pairs_needed": 10,
        "recommended_pairs": 20,
        "checkerboard_history": []
    })

    # Count existing calibration files
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