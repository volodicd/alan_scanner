import cv2
import numpy as np
import time
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stereo_vision.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StereoVision:
    def __init__(self, left_cam_idx=0, right_cam_idx=1, width=640, height=480):
        """Initialize stereo vision system with two cameras."""
        self.left_cam_idx = left_cam_idx
        self.right_cam_idx = right_cam_idx
        self.width = width
        self.height = height
        self.left_cam = None
        self.right_cam = None

        # Camera calibration matrices (populated after calibration)
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.R = None
        self.T = None
        self.Q = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None

        # Configuration for SGBM algorithm
        self.sgbm_params = {
            'window_size': 11,
            'min_disp': 0,
            'num_disp': 128,  # Must be multiple of 16
            'uniqueness_ratio': 15,
            'speckle_window_size': 100,
            'speckle_range': 32
        }

    def open_cameras(self):
        """Open and configure both cameras."""
        logger.info("Opening cameras (Left: %d, Right: %d)", self.left_cam_idx, self.right_cam_idx)
        self.left_cam = cv2.VideoCapture(self.left_cam_idx)
        self.right_cam = cv2.VideoCapture(self.right_cam_idx)

        if not self.left_cam.isOpened() or not self.right_cam.isOpened():
            logger.error("Failed to open one or both cameras")
            raise RuntimeError("Failed to open one or both cameras.")

        # Set resolution for both cameras
        logger.info("Setting camera resolution to %dx%d", self.width, self.height)
        for cam in [self.left_cam, self.right_cam]:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cam.set(cv2.CAP_PROP_FPS, 30)

        # Allow cameras to warm up
        logger.info("Allowing cameras to warm up for 1 second")
        time.sleep(1)
        logger.info("Cameras initialized successfully")

    def close_cameras(self):
        """Close the cameras."""
        if self.left_cam:
            self.left_cam.release()
        if self.right_cam:
            self.right_cam.release()
        self.left_cam = None
        self.right_cam = None
        logger.info("Cameras closed")

    def capture_frames(self):
        """Capture frames from both cameras."""
        if not self.left_cam or not self.right_cam:
            self.open_cameras()

        ret_left, left_frame = self.left_cam.read()
        ret_right, right_frame = self.right_cam.read()

        if not ret_left or not ret_right:
            logger.error("Failed to capture frames from one or both cameras")
            return None, None

        return left_frame, right_frame

    def calibrate_cameras(self, checkerboard_size=(7, 6), square_size=0.025, auto_capture=True, stability_seconds=3.0):
        """
        Calibration with checkerboard detection and automatic frame capture.

        Args:
            checkerboard_size: (columns, rows) of inner corners in the checkerboard.
            square_size: Physical size of each checkerboard square in meters.
            auto_capture: If True, automatically capture frames when checkerboard is stable.
            stability_seconds: Number of seconds checkerboard needs to be detected before auto-capturing.
        """
        already_opened = self.left_cam is not None and self.right_cam is not None
        if not already_opened:
            self.open_cameras()

        # Create output directory for calibration images
        os.makedirs('static/calibration', exist_ok=True)
        logger.info("Starting camera calibration process")
        logger.info("Checkerboard size: %dx%d, Square size: %.3f meters",
                    checkerboard_size[0], checkerboard_size[1], square_size)
        logger.info("Auto-capture: %s, Stability time: %.1f seconds",
                    "enabled" if auto_capture else "disabled", stability_seconds)

        # Termination criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare 3D object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size  # Scale to real-world units

        # Arrays to store points
        objpoints = []
        left_imgpoints = []
        right_imgpoints = []

        # Auto-capture variables
        stable_since = 0
        is_stable = False
        last_capture_time = 0
        frame_count = 0
        needed_frames = 20

        try:
            while frame_count < needed_frames:
                left_frame, right_frame = self.capture_frames()
                if left_frame is None or right_frame is None:
                    logger.warning("Failed to capture frames")
                    time.sleep(0.5)
                    continue

                # Convert to grayscale
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                # Apply preprocessing to improve detection
                left_gray = cv2.GaussianBlur(left_gray, (5, 5), 0)
                right_gray = cv2.GaussianBlur(right_gray, (5, 5), 0)

                # Check for checkerboard
                pattern_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                found_left, left_corners = cv2.findChessboardCorners(
                    left_gray, checkerboard_size, pattern_flags)
                found_right, right_corners = cv2.findChessboardCorners(
                    right_gray, checkerboard_size, pattern_flags)

                # Track detection for auto-capture
                current_time = time.time()
                both_found = found_left and found_right

                # For auto-capture, track stability
                if both_found:
                    if not is_stable:
                        stable_since = current_time
                        is_stable = True

                    # Calculate stable duration
                    stable_duration = current_time - stable_since

                    # Auto-capture if stable for required duration and enough time has passed since last capture
                    min_capture_interval = 1.0  # Minimum seconds between captures
                    time_since_last_capture = current_time - last_capture_time

                    if (auto_capture and stable_duration >= stability_seconds and
                            time_since_last_capture > min_capture_interval):
                        # Process frame
                        logger.info("Auto-capturing after %.1f seconds of stability", stable_duration)

                        # Refine corner locations
                        left_corners = cv2.cornerSubPix(
                            left_gray, left_corners, (11, 11), (-1, -1), criteria)
                        right_corners = cv2.cornerSubPix(
                            right_gray, right_corners, (11, 11), (-1, -1), criteria)

                        # Store points
                        objpoints.append(objp)
                        left_imgpoints.append(left_corners)
                        right_imgpoints.append(right_corners)

                        # Save calibration images
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        left_path = f'static/calibration/left_{timestamp}.jpg'
                        right_path = f'static/calibration/right_{timestamp}.jpg'

                        # Draw corners on the images
                        left_display = left_frame.copy()
                        right_display = right_frame.copy()
                        cv2.drawChessboardCorners(
                            left_display, checkerboard_size, left_corners, found_left)
                        cv2.drawChessboardCorners(
                            right_display, checkerboard_size, right_corners, found_right)

                        cv2.imwrite(left_path, left_display)
                        cv2.imwrite(right_path, right_display)

                        # Update state
                        frame_count += 1
                        last_capture_time = current_time
                        is_stable = False  # Reset stability to prevent rapid captures

                        logger.info("Captured calibration frame pair %d/%d", frame_count, needed_frames)
                else:
                    # Reset stability tracking
                    is_stable = False

                # Add a small delay
                time.sleep(0.05)

            if frame_count < 10:
                logger.warning("Not enough calibration pairs collected (minimum 10 needed)")
                return False

            logger.info("Starting stereo calibration with %d image pairs...", frame_count)
            image_size = left_gray.shape[::-1]  # (width, height)

            # Run stereo calibration
            ret, self.camera_matrix_left, self.dist_coeffs_left, \
                self.camera_matrix_right, self.dist_coeffs_right, \
                self.R, self.T, E, F = cv2.stereoCalibrate(
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
            logger.info("Stereo calibration completed!")
            logger.info("Calibration RMS error: %.6f", ret)

            # Stereo rectification
            logger.info("Computing stereo rectification parameters...")
            self.R1, self.R2, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(
                self.camera_matrix_left, self.dist_coeffs_left,
                self.camera_matrix_right, self.dist_coeffs_right,
                image_size, self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
            )

            # Save calibration parameters to file
            logger.info("Saving calibration data to file...")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            calibration_data = {
                'camera_matrix_left': self.camera_matrix_left,
                'dist_coeffs_left': self.dist_coeffs_left,
                'camera_matrix_right': self.camera_matrix_right,
                'dist_coeffs_right': self.dist_coeffs_right,
                'R': self.R,
                'T': self.T,
                'Q': self.Q,
                'R1': self.R1,
                'R2': self.R2,
                'P1': self.P1,
                'P2': self.P2,
                'image_size': image_size,
                'calibration_date': timestamp,
                'rms_error': float(ret),
                'frame_count': frame_count
            }

            # Save the main calibration file
            np.save('stereo_calibration.npy', calibration_data)

            # Also save a backup with timestamp
            backup_file = f'static/calibration/stereo_calibration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npy'
            np.save(backup_file, calibration_data)

            logger.info("Saved calibration data to stereo_calibration.npy and %s", backup_file)
            return True

        except Exception as e:
            logger.error("Calibration failed: %s", str(e))
            return False

        finally:
            if not already_opened:
                self.close_cameras()

    def load_calibration(self):
        """Load calibration parameters from file."""
        try:
            logger.info("Loading calibration from stereo_calibration.npy")
            calibration_data = np.load('stereo_calibration.npy', allow_pickle=True).item()

            # Convert lists back to numpy arrays if needed
            if isinstance(calibration_data['camera_matrix_left'], list):
                self.camera_matrix_left = np.array(calibration_data['camera_matrix_left'], dtype=np.float32)
            else:
                self.camera_matrix_left = calibration_data['camera_matrix_left'].astype(np.float32)

            if isinstance(calibration_data['dist_coeffs_left'], list):
                self.dist_coeffs_left = np.array(calibration_data['dist_coeffs_left'], dtype=np.float32)
            else:
                self.dist_coeffs_left = calibration_data['dist_coeffs_left'].astype(np.float32)

            if isinstance(calibration_data['camera_matrix_right'], list):
                self.camera_matrix_right = np.array(calibration_data['camera_matrix_right'], dtype=np.float32)
            else:
                self.camera_matrix_right = calibration_data['camera_matrix_right'].astype(np.float32)

            if isinstance(calibration_data['dist_coeffs_right'], list):
                self.dist_coeffs_right = np.array(calibration_data['dist_coeffs_right'], dtype=np.float32)
            else:
                self.dist_coeffs_right = calibration_data['dist_coeffs_right'].astype(np.float32)

            # Load other parameters similarly
            for param in ['R', 'T', 'Q', 'R1', 'R2', 'P1', 'P2']:
                if param in calibration_data:
                    if isinstance(calibration_data[param], list):
                        setattr(self, param, np.array(calibration_data[param], dtype=np.float32))
                    else:
                        setattr(self, param, calibration_data[param].astype(np.float32))

            logger.info("Calibration loaded successfully")
            return True

        except FileNotFoundError:
            logger.error("Calibration file not found. Please run calibration first.")
            return False
        except Exception as e:
            logger.error("Error loading calibration: %s", str(e))
            return False

    def compute_disparity_map(self, left_img, right_img):
        """
        Compute the disparity map using SGBM.

        Args:
            left_img: Left camera image.
            right_img: Right camera image.

        Returns:
            (disparity, disparity_color) where disparity is float32 and disparity_color is a color map.
        """
        # Convert to grayscale if necessary
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img

        # Create a StereoSGBM matcher
        window_size = self.sgbm_params['window_size']
        min_disp = self.sgbm_params['min_disp']
        num_disp = self.sgbm_params['num_disp']

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=self.sgbm_params['uniqueness_ratio'],
            speckleWindowSize=self.sgbm_params['speckle_window_size'],
            speckleRange=self.sgbm_params['speckle_range'],
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute the disparity map
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Normalize for visualization
        disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255,
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

        return disparity, disp_color

    def get_rectified_images(self, left_img, right_img):
        """
        Get rectified images using the calibration parameters.
        """
        if not all([self.camera_matrix_left, self.dist_coeffs_left,
                    self.R1, self.P1, self.camera_matrix_right,
                    self.dist_coeffs_right, self.R2, self.P2]):
            logger.error("Calibration parameters not loaded")
            return left_img, right_img

        # Create rectification maps
        h, w = left_img.shape[:2]

        mapL1, mapL2 = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.R1, self.P1, (w, h), cv2.CV_32FC1)

        mapR1, mapR2 = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right,
            self.R2, self.P2, (w, h), cv2.CV_32FC1)

        # Rectify images
        left_rect = cv2.remap(left_img, mapL1, mapL2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, mapR1, mapR2, cv2.INTER_LINEAR)

        return left_rect, right_rect

    def set_sgbm_params(self, params):
        """Update SGBM parameters."""
        for key, value in params.items():
            if key in self.sgbm_params:
                self.sgbm_params[key] = value

        # Ensure num_disp is multiple of 16
        if 'num_disp' in params:
            self.sgbm_params['num_disp'] = (self.sgbm_params['num_disp'] // 16) * 16
            if self.sgbm_params['num_disp'] < 16:
                self.sgbm_params['num_disp'] = 16