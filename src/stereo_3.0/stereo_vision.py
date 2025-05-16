import cv2
import numpy as np
import time
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class StereoVision:
    def __init__(self, left_cam_idx=0, right_cam_idx=1, width=640, height=480):
        self.left_cam_idx = left_cam_idx
        self.right_cam_idx = right_cam_idx
        self.width = width
        self.height = height
        self.left_cam = None
        self.right_cam = None

        # Calibration parameters
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

        # Rectification maps
        self.mapL1 = None
        self.mapL2 = None
        self.mapR1 = None
        self.mapR2 = None
        self.maps_initialized = False

        # SGBM parameters with defaults
        self.sgbm_params = {
            'window_size': 11,
            'min_disp': 0,
            'num_disp': 128,
            'uniqueness_ratio': 15,
            'speckle_window_size': 100,
            'speckle_range': 32
        }

        # Calibration directory
        os.makedirs('data/calibration', exist_ok=True)
        os.makedirs('data/captures', exist_ok=True)

    def __enter__(self):
        """Context manager entry method"""
        self.open_cameras()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method"""
        self.close_cameras()
        return False  # Don't suppress exceptions

    def open_cameras(self):
        """Initialize and configure cameras"""
        logger.info(f"Opening cameras (Left: {self.left_cam_idx}, Right: {self.right_cam_idx})")

        # Close any existing cameras first
        self.close_cameras()

        try:
            self.left_cam = cv2.VideoCapture(self.left_cam_idx)
            self.right_cam = cv2.VideoCapture(self.right_cam_idx)

            if not self.left_cam.isOpened() or not self.right_cam.isOpened():
                logger.error("Failed to open one or both cameras")
                self.close_cameras()
                raise RuntimeError("Failed to open one or both cameras")

            # Set camera parameters
            for cam in [self.left_cam, self.right_cam]:
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cam.set(cv2.CAP_PROP_FPS, 30)

            # Allow settings to apply
            time.sleep(0.5)

            logger.info(f"Cameras initialized with resolution {self.width}x{self.height}")
            return True
        except Exception as e:
            logger.error(f"Error opening cameras: {str(e)}")
            self.close_cameras()
            raise

    def close_cameras(self):
        """Release camera resources"""
        if hasattr(self, 'left_cam') and self.left_cam:
            self.left_cam.release()
            self.left_cam = None

        if hasattr(self, 'right_cam') and self.right_cam:
            self.right_cam.release()
            self.right_cam = None

        logger.info("Cameras closed")

    def capture_frames(self):
        """Capture frames from both cameras"""
        if not self.left_cam or not self.right_cam:
            self.open_cameras()

        ret_left, left_frame = self.left_cam.read()
        ret_right, right_frame = self.right_cam.read()

        if not ret_left or not ret_right:
            logger.error("Failed to capture frames from one or both cameras")
            return None, None

        return left_frame, right_frame

    def calibrate(self, checkerboard_size=(7, 6), square_size=0.025, num_samples=20):
        """Run calibration process with specified checkerboard"""
        logger.info(f"Starting calibration with checkerboard size {checkerboard_size} and square size {square_size}m")

        # Set up object points
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size  # Scale to real-world units

        # Arrays to store points
        objpoints = []
        left_imgpoints = []
        right_imgpoints = []

        # Sample collection variables
        frame_count = 0
        last_capture_time = 0
        min_capture_interval = 1.0  # Minimum seconds between captures

        try:
            while frame_count < num_samples:
                # Capture and preprocess frames
                left_frame, right_frame = self.capture_frames()
                if left_frame is None or right_frame is None:
                    continue

                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                # Apply blur for better detection
                left_gray = cv2.GaussianBlur(left_gray, (5, 5), 0)
                right_gray = cv2.GaussianBlur(right_gray, (5, 5), 0)

                # Find checkerboard corners
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                found_left, left_corners = cv2.findChessboardCorners(left_gray, checkerboard_size, flags)
                found_right, right_corners = cv2.findChessboardCorners(right_gray, checkerboard_size, flags)

                current_time = time.time()
                if found_left and found_right and (current_time - last_capture_time > min_capture_interval):
                    # Refine corner positions
                    left_corners = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), criteria)
                    right_corners = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)

                    # Store points
                    objpoints.append(objp)
                    left_imgpoints.append(left_corners)
                    right_imgpoints.append(right_corners)

                    # Save images
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    left_path = f'data/calibration/left_{timestamp}.jpg'
                    right_path = f'data/calibration/right_{timestamp}.jpg'

                    # Draw and save with corners
                    left_display = left_frame.copy()
                    right_display = right_frame.copy()
                    cv2.drawChessboardCorners(left_display, checkerboard_size, left_corners, found_left)
                    cv2.drawChessboardCorners(right_display, checkerboard_size, right_corners, found_right)

                    cv2.imwrite(left_path, left_display)
                    cv2.imwrite(right_path, right_display)

                    frame_count += 1
                    last_capture_time = current_time

                    logger.info(f"Captured calibration frame pair {frame_count}/{num_samples}")

                # Prevent CPU overuse
                time.sleep(0.1)

            # Verify sufficient samples
            if frame_count < 10:
                logger.warning("Not enough calibration pairs collected (minimum 10 needed)")
                return False

            # Run stereo calibration
            logger.info(f"Starting stereo calibration with {frame_count} image pairs...")
            image_size = left_gray.shape[::-1]  # (width, height)

            ret, self.camera_matrix_left, self.dist_coeffs_left, \
                self.camera_matrix_right, self.dist_coeffs_right, \
                self.R, self.T, E, F = cv2.stereoCalibrate(
                objpoints, left_imgpoints, right_imgpoints,
                None, None, None, None, image_size,
                criteria=criteria, flags=0
            )

            logger.info(f"Stereo calibration completed with RMS error: {ret:.6f}")

            # Compute rectification parameters
            self.R1, self.R2, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(
                self.camera_matrix_left, self.dist_coeffs_left,
                self.camera_matrix_right, self.dist_coeffs_right,
                image_size, self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
            )

            # Initialize rectification maps
            self._init_rectification_maps(image_size)

            # Save calibration data
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

            # Save main calibration file
            np.save('data/calibration/stereo_calibration.npy', calibration_data)

            # Save backup with timestamp
            backup_file = f'data/calibration/stereo_calibration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npy'
            np.save(backup_file, calibration_data)

            logger.info(f"Saved calibration data to stereo_calibration.npy and backup")
            return True

        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            return False

    def load_calibration(self, file_path='data/calibration/stereo_calibration.npy'):
        """Load calibration parameters from file"""
        try:
            logger.info(f"Loading calibration from {file_path}")
            calibration_data = np.load(file_path, allow_pickle=True).item()

            # Load all parameters
            param_keys = [
                'camera_matrix_left', 'dist_coeffs_left',
                'camera_matrix_right', 'dist_coeffs_right',
                'R', 'T', 'Q', 'R1', 'R2', 'P1', 'P2'
            ]

            for key in param_keys:
                if key in calibration_data:
                    data = calibration_data[key]
                    if isinstance(data, list):
                        setattr(self, key, np.array(data, dtype=np.float32))
                    else:
                        setattr(self, key, data.astype(np.float32))

            # Initialize rectification maps if available
            if not self.maps_initialized and 'image_size' in calibration_data:
                self._init_rectification_maps(calibration_data['image_size'])

            logger.info("Calibration loaded successfully")
            return True

        except FileNotFoundError:
            logger.error(f"Calibration file not found at {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading calibration: {str(e)}")
            return False

    def _init_rectification_maps(self, image_size):
        """Initialize rectification maps for efficient remapping"""
        required_params = [
            self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1,
            self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2
        ]

        if not all(required_params):
            logger.error("Cannot initialize rectification maps - missing calibration parameters")
            return False

        try:
            w, h = image_size
            logger.info(f"Initializing rectification maps for {w}x{h} images")

            self.mapL1, self.mapL2 = cv2.initUndistortRectifyMap(
                self.camera_matrix_left, self.dist_coeffs_left,
                self.R1, self.P1, (w, h), cv2.CV_32FC1
            )

            self.mapR1, self.mapR2 = cv2.initUndistortRectifyMap(
                self.camera_matrix_right, self.dist_coeffs_right,
                self.R2, self.P2, (w, h), cv2.CV_32FC1
            )

            self.maps_initialized = True
            logger.info("Rectification maps initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize rectification maps: {str(e)}")
            return False

    def get_rectified_images(self, left_img, right_img):
        """Apply rectification to stereo image pair"""
        if not self.maps_initialized:
            # Try to initialize maps if possible
            h, w = left_img.shape[:2]
            params_available = all([
                self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1,
                self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2
            ])

            if params_available:
                self._init_rectification_maps((w, h))
            else:
                logger.warning("Rectification maps not initialized - returning original images")
                return left_img, right_img

        if not self.maps_initialized:
            return left_img, right_img

        # Apply rectification
        left_rect = cv2.remap(left_img, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.mapR1, self.mapR2, cv2.INTER_LINEAR)

        return left_rect, right_rect

    def compute_disparity_map(self, left_img, right_img):
        """Calculate disparity map using SGBM algorithm"""
        # Convert to grayscale if needed
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img

        # Configure SGBM matcher
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

        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Create colormap for visualization
        disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255,
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

        return disparity, disp_color

    def analyze_disparity(self, disparity):
        """Analyze disparity map to detect objects and calculate distances.
        Returns tuple: (is_object, distance_object, grid_distances)
        """
        if self.Q is None or self.T is None or self.P1 is None:
            logger.error("Cannot analyze disparity - missing calibration parameters")
            return False, 0, [1000] * 16

        # Create valid disparity mask
        valid_mask = (disparity > 0)

        # Default values
        is_object = False
        distance_object = 1000  # cm
        grid_distances = [1000] * 16  # 16 grid squares

        # Check for sufficient valid points
        if np.sum(valid_mask) < 100:
            logger.debug("Not enough valid disparity points for analysis")
            return is_object, distance_object, grid_distances

        # Image dimensions
        h, w = disparity.shape

        # Extract calibration parameters for distance calculation
        baseline = abs(self.T[0][0])  # Baseline in meters
        focal_length = self.P1[0, 0]  # Focal length in pixels

        # Analyze center region
        center_y, center_x = h // 2, w // 2
        center_size = min(h, w) // 4
        center_region = disparity[
                        center_y - center_size:center_y + center_size,
                        center_x - center_size:center_x + center_size
                        ]
        center_mask = valid_mask[
                      center_y - center_size:center_y + center_size,
                      center_x - center_size:center_x + center_size
                      ]

        # Check for object in center
        if np.sum(center_mask) > 10:
            valid_disparities = center_region[center_mask]
            avg_disparity = np.mean(valid_disparities)

            if avg_disparity > 0:
                # Distance in cm
                distance = int((baseline * focal_length / avg_disparity) * 100)
                distance_object = min(max(distance, 10), 1000)  # Clip between 10cm and 10m
                is_object = distance < 200  # Object closer than 2 meters

        # Calculate grid distances (4x4 grid = 16 regions)
        cell_h, cell_w = h // 4, w // 4
        grid_distances = []

        for i in range(4):
            for j in range(4):
                # Extract grid cell
                cell = disparity[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                cell_mask = valid_mask[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]

                if np.sum(cell_mask) > 10:
                    valid_disp = cell[cell_mask]
                    avg_disp = np.mean(valid_disp)

                    if avg_disp > 0:
                        dist = int((baseline * focal_length / avg_disp) * 100)
                        grid_distances.append(min(max(dist, 10), 1000))
                    else:
                        grid_distances.append(1000)
                else:
                    grid_distances.append(1000)

        return is_object, distance_object, grid_distances

    def set_sgbm_params(self, params):
        """Update SGBM parameters"""
        for key, value in params.items():
            if key in self.sgbm_params:
                self.sgbm_params[key] = value

        # Ensure num_disp is multiple of 16
        if 'num_disp' in params:
            self.sgbm_params['num_disp'] = (self.sgbm_params['num_disp'] // 16) * 16
            if self.sgbm_params['num_disp'] < 16:
                self.sgbm_params['num_disp'] = 16