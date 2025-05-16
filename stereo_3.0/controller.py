import base64
import logging
import os
import threading
import time

import cv2
import numpy as np

from stereo_vision import StereoVision

logger = logging.getLogger(__name__)


class VisionController:
    def __init__(self):
        self.stereo_vision = None
        self.lock = threading.RLock()
        self.is_running = False
        self.worker_thread = None

        # Vision data storage
        self.last_update_time = 0
        self.vision_data = {
            "is_object": False,
            "distance_object": 0,
            "objs": [1000] * 16,
            "final": False
        }

        # Frame storage for streaming
        self.current_frames = {
            "left": None,
            "right": None,
            "disparity": None,
            "timestamp": 0
        }

        # Create needed directories
        os.makedirs('data/calibration', exist_ok=True)
        os.makedirs('data/captures', exist_ok=True)

    def initialize(self, left_cam_idx=0, right_cam_idx=1, width=640, height=480):
        """Initialize stereo vision system with specified parameters"""
        with self.lock:
            # Stop any running processing
            if self.is_running:
                self.stop_processing()

            # Create new StereoVision instance
            try:
                self.stereo_vision = StereoVision(
                    left_cam_idx=left_cam_idx,
                    right_cam_idx=right_cam_idx,
                    width=width,
                    height=height
                )
                self.stereo_vision.open_cameras()

                # Try to load calibration
                calibration_loaded = self.stereo_vision.load_calibration()
                logger.info(f"Calibration {'loaded successfully' if calibration_loaded else 'not found'}")

                return True
            except Exception as e:
                logger.error(f"Vision initialization error: {str(e)}")
                if self.stereo_vision:
                    self.stereo_vision.close_cameras()
                    self.stereo_vision = None
                return False

    def start_processing(self):
        """Start the vision processing thread"""
        with self.lock:
            if self.is_running:
                logger.info("Processing already running")
                return False

            if not self.stereo_vision:
                logger.error("Cannot start processing - stereo vision not initialized")
                return False

            self.is_running = True

            # Start worker thread
            self.worker_thread = threading.Thread(target=self._process_frames)
            self.worker_thread.daemon = True
            self.worker_thread.start()

            logger.info("Vision processing started")
            return True

    def stop_processing(self):
        """Stop the vision processing thread"""
        with self.lock:
            if not self.is_running:
                logger.info("Processing already stopped")
                return False

            # Signal thread to stop
            self.is_running = False

            # Wait for thread to terminate
            if self.worker_thread:
                self.worker_thread.join(timeout=2.0)
                self.worker_thread = None

            # Close camera resources
            if self.stereo_vision:
                self.stereo_vision.close_cameras()

            logger.info("Vision processing stopped")
            return True

    def _process_frames(self):
        """Main processing loop (runs in worker thread)"""
        if not self.stereo_vision:
            logger.error("StereoVision not initialized")
            with self.lock:
                self.is_running = False
            return

        logger.info("Vision processing thread started")

        try:
            # Use context manager for safe camera handling
            with self.stereo_vision:
                # Check calibration status
                has_calibration = self.stereo_vision.maps_initialized or self.stereo_vision.load_calibration()

                # Processing loop
                while True:
                    # Check if we should continue running
                    with self.lock:
                        if not self.is_running:
                            break

                    try:
                        # Capture frames
                        left_frame, right_frame = self.stereo_vision.capture_frames()

                        if left_frame is None or right_frame is None:
                            logger.warning("Failed to capture frames")
                            time.sleep(0.1)
                            continue

                        # Update current timestamp
                        current_time = time.time()

                        # Process images if calibration is available
                        if has_calibration:
                            # Rectify images
                            left_rect, right_rect = self.stereo_vision.get_rectified_images(left_frame, right_frame)

                            # Compute disparity map
                            disparity, disparity_color = self.stereo_vision.compute_disparity_map(left_rect, right_rect)

                            # Analyze disparity for object detection
                            is_object, distance_object, grid_distances = self.stereo_vision.analyze_disparity(disparity)

                            # Update vision data (thread-safe)
                            with self.lock:
                                self.vision_data = {
                                    "is_object": is_object,
                                    "distance_object": distance_object,
                                    "objs": grid_distances,
                                    "final": is_object  # Consider using same value for now
                                }
                                self.last_update_time = current_time

                                # Store current frames for streaming
                                self.current_frames = {
                                    "left": left_rect,
                                    "right": right_rect,
                                    "disparity": disparity_color,
                                    "timestamp": current_time
                                }
                        else:
                            # Store raw frames if no calibration
                            with self.lock:
                                self.current_frames = {
                                    "left": left_frame,
                                    "right": right_frame,
                                    "disparity": None,
                                    "timestamp": current_time
                                }

                        # Control frame rate to avoid CPU overuse
                        time.sleep(0.03)  # ~30 FPS target

                    except Exception as e:
                        logger.error(f"Frame processing error: {str(e)}")
                        time.sleep(0.5)  # Slower retry on error

        except Exception as e:
            logger.error(f"Vision thread exception: {str(e)}")

        finally:
            # Ensure state is updated when thread exits
            with self.lock:
                self.is_running = False
            logger.info("Vision processing thread stopped")

    def get_vision_data(self):
        """Get current vision data for TurtleBot integration"""
        with self.lock:
            current_time = time.time()
            result = self.vision_data.copy()
            return result

    def get_encoded_frames(self, quality=80, full_size=True):
        """Get current frames as base64 encoded JPEG images"""
        with self.lock:
            frames = self.current_frames

            # Важлива перевірка на наявність кадрів
            if frames["left"] is None:
                return None

            result = {}

            for key in ["left", "right", "disparity"]:
                if frames[key] is not None:
                    try:
                        # Apply scaling if not full size
                        frame = frames[key]
                        if not full_size:
                            h, w = frame.shape[:2]
                            new_w = int(w * 0.5)
                            new_h = int(h * 0.5)
                            frame = cv2.resize(frame, (new_w, new_h))

                        # Encode to JPEG
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                        result[key] = base64.b64encode(buffer).decode('utf-8')
                    except Exception as e:
                        logger.error(f"Error encoding {key} frame: {str(e)}")

            result["timestamp"] = frames["timestamp"]
            return result

    def capture_and_save(self):
        """Capture and save current frames"""
        with self.lock:
            if not self.stereo_vision:
                return None

        try:
            # Use context manager for safe camera handling
            with self.stereo_vision:
                left_frame, right_frame = self.stereo_vision.capture_frames()
                if left_frame is None or right_frame is None:
                    logger.error("Failed to capture frames")
                    return None

                # Generate timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                # Save raw captures
                left_path = f"data/captures/left_{timestamp}.jpg"
                right_path = f"data/captures/right_{timestamp}.jpg"

                cv2.imwrite(left_path, left_frame)
                cv2.imwrite(right_path, right_frame)

                # Result object
                result = {
                    "timestamp": timestamp,
                    "left_path": left_path,
                    "right_path": right_path
                }

                # Process and save disparity if calibration available
                try:
                    has_calibration = self.stereo_vision.load_calibration()
                    if has_calibration:
                        # Get rectified images
                        left_rect, right_rect = self.stereo_vision.get_rectified_images(left_frame, right_frame)

                        # Compute disparity
                        _, disparity_color = self.stereo_vision.compute_disparity_map(left_rect, right_rect)

                        # Save disparity image
                        disparity_path = f"data/captures/disparity_{timestamp}.jpg"
                        cv2.imwrite(disparity_path, disparity_color)

                        # Add to result
                        result["disparity_path"] = disparity_path
                        result["has_calibration"] = True
                    else:
                        result["has_calibration"] = False
                except Exception as e:
                    logger.error(f"Error processing disparity: {str(e)}")
                    result["has_calibration"] = False

                return result

        except Exception as e:
            logger.error(f"Error in capture and save: {str(e)}")
            return None

    def run_calibration(self, checkerboard_size=(7, 6), square_size=0.025, num_samples=20):
        """Run the calibration process"""
        with self.lock:
            if self.is_running:
                logger.error("Cannot calibrate while processing is running")
                return False, "Stop vision processing before calibration"

            if not self.stereo_vision:
                self.stereo_vision = StereoVision()

            try:
                result = self.stereo_vision.calibrate(
                    checkerboard_size=checkerboard_size,
                    square_size=square_size,
                    num_samples=num_samples
                )

                if result:
                    # Get calibration information
                    try:
                        calibration_data = np.load('data/calibration/stereo_calibration.npy', allow_pickle=True).item()
                        info = {
                            'date': calibration_data.get('calibration_date', 'Unknown'),
                            'frame_count': calibration_data.get('frame_count', 0),
                            'rms_error': calibration_data.get('rms_error', 0.0)
                        }
                        return True, info
                    except Exception as e:
                        logger.error(f"Error reading calibration info: {str(e)}")
                        return True, None
                else:
                    return False, "Calibration failed"

            except Exception as e:
                logger.error(f"Calibration error: {str(e)}")
                return False, str(e)

    def get_calibration_status(self):
        """Check if calibration file exists and load info"""
        calibration_path = 'data/calibration/stereo_calibration.npy'

        if not os.path.exists(calibration_path):
            return {
                'has_calibration': False,
                'calibration_info': None
            }

        try:
            calibration_data = np.load(calibration_path, allow_pickle=True).item()
            info = {
                'date': calibration_data.get('calibration_date', 'Unknown'),
                'frame_count': calibration_data.get('frame_count', 0),
                'rms_error': calibration_data.get('rms_error', 0.0),
                'image_size': calibration_data.get('image_size', [0, 0])
            }
            return {
                'has_calibration': True,
                'calibration_info': info
            }
        except Exception as e:
            logger.error(f"Error loading calibration info: {str(e)}")
            return {
                'has_calibration': True,
                'calibration_info': {'error': str(e)}
            }

    def detect_checkerboard(self, checkerboard_size=(7, 6)):
        """Test checkerboard detection with current camera feed"""
        with self.lock:
            if not self.stereo_vision:
                return False, "Stereo vision not initialized", None, None

        try:
            with self.stereo_vision:
                # Capture frames
                left_frame, right_frame = self.stereo_vision.capture_frames()
                if left_frame is None or right_frame is None:
                    return False, "Failed to capture frames", None, None

                # Convert to grayscale
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                # Apply preprocessing
                left_gray = cv2.GaussianBlur(left_gray, (5, 5), 0)
                right_gray = cv2.GaussianBlur(right_gray, (5, 5), 0)

                # Check for checkerboard
                pattern_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

                found_left, left_corners = cv2.findChessboardCorners(left_gray, checkerboard_size, pattern_flags)
                found_right, right_corners = cv2.findChessboardCorners(right_gray, checkerboard_size, pattern_flags)

                # Create display images
                left_display = left_frame.copy()
                right_display = right_frame.copy()

                # Draw corners if found
                if found_left:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    left_corners = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(left_display, checkerboard_size, left_corners, found_left)

                if found_right:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    right_corners = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(right_display, checkerboard_size, right_corners, found_right)

                # Encode display images
                _, left_buffer = cv2.imencode('.jpg', left_display, [cv2.IMWRITE_JPEG_QUALITY, 80])
                _, right_buffer = cv2.imencode('.jpg', right_display, [cv2.IMWRITE_JPEG_QUALITY, 80])

                left_b64 = base64.b64encode(left_buffer).decode('utf-8')
                right_b64 = base64.b64encode(right_buffer).decode('utf-8')

                return True, "Checkerboard detection complete", {
                    'left_found': found_left,
                    'right_found': found_right,
                    'both_found': found_left and found_right,
                    'left_image': left_b64,
                    'right_image': right_b64,
                    'checkerboard_size': checkerboard_size
                }, None

        except cv2.error as e:
            logger.error(f"OpenCV error in checkerboard detection: {str(e)}")
            return False, f"Camera error: {str(e)}", None, None
        except Exception as e:
            logger.error(f"Error in checkerboard detection: {str(e)}")
            return False, str(e), None, None

    def get_config(self):
        """Get current configuration"""
        with self.lock:
            if not self.stereo_vision:
                # Return default config
                return {
                    "left_cam_idx": 0,
                    "right_cam_idx": 1,
                    "width": 640,
                    "height": 480,
                    "sgbm_params": {
                        "window_size": 11,
                        "min_disp": 0,
                        "num_disp": 128,
                        "uniqueness_ratio": 15,
                        "speckle_window_size": 100,
                        "speckle_range": 32
                    }
                }

            # Return actual config
            return {
                "left_cam_idx": self.stereo_vision.left_cam_idx,
                "right_cam_idx": self.stereo_vision.right_cam_idx,
                "width": self.stereo_vision.width,
                "height": self.stereo_vision.height,
                "sgbm_params": self.stereo_vision.sgbm_params.copy()
            }

    def update_config(self, config):
        """Update configuration parameters"""
        with self.lock:
            if not self.stereo_vision:
                self.stereo_vision = StereoVision()

            needs_reinit = False

            # Update camera indices if provided
            if 'left_cam_idx' in config and 'right_cam_idx' in config:
                left_idx = int(config['left_cam_idx'])
                right_idx = int(config['right_cam_idx'])

                if left_idx == right_idx:
                    return False, "Left and right camera indices must be different"

                if left_idx != self.stereo_vision.left_cam_idx or right_idx != self.stereo_vision.right_cam_idx:
                    self.stereo_vision.left_cam_idx = left_idx
                    self.stereo_vision.right_cam_idx = right_idx
                    needs_reinit = True

            # Update resolution if provided
            if 'width' in config and 'height' in config:
                width = int(config['width'])
                height = int(config['height'])

                if width != self.stereo_vision.width or height != self.stereo_vision.height:
                    self.stereo_vision.width = width
                    self.stereo_vision.height = height
                    needs_reinit = True

            # Update SGBM parameters if provided
            if 'sgbm_params' in config and isinstance(config['sgbm_params'], dict):
                self.stereo_vision.set_sgbm_params(config['sgbm_params'])

            # Reinitialize cameras if needed
            if needs_reinit and self.is_running:
                # Stop and restart processing
                was_running = self.is_running
                self.stop_processing()
                self.initialize(
                    left_cam_idx=self.stereo_vision.left_cam_idx,
                    right_cam_idx=self.stereo_vision.right_cam_idx,
                    width=self.stereo_vision.width,
                    height=self.stereo_vision.height
                )
                if was_running:
                    self.start_processing()

            return True, "Configuration updated"