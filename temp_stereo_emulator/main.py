import base64
import cv2
import flask
import json
import logging
import numpy as np
import os
import random
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify
from logging.handlers import RotatingFileHandler
from waitress import serve

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            "vision_service_emulator.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create required directories
os.makedirs('data/calibration', exist_ok=True)
os.makedirs('data/captures', exist_ok=True)


class StereoVisionEmulator:
    """Emulates the StereoVision class with mock data"""

    def __init__(self, left_cam_idx=0, right_cam_idx=1, width=640, height=480):
        self.left_cam_idx = left_cam_idx
        self.right_cam_idx = right_cam_idx
        self.width = width
        self.height = height

        # Generate mock calibration data
        self.maps_initialized = False
        self.camera_matrix_left = np.array([[500, 0, width / 2], [0, 500, height / 2], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs_left = np.zeros((1, 5), dtype=np.float32)
        self.camera_matrix_right = np.array([[500, 0, width / 2], [0, 500, height / 2], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs_right = np.zeros((1, 5), dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.T = np.array([[0.1], [0], [0]], dtype=np.float32)  # 10cm baseline
        self.Q = np.array([
            [1, 0, 0, -width / 2],
            [0, 1, 0, -height / 2],
            [0, 0, 0, 500],
            [0, 0, 10, 0]
        ], dtype=np.float32)
        self.R1 = self.R.copy()
        self.R2 = self.R.copy()
        self.P1 = np.array([[500, 0, width / 2, 0], [0, 500, height / 2, 0], [0, 0, 1, 0]], dtype=np.float32)
        self.P2 = np.array([[500, 0, width / 2, -50], [0, 500, height / 2, 0], [0, 0, 1, 0]], dtype=np.float32)

        # SGBM parameters
        self.sgbm_params = {
            'window_size': 11,
            'min_disp': 0,
            'num_disp': 128,
            'uniqueness_ratio': 15,
            'speckle_window_size': 100,
            'speckle_range': 32
        }

        # Mock objects in the scene - [(x, y, distance_cm), ...]
        self.mock_objects = [
            (width // 2, height // 2, 80),  # Center object
            (width // 4, height // 4, 150),  # Upper left object
            (3 * width // 4, 3 * height // 4, 120)  # Lower right object
        ]

        # Save mock calibration file if it doesn't exist
        self.save_mock_calibration()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def save_mock_calibration(self):
        """Save mock calibration data to file"""
        calibration_file = 'data/calibration/stereo_calibration.npy'
        if not os.path.exists(calibration_file):
            # Create mock calibration data
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
                'image_size': (self.width, self.height),
                'calibration_date': timestamp,
                'rms_error': 0.23,  # Mock RMS error
                'frame_count': 20
            }
            np.save(calibration_file, calibration_data)
            logger.info(f"Created mock calibration file: {calibration_file}")
            self.maps_initialized = True

    def open_cameras(self):
        """Simulate opening cameras"""
        logger.info(f"Opening emulated cameras (Left: {self.left_cam_idx}, Right: {self.right_cam_idx})")
        # Nothing to do in emulation mode
        return True

    def close_cameras(self):
        """Simulate closing cameras"""
        logger.info("Closing emulated cameras")
        # Nothing to do in emulation mode

    def load_calibration(self, file_path='data/calibration/stereo_calibration.npy'):
        """Load calibration or create mock data if not found"""
        if os.path.exists(file_path):
            try:
                calibration_data = np.load(file_path, allow_pickle=True).item()
                logger.info("Loaded calibration data from file")
                self.maps_initialized = True
                return True
            except Exception as e:
                logger.error(f"Error loading calibration: {str(e)}")

        # Create mock calibration if not found
        self.save_mock_calibration()
        return True

    def generate_mock_frame(self, is_left=True, add_text=True):
        """Generate a mock frame with synthetic content"""
        # Create a base frame with gradient background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Add gradient background
        for y in range(self.height):
            for x in range(self.width):
                frame[y, x, 0] = 50 + (x * 150) // self.width  # Blue channel
                frame[y, x, 1] = 50 + (y * 150) // self.height  # Green channel
                frame[y, x, 2] = 100  # Red channel

        # Add a moving element (circle) to make frames dynamic
        circle_x = int(self.width / 2 + self.width / 4 * np.sin(time.time()))
        circle_y = int(self.height / 2 + self.height / 4 * np.cos(time.time()))
        cv2.circle(frame, (circle_x, circle_y), 30, (0, 0, 255), -1)

        # Add mock objects with displacement in right camera to simulate disparity
        for obj_x, obj_y, distance in self.mock_objects:
            # Calculate disparity based on distance (larger disparity for closer objects)
            # Baseline * focal_length / distance = disparity
            disparity = int(0.1 * 500 / (distance / 100))  # 10cm baseline, 500px focal length

            # Draw object with appropriate disparity
            if is_left:
                cv2.rectangle(frame, (obj_x - 20, obj_y - 20), (obj_x + 20, obj_y + 20), (0, 255, 0), -1)
            else:
                # Shift horizontally in right camera by disparity amount
                shifted_x = obj_x - disparity
                cv2.rectangle(frame, (shifted_x - 20, obj_y - 20), (shifted_x + 20, obj_y + 20), (0, 255, 0), -1)

        # Add text label to indicate which camera
        if add_text:
            label = "LEFT" if is_left else "RIGHT"
            cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(frame, timestamp, (20, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def capture_frames(self):
        """Generate mock frames for both cameras"""
        left_frame = self.generate_mock_frame(is_left=True)
        right_frame = self.generate_mock_frame(is_left=False)
        return left_frame, right_frame

    def get_rectified_images(self, left_img, right_img):
        """Simulate rectification"""
        # For simulation, we'll assume the generated images are already rectified
        return left_img, right_img

    def compute_disparity_map(self, left_img, right_img):
        """Generate a mock disparity map"""
        # Create a base disparity map
        disparity = np.zeros((self.height, self.width), dtype=np.float32)

        # Convert to grayscale
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img

        # Add disparity for each mock object
        for obj_x, obj_y, distance in self.mock_objects:
            # Calculate disparity based on distance
            disp_value = int(0.1 * 500 / (distance / 100))  # 10cm baseline, 500px focal length

            # Create a circular region of disparity
            radius = 40
            for y in range(max(0, obj_y - radius), min(self.height, obj_y + radius)):
                for x in range(max(0, obj_x - radius), min(self.width, obj_x + radius)):
                    # Calculate distance from center of object
                    dx = x - obj_x
                    dy = y - obj_y
                    dist = np.sqrt(dx ** 2 + dy ** 2)

                    if dist < radius:
                        # Fade disparity at edges
                        factor = 1 - (dist / radius)
                        disparity[y, x] = disp_value * factor

        # Add noise
        noise = np.random.normal(0, 0.5, disparity.shape).astype(np.float32)
        disparity += noise

        # Create colored disparity visualization
        disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

        return disparity, disp_color

    def analyze_disparity(self, disparity):
        """Analyze mock disparity map"""
        # Mock values for TurtleBot integration
        center_object = self.mock_objects[0]  # Use the center object as reference
        is_object = center_object[2] < 100  # Object detected if closer than 1m
        distance_object = center_object[2]

        # Generate grid distances (4x4 grid)
        grid_distances = []
        cell_h, cell_w = self.height // 4, self.width // 4

        for i in range(4):
            for j in range(4):
                cell_center_x = j * cell_w + cell_w // 2
                cell_center_y = i * cell_h + cell_h // 2

                # Find the nearest object to this cell
                nearest_dist = 1000
                for obj_x, obj_y, dist in self.mock_objects:
                    dx = obj_x - cell_center_x
                    dy = obj_y - cell_center_y
                    cell_obj_dist = np.sqrt(dx ** 2 + dy ** 2)

                    if cell_obj_dist < 100:  # If object is near this cell
                        nearest_dist = min(nearest_dist, dist)

                grid_distances.append(nearest_dist)

        return is_object, distance_object, grid_distances

    def calibrate(self, checkerboard_size=(7, 6), square_size=0.025, num_samples=20):
        """Simulate calibration process"""
        logger.info(f"Simulating calibration with checkerboard size {checkerboard_size}")
        time.sleep(2)  # Simulate processing time

        # Create mock calibration data
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
            'image_size': (self.width, self.height),
            'calibration_date': timestamp,
            'rms_error': 0.23,  # Mock RMS error
            'frame_count': num_samples
        }

        # Save calibration data
        np.save('data/calibration/stereo_calibration.npy', calibration_data)
        self.maps_initialized = True

        return True

    def set_sgbm_params(self, params):
        """Update SGBM parameters"""
        for key, value in params.items():
            if key in self.sgbm_params:
                self.sgbm_params[key] = value


class VisionControllerEmulator:
    """Emulates the VisionController class"""

    def __init__(self):
        self.stereo_vision = None
        self.lock = threading.RLock()
        self.is_running = False
        self.worker_thread = None

        # Mock data
        self.last_update_time = 0
        self.vision_data = {
            "is_object": False,
            "distance_object": 0,
            "objs": [1000] * 16,
            "final": False
        }

        # Frame storage
        self.current_frames = {
            "left": None,
            "right": None,
            "disparity": None,
            "timestamp": 0
        }

        # Create directories
        os.makedirs('data/calibration', exist_ok=True)
        os.makedirs('data/captures', exist_ok=True)

    def initialize(self, left_cam_idx=0, right_cam_idx=1, width=640, height=480):
        """Initialize with mock stereo vision"""
        with self.lock:
            if self.is_running:
                self.stop_processing()

            try:
                self.stereo_vision = StereoVisionEmulator(
                    left_cam_idx=left_cam_idx,
                    right_cam_idx=right_cam_idx,
                    width=width,
                    height=height
                )

                # Load calibration
                calibration_loaded = self.stereo_vision.load_calibration()
                logger.info(f"Calibration {'loaded successfully' if calibration_loaded else 'not found'}")

                return True
            except Exception as e:
                logger.error(f"Vision initialization error: {str(e)}")
                self.stereo_vision = None
                return False

    def start_processing(self):
        """Start mock processing"""
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
        """Stop mock processing"""
        with self.lock:
            if not self.is_running:
                logger.info("Processing already stopped")
                return False

            self.is_running = False

            if self.worker_thread:
                self.worker_thread.join(timeout=2.0)
                self.worker_thread = None

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
            has_calibration = self.stereo_vision.maps_initialized or self.stereo_vision.load_calibration()

            while True:
                with self.lock:
                    if not self.is_running:
                        break

                try:
                    # Generate mock frames
                    left_frame, right_frame = self.stereo_vision.capture_frames()
                    current_time = time.time()

                    if has_calibration:
                        # Process mock images
                        left_rect, right_rect = self.stereo_vision.get_rectified_images(left_frame, right_frame)
                        disparity, disparity_color = self.stereo_vision.compute_disparity_map(left_rect, right_rect)
                        is_object, distance_object, grid_distances = self.stereo_vision.analyze_disparity(disparity)

                        # Update data
                        with self.lock:
                            self.vision_data = {
                                "is_object": is_object,
                                "distance_object": distance_object,
                                "objs": grid_distances,
                                "final": is_object
                            }
                            self.last_update_time = current_time

                            self.current_frames = {
                                "left": left_rect,
                                "right": right_rect,
                                "disparity": disparity_color,
                                "timestamp": current_time
                            }
                    else:
                        with self.lock:
                            self.current_frames = {
                                "left": left_frame,
                                "right": right_frame,
                                "disparity": None,
                                "timestamp": current_time
                            }

                    # Control frame rate
                    time.sleep(0.1)  # 10 FPS to reduce CPU usage in emulation

                except Exception as e:
                    logger.error(f"Frame processing error: {str(e)}")
                    time.sleep(0.5)

        except Exception as e:
            logger.error(f"Vision thread exception: {str(e)}")

        finally:
            with self.lock:
                self.is_running = False
            logger.info("Vision processing thread stopped")

    def get_vision_data(self):
        """Get current vision data"""
        with self.lock:
            # Randomly vary the data slightly to simulate real-time changes
            result = self.vision_data.copy()

            # Make object randomly appear/disappear
            if random.random() < 0.05:  # 5% chance to change
                result["is_object"] = not result["is_object"]

            # Vary distance slightly
            if result["distance_object"] > 0:
                result["distance_object"] = max(10, result["distance_object"] + random.randint(-5, 5))

            # Vary grid distances
            result["objs"] = [max(10, d + random.randint(-10, 10)) for d in result["objs"]]

            # Update final status
            result["final"] = result["is_object"]

            return result

    def get_encoded_frames(self, quality=80, full_size=True):
        """Get base64 encoded frames"""
        with self.lock:
            # If no frames exist yet, generate them
            if self.current_frames["left"] is None:
                if self.stereo_vision:
                    left_frame, right_frame = self.stereo_vision.capture_frames()
                    disparity, disparity_color = self.stereo_vision.compute_disparity_map(left_frame, right_frame)
                    self.current_frames = {
                        "left": left_frame,
                        "right": right_frame,
                        "disparity": disparity_color,
                        "timestamp": time.time()
                    }
                else:
                    return None

            frames = self.current_frames
            result = {}

            for key in ["left", "right", "disparity"]:
                if frames[key] is not None:
                    try:
                        frame = frames[key]
                        if not full_size:
                            h, w = frame.shape[:2]
                            new_w = int(w * 0.5)
                            new_h = int(h * 0.5)
                            frame = cv2.resize(frame, (new_w, new_h))

                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                        result[key] = base64.b64encode(buffer).decode('utf-8')
                    except Exception as e:
                        logger.error(f"Error encoding {key} frame: {str(e)}")

            result["timestamp"] = frames["timestamp"]
            return result

    def capture_and_save(self):
        """Capture and save frames"""
        try:
            if not self.stereo_vision:
                return None

            left_frame, right_frame = self.stereo_vision.capture_frames()

            # Generate timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Save captures
            left_path = f"data/captures/left_{timestamp}.jpg"
            right_path = f"data/captures/right_{timestamp}.jpg"

            cv2.imwrite(left_path, left_frame)
            cv2.imwrite(right_path, right_frame)

            result = {
                "timestamp": timestamp,
                "left_path": left_path,
                "right_path": right_path
            }

            # Process disparity if calibration available
            try:
                has_calibration = self.stereo_vision.load_calibration()
                if has_calibration:
                    # Compute disparity
                    _, disparity_color = self.stereo_vision.compute_disparity_map(left_frame, right_frame)

                    # Save disparity image
                    disparity_path = f"data/captures/disparity_{timestamp}.jpg"
                    cv2.imwrite(disparity_path, disparity_color)

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
        """Run mock calibration"""
        with self.lock:
            if self.is_running:
                logger.error("Cannot calibrate while processing is running")
                return False, "Stop vision processing before calibration"

            if not self.stereo_vision:
                self.stereo_vision = StereoVisionEmulator()

            try:
                result = self.stereo_vision.calibrate(
                    checkerboard_size=checkerboard_size,
                    square_size=square_size,
                    num_samples=num_samples
                )

                if result:
                    # Get calibration information
                    calibration_data = np.load('data/calibration/stereo_calibration.npy', allow_pickle=True).item()
                    info = {
                        'date': calibration_data.get('calibration_date', 'Unknown'),
                        'frame_count': calibration_data.get('frame_count', 0),
                        'rms_error': calibration_data.get('rms_error', 0.0)
                    }
                    return True, info
                else:
                    return False, "Calibration failed"

            except Exception as e:
                logger.error(f"Calibration error: {str(e)}")
                return False, str(e)

    def get_calibration_status(self):
        """Check calibration file"""
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
        """Test checkerboard detection"""
        try:
            if not self.stereo_vision:
                return False, "Stereo vision not initialized", None, None

            # Capture frames
            left_frame, right_frame = self.stereo_vision.capture_frames()

            # Generate mock detection results
            found_left = random.random() > 0.3  # 70% chance of finding checkerboard
            found_right = random.random() > 0.3

            # Create display images with checkerboard corners drawn
            left_display = left_frame.copy()
            right_display = right_frame.copy()

            # Draw checkerboard points if found
            if found_left:
                # Draw a fake checkerboard pattern
                for i in range(checkerboard_size[0]):
                    for j in range(checkerboard_size[1]):
                        x = 100 + i * 50
                        y = 100 + j * 50
                        cv2.circle(left_display, (x, y), 5, (0, 255, 0), -1)

            if found_right:
                # Draw a fake checkerboard pattern (shifted for right camera)
                for i in range(checkerboard_size[0]):
                    for j in range(checkerboard_size[1]):
                        x = 100 + i * 50 - 15  # Shift by 15px to simulate disparity
                        y = 100 + j * 50
                        cv2.circle(right_display, (x, y), 5, (0, 255, 0), -1)

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
        """Update configuration"""
        with self.lock:
            if not self.stereo_vision:
                self.stereo_vision = StereoVisionEmulator()

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

            # Reinitialize if needed
            if needs_reinit and self.is_running:
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


# Initialize Flask app
app = Flask(__name__)

# Initialize the controller
vision_controller = VisionControllerEmulator()

# Route debugging helper
@app.route('/debug/routes', methods=['GET'])
def debug_routes():
    """List all registered routes for debugging"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": [method for method in rule.methods if method not in ['HEAD', 'OPTIONS']],
            "path": str(rule)
        })
    return jsonify(routes)

# Define ALL routes directly on the app

@app.route('/health', methods=['GET'])
def app_health_check():
    """Root health check endpoint"""
    return {"status": "healthy"}, 200

@app.route('/api/health', methods=['GET'])
def api_health_check():
    """API health check endpoint"""
    return {"status": "healthy"}, 200

@app.route('/api/vision/initialize', methods=['POST'])
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

@app.route('/api/vision/start', methods=['POST'])
def start_vision():
    """Start the vision processing thread"""
    success = vision_controller.start_processing()
    return jsonify({
        'success': success,
        'message': 'Vision processing started' if success else 'Vision processing already running or not initialized'
    })

@app.route('/api/vision/stop', methods=['POST'])
def stop_vision():
    """Stop the vision processing thread"""
    success = vision_controller.stop_processing()
    return jsonify({
        'success': success,
        'message': 'Vision processing stopped' if success else 'Vision processing not running'
    })

@app.route('/api/vision/data', methods=['GET'])
def get_vision_data():
    """Get current vision analysis data"""
    return jsonify(vision_controller.get_vision_data())

@app.route('/api/turtlebot/vision', methods=['GET'])
def get_turtlebot_data():
    """TurtleBot-specific endpoint with required format"""
    data = vision_controller.get_vision_data()
    # Return only the exact fields needed by TurtleBot
    return jsonify({
        'is_object': data['is_object'],
        'distance_object': data['distance_object'],
        'objs': data['objs'],
        'final': data['final']
    })

@app.route('/api/frames', methods=['GET'])
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

@app.route('/api/capture', methods=['POST'])
def capture_frame():
    """Capture and save current frames"""
    result = vision_controller.capture_and_save()

    if not result:
        return jsonify({'success': False, 'message': 'Failed to capture frames'}), 500

    return jsonify({
        'success': True,
        **result
    })

@app.route('/api/calibrate', methods=['POST'])
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

@app.route('/api/calibrate/status', methods=['GET'])
def get_calibration_status():
    """Get current calibration status"""
    status = vision_controller.get_calibration_status()
    return jsonify({
        'success': True,
        **status
    })

@app.route('/api/calibrate/detect', methods=['POST'])
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

@app.route('/api/config', methods=['GET', 'POST'])
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

@app.route('/api/system/info', methods=['GET'])
def get_system_info():
    """Get system information including available cameras"""
    import platform

    # Get system info
    system_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'opencv_version': cv2.__version__
    }

    # Mock available cameras
    available_cameras = [
        {'index': 0, 'resolution': '640x480', 'fps': 30.0},
        {'index': 1, 'resolution': '640x480', 'fps': 30.0},
    ]

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

# Add a catch-all route for debugging
@app.route('/<path:path>', methods=['GET', 'POST'])
def catch_all(path):
    """Catch-all route for debugging"""
    return jsonify({
        'success': False,
        'message': f'Route not found: /{path}',
        'method': request.method
    }), 404

# Main entry point
if __name__ == '__main__':
    logger.info("Starting Vision Service Emulator")

    # Initialize the vision controller with defaults
    vision_controller.initialize()

    # Use waitress for production-grade serving
    serve(app, host='0.0.0.0', port=5050)