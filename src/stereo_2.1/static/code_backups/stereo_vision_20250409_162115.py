import cv2
import numpy as np
import time
import logging
import os
from datetime import datetime
import torch

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
        """Initialize stereo vision system with two Logitech C270 cameras."""
        self.left_cam_idx = left_cam_idx
        self.right_cam_idx = right_cam_idx
        self.width = width
        self.height = height

        # Camera calibration matrices (populated after calibration)
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.R = None
        self.T = None
        self.Q = None

        # Deep learning models
        self.dl_model = None
        self.use_dl = False
        self.dl_model_name = None

        # Configuration
        self.disparity_method = 'sgbm'  # 'sgbm' or 'dl'
        self.sgbm_params = {
            'window_size': 11,
            'min_disp': 0,
            'num_disp': 112,  # Must be multiple of 16
            'uniqueness_ratio': 15,
            'speckle_window_size': 100,
            'speckle_range': 32
        }
        self.dl_params = {
            'max_disp': 256,
            'mixed_precision': True,
            'downscale_factor': 1.0  # 1.0 means no downscaling
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

    def basic_test(self):
        """Basic test to verify both cameras are working properly."""
        self.open_cameras()

        print("Basic stereo camera test starting...")
        print("Press 'c' to capture a stereo pair, 'q' to exit")

        try:
            while True:
                # Capture frames from both cameras
                ret_left, left_frame = self.left_cam.read()
                ret_right, right_frame = self.right_cam.read()

                if not ret_left or not ret_right:
                    print("Failed to capture frames from one or both cameras.")
                    continue

                # Display the frames
                cv2.imshow('Left Camera', left_frame)
                cv2.imshow('Right Camera', right_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Save the captured stereo pair
                    timestamp = int(time.time())
                    cv2.imwrite(f'left_{timestamp}.jpg', left_frame)
                    cv2.imwrite(f'right_{timestamp}.jpg', right_frame)
                    print(f"Saved stereo pair as left_{timestamp}.jpg and right_{timestamp}.jpg")

                    # Display side-by-side comparison
                    stereo_pair = np.hstack((left_frame, right_frame))
                    cv2.imshow('Stereo Pair', stereo_pair)
                    cv2.imwrite(f'stereo_pair_{timestamp}.jpg', stereo_pair)

        finally:
            self.left_cam.release()
            self.right_cam.release()
            cv2.destroyAllWindows()

    def calibrate_cameras(self, checkerboard_size=(7, 3), square_size=0.025, auto_capture=True, stability_seconds=3.0):
        """
        Improved calibration with better checkerboard detection and automatic frame capture.

        Args:
            checkerboard_size: (columns, rows) of inner corners in the checkerboard.
            square_size: Physical size of each checkerboard square in meters.
            auto_capture: If True, automatically capture frames when checkerboard is stable.
            stability_seconds: Number of seconds checkerboard needs to be detected before auto-capturing.
        """
        self.open_cameras()

        # Create output directory for calibration images if it doesn't exist
        os.makedirs('calibration_images', exist_ok=True)
        logger.info("Starting camera calibration process")
        logger.info("Checkerboard size: %dx%d, Square size: %.3f meters",
                    checkerboard_size[0], checkerboard_size[1], square_size)
        logger.info("Auto-capture: %s, Stability time: %.1f seconds",
                    "enabled" if auto_capture else "disabled", stability_seconds)

        # Termination criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        logger.debug("Corner refinement criteria: max_iter=30, epsilon=0.001")

        # Prepare 3D object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size  # Scale to real-world units

        # Arrays to store points
        objpoints = []
        left_imgpoints = []
        right_imgpoints = []

        # Auto-capture variables
        stable_detection_start = None
        last_detection_status = False

        logger.info("\n===== ENHANCED CALIBRATION =====")
        logger.info("Looking for a %dx%d internal corner pattern",
                    checkerboard_size[0], checkerboard_size[1])

        if auto_capture:
            print("\n===== ENHANCED CALIBRATION WITH AUTO-CAPTURE =====")
            print(f"Looking for a {checkerboard_size[0]}x{checkerboard_size[1]} internal corner pattern")
            print(f"Auto-capture will trigger after {stability_seconds} seconds of stable detection")
            print("Press 'c' to force capture a frame, 'd' to debug detection, 'q' when done.")
        else:
            print("\n===== ENHANCED CALIBRATION =====")
            print(f"Looking for a {checkerboard_size[0]}x{checkerboard_size[1]} internal corner pattern")
            print("Press 'c' to capture a frame, 'd' to debug detection, 'q' when done.")

        frame_count = 0
        needed_frames = 20
        checkerboard_detection_count = 0

        # Initialize last_detection_time for auto-capture
        last_detection_time = 0
        stable_since = 0
        is_stable = False

        try:
            while True:
                ret_left, left_frame = self.left_cam.read()
                ret_right, right_frame = self.right_cam.read()

                if not ret_left or not ret_right:
                    logger.warning("Failed to read frames from cameras")
                    continue

                # Convert to grayscale
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                # Apply preprocessing to improve detection
                left_gray = cv2.GaussianBlur(left_gray, (5, 5), 0)
                right_gray = cv2.GaussianBlur(right_gray, (5, 5), 0)

                # Display the frames
                cv2.imshow('Left Camera', left_frame)
                cv2.imshow('Right Camera', right_frame)

                # Check for checkerboard in both frames every frame
                pattern_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                found_left, left_corners = cv2.findChessboardCorners(
                    left_gray, checkerboard_size, pattern_flags)
                found_right, right_corners = cv2.findChessboardCorners(
                    right_gray, checkerboard_size, pattern_flags)

                # Track detection for auto-capture
                current_time = time.time()
                both_found = found_left and found_right

                # Display text about detection status
                status_frame = np.zeros((100, 400, 3), dtype=np.uint8)
                if both_found:
                    cv2.putText(status_frame, "DETECTED!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    checkerboard_detection_count += 1

                    # For auto-capture, track stability
                    if auto_capture:
                        if not is_stable:
                            stable_since = current_time
                            is_stable = True

                        # Calculate and display time stable
                        stable_duration = current_time - stable_since
                        cv2.putText(status_frame, f"Stable: {stable_duration:.1f}s", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Auto-capture if stable for required duration
                        if stable_duration >= stability_seconds:
                            logger.info("Auto-capturing after %.1f seconds of stability", stable_duration)
                            # Set the flag to trigger capture
                            # This will be processed in the next iteration
                            key = ord('c')
                            # Reset stability to avoid multiple captures
                            is_stable = False
                else:
                    cv2.putText(status_frame, "NOT DETECTED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Reset stability tracking
                    is_stable = False

                # Show detection rate for debugging
                detection_rate = (checkerboard_detection_count / (checkerboard_detection_count + 1)) * 100
                cv2.putText(status_frame, f"Detection rate: {detection_rate:.1f}%",
                            (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow('Detection Status', status_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    # Debug mode - try different preprocessing and show results
                    logger.info("Debug mode activated - testing different preprocessing and detection settings")
                    print("Debug mode activated - testing different preprocessing...")

                    # Test 1: Adaptive thresholding
                    logger.debug("Testing adaptive thresholding")
                    left_thresh1 = cv2.adaptiveThreshold(left_gray, 255,
                                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY, 11, 2)
                    right_thresh1 = cv2.adaptiveThreshold(right_gray, 255,
                                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY, 11, 2)

                    # Test 2: Simple thresholding
                    logger.debug("Testing simple thresholding")
                    _, left_thresh2 = cv2.threshold(left_gray, 128, 255, cv2.THRESH_BINARY)
                    _, right_thresh2 = cv2.threshold(right_gray, 128, 255, cv2.THRESH_BINARY)

                    # Test 3: Canny edge detection
                    logger.debug("Testing Canny edge detection")
                    left_edges = cv2.Canny(left_gray, 50, 150)
                    right_edges = cv2.Canny(right_gray, 50, 150)

                    # Display preprocessed images
                    cv2.imshow('Left Adaptive Threshold', left_thresh1)
                    cv2.imshow('Right Adaptive Threshold', right_thresh1)
                    cv2.imshow('Left Simple Threshold', left_thresh2)
                    cv2.imshow('Right Simple Threshold', right_thresh2)
                    cv2.imshow('Left Edges', left_edges)
                    cv2.imshow('Right Edges', right_edges)

                    # Try to find checkerboard with different settings
                    logger.info("Trying different corner detection settings...")
                    print("Trying different corner detection settings...")

                    # Create a debug report window
                    debug_report = np.zeros((600, 800, 3), dtype=np.uint8)
                    cv2.putText(debug_report, "OpenCV Checkerboard Detection Debug", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Try with normal settings
                    logger.debug("Testing with default settings (no flags)")
                    start_time = time.time()
                    found_left1, _ = cv2.findChessboardCorners(
                        left_gray, checkerboard_size, None)
                    found_right1, _ = cv2.findChessboardCorners(
                        right_gray, checkerboard_size, None)
                    time1 = time.time() - start_time
                    logger.debug("Default settings time: %.3f seconds", time1)

                    # Try with adaptive threshold
                    logger.debug("Testing with pre-thresholded images")
                    start_time = time.time()
                    found_left2, _ = cv2.findChessboardCorners(
                        left_thresh1, checkerboard_size, None)
                    found_right2, _ = cv2.findChessboardCorners(
                        right_thresh1, checkerboard_size, None)
                    time2 = time.time() - start_time
                    logger.debug("Thresholded images time: %.3f seconds", time2)

                    # Try with different flags
                    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                    logger.debug("Testing with flags: ADAPTIVE_THRESH + NORMALIZE_IMAGE")
                    start_time = time.time()
                    found_left3, _ = cv2.findChessboardCorners(
                        left_gray, checkerboard_size, flags)
                    found_right3, _ = cv2.findChessboardCorners(
                        right_gray, checkerboard_size, flags)
                    time3 = time.time() - start_time
                    logger.debug("Standard flags time: %.3f seconds", time3)

                    # Add fast check flag
                    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
                    logger.debug("Testing with flags: ADAPTIVE_THRESH + NORMALIZE_IMAGE + FAST_CHECK")
                    start_time = time.time()
                    found_left4, _ = cv2.findChessboardCorners(
                        left_gray, checkerboard_size, flags)
                    found_right4, _ = cv2.findChessboardCorners(
                        right_gray, checkerboard_size, flags)
                    time4 = time.time() - start_time
                    logger.debug("Fast check flags time: %.3f seconds", time4)

                    # Add results to debug report
                    cv2.putText(debug_report, f"Default: L={found_left1}, R={found_right1} ({time1:.3f}s)",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if found_left1 and found_right1 else (0, 0, 255), 1)
                    cv2.putText(debug_report, f"Threshold: L={found_left2}, R={found_right2} ({time2:.3f}s)",
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if found_left2 and found_right2 else (0, 0, 255), 1)
                    cv2.putText(debug_report, f"Std Flags: L={found_left3}, R={found_right3} ({time3:.3f}s)",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if found_left3 and found_right3 else (0, 0, 255), 1)
                    cv2.putText(debug_report, f"Fast Check: L={found_left4}, R={found_right4} ({time4:.3f}s)",
                                (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if found_left4 and found_right4 else (0, 0, 255), 1)

                    # Try with different sizes
                    test_sizes = [(7, 6), (9, 6), (8, 5), (10, 7)]
                    logger.info("Testing with alternative checkerboard sizes")
                    y_pos = 240
                    for test_size in test_sizes:
                        logger.debug("Testing with size %dx%d...", test_size[0], test_size[1])
                        print(f"Testing with size {test_size}...")
                        start_time = time.time()
                        found_left, _ = cv2.findChessboardCorners(
                            left_gray, test_size, flags)
                        found_right, _ = cv2.findChessboardCorners(
                            right_gray, test_size, flags)
                        time_taken = time.time() - start_time
                        if found_left and found_right:
                            logger.info("SUCCESS! Size %dx%d detected in both images (%.3fs)",
                                        test_size[0], test_size[1], time_taken)
                            print(f"SUCCESS! Size {test_size} detected in both images.")
                        else:
                            logger.debug("Size %dx%d: Left=%s, Right=%s (%.3fs)",
                                         test_size[0], test_size[1], found_left, found_right, time_taken)
                            print(f"Size {test_size}: Left detected: {found_left}, Right detected: {found_right}")

                        # Add to debug report
                        cv2.putText(debug_report,
                                    f"Size {test_size}: L={found_left}, R={found_right} ({time_taken:.3f}s)",
                                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0) if found_left and found_right else (0, 0, 255), 1)
                        y_pos += 40

                    # Add recommendations based on results
                    cv2.putText(debug_report, "Recommendations:", (20, y_pos + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Find best settings
                    best_setting = 0
                    best_found = False
                    best_time = float('inf')

                    settings = [
                        (found_left1 and found_right1, time1, "Default settings"),
                        (found_left2 and found_right2, time2, "Pre-thresholded"),
                        (found_left3 and found_right3, time3, "Standard flags"),
                        (found_left4 and found_right4, time4, "Fast check flags")
                    ]

                    for i, (found, detection_time, setting_name) in enumerate(settings):
                        if found and detection_time < best_time:
                            best_setting = i
                            best_found = True
                            best_time = detection_time

                    if best_found:
                        recommendation = f"Best detection: {settings[best_setting][2]} ({best_time:.3f}s)"
                        logger.info(recommendation)
                        cv2.putText(debug_report, recommendation, (40, y_pos + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    else:
                        recommendation = "No successful detection method found. Try improving lighting or pattern position."
                        logger.warning(recommendation)
                        cv2.putText(debug_report, recommendation, (40, y_pos + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                    # Display debug report
                    cv2.imshow('Detection Debug Report', debug_report)

                    # Save debug report
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_report_path = f'calibration_images/debug_report_{timestamp}.png'
                    cv2.imwrite(debug_report_path, debug_report)
                    logger.info("Saved debug report to %s", debug_report_path)

                    print("Detection test results with original size:")
                    print(f"Normal: Left={found_left1}, Right={found_right1}")
                    print(f"Adaptive Threshold: Left={found_left2}, Right={found_right2}")
                    print(f"Different flags: Left={found_left3}, Right={found_right3}")
                    print(f"Fast check: Left={found_left4}, Right={found_right4}")
                    print("Press any key to continue...")
                    cv2.waitKey(0)
                    cv2.destroyWindow('Left Adaptive Threshold')
                    cv2.destroyWindow('Right Adaptive Threshold')
                    cv2.destroyWindow('Left Simple Threshold')
                    cv2.destroyWindow('Right Simple Threshold')
                    cv2.destroyWindow('Left Edges')
                    cv2.destroyWindow('Right Edges')
                    cv2.destroyWindow('Detection Debug Report')

                elif key == ord('c'):
                    # Try multiple approaches to find checkerboard
                    logger.info("Manual calibration frame capture triggered")
                    pattern_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

                    # First attempt with normal grayscale
                    logger.debug("Attempting checkerboard detection with standard flags")
                    start_time = time.time()
                    found_left, left_corners = cv2.findChessboardCorners(
                        left_gray, checkerboard_size, pattern_flags)
                    found_right, right_corners = cv2.findChessboardCorners(
                        right_gray, checkerboard_size, pattern_flags)
                    detection_time = time.time() - start_time
                    logger.debug("Initial detection attempt took %.3f seconds: Left=%s, Right=%s",
                                 detection_time, found_left, found_right)

                    # If not found, try with additional filtering
                    if not (found_left and found_right):
                        logger.debug("Standard detection failed, trying with enhanced contrast...")
                        # Try with enhanced contrast
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        left_gray = clahe.apply(left_gray)
                        right_gray = clahe.apply(right_gray)

                        # Add fast check to pattern_flags
                        pattern_flags |= cv2.CALIB_CB_FAST_CHECK

                        start_time = time.time()
                        found_left, left_corners = cv2.findChessboardCorners(
                            left_gray, checkerboard_size, pattern_flags)
                        found_right, right_corners = cv2.findChessboardCorners(
                            right_gray, checkerboard_size, pattern_flags)
                        detection_time = time.time() - start_time
                        logger.debug("Enhanced detection attempt took %.3f seconds: Left=%s, Right=%s",
                                     detection_time, found_left, found_right)

                    if found_left and found_right:
                        logger.info("Checkerboard detected in both frames! Refining corners...")
                        # Refine corner locations
                        start_time = time.time()
                        left_corners = cv2.cornerSubPix(
                            left_gray, left_corners, (11, 11), (-1, -1), criteria)
                        right_corners = cv2.cornerSubPix(
                            right_gray, right_corners, (11, 11), (-1, -1), criteria)
                        refinement_time = time.time() - start_time
                        logger.debug("Corner refinement took %.3f seconds", refinement_time)

                        # Draw and show the corners
                        left_display = left_frame.copy()
                        right_display = right_frame.copy()
                        cv2.drawChessboardCorners(
                            left_display, checkerboard_size, left_corners, found_left)
                        cv2.drawChessboardCorners(
                            right_display, checkerboard_size, right_corners, found_right)
                        cv2.imshow('Left Corners', left_display)
                        cv2.imshow('Right Corners', right_display)

                        # Store object points and corresponding image points
                        objpoints.append(objp)
                        left_imgpoints.append(left_corners)
                        right_imgpoints.append(right_corners)

                        frame_count += 1
                        logger.info("Captured calibration frame pair %d/%d", frame_count, needed_frames)
                        print(f"Collected frame pair {frame_count}/{needed_frames}")

                        # Save the successful frame with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        left_path = f'calibration_images/calib_{timestamp}_left.jpg'
                        right_path = f'calibration_images/calib_{timestamp}_right.jpg'
                        cv2.imwrite(left_path, left_display)
                        cv2.imwrite(right_path, right_display)
                        logger.debug("Saved calibration images to %s and %s", left_path, right_path)

                        # Display a success message on screen
                        success_msg = np.zeros((100, 400, 3), dtype=np.uint8)
                        cv2.putText(success_msg, f"Frame {frame_count}/{needed_frames} Captured!",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(success_msg, f"Corners: L={len(left_corners)}, R={len(right_corners)}",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow('Capture Success', success_msg)
                        cv2.waitKey(500)
                        cv2.destroyWindow('Capture Success')

                        # Reset the stability tracking to avoid immediate recapture
                        is_stable = False

                    else:
                        logger.warning("Checkerboard corners not detected in both frames")
                        print("Checkerboard corners not detected. Please try again.")
                        print("Tips: Ensure good lighting, avoid glare, and hold pattern still.")
                        print("Try a different angle or distance from the cameras.")

                        # Display a failure message on screen
                        fail_msg = np.zeros((150, 450, 3), dtype=np.uint8)
                        cv2.putText(fail_msg, "Checkerboard Not Detected!",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(fail_msg, "• Check lighting and avoid glare",
                                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 1)
                        cv2.putText(fail_msg, "• Hold pattern still and flat",
                                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 1)
                        cv2.putText(fail_msg, "• Try different distance/angle",
                                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 1)
                        cv2.imshow('Capture Failed', fail_msg)
                        cv2.waitKey(1500)
                        cv2.destroyWindow('Capture Failed')

                if frame_count >= needed_frames:
                    print("Collected enough frames, press 'q' to finish calibration.")

            if frame_count < 10:
                logger.warning("Not enough calibration pairs collected (minimum 10 needed)")
                print("Not enough calibration pairs; results may be poor.")
                return False

            logger.info("Starting stereo calibration with %d image pairs...", frame_count)
            print("Running stereo calibration...")

            # Create an on-screen status window for progress
            status_window = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(status_window, "Stereo Calibration in Progress", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(status_window, "Step 1/3: Computing camera parameters...", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(status_window, f"Processing {frame_count} image pairs", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(status_window, "Please wait...", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imshow('Calibration Status', status_window)
            cv2.waitKey(100)  # Brief pause to ensure window displays

            # Continue with stereo calibration with detailed logging
            image_size = left_gray.shape[::-1]  # (width, height)
            logger.info("Image size for calibration: %dx%d", image_size[0], image_size[1])

            # Log the number of corners in each image
            logger.debug("Corners detected in each image pair:")
            for i, (left_pts, right_pts) in enumerate(zip(left_imgpoints, right_imgpoints)):
                logger.debug("Pair %d: Left=%d corners, Right=%d corners",
                             i + 1, len(left_pts), len(right_pts))

            # Log calibration start with detailed configuration
            flags = 0  # let OpenCV refine intrinsics
            logger.info("Starting OpenCV stereoCalibrate with flags=%d", flags)
            logger.info("This may take some time depending on the number of image pairs...")

            start_time = time.time()
            try:
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
                    flags=flags
                )
                calibration_time = time.time() - start_time
                logger.info("Stereo calibration completed in %.2f seconds!", calibration_time)
                logger.info("Calibration RMS error: %.6f", ret)

                # Update status window
                cv2.putText(status_window, f"Step 1/3: Complete! RMS error: {ret:.6f}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(status_window, "Step 2/3: Computing rectification...", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.imshow('Calibration Status', status_window)
                cv2.waitKey(100)

                print("Calibration complete!")
                print(f"RMS error: {ret:.6f}")
                print("Refining stereo rectification parameters...")

                # Log camera matrices
                logger.debug("Left camera matrix:\n%s", str(self.camera_matrix_left))
                logger.debug("Left distortion coefficients: %s", str(self.dist_coeffs_left.ravel()))
                logger.debug("Right camera matrix:\n%s", str(self.camera_matrix_right))
                logger.debug("Right distortion coefficients: %s", str(self.dist_coeffs_right.ravel()))
                logger.debug("Rotation matrix:\n%s", str(self.R))
                logger.debug("Translation vector: %s", str(self.T.ravel()))

                # Stereo rectification with timing
                logger.info("Computing stereo rectification parameters...")
                start_time = time.time()
                R1, R2, P1, P2, self.Q, roi1, roi2 = cv2.stereoRectify(
                    self.camera_matrix_left, self.dist_coeffs_left,
                    self.camera_matrix_right, self.dist_coeffs_right,
                    image_size, self.R, self.T,
                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
                )
                rect_time = time.time() - start_time
                logger.info("Rectification computed in %.2f seconds", rect_time)

                # Log the rectification regions of interest
                logger.debug("Left ROI: %s", str(roi1))
                logger.debug("Right ROI: %s", str(roi2))
                logger.debug("Q (disparity-to-depth) matrix:\n%s", str(self.Q))

                # Update status window
                cv2.putText(status_window, "Step 2/3: Complete!", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(status_window, "Step 3/3: Saving calibration data...", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.imshow('Calibration Status', status_window)
                cv2.waitKey(100)

                # Save calibration parameters to file with timing
                logger.info("Saving calibration data to file...")
                start_time = time.time()

                # Create a timestamp for the calibration
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                calibration_data = {
                    'camera_matrix_left': self.camera_matrix_left,
                    'dist_coeffs_left': self.dist_coeffs_left,
                    'camera_matrix_right': self.camera_matrix_right,
                    'dist_coeffs_right': self.dist_coeffs_right,
                    'R': self.R,
                    'T': self.T,
                    'Q': self.Q,
                    'R1': R1,
                    'R2': R2,
                    'P1': P1,
                    'P2': P2,
                    'image_size': image_size,
                    'calibration_date': timestamp,
                    'rms_error': float(ret),
                    'frame_count': frame_count
                }

                # Save the main calibration file
                np.save('stereo_calibration.npy', calibration_data)

                # Also save a backup with timestamp
                backup_file = f'calibration_images/stereo_calibration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npy'
                np.save(backup_file, calibration_data)

                save_time = time.time() - start_time
                logger.info("Calibration data saved in %.2f seconds", save_time)
                logger.info("Saved calibration data to stereo_calibration.npy and %s", backup_file)
                print("Saved calibration data to stereo_calibration.npy")
                print(f"Backup saved to {backup_file}")

                # Final status update
                cv2.putText(status_window, "Step 3/3: Complete!", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(status_window, "Calibration Successful!", (150, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Calibration Status', status_window)
                cv2.waitKey(1500)
                cv2.destroyWindow('Calibration Status')

                return True

            except Exception as e:
                logger.error("Calibration failed with error: %s", str(e))
                # Show error on status window
                cv2.rectangle(status_window, (0, 0), (500, 200), (0, 0, 100), -1)
                cv2.putText(status_window, "Calibration Failed!", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(status_window, str(e), (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(status_window, "See log file for details", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 1)
                cv2.imshow('Calibration Status', status_window)
                cv2.waitKey(3000)
                cv2.destroyWindow('Calibration Status')
                print(f"Calibration failed: {str(e)}")
                return False

        finally:
            self.left_cam.release()
            self.right_cam.release()
            cv2.destroyAllWindows()

    def load_calibration(self):
        """Load calibration parameters from file."""
        try:
            logger.info("Loading calibration from stereo_calibration.npy")
            calibration_data = np.load('stereo_calibration.npy', allow_pickle=True).item()

            # Convert lists back to numpy arrays (since they were saved with .tolist())
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

            if isinstance(calibration_data['R'], list):
                self.R = np.array(calibration_data['R'], dtype=np.float32)
            else:
                self.R = calibration_data['R'].astype(np.float32)

            if isinstance(calibration_data['T'], list):
                self.T = np.array(calibration_data['T'], dtype=np.float32)
            else:
                self.T = calibration_data['T'].astype(np.float32)

            if isinstance(calibration_data['Q'], list):
                self.Q = np.array(calibration_data['Q'], dtype=np.float32)
            else:
                self.Q = calibration_data['Q'].astype(np.float32)

            if 'R1' in calibration_data:
                if isinstance(calibration_data['R1'], list):
                    self.R1 = np.array(calibration_data['R1'], dtype=np.float32)
                else:
                    self.R1 = calibration_data['R1'].astype(np.float32)

            if 'R2' in calibration_data:
                if isinstance(calibration_data['R2'], list):
                    self.R2 = np.array(calibration_data['R2'], dtype=np.float32)
                else:
                    self.R2 = calibration_data['R2'].astype(np.float32)

            if 'P1' in calibration_data:
                if isinstance(calibration_data['P1'], list):
                    self.P1 = np.array(calibration_data['P1'], dtype=np.float32)
                else:
                    self.P1 = calibration_data['P1'].astype(np.float32)

            if 'P2' in calibration_data:
                if isinstance(calibration_data['P2'], list):
                    self.P2 = np.array(calibration_data['P2'], dtype=np.float32)
                else:
                    self.P2 = calibration_data['P2'].astype(np.float32)

            # Log some information to help with debugging
            logger.info("Calibration loaded successfully")
            logger.debug("Image size: %s", str(calibration_data.get('image_size', 'unknown')))
            logger.debug("Calibration date: %s", calibration_data.get('calibration_date', 'unknown'))
            logger.debug("RMS error: %.6f", calibration_data.get('rms_error', 0.0))

            # Verify we have valid numpy arrays
            if (isinstance(self.camera_matrix_left, np.ndarray) and
                    isinstance(self.dist_coeffs_left, np.ndarray) and
                    isinstance(self.camera_matrix_right, np.ndarray) and
                    isinstance(self.dist_coeffs_right, np.ndarray)):
                logger.debug("All matrices verified as numpy arrays")
            else:
                logger.error("Matrix conversion failed: not all matrices are numpy arrays")
                return False

            print("Calibration loaded successfully.")
            return True
        except FileNotFoundError:
            logger.error("Calibration file not found. Please run calibration first.")
            print("Calibration file not found. Please run calibration first.")
            return False
        except Exception as e:
            logger.error("Error loading calibration: %s", str(e))
            print(f"Error loading calibration: {str(e)}")
            return False

    def initialize_dl_model(self, model_name='raft_stereo', weights_path=None):
        """
        Initialize deep learning model for disparity estimation
        
        Args:
            model_name (str): Name of the model to initialize ('raft_stereo' or 'crestereo')
            weights_path (str, optional): Path to model weights file. If None, will search in default locations
            
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        try:
            logger.info(f"Initializing deep learning model: {model_name}")
            
            if model_name == 'raft_stereo':
                # Import RaftStereoWrapper
                try:
                    from models.raft_stereo_wrapper import RaftStereoWrapper
                except ImportError:
                    logger.error("Failed to import RaftStereoWrapper. Make sure RAFT-Stereo is properly installed")
                    return False

                # Create and initialize model
                logger.info("Creating RAFT-Stereo model instance")
                self.dl_model = RaftStereoWrapper(max_disp=self.dl_params['max_disp'])

                # Load model weights
                logger.info(f"Loading RAFT-Stereo model weights from {weights_path if weights_path else 'default locations'}")
                success = self.dl_model.load_model(weights_path)
                if not success:
                    logger.error("Failed to load RAFT-Stereo model weights")
                    return False

            elif model_name == 'crestereo':
                # Your existing CREStereo initialization code
                logger.error("CREStereo model not yet implemented")
                return False

            else:
                logger.error(f"Unknown model: {model_name}")
                return False

            # Update model information
            self.dl_model_name = model_name
            self.use_dl = True
            self.disparity_method = 'dl'

            logger.info(f"Successfully initialized {model_name} model")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize deep learning model: {str(e)}")
            logger.exception("Stack trace:")
            return False

    def compute_disparity_sgbm(self, left_img, right_img):
        """
        Compute the disparity map using OpenCV's StereoSGBM

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

    # In compute_disparity_dl method of StereoVision class, add support for RAFT-Stereo:

    def compute_disparity_dl(self, left_img, right_img):
        """
        Compute the disparity map using deep learning model

        Args:
            left_img: Left camera image.
            right_img: Right camera image.

        Returns:
            (disparity, disparity_color) where disparity is float32 and disparity_color is a color map.
        """
        if self.dl_model is None:
            logger.warning("Deep learning model not initialized, falling back to SGBM")
            return self.compute_disparity_sgbm(left_img, right_img)

        try:
            # Process the images for the model
            h, w = left_img.shape[:2]

            # Downscale images if needed for faster processing
            if self.dl_params['downscale_factor'] != 1.0:
                scale = self.dl_params['downscale_factor']
                new_h, new_w = int(h * scale), int(w * scale)
                left_img_scaled = cv2.resize(left_img, (new_w, new_h))
                right_img_scaled = cv2.resize(right_img, (new_w, new_h))
            else:
                left_img_scaled = left_img
                right_img_scaled = right_img

            # Run inference
            start_time = time.time()

            # Use the model's inference method
            # All DL models should implement a consistent .inference() method
            disparity = self.dl_model.inference(left_img_scaled, right_img_scaled)

            inference_time = time.time() - start_time
            logger.debug(f"Deep learning inference took {inference_time:.3f} seconds using {self.dl_model_name}")

            # Resize back to original resolution if downscaled
            if self.dl_params['downscale_factor'] != 1.0:
                disparity = cv2.resize(disparity, (w, h))
                # Scale disparity values for the new resolution
                disparity = disparity * (1.0 / self.dl_params['downscale_factor'])

            # Normalize for visualization
            disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255,
                                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

            return disparity, disp_color

        except Exception as e:
            logger.error(f"Error in deep learning disparity computation: {str(e)}")
            logger.error("Falling back to SGBM method")
            return self.compute_disparity_sgbm(left_img, right_img)

    def compute_disparity_map(self, left_img, right_img):
        """
        Compute the disparity map for a stereo image pair using selected method.

        Args:
            left_img: Left camera image.
            right_img: Right camera image.

        Returns:
            (disparity, disparity_color) where disparity is float32 and disparity_color is a color map.
        """
        # Use selected method
        if self.disparity_method == 'dl' and self.dl_model is not None:
            return self.compute_disparity_dl(left_img, right_img)
        else:
            return self.compute_disparity_sgbm(left_img, right_img)

    def compute_confidence_map(self, disparity, left_img, right_img):
        """
        Compute confidence map for disparity estimates

        Args:
            disparity: Disparity map
            left_img: Left image
            right_img: Right image

        Returns:
            confidence: Confidence map (0-1 range, higher is better)
        """
        try:
            from models.utils import compute_confidence_map
            return compute_confidence_map(disparity, left_img, right_img)
        except ImportError:
            # Fallback to a simple left-right consistency check if module not available
            H, W = disparity.shape
            confidence = np.ones((H, W), dtype=np.float32)

            # Convert to grayscale for matching
            if len(left_img.shape) == 3:
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_img
                right_gray = right_img

            # Create a right-to-left disparity map
            stereo_right = cv2.StereoSGBM_create(
                minDisparity=-self.sgbm_params['num_disp'],
                numDisparities=self.sgbm_params['num_disp'],
                blockSize=self.sgbm_params['window_size'],
                P1=8 * 3 * self.sgbm_params['window_size'] ** 2,
                P2=32 * 3 * self.sgbm_params['window_size'] ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=self.sgbm_params['uniqueness_ratio'],
                speckleWindowSize=self.sgbm_params['speckle_window_size'],
                speckleRange=self.sgbm_params['speckle_range'],
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
            right_disparity = stereo_right.compute(right_gray, left_gray).astype(np.float32) / 16.0
            right_disparity = -right_disparity  # Flip sign to make it positive

            # Check consistency
            for y in range(H):
                for x in range(W):
                    d = disparity[y, x]
                    # Skip invalid disparities
                    if d <= 0:
                        confidence[y, x] = 0
                        continue

                    # Find corresponding pixel in right image
                    x_r = int(x - d)
                    if x_r < 0 or x_r >= W:
                        confidence[y, x] = 0
                        continue

                    # Get disparity at corresponding right image point
                    d_r = right_disparity[y, x_r]

                    # Check consistency (within 1 pixel threshold)
                    if abs(d - d_r) > 1.0:
                        confidence[y, x] = 0

            return confidence

    def process_stereo_frames(self):
        """
        Continuously capture stereo frames, rectify them, and compute a disparity map.
        Requires prior calibration.
        """
        if self.Q is None:
            success = self.load_calibration()
            if not success:
                print("Please run calibration first.")
                return

        self.open_cameras()

        # Build rectification maps
        mapL1, mapL2 = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.R1, self.P1,
            (self.width, self.height), cv2.CV_32FC1
        )
        mapR1, mapR2 = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right,
            self.R2, self.P2,
            (self.width, self.height), cv2.CV_32FC1
        )

        print("Processing stereo streams. Press 'q' to quit.")
        print("Press 'd' to toggle between SGBM and Deep Learning methods.")

        try:
            while True:
                ret_left, left_frame = self.left_cam.read()
                ret_right, right_frame = self.right_cam.read()

                if not ret_left or not ret_right:
                    continue

                # Rectify images
                left_rect = cv2.remap(left_frame, mapL1, mapL2, cv2.INTER_LINEAR)
                right_rect = cv2.remap(right_frame, mapR1, mapR2, cv2.INTER_LINEAR)

                # Optional: draw horizontal lines to check alignment
                step = 40
                for i in range(0, left_rect.shape[0], step):
                    cv2.line(left_rect, (0, i), (left_rect.shape[1], i), (0, 255, 0), 1)
                    cv2.line(right_rect, (0, i), (right_rect.shape[1], i), (0, 255, 0), 1)

                # Compute and display disparity map
                disparity, disp_color = self.compute_disparity_map(left_rect, right_rect)

                # Add method label
                method_text = "SGBM" if self.disparity_method == 'sgbm' else "Deep Learning"
                cv2.putText(disp_color, f"Method: {method_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow('Left Rectified', left_rect)
                cv2.imshow('Right Rectified', right_rect)
                cv2.imshow('Disparity Map', disp_color)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    # Toggle between SGBM and DL methods
                    if self.disparity_method == 'sgbm':
                        if self.dl_model is not None:
                            self.disparity_method = 'dl'
                            print("Switched to Deep Learning disparity method")
                        else:
                            print("Deep Learning model not available, trying to initialize...")
                            if self.initialize_dl_model():
                                self.disparity_method = 'dl'
                                print("Switched to Deep Learning disparity method")
                            else:
                                print("Failed to initialize Deep Learning model")
                    else:
                        self.disparity_method = 'sgbm'
                        print("Switched to SGBM disparity method")

        finally:
            self.left_cam.release()
            self.right_cam.release()
            cv2.destroyAllWindows()


def main():
    stereo = StereoVision(left_cam_idx=0, right_cam_idx=1)

    print("Stereo Vision Test")
    print("1. Basic camera test")
    print("2. Camera calibration (manual capture)")
    print("3. Camera calibration (auto-capture after 3 seconds of stability)")
    print("4. Process stereo frames with OpenCV SGBM")
    print("5. Process stereo frames with CREStereo deep learning model")

    choice = input("Enter your choice (1-5): ")
    if choice == '1':
        stereo.basic_test()
    elif choice == '2':
        stereo.calibrate_cameras(auto_capture=False)
    elif choice == '3':
        # Get stability seconds
        try:
            stability_seconds = float(input("Enter stability threshold in seconds (default 3.0): ") or "3.0")
            if stability_seconds < 0.5:
                logger.warning("Stability threshold too low, setting to minimum 0.5 seconds")
                stability_seconds = 0.5
            elif stability_seconds > 10.0:
                logger.warning("Stability threshold too high, setting to maximum 10.0 seconds")
                stability_seconds = 10.0
        except ValueError:
            logger.warning("Invalid input for stability threshold, using default 3.0 seconds")
            stability_seconds = 3.0

        logger.info("Starting auto-capture calibration with %.1f second stability threshold", stability_seconds)
        stereo.calibrate_cameras(auto_capture=True, stability_seconds=stability_seconds)
    elif choice == '4':
        stereo.disparity_method = 'sgbm'
        stereo.process_stereo_frames()
    elif choice == '5':
        # Initialize deep learning model
        if stereo.initialize_dl_model():
            stereo.process_stereo_frames()
        else:
            print("Failed to initialize deep learning model. Using SGBM instead.")
            stereo.disparity_method = 'sgbm'
            stereo.process_stereo_frames()
    else:
        print("Invalid choice")

    logger.info("Program terminated")


if __name__ == "__main__":
    main()