import os
import logging.handlers

# Make sure necessary directories exist
os.makedirs('static/captures', exist_ok=True)
os.makedirs('static/calibration', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            "stereo_vision_app.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reduce verbosity for third-party libraries
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)
logging.getLogger('socketio').setLevel(logging.WARNING)

# Global variables
stereo_vision = None  # StereoVision instance
is_streaming = False  # Whether streaming is active

# Default configuration
current_config = {
    "left_cam_idx": 0,
    "right_cam_idx": 1,
    "width": 640,
    "height": 480,
    "calibration_checkerboard_size": (11, 12),
    "calibration_square_size": 0.004,
    "auto_capture": True,
    "stability_seconds": 1.0, #better not to set less than 1s
    "sgbm_params": {
        "window_size": 11,
        "min_disp": 0,
        "num_disp": 128,
        "uniqueness_ratio": 15,
        "speckle_window_size": 100,
        "speckle_range": 32
    }
}

# Calibration state tracking
calibration_state = {
    "is_stable": False,
    "stable_since": 0,
    "last_capture_time": 0,
    "captured_pairs": 0,
    "min_pairs_needed": 10,
    "recommended_pairs": 20
}