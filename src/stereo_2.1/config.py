import  os
import logging
import logging.handlers

os.makedirs('static/captures', exist_ok=True)
os.makedirs('static/maps', exist_ok=True)
os.makedirs('static/logs', exist_ok=True)
os.makedirs('static/calibration', exist_ok=True)
os.makedirs('static/code_backups', exist_ok=True)


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            "stereo_vision_app.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


logging.getLogger('werkzeug').setLevel(logging.WARNING)  # Reduce Flask debug logs
logging.getLogger('engineio').setLevel(logging.WARNING)  # Reduce SocketIO logs
logging.getLogger('socketio').setLevel(logging.WARNING)  # Reduce SocketIO logs


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
    "calibration_checkerboard_size": (11, 12),
    "calibration_square_size": 0.004,
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
    "disparity_method": "dl",
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
    "min_pairs_needed": 20,
    "recommended_pairs": 50,
    "checkerboard_history": []  # To track detection stability
}

code_versions = []