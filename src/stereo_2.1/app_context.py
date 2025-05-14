# app_context.py
import threading
import logging

logger = logging.getLogger(__name__)


class AppContext:
    """Central place for application state to avoid global variables"""

    def __init__(self):
        self.stereo_vision = None
        self.is_streaming = False
        self.stream_thread = None
        self.stream_thread_active = False
        self.lock = threading.Lock()  # Single lock for all state
        self.socketio = None

        # For FPS calculation
        self.frame_times = []
        self.last_fps_update = 0
        self.current_fps = 0

        # Configuration
        self.config = {
            "left_cam_idx": 0,
            "right_cam_idx": 1,
            "width": 640,
            "height": 480,
            "calibration_checkerboard_size": (11, 12),
            "calibration_square_size": 0.004,
            "auto_capture": True,
            "stability_seconds": 1.0,
            "sgbm_params": {
                "window_size": 11,
                "min_disp": 0,
                "num_disp": 128,
                "uniqueness_ratio": 15,
                "speckle_window_size": 100,
                "speckle_range": 32
            }
        }

        # Calibration state
        self.calibration_in_progress = False
        self.calibration_state = {
            "is_stable": False,
            "stable_since": 0,
            "last_capture_time": 0,
            "captured_pairs": 0,
            "min_pairs_needed": 10,
            "recommended_pairs": 20
        }

    def init_socketio(self, io):
        """Initialize the SocketIO instance"""
        self.socketio = io

    def emit(self, event, data):
        """Emit an event via SocketIO if available"""
        if self.socketio:
            self.socketio.emit(event, data)
        else:
            logger.warning(f"Attempted to emit '{event}' but socketio is not initialized!")

    def calculate_fps(self):
        """Calculate FPS from stored frame times"""
        if len(self.frame_times) < 2:
            return 0.0

        # Calculate time difference between oldest and newest frame
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff <= 0:
            return 0.0

        # Calculate FPS based on number of frames and time elapsed
        return (len(self.frame_times) - 1) / time_diff


# Single instance to be imported by other modules
app_ctx = AppContext()