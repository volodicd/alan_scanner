import requests
import logging
import time

logger = logging.getLogger(__name__)


class VisionClient:
    def __init__(self, base_url="http://localhost:5050"):
        self.base_url = base_url
        self.timeout = 5  # Default timeout in seconds

    def _make_request(self, method, endpoint, data=None, params=None, timeout=None):
        """Helper method for making HTTP requests with error handling"""
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.timeout

        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=timeout)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Check for HTTP errors
            response.raise_for_status()

            # Parse JSON response
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error for {url}: {str(e)}")
            return {"success": False, "message": f"Connection error: {str(e)}"}
        except ValueError as e:
            logger.error(f"Invalid JSON response from {url}: {str(e)}")
            return {"success": False, "message": "Invalid response format"}
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {str(e)}")
            return {"success": False, "message": f"Unexpected error: {str(e)}"}

    def initialize_vision(self, left_cam_idx=0, right_cam_idx=1, width=640, height=480):
        """Initialize the vision system"""
        return self._make_request(
            "POST",
            "/api/vision/initialize",
            data={
                "left_cam_idx": left_cam_idx,
                "right_cam_idx": right_cam_idx,
                "width": width,
                "height": height
            }
        )

    def start_vision(self):
        """Start vision processing"""
        return self._make_request("POST", "/api/vision/start")

    def stop_vision(self):
        """Stop vision processing"""
        return self._make_request("POST", "/api/vision/stop")

    def get_vision_data(self):
        """Get current vision analysis data"""
        return self._make_request("GET", "/api/vision/data")

    def get_frames(self, quality=80, full_size=True):
        """Get current camera frames"""
        return self._make_request(
            "GET",
            "/api/frames",
            params={
                "quality": quality,
                "full_size": "true" if full_size else "false"
            }
        )

    def capture_frame(self):
        """Capture and save current frames"""
        return self._make_request("POST", "/api/capture")

    def run_calibration(self, checkerboard_size=(7, 6), square_size=0.025, num_samples=20):
        """Run the calibration process"""
        return self._make_request(
            "POST",
            "/api/calibrate",
            data={
                "checkerboard_size": checkerboard_size,
                "square_size": square_size,
                "num_samples": num_samples
            },
            timeout=120  # Longer timeout for calibration
        )

    def get_calibration_status(self):
        """Get current calibration status"""
        return self._make_request("GET", "/api/calibrate/status")

    def detect_checkerboard(self, checkerboard_size=(7, 6)):
        """Test checkerboard detection"""
        return self._make_request(
            "POST",
            "/api/calibrate/detect",
            data={
                "checkerboard_size": checkerboard_size
            }
        )

    def get_config(self):
        """Get current configuration"""
        return self._make_request("GET", "/api/config")

    def update_config(self, config):
        """Update configuration"""
        return self._make_request("POST", "/api/config", data=config)

    def get_system_info(self):
        """Get system information"""
        return self._make_request("GET", "/api/system/info")

    def check_health(self):
        """Check if vision service is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False