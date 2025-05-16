import logging

import requests

logger = logging.getLogger(__name__)


class VisionClient:
    def __init__(self, base_url="http://localhost:5050"):
        self.base_url = base_url
        self.timeout = 5  # Default timeout in seconds

    def _make_request(self, method, endpoint, data=None, params=None, timeout=None):
        """Super mega universal ultra function to make reuest to stereo api(and to the turtlebot as well)"""
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.timeout

        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=timeout)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=timeout)
            else:
                raise ValueError(f"Wrong method, bro: {method}")

            # Check for HTTP errors
            response.raise_for_status()

            # Parse JSON response
            return response.json()

        except Exception as e:
            logger.error(f"Noooo, errrorr, nooo: {url}: {str(e)}")
            return {"success": False, "message": f"Haram bro, error: {str(e)}"}

    def initialize_vision(self, left_cam_idx=0, right_cam_idx=1, width=640, height=480):
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
        return self._make_request("POST", "/api/vision/start")

    def stop_vision(self):
        return self._make_request("POST", "/api/vision/stop")

    def get_vision_data(self):
        return self._make_request("GET", "/api/vision/data")

    def get_frames(self, quality=100, full_size=True):
        return self._make_request(
            "GET",
            "/api/frames",
            params={"quality": quality, "full_size": "true" if full_size else "false"}
        )

    def capture_frame(self):
        return self._make_request("POST", "/api/capture")

    def run_calibration(self, checkerboard_size=(11, 12), square_size=0.004, num_samples=100):
        return self._make_request(
            "POST",
            "/api/calibrate",
            data={
                "checkerboard_size": checkerboard_size,
                "square_size": square_size,
                "num_samples": num_samples
            },
            timeout=180
        )

    def get_calibration_status(self):
        return self._make_request("GET", "/api/calibrate/status")

    def detect_checkerboard(self, checkerboard_size=(11, 12)):
        return self._make_request(
            "POST",
            "/api/calibrate/detect",
            data={"checkerboard_size": checkerboard_size}
        )

    def get_config(self):
        return self._make_request("GET", "/api/config")

    def update_config(self, config):
        return self._make_request("POST", "/api/config", data=config)

    def get_system_info(self):
        return self._make_request("GET", "/api/system/info")

    def check_health(self):
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False