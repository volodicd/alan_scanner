# routes/__init__.py
from .camera_routes import register_camera_routes
from .calibration_routes import register_calibration_routes
from .settings_routes import register_settings_routes

def register_all_routes(app):
    """Register all application routes."""
    register_camera_routes(app)
    register_calibration_routes(app)
    register_settings_routes(app)