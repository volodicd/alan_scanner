# utils.py
import os
import logging

logger = logging.getLogger(__name__)

def ensure_app_directories():
    """Create all necessary directories for the application"""
    dirs = [
        'static/captures',
        'static/calibration',
        'static/img',
        'static/css',
        'static/js',
    ]
    for directory in dirs:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")