import os
import logging

logger = logging.getLogger(__name__)


def ensure_directories():
    """Create all necessary directories for the application"""
    dirs = [
        'data/captures',
        'data/calibration',
    ]

    for directory in dirs:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")