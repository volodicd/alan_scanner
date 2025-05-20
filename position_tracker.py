#!/usr/bin/env python3

import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionTracker:
    """Shared position tracker class that can be imported by other modules"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PositionTracker, cls).__new__(cls)
                cls._instance.x = 0.0  # cm
                cls._instance.y = 0.0  # cm
                cls._instance.heading = 0  # degrees (0 = east, 90 = north)
                cls._instance.position_lock = threading.Lock()
                cls._instance.start_flag = False  # Flag to start exploration
                cls._instance.initial_position = (0.0, 0.0, 0)  # Store initial position
                logger.info("Position tracker initialized at (0, 0)")
        return cls._instance
    
    def update_position(self, x, y, heading=None):
        """Update the robot's position"""
        with self.position_lock:
            self.x = float(x)
            self.y = float(y)
            if heading is not None:
                self.heading = float(heading) % 360
            
            logger.info(f"Position updated to ({self.x:.1f}, {self.y:.1f})")
            return True
    
    def get_position(self):
        """Get current position (x, y only)"""
        with self.position_lock:
            return {
                'x': round(self.x, 1),
                'y': round(self.y, 1)
            }
    
    def reset_position(self):
        """Reset position to origin"""
        with self.position_lock:
            self.x = 0.0
            self.y = 0.0
            self.heading = 0
            self.initial_position = (0.0, 0.0, 0)
            logger.info("Position reset to origin (0, 0)")
            return True
            
    def set_start_flag(self, value):
        """Set the start flag"""
        with self.position_lock:
            self.start_flag = bool(value)
            logger.info(f"Start flag set to {self.start_flag}")
            return self.start_flag
            
    def get_start_flag(self):
        """Get the current start flag value"""
        with self.position_lock:
            return self.start_flag
    
    def set_initial_position(self, x, y, heading):
        """Remember the initial position"""
        with self.position_lock:
            self.initial_position = (float(x), float(y), float(heading))
            logger.info(f"Initial position set to {self.initial_position}")
    
    def get_initial_position(self):
        """Get the initial position"""
        with self.position_lock:
            return self.initial_position

# Create a global instance
position_tracker = PositionTracker()
