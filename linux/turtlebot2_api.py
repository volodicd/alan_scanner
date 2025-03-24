#!/usr/bin/env python3

import math
import time
from enum import Enum
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from sensor_msgs.msg import BatteryState

from kobuki_ros_interfaces.msg import BumperEvent, CliffEvent, WheelDropEvent
from kobuki_ros_interfaces.msg import SensorState, Sound


class Direction(Enum):
    """Movement directions for the robot."""
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STOP = 4


class ObstacleLocation(Enum):
    """Locations where obstacles can be detected."""
    NONE = 0
    LEFT = 1
    CENTER = 2
    RIGHT = 3
    MULTIPLE = 4


class TurtleBot2:
    """
    A simplified API for controlling the TurtleBot2 robot with ROS2.
    
    This class provides methods for movement, obstacle detection, battery monitoring,
    and basic navigation for room exploration.
    """
    
    def __init__(self):
        """Initialize the TurtleBot2 API."""
        # Initialize ROS2
        rclpy.init()
        
        # Create node
        self.node = Node('turtlebot2_api')
        
        # Movement parameters
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 1.0  # rad/s
        self.movement_timeout = 5.0  # seconds
        
        # Robot state
        self.is_moving = False
        self.current_direction = Direction.STOP
        self.battery_percentage = 100.0
        self.obstacle_detected = ObstacleLocation.NONE
        self.odometry = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.start_position = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.visited_cells = set()
        self.cell_size = 0.5  # meters
        
        # Create mutex for thread safety
        self.lock = Lock()
        
        # Create publishers
        self.cmd_vel_pub = self.node.create_publisher(
            Twist, 'commands/velocity', 10)
        self.sound_pub = self.node.create_publisher(
            Sound, 'commands/sound', 10)
        self.reset_pub = self.node.create_publisher(
            Empty, 'commands/reset_odometry', 10)
        
        # Create subscribers
        self.bumper_sub = self.node.create_subscription(
            BumperEvent, 'events/bumper', self._bumper_callback, 10)
        self.cliff_sub = self.node.create_subscription(
            CliffEvent, 'events/cliff', self._cliff_callback, 10)
        self.wheel_drop_sub = self.node.create_subscription(
            WheelDropEvent, 'events/wheel_drop', self._wheel_drop_callback, 10)
        self.odom_sub = self.node.create_subscription(
            Odometry, 'odom', self._odometry_callback, 10)
        self.battery_sub = self.node.create_subscription(
            SensorState, 'sensors/core', self._sensor_state_callback, 10)
        
        # Reset odometry at startup
        self.reset_odometry()
        
        self.node.get_logger().info('TurtleBot2 API initialized')
    
    def shutdown(self):
        """Shutdown the TurtleBot2 API."""
        self.stop()
        self.node.destroy_node()
        rclpy.shutdown()
    
    # Movement commands
    
    def move(self, direction: Direction, duration: Optional[float] = None):
        """
        Move the robot in the specified direction.
        
        Args:
            direction: The direction to move (FORWARD, BACKWARD, LEFT, RIGHT, STOP)
            duration: Optional duration in seconds. If None, robot will move until stop() is called.
        """
        with self.lock:
            msg = Twist()
            
            if direction == Direction.FORWARD:
                msg.linear.x = self.linear_speed
                msg.angular.z = 0.0
            elif direction == Direction.BACKWARD:
                msg.linear.x = -self.linear_speed
                msg.angular.z = 0.0
            elif direction == Direction.LEFT:
                msg.linear.x = 0.0
                msg.angular.z = self.angular_speed
            elif direction == Direction.RIGHT:
                msg.linear.x = 0.0
                msg.angular.z = -self.angular_speed
            elif direction == Direction.STOP:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
            
            self.cmd_vel_pub.publish(msg)
            self.current_direction = direction
            self.is_moving = direction != Direction.STOP
        
        if duration is not None:
            time.sleep(duration)
            self.stop()
    
    def stop(self):
        """Stop the robot's movement."""
        self.move(Direction.STOP)

    def move_distance(self, distance: float, speed: Optional[float] = None) -> bool:
        """Move the robot forward or backward by a specific distance."""
        if speed is None:
            speed = self.linear_speed

        # Set direction based on distance sign
        direction = Direction.FORWARD if distance > 0 else Direction.BACKWARD

        # Calculate time needed to move the distance
        movement_time = abs(distance) / abs(speed)

        # Send movement command
        msg = Twist()
        msg.linear.x = speed if direction == Direction.FORWARD else -speed
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

        # Wait for calculated duration, checking for obstacles periodically
        time_step = 0.1  # seconds
        elapsed_time = 0
        obstacle_detected = False

        try:
            while elapsed_time < movement_time and not obstacle_detected:
                time.sleep(time_step)
                elapsed_time += time_step
                rclpy.spin_once(self.node, timeout_sec=0.01)
                obstacle_detected = self.obstacle_detected != ObstacleLocation.NONE

                if obstacle_detected:
                    self.stop()
                    return False
        finally:
            self.stop()

        return True


    def rotate(self, angle: float) -> bool:
        # Determine direction
        direction = Direction.LEFT if angle > 0 else Direction.RIGHT

        # Set angular velocity
        actual_speed = self.angular_speed if direction == Direction.LEFT else -self.angular_speed

        # Calculate time needed to rotate
        rotation_time = abs(angle) / abs(actual_speed)

        # Send movement command
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = actual_speed
        self.cmd_vel_pub.publish(msg)

        # Wait for calculated duration
        time.sleep(rotation_time)

        # Stop rotation
        self.stop()
        return True
    
    def reset_odometry(self):
        """Reset the robot's odometry."""
        self.reset_pub.publish(Empty())
        self.start_position = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.visited_cells = set()
        self.visited_cells.add((0, 0))  # Mark current position as visited
    
    # Sensor reading methods
    
    def get_battery_percentage(self) -> float:
        """Get the current battery percentage."""
        return self.battery_percentage
    
    def is_obstacle_detected(self) -> ObstacleLocation:
        """
        Check if an obstacle is detected by bumpers or cliff sensors.
        
        Returns:
            ObstacleLocation enum indicating where the obstacle is detected
        """
        return self.obstacle_detected
    
    def get_current_position(self) -> Dict[str, float]:
        """
        Get the current position of the robot.
        
        Returns:
            Dictionary containing x, y coordinates and theta (orientation) in radians
        """
        return self.odometry
    
    # Room exploration methods
    
    def explore_room(self, max_duration: float = 300.0) -> bool:
        """
        Explore a room by moving to unvisited areas.
        
        This uses a simple frontier-based exploration strategy:
        1. Move forward until obstacle detected
        2. Mark cells as visited along the way
        3. When obstacle detected, rotate to find clear path
        4. Repeat until timeout or a sufficient portion of the room is explored
        
        Args:
            max_duration: Maximum time in seconds to explore
            
        Returns:
            True if exploration completed successfully, False if interrupted
        """
        self.node.get_logger().info('Starting room exploration')
        
        # Reset odometry to start fresh
        self.reset_odometry()
        
        start_time = time.time()
        try:
            while time.time() - start_time < max_duration:
                # Check battery
                if self.battery_percentage < 15:
                    self.node.get_logger().warn('Battery low, stopping exploration')
                    return False
                
                # Try to move forward
                if self.move_distance(1.0):
                    # Successfully moved, mark cells as visited
                    self._mark_current_cell_visited()
                else:
                    # Obstacle detected, try to find clear path
                    found_clear_path = False
                    
                    # Try rotating left first
                    self.rotate(math.pi/2)  # 90 degrees
                    if self.obstacle_detected == ObstacleLocation.NONE:
                        found_clear_path = True
                    
                    # If left rotation didn't work, try right
                    if not found_clear_path:
                        self.rotate(-math.pi)  # -180 degrees (turning right from previous position)
                        if self.obstacle_detected == ObstacleLocation.NONE:
                            found_clear_path = True
                    
                    # If still no clear path, we're stuck, try backing up and turning
                    if not found_clear_path:
                        self.move_distance(-0.5)  # Back up
                        self.rotate(math.pi)  # Turn around
                
                # Process some callbacks
                rclpy.spin_once(self.node, timeout_sec=0.1)
                
                # Check if we have explored enough of the room
                # Currently using a simplified metric of number of cells visited
                if len(self.visited_cells) > 20:  # This threshold can be adjusted
                    self.node.get_logger().info(f'Explored {len(self.visited_cells)} cells, completing exploration')
                    return True
        finally:
            self.stop()
        
        self.node.get_logger().info(f'Exploration timeout, visited {len(self.visited_cells)} cells')
        return False
    
    def return_to_start(self) -> bool:
        """
        Return to the starting point after exploration.
        
        Returns:
            True if successfully returned to start, False otherwise
        """
        # Calculate vector to starting point
        dx = self.start_position['x'] - self.odometry['x']
        dy = self.start_position['y'] - self.odometry['y']
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Calculate angle to starting point
        angle_to_start = math.atan2(dy, dx)
        current_angle = self.odometry['theta']
        angle_to_rotate = self._angle_diff(current_angle, angle_to_start)
        
        # Rotate towards starting point
        self.rotate(angle_to_rotate)
        
        # Move to starting point
        return self.move_distance(distance)
    
    # Helper methods
    
    def _mark_current_cell_visited(self):
        """Mark the current cell as visited."""
        # Convert continuous coordinates to discrete cell coordinates
        cell_x = int(self.odometry['x'] / self.cell_size)
        cell_y = int(self.odometry['y'] / self.cell_size)
        
        # Add to set of visited cells
        self.visited_cells.add((cell_x, cell_y))
    
    def _angle_diff(self, angle1: float, angle2: float) -> float:
        """
        Calculate the shortest angular difference between two angles.
        
        Args:
            angle1: First angle in radians
            angle2: Second angle in radians
            
        Returns:
            Shortest angle difference in radians
        """
        diff = (angle2 - angle1 + math.pi) % (2 * math.pi) - math.pi
        return diff
    
    # Callback methods
    
    def _bumper_callback(self, msg: BumperEvent):
        """Process bumper events."""
        with self.lock:
            if msg.state == BumperEvent.PRESSED:
                if msg.bumper == BumperEvent.LEFT:
                    self.obstacle_detected = ObstacleLocation.LEFT
                elif msg.bumper == BumperEvent.CENTER:
                    self.obstacle_detected = ObstacleLocation.CENTER
                elif msg.bumper == BumperEvent.RIGHT:
                    self.obstacle_detected = ObstacleLocation.RIGHT
                
                # Stop the robot when a bumper is pressed
                if self.is_moving:
                    self.stop()
            else:  # RELEASED
                # Only clear the obstacle if this specific bumper was the one detecting
                if ((msg.bumper == BumperEvent.LEFT and self.obstacle_detected == ObstacleLocation.LEFT) or
                    (msg.bumper == BumperEvent.CENTER and self.obstacle_detected == ObstacleLocation.CENTER) or
                    (msg.bumper == BumperEvent.RIGHT and self.obstacle_detected == ObstacleLocation.RIGHT)):
                    self.obstacle_detected = ObstacleLocation.NONE
    
    def _cliff_callback(self, msg: CliffEvent):
        """Process cliff detection events."""
        with self.lock:
            if msg.state == CliffEvent.CLIFF:
                if msg.sensor == CliffEvent.LEFT:
                    self.obstacle_detected = ObstacleLocation.LEFT
                elif msg.sensor == CliffEvent.CENTER:
                    self.obstacle_detected = ObstacleLocation.CENTER
                elif msg.sensor == CliffEvent.RIGHT:
                    self.obstacle_detected = ObstacleLocation.RIGHT
                
                # Stop the robot when a cliff is detected
                if self.is_moving:
                    self.stop()
            else:  # FLOOR
                # Only clear the obstacle if this specific cliff sensor was the one detecting
                if ((msg.sensor == CliffEvent.LEFT and self.obstacle_detected == ObstacleLocation.LEFT) or
                    (msg.sensor == CliffEvent.CENTER and self.obstacle_detected == ObstacleLocation.CENTER) or
                    (msg.sensor == CliffEvent.RIGHT and self.obstacle_detected == ObstacleLocation.RIGHT)):
                    self.obstacle_detected = ObstacleLocation.NONE
    
    def _wheel_drop_callback(self, msg: WheelDropEvent):
        """Process wheel drop events."""
        with self.lock:
            if msg.state == WheelDropEvent.DROPPED:
                # Stop the robot when a wheel is dropped
                if self.is_moving:
                    self.stop()
                self.node.get_logger().warn('Wheel drop detected, stopping robot')
    
    def _odometry_callback(self, msg: Odometry):
        """Process odometry data."""
        with self.lock:
            # Extract position and orientation
            self.odometry['x'] = msg.pose.pose.position.x
            self.odometry['y'] = msg.pose.pose.position.y
            
            # Convert quaternion to Euler angles (yaw/theta)
            q = msg.pose.pose.orientation
            self.odometry['theta'] = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                               1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
    def _sensor_state_callback(self, msg: SensorState):
        """Process core sensor state updates."""
        with self.lock:
            # Battery level (0-100%)
            battery_voltage = msg.battery * 0.1  # convert from 0.1V to V
            
            # Simple linear mapping from voltage to percentage
            # Assuming 16.5V is 100% and 14.0V is 0%
            self.battery_percentage = max(0, min(100, (battery_voltage - 14.0) / (16.5 - 14.0) * 100))
            
            # Additional bumper/cliff processing from SensorState if needed
            if msg.bumper > 0 or msg.cliff > 0:
                if self.is_moving:
                    self.stop()
                
                # Determine obstacle location
                if (msg.bumper & SensorState.BUMPER_LEFT) or (msg.cliff & SensorState.CLIFF_LEFT):
                    self.obstacle_detected = ObstacleLocation.LEFT
                elif (msg.bumper & SensorState.BUMPER_CENTRE) or (msg.cliff & SensorState.CLIFF_CENTRE):
                    self.obstacle_detected = ObstacleLocation.CENTER
                elif (msg.bumper & SensorState.BUMPER_RIGHT) or (msg.cliff & SensorState.CLIFF_RIGHT):
                    self.obstacle_detected = ObstacleLocation.RIGHT
                else:
                    self.obstacle_detected = ObstacleLocation.MULTIPLE


# Example usage
def main():
    try:
        # Create and initialize TurtleBot2 API
        turtle = TurtleBot2()
        
        # Example 1: Basic Movement
        print("Moving forward...")
        turtle.move(Direction.FORWARD, 2.0)
        
        print("Turning left...")
        turtle.rotate(math.pi/2)  # 90 degrees left
        
        print("Moving forward...")
        turtle.move(Direction.FORWARD, 2.0)
        
        # Example 2: Room Exploration
        print("Exploring room...")
        turtle.explore_room(max_duration=60.0)
        
        print("Returning to start point...")
        turtle.return_to_start()
        
    finally:
        # Ensure proper shutdown
        if 'turtle' in locals():
            turtle.shutdown()


if __name__ == "__main__":
    main()