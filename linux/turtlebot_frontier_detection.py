#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from kobuki_ros_interfaces.msg import BumperEvent
import requests
import time
import math
import random

# Import the shared position tracker
from position_tracker import position_tracker

class TurtleBotFrontierDetection(Node):
    def __init__(self):
        super().__init__('turtlebot_frontier_detection')
        
        # Publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, 'commands/velocity', 10)
        self.bumper_sub = self.create_subscription(BumperEvent, 'events/bumper', self.bumper_callback, 10)
        
        # Vision API endpoint
        self.vision_api_url = "http://localhost:5050/api/turtlebot/vision"
        
        # Robot state - initialize from position tracker
        self.x = 0.0  # starting x position in cm
        self.y = 0.0  # starting y position in cm
        self.heading = 0  # degrees (0 = east, 90 = north)
        self.obstacle_detected = False
        
        # Map state
        self.grid_size = 50  # cm
        self.map_data = {}  # Dictionary to store map data
        self.position_history = {}  # Dictionary to store positions after each rotation
        self.rotation_count = 0  # Counter for rotations
        self.frontiers = []  # List of frontier points
        self.visited_positions = set()  # Set of visited grid positions
        
        # Constants
        self.MIN_FRONTIER_DIST = 100  # Minimum distance to consider a frontier (cm)
        self.MAX_OBSTACLE_DIST = 50   # Maximum distance to consider an obstacle (cm)
        self.ROTATION_ANGLE = 45      # Angle to rotate for scanning (degrees)
        
        # Reset position tracker and store initial position
        position_tracker.reset_position()
        position_tracker.set_initial_position(self.x, self.y, self.heading)
        
        self.get_logger().info('TurtleBot Frontier Detection initialized at (0, 0)')
        self.get_logger().info('Waiting for start flag to be set to True...')

    def bumper_callback(self, msg):
        """Handle bumper events"""
        if msg.state == BumperEvent.PRESSED:
            self.obstacle_detected = True
            self.get_logger().info('Bumper pressed - obstacle detected')
            self.stop()
            
            # Handle obstacle
            self.move_backward(1.0)
            self.rotate("left", 45)
            self.obstacle_detected = False
            
            # Store current position after rotation
            self.rotation_count += 1
            self.position_history[self.rotation_count] = (self.x, self.y, self.heading)

    def move_forward(self, duration=0.5):
        """Move forward"""
        msg = Twist()
        msg.linear.x = 0.2  # m/s
        self.vel_pub.publish(msg)
        
        if duration:
            time.sleep(duration)
            # Update position
            distance = duration * 0.2 * 100  # Convert to cm
            self.x += distance * math.cos(math.radians(self.heading))
            self.y += distance * math.sin(math.radians(self.heading))
            
            # Record grid position as visited
            grid_x = round(self.x / self.grid_size) * self.grid_size
            grid_y = round(self.y / self.grid_size) * self.grid_size
            self.visited_positions.add((grid_x, grid_y))
            
            # Update shared position tracker
            position_tracker.update_position(self.x, self.y, self.heading)
            
            self.get_logger().debug(f'Position updated to ({self.x:.1f}, {self.y:.1f})')
            self.stop()

    def move_backward(self, duration=0.5):
        """Move backward"""
        msg = Twist()
        msg.linear.x = -0.2  # m/s
        self.vel_pub.publish(msg)
        
        if duration:
            time.sleep(duration)
            # Update position
            distance = duration * 0.2 * 100  # Convert to cm
            self.x -= distance * math.cos(math.radians(self.heading))
            self.y -= distance * math.sin(math.radians(self.heading))
            
            # Update shared position tracker
            position_tracker.update_position(self.x, self.y, self.heading)
            
            self.stop()

    def rotate(self, direction, angle):
        """Rotate the robot"""
        # Update heading
        if direction == "left":
            self.heading = (self.heading + angle) % 360
            angular_vel = 1.0  # rad/s
        else:
            self.heading = (self.heading - angle) % 360
            angular_vel = -1.0  # rad/s
            
        # Execute rotation
        msg = Twist()
        msg.angular.z = angular_vel
        self.vel_pub.publish(msg)
        
        # Calculate time needed to rotate
        duration = math.radians(angle) / abs(angular_vel)
        time.sleep(duration)
        self.stop()
        
        # Store current position after rotation
        self.rotation_count += 1
        self.position_history[self.rotation_count] = (self.x, self.y, self.heading)
        
        # Update shared position tracker
        position_tracker.update_position(self.x, self.y, self.heading)
        
        self.get_logger().info(f'Rotation {self.rotation_count}: ({self.x:.1f}, {self.y:.1f}, {self.heading}°)')

    def stop(self):
        """Stop the robot"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.vel_pub.publish(msg)

    def get_vision_data(self):
        """Get vision data from API"""
        try:
            response = requests.get(self.vision_api_url, timeout=1.0)
            if response.status_code == 200:
                return response.json()
            else:
                self.get_logger().warning(f'Failed to get vision data: {response.status_code}')
                return None
        except Exception as e:
            self.get_logger().error(f'Error getting vision data: {str(e)}')
            return None

    def update_map_with_vision(self):
        """Update map with current vision data"""
        vision_data = self.get_vision_data()
        if not vision_data:
            return False
        
        # Extract vision data
        is_object = vision_data.get('is_object', False)
        distance_object = vision_data.get('distance_object', 1000)
        grid_distances = vision_data.get('objs', [1000] * 16)
        
        # Update map data for current position
        grid_x = round(self.x / self.grid_size) * self.grid_size
        grid_y = round(self.y / self.grid_size) * self.grid_size
        
        # Process the 4x4 grid of distances
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                distance = grid_distances[idx]
                
                # Calculate direction for this grid cell (0-15)
                # Convert grid index to angle offset (-45 to +45 degrees horizontally, -45 to +45 vertically)
                angle_h = -45 + j * 30
                angle_v = -45 + i * 30
                
                # Calculate absolute angle by combining robot heading with relative angles
                abs_angle = (self.heading + angle_h) % 360
                
                # Simplified calculation for point at this distance and angle
                point_distance = min(distance, 500)  # Cap at 500cm
                if point_distance < 900:  # Only record meaningful readings
                    # Calculate point position
                    point_x = self.x + point_distance * math.cos(math.radians(abs_angle))
                    point_y = self.y + point_distance * math.sin(math.radians(abs_angle))
                    
                    # Round to grid
                    point_grid_x = round(point_x / self.grid_size) * self.grid_size
                    point_grid_y = round(point_y / self.grid_size) * self.grid_size
                    
                    # Store in map data
                    map_key = (point_grid_x, point_grid_y)
                    self.map_data[map_key] = {
                        'distance': point_distance,
                        'visited': (point_grid_x, point_grid_y) in self.visited_positions,
                        'last_updated': time.time()
                    }
        
        return True

    def detect_frontiers(self):
        """Detect frontiers in the current map"""
        self.frontiers = []
        
        # Look for grid cells that border unexplored areas
        for (x, y), data in self.map_data.items():
            if data['visited']:
                continue  # Skip visited cells
                
            # Check if this is a frontier (borders unexplored area)
            is_frontier = False
            
            # Check adjacent cells
            for dx, dy in [(self.grid_size, 0), (-self.grid_size, 0), (0, self.grid_size), (0, -self.grid_size)]:
                adjacent = (x + dx, y + dy)
                
                # If adjacent cell not in map, this might be a frontier
                if adjacent not in self.map_data:
                    is_frontier = True
                    break
            
            if is_frontier and data['distance'] > self.MIN_FRONTIER_DIST:
                # Calculate distance to this frontier
                dist_to_frontier = math.sqrt((x - self.x)**2 + (y - self.y)**2)
                self.frontiers.append((x, y, dist_to_frontier))
        
        # Sort frontiers by distance
        self.frontiers.sort(key=lambda f: f[2])
        
        self.get_logger().info(f'Detected {len(self.frontiers)} frontiers')
        return len(self.frontiers) > 0

    def move_to_frontier(self):
        """Move to the closest frontier"""
        if not self.frontiers:
            return False
            
        # Get closest frontier
        frontier_x, frontier_y, _ = self.frontiers[0]
        self.get_logger().info(f'Moving to frontier at ({frontier_x}, {frontier_y})')
        
        # Calculate direction to frontier
        dx = frontier_x - self.x
        dy = frontier_y - self.y
        target_heading = math.degrees(math.atan2(dy, dx)) % 360
        
        # Calculate distance to frontier
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Turn toward frontier
        heading_diff = (target_heading - self.heading + 180) % 360 - 180
        
        if abs(heading_diff) > 10:
            # Need to turn
            direction = "left" if heading_diff > 0 else "right"
            self.rotate(direction, min(abs(heading_diff), 45))
        
        # Move forward
        steps = min(int(distance / 50), 3)  # Move in smaller increments
        for _ in range(steps):
            # Check for obstacles
            if self.check_for_obstacles():
                return False
                
            # Move forward one step
            self.move_forward(0.5)
            
            # Update map
            self.update_map_with_vision()
            
        return True

    def return_to_start_position(self):
        """Navigate back to the start position"""
        initial_x, initial_y, initial_heading = position_tracker.get_initial_position()
        self.get_logger().info(f'Returning to start position ({initial_x}, {initial_y})...')
        
        # Navigate until we're close to the starting point
        while True:
            # Calculate direction to start
            dx = initial_x - self.x
            dy = initial_y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If we're close enough to the start, break
            if distance < 20:  # Within 20cm
                self.get_logger().info('Reached start position')
                break
                
            # Calculate target heading
            target_heading = math.degrees(math.atan2(dy, dx)) % 360
            
            # Turn toward target
            heading_diff = (target_heading - self.heading + 180) % 360 - 180
            
            if abs(heading_diff) > 10:
                # Need to turn
                direction = "left" if heading_diff > 0 else "right"
                self.rotate(direction, min(abs(heading_diff), 45))
            
            # Check for obstacles
            if self.check_for_obstacles():
                # If obstacle detected, try to go around
                self.rotate("left", 45)
                self.move_forward(0.5)
                continue
            
            # Move forward
            step_distance = min(distance, 50)  # Move at most 50cm at a time
            step_duration = step_distance / 20  # 20cm/s = 0.2m/s
            self.move_forward(step_duration)
            
            # Process ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.01)
            
        # Finally, rotate to the initial heading
        current_diff = (initial_heading - self.heading + 180) % 360 - 180
        if abs(current_diff) > 10:
            direction = "left" if current_diff > 0 else "right"
            self.rotate(direction, abs(current_diff))
            
        self.get_logger().info('Successfully returned to start position and orientation')
        return True

    def check_for_obstacles(self):
        """Check for obstacles using vision data"""
        vision_data = self.get_vision_data()
        if not vision_data:
            return False
            
        # Check if there's an object directly in front
        is_object = vision_data.get('is_object', False)
        distance_object = vision_data.get('distance_object', 1000)
        
        if is_object and distance_object < self.MAX_OBSTACLE_DIST:
            self.get_logger().info(f'Obstacle detected at distance {distance_object}cm')
            return True
            
        return False

    def scan_surroundings(self):
        """Scan surroundings by rotating 360 degrees"""
        self.get_logger().info('Scanning surroundings...')
        
        # Rotate in increments
        full_rotation = 360
        increment = self.ROTATION_ANGLE
        
        for _ in range(int(full_rotation / increment)):
            # Update map at current orientation
            self.update_map_with_vision()
            
            # Rotate to next position
            self.rotate("left", increment)
            
            # Process ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.01)
            
        # Final update after completing rotation
        self.update_map_with_vision()
        self.get_logger().info('Scan complete')

    def wait_for_start(self):
        """Wait until the start flag is set to True"""
        self.get_logger().info('Waiting for start signal...')
        
        while rclpy.ok() and not position_tracker.get_start_flag():
            # Process ROS callbacks while waiting
            rclpy.spin_once(self, timeout_sec=0.5)
            
        self.get_logger().info('Start signal received! Beginning exploration...')
        return True

    def explore(self):
        """Main exploration loop using frontier detection"""
        # Wait for start signal before beginning
        if not self.wait_for_start():
            return
            
        self.get_logger().info('Starting exploration')
        
        try:
            # Initial scan
            self.scan_surroundings()
            
            # Use a counter to limit exploration time
            exploration_counter = 0
            max_exploration_steps = 100  # Adjust based on desired exploration duration
            
            while rclpy.ok() and exploration_counter < max_exploration_steps:
                # Update map with current vision data
                self.update_map_with_vision()
                
                # Detect frontiers
                frontiers_exist = self.detect_frontiers()
                
                if frontiers_exist:
                    # Move to frontier
                    success = self.move_to_frontier()
                    
                    if not success:
                        # If failed to move to frontier, try scanning
                        self.scan_surroundings()
                else:
                    # No frontiers found, scan again
                    self.get_logger().info('No frontiers found, performing scan')
                    self.scan_surroundings()
                    
                    # If still no frontiers, try random exploration
                    if not self.detect_frontiers():
                        self.get_logger().info('Still no frontiers, moving randomly')
                        self.rotate("left", random.randint(30, 120))
                        if not self.check_for_obstacles():
                            self.move_forward(1.0)
                
                # Print exploration statistics
                self.get_logger().info(f'Explored {len(self.visited_positions)} grid cells')
                self.get_logger().info(f'Rotation history contains {len(self.position_history)} entries')
                
                # Increment counter
                exploration_counter += 1
                
                # Process ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.1)
                
            self.get_logger().info('Exploration complete')
            
            # Return to start position
            self.return_to_start_position()
                
        except KeyboardInterrupt:
            self.get_logger().info('Exploration stopped by user')
            # Try to return to start position even if interrupted
            try:
                self.return_to_start_position()
            except Exception as e:
                self.get_logger().error(f'Failed to return to start: {str(e)}')
        finally:
            self.stop()
            self.get_logger().info('Final position: ({:.1f}, {:.1f})'.format(self.x, self.y))
            self.get_logger().info(f'Total rotations: {self.rotation_count}')

def main():
    rclpy.init()
    node = TurtleBotFrontierDetection()
    
    try:
        node.explore()
    finally:
        node.stop()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
