#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from kobuki_ros_interfaces.msg import BumperEvent
import requests
import time
import math
import random
import numpy as np
from collections import deque
import threading

# Import the shared position tracker
from position_tracker import position_tracker

class WavefrontFrontierDetector:
    """Implementation of the Wavefront Frontier Detector algorithm"""
    
    def __init__(self, grid_resolution=50):
        # Grid resolution in cm
        self.grid_resolution = grid_resolution
        
        # Map representation
        self.grid = {}  # Dictionary-based sparse grid
        
        # Cell classifications
        self.UNKNOWN = 0
        self.OPEN_SPACE = 1
        self.OCCUPIED = 2
        
        # Lists for WFD algorithm
        self.map_open_list = set()
        self.map_close_list = set()
        self.frontier_open_list = set()
        self.frontier_close_list = set()
        
        # Store frontiers
        self.frontiers = []
    
    def update_grid(self, x, y, cell_type):
        """Update a cell in the grid"""
        grid_x = round(x / self.grid_resolution) * self.grid_resolution
        grid_y = round(y / self.grid_resolution) * self.grid_resolution
        self.grid[(grid_x, grid_y)] = cell_type
    
    def get_cell_type(self, x, y):
        """Get the type of a cell in the grid"""
        grid_x = round(x / self.grid_resolution) * self.grid_resolution
        grid_y = round(y / self.grid_resolution) * self.grid_resolution
        return self.grid.get((grid_x, grid_y), self.UNKNOWN)
    
    def is_frontier_point(self, x, y):
        """Check if a point is a frontier point (unknown with at least one open neighbor)"""
        # If the cell is not unknown, it's not a frontier
        if self.get_cell_type(x, y) != self.UNKNOWN:
            return False
        
        # Check if it has at least one open space neighbor
        directions = [
            (0, self.grid_resolution), (self.grid_resolution, 0),
            (0, -self.grid_resolution), (-self.grid_resolution, 0)
        ]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.get_cell_type(nx, ny) == self.OPEN_SPACE:
                return True
        
        return False
    
    def detect_frontiers(self, robot_x, robot_y):
        """Detect frontiers using Wavefront Frontier Detector algorithm"""
        # Clear previous frontier data
        self.map_open_list.clear()
        self.map_close_list.clear()
        self.frontier_open_list.clear()
        self.frontier_close_list.clear()
        self.frontiers.clear()
        
        # Initialize queue for first BFS (map exploration)
        queue_m = deque()
        
        # Start from robot position
        start_x = round(robot_x / self.grid_resolution) * self.grid_resolution
        start_y = round(robot_y / self.grid_resolution) * self.grid_resolution
        
        # Add robot position to queue
        queue_m.append((start_x, start_y))
        self.map_open_list.add((start_x, start_y))
        
        # Neighbor directions
        directions = [
            (0, self.grid_resolution), (self.grid_resolution, 0),
            (0, -self.grid_resolution), (-self.grid_resolution, 0)
        ]
        
        # First BFS: find frontier points
        while queue_m:
            p_x, p_y = queue_m.popleft()
            
            # If already processed, skip
            if (p_x, p_y) in self.map_close_list:
                continue
            
            # Mark as processed
            self.map_close_list.add((p_x, p_y))
            
            # If it's a frontier point, start second BFS to find the whole frontier
            if self.is_frontier_point(p_x, p_y):
                # Initialize new frontier
                new_frontier = []
                
                # Initialize queue for second BFS (frontier extraction)
                queue_f = deque()
                queue_f.append((p_x, p_y))
                self.frontier_open_list.add((p_x, p_y))
                
                # Second BFS: extract connected frontier points
                while queue_f:
                    q_x, q_y = queue_f.popleft()
                    
                    # If already in map_close_list or frontier_close_list, skip
                    if ((q_x, q_y) in self.map_close_list or 
                        (q_x, q_y) in self.frontier_close_list):
                        continue
                    
                    # Mark as processed in frontier list
                    self.frontier_close_list.add((q_x, q_y))
                    
                    # If it's a frontier point, add to the current frontier
                    if self.is_frontier_point(q_x, q_y):
                        new_frontier.append((q_x, q_y))
                        
                        # Add neighbors to queue_f
                        for dx, dy in directions:
                            w_x, w_y = q_x + dx, q_y + dy
                            
                            # Only add if not already in open or closed lists
                            if ((w_x, w_y) not in self.frontier_open_list and 
                                (w_x, w_y) not in self.frontier_close_list):
                                queue_f.append((w_x, w_y))
                                self.frontier_open_list.add((w_x, w_y))
                
                # Mark all points in this frontier as closed in map list to avoid reprocessing
                for fx, fy in new_frontier:
                    self.map_close_list.add((fx, fy))
                
                # Only add if the frontier has enough points
                if len(new_frontier) >= 3:  # Minimum size to be considered a frontier
                    self.frontiers.append(new_frontier)
            
            # Add neighbors to queue_m for map exploration
            for dx, dy in directions:
                v_x, v_y = p_x + dx, p_y + dy
                
                # Only add if not already in open or closed lists and is open space
                if ((v_x, v_y) not in self.map_open_list and 
                    (v_x, v_y) not in self.map_close_list and
                    self.get_cell_type(v_x, v_y) == self.OPEN_SPACE):
                    queue_m.append((v_x, v_y))
                    self.map_open_list.add((v_x, v_y))
        
        # Calculate median points for each frontier
        frontier_medians = []
        for frontier in self.frontiers:
            if frontier:
                # Calculate median x and y
                xs = [x for x, y in frontier]
                ys = [y for x, y in frontier]
                median_x = sorted(xs)[len(xs) // 2]
                median_y = sorted(ys)[len(ys) // 2]
                
                # Calculate distance to robot
                dist = math.sqrt((median_x - robot_x)**2 + (median_y - robot_y)**2)
                
                frontier_medians.append((median_x, median_y, dist))
        
        # Sort frontiers by distance from robot
        frontier_medians.sort(key=lambda f: f[2])
        
        return frontier_medians

class TurtleBotWFD(Node):
    def __init__(self):
        super().__init__('turtlebot_wfd')
        
        # Publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, 'commands/velocity', 10)
        self.bumper_sub = self.create_subscription(BumperEvent, 'events/bumper', self.bumper_callback, 10)
        
        # Vision API endpoint
        self.vision_api_url = "http://localhost:5000/api/turtlebot/vision"
        
        # Robot state - initialize from position tracker
        self.x = 0.0  # starting x position in cm
        self.y = 0.0  # starting y position in cm
        self.heading = 0  # degrees (0 = east, 90 = north)
        self.obstacle_detected = False
        
        # Map state
        self.grid_size = 50  # cm
        self.position_history = {}  # Dictionary to store positions after each rotation
        self.rotation_count = 0  # Counter for rotations
        self.visited_positions = set()  # Set of visited grid positions
        
        # Create Wavefront Frontier Detector
        self.wfd = WavefrontFrontierDetector(grid_resolution=self.grid_size)
        
        # Exploration state tracking
        self.consecutive_no_frontier_scans = 0
        self.max_consecutive_no_frontier_scans = 3
        
        # Constants
        self.MIN_FRONTIER_DIST = 100  # Minimum distance to consider a frontier (cm)
        self.MAX_OBSTACLE_DIST = 50   # Maximum distance to consider an obstacle (cm)
        self.ROTATION_ANGLE = 45      # Angle to rotate for scanning (degrees)
        
        # Reset position tracker and store initial position
        position_tracker.reset_position()
        position_tracker.set_initial_position(self.x, self.y, self.heading)
        position_tracker.set_finished_flag(False)  # Ensure finished flag is reset
        
        self.get_logger().info('TurtleBot with Wavefront Frontier Detector initialized at (0, 0)')
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
            
            # Update grid with open space
            self.wfd.update_grid(self.x, self.y, self.wfd.OPEN_SPACE)
            
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
            
            # Update grid with open space
            self.wfd.update_grid(self.x, self.y, self.wfd.OPEN_SPACE)
            
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
        
        # Update grid for current position (mark as open space)
        grid_x = round(self.x / self.grid_size) * self.grid_size
        grid_y = round(self.y / self.grid_size) * self.grid_size
        self.wfd.update_grid(grid_x, grid_y, self.wfd.OPEN_SPACE)
        
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
                    
                    # If distance is short, mark as obstacle
                    if point_distance < self.MAX_OBSTACLE_DIST:
                        self.wfd.update_grid(point_grid_x, point_grid_y, self.wfd.OCCUPIED)
                    else:
                        # Mark as open space if distance is valid
                        self.wfd.update_grid(point_grid_x, point_grid_y, self.wfd.OPEN_SPACE)
                        
                        # Add a few cells beyond this as unknown (potential frontiers)
                        for k in range(1, 4):
                            beyond_x = point_x + k * self.grid_size * math.cos(math.radians(abs_angle))
                            beyond_y = point_y + k * self.grid_size * math.sin(math.radians(abs_angle))
                            beyond_grid_x = round(beyond_x / self.grid_size) * self.grid_size
                            beyond_grid_y = round(beyond_y / self.grid_size) * self.grid_size
                            
                            # Only mark as unknown if not already classified
                            if self.wfd.get_cell_type(beyond_grid_x, beyond_grid_y) == self.wfd.UNKNOWN:
                                self.wfd.update_grid(beyond_grid_x, beyond_grid_y, self.wfd.UNKNOWN)
        
        return True

    def detect_frontiers(self):
        """Use WFD algorithm to detect frontiers"""
        # First update map with vision data
        self.update_map_with_vision()
        
        # Use Wavefront Frontier Detector to find frontiers
        frontier_medians = self.wfd.detect_frontiers(self.x, self.y)
        
        self.get_logger().info(f'Detected {len(frontier_medians)} frontiers')
        
        # Store frontiers
        self.frontiers = frontier_medians
        
        return len(frontier_medians) > 0

    def is_exploration_complete(self):
        """Determine if exploration is complete based on frontier detection"""
        # If we've done multiple full scans and still found no frontiers, 
        # consider exploration complete
        if self.consecutive_no_frontier_scans >= self.max_consecutive_no_frontier_scans:
            self.get_logger().info(f"No frontiers found after {self.consecutive_no_frontier_scans} complete scans")
            self.get_logger().info("Exploration appears to be complete - all accessible areas mapped")
            return True
        
        # Exploration is not yet complete
        return False

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
        
        # Set the finished flag to true
        position_tracker.set_finished_flag(True)
        
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
        """Main exploration loop using Wavefront Frontier Detection"""
        # Make sure the finished flag is initially false
        position_tracker.set_finished_flag(False)
        
        # Wait for start signal before beginning
        if not self.wait_for_start():
            return
            
        self.get_logger().info('Starting exploration with Wavefront Frontier Detector')
        
        try:
            # Initial scan
            self.scan_surroundings()
            
            # Reset exploration tracking variables
            self.consecutive_no_frontier_scans = 0
            
            # Main exploration loop - continues until we determine exploration is complete
            while rclpy.ok():
                # Detect frontiers using WFD
                frontiers_exist = self.detect_frontiers()
                
                if frontiers_exist:
                    # Found frontiers - reset consecutive scan counter
                    self.consecutive_no_frontier_scans = 0
                    
                    # Move to frontier
                    success = self.move_to_frontier()
                    
                    if not success:
                        # If failed to move to frontier, try scanning
                        self.scan_surroundings()
                else:
                    # No frontiers found, scan again
                    self.get_logger().info('No frontiers found, performing full scan')
                    self.scan_surroundings()
                    
                    # Check again after scanning
                    if not self.detect_frontiers():
                        # Increment consecutive no-frontier counter
                        self.consecutive_no_frontier_scans += 1
                        self.get_logger().info(f'Still no frontiers after scan ({self.consecutive_no_frontier_scans}/{self.max_consecutive_no_frontier_scans})')
                        
                        # Try random movement to discover new areas
                        self.get_logger().info('Moving randomly to try to discover new areas')
                        self.rotate("left", random.randint(30, 120))
                        if not self.check_for_obstacles():
                            self.move_forward(1.0)
                
                # Print exploration statistics
                self.get_logger().info(f'Explored {len(self.visited_positions)} grid cells')
                self.get_logger().info(f'Rotation history contains {len(self.position_history)} entries')
                
                # Check if exploration is complete based on frontier analysis
                if self.is_exploration_complete():
                    self.get_logger().info('WAVEFRONT FRONTIER DETECTION COMPLETE')
                    break
                
                # Process ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.1)
                
            self.get_logger().info('Exploration complete')
            
            # Return to start position
            self.return_to_start_position()
            
            # Set finish flag (also set in return_to_start_position, but setting again for certainty)
            position_tracker.set_finished_flag(True)
                
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
    node = TurtleBotWFD()
    
    try:
        node.explore()
    finally:
        node.stop()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
