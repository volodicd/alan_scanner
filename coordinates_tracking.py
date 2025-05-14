#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from kobuki_ros_interfaces.msg import BumperEvent
import time
import math
import requests
from requests.exceptions import RequestException
import numpy as np

class SimpleFrontierExplorer(Node):
    def __init__(self):
        super().__init__('simple_frontier_explorer')

        # Publishers and Subscribers
        self.vel_pub = self.create_publisher(Twist, 'commands/velocity', 10)
        self.bumper_sub = self.create_subscription(BumperEvent, 'events/bumper', self.bumper_callback, 10)

        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        
        # Simple grid map (much smaller)
        self.grid_size = 20  # 2m x 2m grid only
        self.cell_size = 0.1  # 10cm cells
        self.visited = set()  # Positions we've been to
        self.obstacles = set()  # Where we hit stuff
        
        # Movement parameters
        self.linear_vel = 0.15
        self.angular_vel = 1.5
        
        # Exploration targets
        self.targets = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Simple cardinal directions
        self.current_target_index = 0
        self.stuck_count = 0
        
        # API for coordinates
        self.coordinate_api = "http://localhost:5002/coordinates"
        
        self.get_logger().info('Simple Frontier Explorer initialized')

    def get_current_cell(self):
        """Get which grid cell we're in"""
        grid_x = int(self.x / self.cell_size + self.grid_size / 2)
        grid_y = int(self.y / self.cell_size + self.grid_size / 2)
        return (grid_x, grid_y)

    def find_next_target(self):
        """Find next unvisited direction to explore"""
        # Try each direction
        for i in range(4):
            # Get target in this direction
            target_x = self.x + self.targets[i][0] * 1.0  # 1 meter in that direction
            target_y = self.y + self.targets[i][1] * 1.0
            
            # Check if we've already been near there
            target_cell = (int(target_x / self.cell_size + self.grid_size / 2),
                          int(target_y / self.cell_size + self.grid_size / 2))
            
            if target_cell not in self.visited and target_cell not in self.obstacles:
                return (target_x, target_y)
        
        # If all explored, go random
        angle = np.random.uniform(0, 2*math.pi)
        return (self.x + 1.0 * math.cos(angle), self.y + 1.0 * math.sin(angle))

    def navigate_to_target(self, target_x, target_y):
        """Simple navigation to target"""
        # Calculate heading to target
        dx = target_x - self.x
        dy = target_y - self.y
        desired_heading = math.atan2(dy, dx)
        heading_error = self.normalize_angle(desired_heading - self.heading)
        
        # Turn or move based on error
        if abs(heading_error) > 0.2:  # Need to turn more
            # Turn toward target
            turn_speed = self.angular_vel if heading_error > 0 else -self.angular_vel
            self.publish_velocity(0.0, turn_speed)
        else:
            # Move forward
            self.publish_velocity(self.linear_vel, 0.0)

    def check_completion(self):
        """Simple check if exploration is done"""
        # If we've visited enough cells, we're done
        coverage = len(self.visited) / (self.grid_size * self.grid_size)
        return coverage > 0.9  # Explored 90% of grid

    def update_position(self, linear, angular, dt):
        """Update robot position"""
        self.heading += angular * dt
        self.heading = self.normalize_angle(self.heading)
        self.x += linear * dt * math.cos(self.heading)
        self.y += linear * dt * math.sin(self.heading)
        
        # Mark current cell as visited
        self.visited.add(self.get_current_cell())

    def send_coordinates(self):
        """Send position to API or print"""
        try:
            data = {'x': round(self.x, 3), 'y': round(self.y, 3)}
            requests.post(self.coordinate_api, json=data, timeout=0.5)
        except RequestException:
            self.get_logger().info(f'Position: x={self.x:.2f}, y={self.y:.2f}')

    def publish_velocity(self, linear, angular):
        """Publish velocity commands"""
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.vel_pub.publish(msg)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def bumper_callback(self, msg):
        """Handle bumper collision"""
        if msg.state == BumperEvent.PRESSED:
            self.get_logger().warn('Bumper hit! Backing up')
            
            # Mark obstacle
            self.obstacles.add(self.get_current_cell())
            
            # Back up and turn
            self.publish_velocity(-0.1, 0.0)
            time.sleep(0.5)
            self.publish_velocity(0.0, self.angular_vel)
            time.sleep(1.0)
            
            # Reset stuck counter
            self.stuck_count = 0

    def explore(self, duration=300):
        """Main exploration loop"""
        start_time = time.time()
        last_update = start_time
        last_velocity = (0.0, 0.0)
        current_target = None
        
        self.get_logger().info('Starting simple frontier exploration')
        
        while time.time() - start_time < duration:
            current_time = time.time()
            dt = current_time - last_update
            last_update = current_time
            
            # Update position
            self.update_position(last_velocity[0], last_velocity[1], dt)
            
            # Check if done
            if self.check_completion():
                self.get_logger().info('Exploration complete!')
                break
            
            # Get new target if needed
            if current_target is None or self.reached_target(current_target):
                current_target = self.find_next_target()
                self.get_logger().info(f'New target: {current_target}')
                self.stuck_count = 0
            
            # Navigate to target
            self.navigate_to_target(current_target[0], current_target[1])
            last_velocity = (self.linear_vel, 0.0)  # Simplified
            
            # Check if stuck
            if self.is_stuck():
                self.get_logger().warn('Robot stuck, finding new target')
                current_target = None
            
            # Send coordinates
            self.send_coordinates()
            
            # Progress log
            if int(current_time) % 10 == 0:
                coverage = len(self.visited) / (self.grid_size * self.grid_size) * 100
                self.get_logger().info(f'Explored: {coverage:.0f}%')
            
            time.sleep(0.1)
        
        # Clean up
        self.publish_velocity(0.0, 0.0)
        self.get_logger().info('Exploration finished')

    def reached_target(self, target):
        """Check if we're close to target"""
        dx = target[0] - self.x
        dy = target[1] - self.y
        return math.sqrt(dx*dx + dy*dy) < 0.2  # 20cm tolerance

    def is_stuck(self):
        """Simple stuck detection"""
        self.stuck_count += 1
        if self.stuck_count > 50:  # About 5 seconds at loop rate
            self.stuck_count = 0
            return True
        return False


def main():
    rclpy.init()
    explorer = SimpleFrontierExplorer()

    try:
        explorer.explore(duration=300)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
