#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from kobuki_ros_interfaces.msg import BumperEvent
import time
import math
import heapq

class TurtleBotAStar(Node):
    def __init__(self):
        super().__init__('turtlebot_astar')
        
        # Publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, 'commands/velocity', 10)
        self.bumper_sub = self.create_subscription(BumperEvent, 'events/bumper', self.bumper_callback, 10)
        
        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.heading = 0  # degrees (0 = east, 90 = north)
        self.obstacle_detected = False
        
        # A* navigation
        self.grid_size = 50
        self.obstacles = set()  # Set of (grid_x, grid_y) positions
        self.path = []
        self.target_x = None
        self.target_y = None
        
        self.get_logger().info('TurtleBot initialized at (0, 0)')

    def bumper_callback(self, msg):
        """Handle bumper events"""
        if msg.state == BumperEvent.PRESSED:
            self.obstacle_detected = True
            
            # Add current position to obstacles
            grid_x = round(self.x / self.grid_size) * self.grid_size
            grid_y = round(self.y / self.grid_size) * self.grid_size
            self.obstacles.add((grid_x, grid_y))
            
            self.get_logger().info(f'Obstacle detected at grid ({grid_x}, {grid_y})')
            self.stop()
            
            # Handle obstacle
            self.move_backward(1.0)
            self.rotate("left", 45)
            self.obstacle_detected = False
            
            # Recalculate path
            if self.target_x is not None:
                self.path = self.find_path(self.x, self.y, self.target_x, self.target_y)

    def move_forward(self, duration=0.5):
        """Move forward"""
        msg = Twist()
        msg.linear.x = 0.2  # m/s
        self.vel_pub.publish(msg)
        
        if duration:
            time.sleep(duration)
            # Update position
            self.x += duration * 0.2 * 100 * math.cos(math.radians(self.heading))
            self.y += duration * 0.2 * 100 * math.sin(math.radians(self.heading))
            self.stop()

    def move_backward(self, duration=0.5):
        """Move backward"""
        msg = Twist()
        msg.linear.x = -0.2  # m/s
        self.vel_pub.publish(msg)
        
        if duration:
            time.sleep(duration)
            # Update position
            self.x -= duration * 0.2 * 100 * math.cos(math.radians(self.heading))
            self.y -= duration * 0.2 * 100 * math.sin(math.radians(self.heading))
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

    def stop(self):
        """Stop the robot"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.vel_pub.publish(msg)

    def find_path(self, start_x, start_y, goal_x, goal_y):
        """Find path using A* algorithm"""
        # Round to grid
        start_x = round(start_x / self.grid_size) * self.grid_size
        start_y = round(start_y / self.grid_size) * self.grid_size
        goal_x = round(goal_x / self.grid_size) * self.grid_size
        goal_y = round(goal_y / self.grid_size) * self.grid_size
        
        # A* algorithm data structures
        open_list = []  # Priority queue (f_score, (x, y))
        closed_set = set()
        g_score = {(start_x, start_y): 0}
        came_from = {}
        
        # Add start to open list
        f = self.heuristic(start_x, start_y, goal_x, goal_y)
        heapq.heappush(open_list, (f, (start_x, start_y)))
        
        # Directions: right, up, left, down
        directions = [(self.grid_size, 0), (0, self.grid_size), 
                     (-self.grid_size, 0), (0, -self.grid_size)]
        
        while open_list:
            # Get node with lowest f-score
            _, current = heapq.heappop(open_list)
            
            # Check if goal reached
            if current == (goal_x, goal_y):
                # Reconstruct path
                path = [(goal_x, goal_y)]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
                
            # Add to closed set
            closed_set.add(current)
            
            # Check neighbors
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                # Skip if out of bounds or obstacle
                if not (-500 <= nx <= 500 and -500 <= ny <= 500) or neighbor in self.obstacles:
                    continue
                    
                # Skip if in closed set
                if neighbor in closed_set:
                    continue
                    
                # Calculate tentative g score
                g = g_score[current] + self.grid_size
                
                # Update if better path found
                if neighbor not in g_score or g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = g
                    f = g + self.heuristic(nx, ny, goal_x, goal_y)
                    heapq.heappush(open_list, (f, neighbor))
        
        # No path found
        return []

    def heuristic(self, x1, y1, x2, y2):
        """Calculate Manhattan distance heuristic"""
        return abs(x1 - x2) + abs(y1 - y2)

    def navigate_to_target(self):
        """Navigate to user-specified target"""
        # Get target coordinates
        print("\nEnter target coordinates:")
        self.target_x = float(input("X coordinate (-500 to 500): "))
        self.target_y = float(input("Y coordinate (-500 to 500): "))
        
        # Make sure coordinates are in range
        if not (-500 <= self.target_x <= 500 and -500 <= self.target_y <= 500):
            print("Coordinates must be between -500 and 500.")
            return
            
        self.get_logger().info(f"Navigating to target ({self.target_x}, {self.target_y})")
        
        # Find initial path
        self.path = self.find_path(self.x, self.y, self.target_x, self.target_y)
        
        if not self.path:
            self.get_logger().warning("No path found to target!")
            return
            
        # Follow path
        for i, (waypoint_x, waypoint_y) in enumerate(self.path):
            # Skip first waypoint (current position)
            if i == 0:
                continue
                
            # Try to reach this waypoint
            while True:
                # Check if obstacle detected
                if self.obstacle_detected:
                    break
                    
                # Calculate direction to waypoint
                dx = waypoint_x - self.x
                dy = waypoint_y - self.y
                target_heading = math.degrees(math.atan2(dy, dx)) % 360
                
                # Calculate distance to waypoint
                distance = math.sqrt(dx*dx + dy*dy)
                
                # If reached waypoint
                if distance < 25:
                    self.get_logger().info(f"Reached waypoint {i}/{len(self.path)-1}")
                    break
                    
                # Turn toward waypoint
                heading_diff = (target_heading - self.heading + 180) % 360 - 180
                
                if abs(heading_diff) > 10:
                    # Need to turn
                    direction = "left" if heading_diff > 0 else "right"
                    self.rotate(direction, min(abs(heading_diff), 30))
                else:
                    # Move forward
                    self.move_forward(0.5)
                    
                # Process ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.01)
                
            # If obstacle encountered, recalculate path
            if self.obstacle_detected:
                self.path = self.find_path(self.x, self.y, self.target_x, self.target_y)
                if not self.path:
                    self.get_logger().warning("No new path found!")
                    break
                # Start from beginning of new path
                i = 0
        
        self.get_logger().info(f"Navigation finished at position ({self.x:.1f}, {self.y:.1f})")

def main():
    rclpy.init()
    bot = TurtleBotAStar()
    
    try:
        bot.navigate_to_target()
    finally:
        bot.stop()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
