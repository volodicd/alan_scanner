#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from kobuki_ros_interfaces.msg import BumperEvent
import time
import math


class SimpleTurtleBot(Node):
    def __init__(self):
        super().__init__('simple_turtlebot')

        # Publishers
        self.vel_pub = self.create_publisher(Twist, 'commands/velocity', 10)

        # Subscribers
        self.bumper_sub = self.create_subscription(
            BumperEvent, 'events/bumper', self.bumper_callback, 10)

        # State
        self.obstacle_detected = False
        self.obstacle_direction = None

        self.get_logger().info('Simple TurtleBot initialized')

    def bumper_callback(self, msg):
        """Handle bumper events"""
        if msg.state == BumperEvent.PRESSED:
            self.obstacle_detected = True
            self.obstacle_direction = msg.bumper  # LEFT, CENTER, or RIGHT
            self.get_logger().info(f'Bumper hit: {self.obstacle_direction}')
            self.stop()
        else:
            self.obstacle_detected = False

    def move_forward(self, duration=None):
        """Move forward"""
        msg = Twist()
        msg.linear.x = 0.2  # Speed in m/s
        msg.angular.z = 0.0
        self.vel_pub.publish(msg)

        if duration:
            time.sleep(duration)
            self.stop()

    def rotate(self, duration):
        """Rotate the robot"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.5  # Turn right by default
        self.vel_pub.publish(msg)

        time.sleep(duration)
        self.stop()

    def stop(self):
        """Stop the robot"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.vel_pub.publish(msg)

    def explore(self, duration=120):
        """Simple exploration algorithm"""
        self.get_logger().info('Starting exploration')
        start_time = time.time()

        while time.time() - start_time < duration:
            # If no obstacle, keep moving forward
            if not self.obstacle_detected:
                self.move_forward()
                time.sleep(0.1)  # Check for obstacles every 0.1 seconds
                rclpy.spin_once(self, timeout_sec=0.01)
            else:
                # If obstacle detected, stop and turn
                self.stop()
                self.get_logger().info('Obstacle detected, turning...')
                self.rotate(2.0)  # Turn for 2 seconds
                self.obstacle_detected = False  # Reset obstacle detection

        self.stop()
        self.get_logger().info('Exploration completed')


def main():
    rclpy.init()
    bot = SimpleTurtleBot()

    try:
        bot.explore(duration=120)  # Explore for 2 minutes
    finally:
        bot.stop()
        rclpy.shutdown()


if __name__ == '__main__':
    main()