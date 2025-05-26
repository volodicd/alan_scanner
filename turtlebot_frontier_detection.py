import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from kobuki_ros_interfaces.msg import BumperEvent
import requests
import time
import math
import random

from position_tracker import position_tracker
from AStar_path_finding import AStarPathfinder
from flask_api import flask_api_server
from wavefront_detector import WavefrontFrontierDetector


class MotionController:
    def __init__(self, node, speed=0.2, angular_speed=1.0, grid_size=50):
        self.node = node
        self.grid_size = grid_size
        self.linear_speed = speed
        self.angular_speed = angular_speed
        self.publisher = node.create_publisher(Twist, 'commands/velocity', 10)

    def stop(self):
        self._publish_velocity(0.0, 0.0)

    def move(self, linear=0.0, angular=0.0, duration=0.5):
        self._publish_velocity(linear, angular)
        time.sleep(duration)
        self.stop()

    def move_forward(self, heading, duration=0.5):
        self.move(self.linear_speed, 0.0, duration)
        distance = duration * self.linear_speed * 100
        dx = distance * math.cos(math.radians(heading))
        dy = distance * math.sin(math.radians(heading))
        return dx, dy

    def move_backward(self, heading, duration=0.5):
        self.move(-self.linear_speed, 0.0, duration)
        distance = duration * self.linear_speed * 100
        dx = -distance * math.cos(math.radians(heading))
        dy = -distance * math.sin(math.radians(heading))
        return dx, dy

    def rotate(self, direction, angle):
        angular_vel = self.angular_speed if direction == "left" else -self.angular_speed
        duration = math.radians(angle) / abs(angular_vel)
        self.move(0.0, angular_vel, duration)

    def _publish_velocity(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.publisher.publish(msg)


class TurtleBot(Node):
    def __init__(self):
        super().__init__('turtlebot_wfd')
        self.motion = MotionController(self)
        self.bumper_sub = self.create_subscription(BumperEvent, 'events/bumper', self.bumper_callback, 10)

        self.grid_size = 50
        self.x, self.y, self.heading = 0.0, 0.0, 0
        self.rotation_count = 0
        self.vision_api_url = "http://localhost:5000/api/turtlebot/vision"

        self.wfd = WavefrontFrontierDetector(self.grid_size)
        self.astar = AStarPathfinder(self.grid_size)

        self.frontiers = []
        self.visited = set()
        self.consecutive_failures = 0

        position_tracker.reset_position()
        position_tracker.set_initial_position(self.x, self.y, self.heading)
        self.get_logger().info("TurtleBot initialized")

    def bumper_callback(self, msg):
        if msg.state == BumperEvent.PRESSED:
            self.get_logger().info("Bumper pressed")
            dx, dy = self.motion.move_backward(self.heading, 1.0)
            self.update_position(dx, dy)
            self.motion.rotate("left", 45)
            self.rotation_count += 1

    def update_position(self, dx, dy):
        self.x += dx
        self.y += dy
        gx, gy = self.to_grid(self.x, self.y)
        self.visited.add((gx, gy))
        self.wfd.update_grid(gx, gy, self.wfd.OPEN_SPACE)
        position_tracker.update_position(self.x, self.y, self.heading)

    def to_grid(self, x, y):
        return (round(x / self.grid_size) * self.grid_size,
                round(y / self.grid_size) * self.grid_size)

    def get_vision_data(self):
        try:
            res = requests.get(self.vision_api_url, timeout=1.0)
            return res.json() if res.status_code == 200 else None
        except:
            return None

    def scan_surroundings(self):
        for _ in range(0, 360, 45):
            self.update_map_with_vision()
            self.motion.rotate("left", 45)

    def update_map_with_vision(self):
        data = self.get_vision_data()
        if not data:
            return

        self.wfd.update_grid(*self.to_grid(self.x, self.y), self.wfd.OPEN_SPACE)

        for idx, dist in enumerate(data.get("objs", [1000]*16)):
            if dist > 900:
                continue
            angle_h = -45 + (idx % 4) * 30
            abs_angle = (self.heading + angle_h) % 360
            px = self.x + min(dist, 500) * math.cos(math.radians(abs_angle))
            py = self.y + min(dist, 500) * math.sin(math.radians(abs_angle))
            gx, gy = self.to_grid(px, py)
            cell = self.wfd.OCCUPIED if dist < 50 else self.wfd.OPEN_SPACE
            self.wfd.update_grid(gx, gy, cell)
            if cell == self.wfd.OCCUPIED:
                self.astar.add_obstacle(gx, gy)

    def explore(self):
        flask_api_server.start()
        self.scan_surroundings()

        while rclpy.ok():
            self.update_map_with_vision()
            self.astar.sync_obstacles_from_grid(self.wfd.grid, self.wfd.OCCUPIED)
            frontiers = self.wfd.detect_frontiers(self.x, self.y)

            if frontiers:
                self.consecutive_failures = 0
                fx, fy, _ = frontiers[0]
                success = self.astar.navigate(self, fx, fy)
                if success:
                    self.update_map_with_vision()
                else:
                    self.scan_surroundings()
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    break
                self.scan_surroundings()

        flask_api_server.stop()
        self.get_logger().info("Exploration complete")


def main():
    rclpy.init()
    node = TurtleBot()
    try:
        node.explore()
    finally:
        node.motion.stop()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
