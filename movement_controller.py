#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool, String
from transforms3d.euler import quat2euler


class TurtlebotMovementController(Node):
    def __init__(self):
        super().__init__('movement_controller')

        # ----------------------------------------------------------------------
        # PUBLISHERS & SUBSCRIBERS
        # ----------------------------------------------------------------------
        # Kobuki listens on /commands/velocity
        self.cmd_vel_pub = self.create_publisher(Twist, '/commands/velocity', 10)

        # Debug
        self.debug_pub = self.create_publisher(String, '/debug/status', 10)
        self.obstacle_debug_pub = self.create_publisher(Bool, '/debug/obstacle_detected', 10)

        # Bumper subscription
        try:
            from kobuki_ros_interfaces.msg import BumperEvent
            self.bumper_sub = self.create_subscription(
                BumperEvent,
                '/events/bumper',
                self.bumper_callback,
                10
            )
            self.get_logger().info("Subscribed to Kobuki bumper events.")
            # We'll store each bumper's state to handle multi-bumper logic
            self.bumper_states = [0, 0, 0]  # left=0, center=1, right=2
        except ImportError:
            self.bumper_sub = None
            self.bumper_states = []
            self.get_logger().warn("kobuki_ros_interfaces not available; no bumper subscription.")

        # Odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Move distance commands
        self.distance_sub = self.create_subscription(
            Float32,
            '/move_distance',
            self.distance_callback,
            10
        )

        # Optional laser scan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # ----------------------------------------------------------------------
        # INTERNAL STATE
        # ----------------------------------------------------------------------
        self.obstacle_detected = False
        self.obstacle_distance = 999.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.prev_x = None
        self.prev_y = None
        self.distance_traveled = 0.0
        self.target_distance = 0.0

        self.current_twist = Twist()

        # Simple state machine: INITIALIZING -> IDLE -> MOVING -> TURNING -> IDLE
        self.state = 'INITIALIZING'
        self.turn_end_time = 0.0

        # ----------------------------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------------------------
        self.declare_parameter('obstacle_threshold', 0.3)
        self.declare_parameter('linear_speed', 0.2)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('update_rate', 10.0)
        self.declare_parameter('cmd_rate', 5.0)

        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.update_rate = self.get_parameter('update_rate').value
        self.cmd_rate = self.get_parameter('cmd_rate').value

        # ----------------------------------------------------------------------
        # TIMERS
        # ----------------------------------------------------------------------
        self.control_timer = self.create_timer(1.0 / self.update_rate, self.control_loop)
        self.cmd_timer = self.create_timer(1.0 / self.cmd_rate, self.publish_cmd_vel)
        self.debug_timer = self.create_timer(1.0, self.print_status)

        # After 3s, switch to IDLE
        self.init_timer = self.create_timer(3.0, self.initialize_complete)

        self.get_logger().info("Movement controller initialized.")

    # ----------------------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------------------
    def initialize_complete(self):
        self.init_timer.cancel()
        self.state = 'IDLE'
        self.get_logger().info("Initialization complete, now IDLE.")

    # ----------------------------------------------------------------------
    # SUBSCRIBER CALLBACKS
    # ----------------------------------------------------------------------
    def bumper_callback(self, msg):
        """
        Kobuki has three bumpers: 0=LEFT, 1=CENTER, 2=RIGHT.
        State: PRESSED=1, RELEASED=0.
        We'll set obstacle_detected=True if ANY bumper is pressed.
        If a bumper is released, we check if all are released => obstacle_detected=False.
        """
        from kobuki_ros_interfaces.msg import BumperEvent
        bumper_id = msg.bumper
        if bumper_id < 0 or bumper_id > 2:
            return

        if msg.state == BumperEvent.PRESSED:
            self.get_logger().warn(f"BUMPER {bumper_id} PRESSED!")
            self.bumper_states[bumper_id] = 1
        else:
            # Released
            self.get_logger().info(f"BUMPER {bumper_id} RELEASED")
            self.bumper_states[bumper_id] = 0

        # Now figure out if ANY bumper is pressed
        if any(self.bumper_states):
            self.obstacle_detected = True
            self.obstacle_distance = 0.05
        else:
            self.obstacle_detected = False
            self.obstacle_distance = 999.0

        self.publish_obstacle_state()

    def scan_callback(self, scan_data):
        """If you do have a LIDAR on /scan, you can detect front obstacle here."""
        if not scan_data.ranges:
            return

        try:
            n = len(scan_data.ranges)
            if n >= 360:
                front_indices = list(range(0, 45)) + list(range(315, 360))
            elif n >= 180:
                mid = n // 2
                front_indices = list(range(mid - 45, mid + 45))
            else:
                front_indices = range(n)

            valid = [scan_data.ranges[i] for i in front_indices
                     if 0.01 < scan_data.ranges[i] < 10.0
                     and not math.isinf(scan_data.ranges[i])]

            if valid:
                dist = min(valid)
                # If bumper is pressed, we keep obstacle_detected = True
                # Otherwise rely on LIDAR to set it:
                if not any(self.bumper_states):
                    # Only use LIDAR if no bumper is pressed
                    self.obstacle_distance = dist
                    self.obstacle_detected = dist < self.obstacle_threshold
                self.publish_obstacle_state()
        except Exception as e:
            self.get_logger().warn(f"Scan callback error: {e}")

    def odom_callback(self, odom):
        pos = odom.pose.pose.position
        self.current_x = pos.x
        self.current_y = pos.y

        q = odom.pose.pose.orientation
        roll, pitch, yaw = quat2euler([q.x, q.y, q.z, q.w])
        self.current_theta = yaw

        if self.prev_x is None:
            self.prev_x = self.current_x
            self.prev_y = self.current_y
            return

        # Update distance traveled if MOVING
        if self.state == 'MOVING':
            dx = self.current_x - self.prev_x
            dy = self.current_y - self.prev_y
            d = math.sqrt(dx * dx + dy * dy)
            if d < 0.5:  # discard jumps
                self.distance_traveled += d
                # Check if we reached the target
                if self.target_distance > 0 and self.distance_traveled >= self.target_distance:
                    self.get_logger().info(f"TARGET REACHED! {self.distance_traveled:.2f} m")
                    self.complete_movement()

        self.prev_x = self.current_x
        self.prev_y = self.current_y

    def distance_callback(self, msg):
        """User commands a distance. Positive => move, zero => stop."""
        dist = msg.data
        if dist > 0:
            self.get_logger().info(f"COMMAND: Move {dist:.2f} m")
            self.set_target_distance(dist)
        else:
            # dist = 0 => STOP
            self.get_logger().info("COMMAND: Stop")
            self.stop_robot()
            old = self.state
            self.state = 'IDLE'
            self.get_logger().info(f"Changed state: {old} -> IDLE")

    # ----------------------------------------------------------------------
    # CONTROL LOOP
    # ----------------------------------------------------------------------
    def control_loop(self):
        """Runs at update_rate Hz, handles turning state, obstacles, etc."""
        if self.state == 'TURNING':
            now = self.get_clock().now().seconds_nanoseconds()[0]
            if now >= self.turn_end_time and self.turn_end_time > 0:
                # done turning
                self.stop_robot()
                old_state = self.state
                self.state = 'MOVING'
                self.get_logger().info(f"Changed state: {old_state} -> MOVING")
                self.move_forward_direct()
            return

        # If we're moving and there's an obstacle => switch to IDLE
        if self.state == 'MOVING' and self.obstacle_detected:
            self.get_logger().info(f"Obstacle detected => stopping movement")
            self.stop_robot()
            old_state = self.state
            self.state = 'IDLE'
            self.get_logger().info(f"Changed state: {old_state} -> IDLE")

    # ----------------------------------------------------------------------
    # STATE CHANGES
    # ----------------------------------------------------------------------
    def set_target_distance(self, dist):
        """Prepare to move forward 'dist' meters."""
        self.stop_robot()
        old = self.state
        self.distance_traveled = 0.0
        self.target_distance = dist
        self.prev_x = self.current_x
        self.prev_y = self.current_y
        self.state = 'MOVING'
        self.get_logger().info(f"Changed state: {old} -> MOVING")
        self.move_forward_direct()
        self.get_logger().info(f"Starting movement for {dist:.2f} m")

    # ----------------------------------------------------------------------
    # MOTION COMMANDS
    # ----------------------------------------------------------------------
    def stop_robot(self):
        """Set twist to zero (cmd_timer keeps publishing)."""
        old_lin = self.current_twist.linear.x
        self.current_twist = Twist()
        self.cmd_vel_pub.publish(self.current_twist)

        if old_lin > 0:
            self.get_logger().info(f"Robot stopped (was moving at {old_lin:.2f} m/s)")

    def publish_cmd_vel(self):
        """Continuously publish the current Twist at self.cmd_rate Hz."""
        self.cmd_vel_pub.publish(self.current_twist)

    # ----------------------------------------------------------------------
    # DEBUG
    # ----------------------------------------------------------------------
    def publish_obstacle_state(self):
        msg = Bool()
        msg.data = self.obstacle_detected
        self.obstacle_debug_pub.publish(msg)

    def print_status(self):
        status = (
            f"STATE: {self.state} | "
            f"Obstacle: {self.obstacle_detected} ({self.obstacle_distance:.2f}m) | "
            f"Distance: {self.distance_traveled:.2f}/{self.target_distance:.2f}m | "
            f"Speed: {self.current_twist.linear.x:.2f}"
        )
        self.get_logger().info(status)
        dbg = String()
        dbg.data = status
        self.debug_pub.publish(dbg)

    # ----------------------------------------------------------------------
    # DIRECT CONTROL FUNCTIONS
    # ----------------------------------------------------------------------
    def move_forward_direct(self, speed=None, distance=None):
        """
        Direct control to move forward.
        Args:
            speed: Override default linear speed. If None, use self.linear_speed
            distance: Optional distance to travel. If None, move until stopped.
        """
        if self.obstacle_detected:
            self.get_logger().warn("Cannot move forward: obstacle detected")
            return False

        # If we're setting a specific distance
        if distance is not None and distance > 0:
            self.set_target_distance(distance)
            return True

        # Otherwise, just move continuously
        use_speed = speed if speed is not None else self.linear_speed
        old_state = self.state
        self.state = 'MOVING'

        self.current_twist = Twist()
        self.current_twist.linear.x = use_speed
        self.get_logger().info(f"Changed state: {old_state} -> MOVING, speed={use_speed:.2f}")
        return True

    def move_backward_direct(self, speed=None, distance=None):
        """
        Direct control to move backward.
        Args:
            speed: Override default linear speed. If None, use self.linear_speed
            distance: Optional distance to travel. If None, move until stopped.
        """
        # For backward movement, we won't check front obstacles
        use_speed = speed if speed is not None else self.linear_speed
        old_state = self.state
        self.state = 'MOVING'

        # If we're setting a specific distance (negative for backward)
        if distance is not None and distance > 0:
            # We'll track distance ourselves since we're moving backward
            self.distance_traveled = 0.0
            self.target_distance = distance
            self.prev_x = self.current_x
            self.prev_y = self.current_y

        self.current_twist = Twist()
        self.current_twist.linear.x = -use_speed  # Negative for backward
        self.get_logger().info(f"Changed state: {old_state} -> MOVING (backward), speed={use_speed:.2f}")
        return True

    def turn_left_direct(self, angular_speed=None, angle_degrees=None):
        """
        Direct control to turn left.
        Args:
            angular_speed: Override default angular speed. If None, use self.angular_speed
            angle_degrees: Optional angle to turn in degrees. If None, turn until stopped.
        """
        use_speed = angular_speed if angular_speed is not None else self.angular_speed
        old_state = self.state
        self.state = 'TURNING'

        # If specific angle requested
        if angle_degrees is not None and angle_degrees > 0:
            angle_rad = math.radians(angle_degrees)
            turn_duration = angle_rad / use_speed

            now = self.get_clock().now().seconds_nanoseconds()[0]
            self.turn_end_time = now + turn_duration
            self.get_logger().info(f"Turning left {angle_degrees} degrees for {turn_duration:.2f} seconds")
        else:
            # Continuous turning
            self.turn_end_time = 0  # Will turn until stopped

        self.current_twist = Twist()
        self.current_twist.angular.z = use_speed  # Positive for left
        self.get_logger().info(f"Changed state: {old_state} -> TURNING (left), speed={use_speed:.2f}")
        return True

    def turn_right_direct(self, angular_speed=None, angle_degrees=None):
        """
        Direct control to turn right.
        Args:
            angular_speed: Override default angular speed. If None, use self.angular_speed
            angle_degrees: Optional angle to turn in degrees. If None, turn until stopped.
        """
        use_speed = angular_speed if angular_speed is not None else self.angular_speed
        old_state = self.state
        self.state = 'TURNING'

        # If specific angle requested
        if angle_degrees is not None and angle_degrees > 0:
            angle_rad = math.radians(angle_degrees)
            turn_duration = angle_rad / use_speed

            now = self.get_clock().now().seconds_nanoseconds()[0]
            self.turn_end_time = now + turn_duration
            self.get_logger().info(f"Turning right {angle_degrees} degrees for {turn_duration:.2f} seconds")
        else:
            # Continuous turning
            self.turn_end_time = 0  # Will turn until stopped

        self.current_twist = Twist()
        self.current_twist.angular.z = -use_speed  # Negative for right
        self.get_logger().info(f"Changed state: {old_state} -> TURNING (right), speed={use_speed:.2f}")
        return True

    def move_for_time(self, linear_speed, duration):
        """
        Move at specified speed for a duration in seconds.
        Positive speed = forward, negative = backward.
        """
        if linear_speed > 0 and self.obstacle_detected:
            self.get_logger().warn("Cannot move forward: obstacle detected")
            return False

        old_state = self.state
        self.state = 'MOVING'

        self.current_twist = Twist()
        self.current_twist.linear.x = linear_speed

        # Set up a timer to stop after duration
        timer = self.create_timer(
            duration,
            lambda: self.complete_movement(timer)
        )

        self.get_logger().info(f"Changed state: {old_state} -> MOVING for {duration}s at speed={linear_speed:.2f}")
        return True

    def complete_movement(self, timer=None):
        """
        Global function to stop robot movement and reset state.
        Can be called directly or as a timer callback.

        Args:
            timer: Optional timer to cancel. If provided, will cancel the timer.
        """
        # If called from a timer, cancel it
        if timer is not None:
            timer.cancel()

        # Stop the robot
        self.stop_robot()

        # Change and log state
        old_state = self.state
        self.state = 'IDLE'
        self.get_logger().info(f"Movement complete. Changed state: {old_state} -> IDLE")

        # Reset movement tracking if needed
        if self.target_distance > 0:
            self.get_logger().info(f"Distance traveled: {self.distance_traveled:.2f}/{self.target_distance:.2f}m")
            self.target_distance = 0.0

    def turn_for_time(self, angular_speed, duration):
        """
        Turn at specified angular speed for a duration in seconds.
        Positive speed = left, negative = right.
        """
        old_state = self.state
        self.state = 'TURNING'

        self.current_twist = Twist()
        self.current_twist.angular.z = angular_speed

        # Set up a timer to stop after duration
        timer = self.create_timer(
            duration,
            lambda: self.complete_movement(timer)
        )

        direction = "left" if angular_speed > 0 else "right"
        self.get_logger().info(f"Changed state: {old_state} -> TURNING {direction} for {duration}s")
        return True

    def set_obstacle_threshold(self, threshold):
        """Set the minimum distance to consider as an obstacle."""
        if threshold <= 0:
            self.get_logger().warn(f"Invalid obstacle threshold: {threshold}")
            return False

        self.obstacle_threshold = threshold
        self.get_logger().info(f"Obstacle threshold set to {threshold}m")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotMovementController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()
        print("Movement controller shut down.")


if __name__ == '__main__':
    main()