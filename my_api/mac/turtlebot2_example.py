#!/usr/bin/env python3

"""
Example script demonstrating the TurtleBot2 API usage for room exploration.
This script shows how to use the TurtleBot2 API to explore a room and navigate
back to the starting point.
"""

import math
import time
import rclpy

from my_api.mac.turtlebot2_api import TurtleBot2, Direction


def explore_room_and_return():
    """Run a room exploration demo and return to the starting point."""
    turtle = None
    try:
        # Initialize the TurtleBot2 API
        turtle = TurtleBot2()
        print("TurtleBot2 initialized")
        print(f"Battery level: {turtle.get_battery_percentage():.1f}%")
        
        # Wait for sensors to initialize
        print("Waiting for sensors to initialize...")
        time.sleep(2)
        
        # First, demonstrate basic movement capabilities
        print("\n--- Basic Movement Demo ---")
        
        # Move forward for 1 second
        print("Moving forward...")
        turtle.move(Direction.FORWARD, 1.0)
        
        # Turn left (90 degrees)
        print("Turning left...")
        turtle.rotate(math.pi/2)
        
        # Move forward for 1 second
        print("Moving forward again...")
        turtle.move(Direction.FORWARD, 1.0)
        
        # Turn right (180 degrees)
        print("Turning around...")
        turtle.rotate(-math.pi)
        
        # Move back to approximately the starting position
        print("Returning to approximate start position...")
        turtle.move(Direction.FORWARD, 1.0)
        
        # Reset odometry to clear any drift before exploration
        print("Resetting odometry...")
        turtle.reset_odometry()
        time.sleep(1)
        
        # Run room exploration algorithm
        print("\n--- Room Exploration Demo ---")
        print("Starting room exploration...")
        success = turtle.explore_room(max_duration=120.0)  # Explore for up to 2 minutes
        
        if success:
            print("Exploration completed successfully!")
        else:
            print("Exploration ended before completion")
            
        # Show exploration statistics
        position = turtle.get_current_position()
        cells_visited = len(turtle.visited_cells)
        print(f"Cells visited: {cells_visited}")
        print(f"Current position: x={position['x']:.2f}, y={position['y']:.2f}, " +
              f"theta={math.degrees(position['theta']):.1f}°")
        
        # Return to the starting point
        print("\n--- Return to Start Demo ---")
        print("Returning to the starting point...")
        if turtle.return_to_start():
            print("Successfully returned to start!")
        else:
            print("Could not return to start (obstacle encountered)")
            
        # Final status
        position = turtle.get_current_position()
        print(f"Final position: x={position['x']:.2f}, y={position['y']:.2f}, " +
              f"theta={math.degrees(position['theta']):.1f}°")
        print(f"Battery level: {turtle.get_battery_percentage():.1f}%")
        
    except KeyboardInterrupt:
        print("\nExploration interrupted by user")
    finally:
        # Clean shutdown
        if turtle is not None:
            print("Shutting down TurtleBot2 API...")
            turtle.shutdown()
            print("Shutdown complete")


if __name__ == "__main__":
    explore_room_and_return()