# TurtleBot2 Python API for ROS2

This Python API provides a simplified interface for controlling the TurtleBot2 robot using ROS2. It's designed to make basic navigation and exploration tasks easy to implement without having to deal with the complexity of ROS2 directly.

## Features

- Basic movement commands (forward, backward, left, right)
- Precise movement by distance and rotation by angle
- Obstacle detection using bumpers and cliff sensors
- Battery status monitoring
- Room exploration with automatic obstacle avoidance
- Return to starting point functionality
- Cell-based tracking of visited areas

## Available Sensors on TurtleBot2

The TurtleBot2 is equipped with the following sensors, which this API makes use of:

- **Bumpers**: Three bumper sensors (left, center, right) for detecting physical contact with objects
- **Cliff Sensors**: Three cliff sensors (left, center, right) for detecting drop-offs or ledges
- **Wheel Drop Sensors**: Two sensors that detect if wheels are not in contact with the ground
- **Encoders**: Wheel encoders that provide odometry data
- **Gyroscope**: Provides orientation data
- **Battery Sensor**: Monitors battery voltage

## Usage

### Basic Example

```python
from my_api.mac.turtlebot2_api import TurtleBot2, Direction
import math

# Initialize the API
turtle = TurtleBot2()

try:
    # Basic movement
    turtle.move(Direction.FORWARD, 2.0)  # Move forward for 2 seconds
    turtle.rotate(math.pi / 2)  # Rotate 90 degrees counter-clockwise

    # Move a specific distance
    turtle.move_distance(1.0)  # Move forward 1 meter

    # Explore a room
    turtle.explore_room(max_duration=120.0)  # Explore for up to 2 minutes

    # Return to the starting point
    turtle.return_to_start()

finally:
    # Always ensure proper shutdown
    turtle.shutdown()
```

### Room Exploration

The API includes a simple room exploration algorithm that:

1. Moves the robot forward until an obstacle is detected
2. When an obstacle is detected, tries to find a clear path by rotating
3. Keeps track of visited areas using a cell-based approach
4. Stops when a sufficient number of cells have been visited or a time limit is reached

```python
# Start the room exploration
success = turtle.explore_room(max_duration=300.0)  # 5 minutes timeout

# Check if exploration was successful
if success:
    print(f"Explored {len(turtle.visited_cells)} cells successfully")
else:
    print("Exploration did not complete")
    
# Return to the start point
turtle.return_to_start()
```

## Installation and Dependencies

This API depends on the following ROS2 packages:

- kobuki_node
- kobuki_ros_interfaces  
- geometry_msgs
- nav_msgs
- std_msgs
- sensor_msgs

Make sure the TurtleBot2 ROS2 stack is properly installed and the robot is operational before using this API.

## Example Scripts

The package includes example scripts:

- `turtlebot2_example.py`: Demonstrates basic movement and room exploration

Run the example with:

```bash
python3 turtlebot2_example.py
```

## Limitations

- Obstacle detection is reactive (contact-based using bumpers)
- No mapping or path planning - uses a simple exploration strategy
- No long-term memory of the environment between runs
- No support for advanced navigation like SLAM or global path planning

## License

This software is released under the BSD license.