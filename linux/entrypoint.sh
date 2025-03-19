#!/bin/bash
set -e

# Source ROS2 setup files
source /opt/ros/humble/setup.bash
source /opt/ros2_ws/install/setup.bash || echo "Warning: Could not source workspace"

# Print ros2 package status
echo "Checking ROS2 packages:"
ros2 pkg list | grep -i kobuki || echo "No kobuki packages found"

# Check if running in WSL
if grep -q WSL /proc/version; then
    echo "Running in WSL environment"
fi

# Check if the robot device exists
if [ -e /dev/kobuki ]; then
    echo "TurtleBot2/Kobuki device found at /dev/kobuki"
    ls -l /dev/kobuki
else
    echo "Warning: TurtleBot2/Kobuki device not found at /dev/kobuki"

    # Look for alternate device paths
    for device in /dev/ttyUSB* /dev/ttyACM*; do
        if [ -e "$device" ]; then
            echo "Found potential TurtleBot2 device at $device"
            echo "Consider creating a symlink: sudo ln -s $device /dev/kobuki"
        fi
    done

    # For testing without physical robot, create a mock device
    if [[ "$1" == "python3" ]]; then
        echo "Setting up virtual device for testing..."
        socat -d PTY,link=/dev/kobuki,raw,echo=0 PTY,link=/dev/virtualbot,raw,echo=0 &
        sleep 1
    fi
fi

# Check network configuration
echo "Network configuration:"
ip addr show
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-Not set}"

# Execute the command
echo "Running command: $@"
exec "$@"