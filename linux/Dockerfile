FROM ros:humble-ros-base

# Set working directory
WORKDIR /app

# Avoid prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install required ROS2 packages and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-vcstool \
    git \
    python3-rosdep \
    usbutils \
    socat \
    psmisc \
    iproute2 \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install kobuki velocity smoother and sophus as per instructions
RUN apt-get update && apt-get install -y \
    ros-humble-kobuki-velocity-smoother \
    ros-humble-sophus \
    ros-humble-teleop-twist-keyboard \
    ros-humble-joy-teleop \
    ros-humble-teleop-twist-joy \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir -p /opt/ros2_ws/src

# Clone TurtleBot2 repository with all submodules
WORKDIR /opt/ros2_ws/src
RUN git clone --recursive https://github.com/idorobotics/turtlebot2_ros2.git && \
    cp -r turtlebot2_ros2/* . && \
    cp -r turtlebot2_ros2/.git* . && \
    rm -rf turtlebot2_ros2

# Initialize rosdep
RUN apt-get update && \
    rosdep init || true && \
    rosdep update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies with rosdep as per instructions
WORKDIR /opt/ros2_ws
RUN . /opt/ros/humble/setup.sh && \
    rosdep install -i --from-path src --rosdistro humble -y || true

# List the available packages
RUN . /opt/ros/humble/setup.sh && \
    echo "Available packages before build:" && \
    find /opt/ros2_ws/src -name package.xml -exec dirname {} \; | sort

# Modify cmd_vel_mux to disable testing
RUN if [ -d "/opt/ros2_ws/src/cmd_vel_mux" ]; then \
    sed -i 's/find_package(ros_testing REQUIRED)/# find_package(ros_testing REQUIRED)/' /opt/ros2_ws/src/cmd_vel_mux/CMakeLists.txt && \
    sed -i '/add_executable(test_mux test\/test_mux.cpp)/,/target_link_libraries(test_mux/d' /opt/ros2_ws/src/cmd_vel_mux/CMakeLists.txt && \
    sed -i '/install(TARGETS test_mux/,/)/d' /opt/ros2_ws/src/cmd_vel_mux/CMakeLists.txt; \
    fi

# Build the workspace using sequential executor and disable testing
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install --executor sequential --cmake-args -DBUILD_TESTING=OFF

# If the build fails, try building with skipping problematic packages
RUN . /opt/ros/humble/setup.sh && \
    if [ ! -d "/opt/ros2_ws/install/kobuki_node" ]; then \
        echo "First build attempt failed, trying with packages-skip option" && \
        colcon build --symlink-install --executor sequential --packages-skip cmd_vel_mux --cmake-args -DBUILD_TESTING=OFF; \
    fi

# List the built packages
RUN . /opt/ros/humble/setup.sh && \
    . /opt/ros2_ws/install/setup.sh && \
    echo "Built packages:" && \
    ros2 pkg list | grep -i kobuki || echo "No kobuki packages were built"

# Find all launch files
RUN . /opt/ros/humble/setup.sh && \
    . /opt/ros2_ws/install/setup.sh && \
    echo "Available launch files:" && \
    find /opt/ros2_ws/install -name "*.launch.py" 2>/dev/null || echo "No launch.py files found" && \
    find /opt/ros2_ws/install -path "*/launch/*" -name "*.py" 2>/dev/null || echo "No launch directory python files found"

# Copy our API to the container
WORKDIR /app
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Source ROS2 setup files\n\
source /opt/ros/humble/setup.bash\n\
source /opt/ros2_ws/install/setup.bash || echo "Warning: Could not source workspace"\n\
\n\
# Print ros2 package status\n\
echo "Checking ROS2 packages:"\n\
ros2 pkg list | grep -i kobuki || echo "No kobuki packages found"\n\
\n\
# Check if running in WSL\n\
if grep -q WSL /proc/version; then\n\
    echo "Running in WSL environment"\n\
fi\n\
\n\
# Check if the robot device exists\n\
if [ -e /dev/kobuki ]; then\n\
    echo "TurtleBot2/Kobuki device found at /dev/kobuki"\n\
    ls -l /dev/kobuki\n\
else\n\
    echo "Warning: TurtleBot2/Kobuki device not found at /dev/kobuki"\n\
    \n\
    # Look for alternate device paths\n\
    for device in /dev/ttyUSB* /dev/ttyACM*; do\n\
        if [ -e "$device" ]; then\n\
            echo "Found potential TurtleBot2 device at $device"\n\
            echo "Consider creating a symlink: sudo ln -s $device /dev/kobuki"\n\
        fi\n\
    done\n\
    \n\
    # For testing without physical robot, create a mock device\n\
    if [[ "$1" == "python3" ]]; then\n\
        echo "Setting up virtual device for testing..."\n\
        socat -d PTY,link=/dev/kobuki,raw,echo=0 PTY,link=/dev/virtualbot,raw,echo=0 &\n\
        sleep 1\n\
    fi\n\
fi\n\
\n\
# Check network configuration\n\
echo "Network configuration:"\n\
ip addr show\n\
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-Not set}"\n\
\n\
# Execute the command\n\
echo "Running command: $@"\n\
exec "$@"\n\
' > /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "turtlebot2_example.py"]