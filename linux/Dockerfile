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

# List the available packages and determine build order
RUN . /opt/ros/humble/setup.sh && \
    echo "Available packages before build:" && \
    find /opt/ros2_ws/src -name package.xml -exec dirname {} \; | sort

# Modify cmd_vel_mux to disable testing
RUN if [ -d "/opt/ros2_ws/src/cmd_vel_mux" ]; then \
    sed -i 's/find_package(ros_testing REQUIRED)/# find_package(ros_testing REQUIRED)/' /opt/ros2_ws/src/cmd_vel_mux/CMakeLists.txt && \
    sed -i '/add_executable(test_mux test\/test_mux.cpp)/,/target_link_libraries(test_mux/d' /opt/ros2_ws/src/cmd_vel_mux/CMakeLists.txt && \
    sed -i '/install(TARGETS test_mux/,/)/d' /opt/ros2_ws/src/cmd_vel_mux/CMakeLists.txt; \
    fi

# Stage 1: Build ecl_tools first
RUN . /opt/ros/humble/setup.sh && \
    echo "Building ecl_tools packages first..." && \
    colcon build --symlink-install --packages-select ecl_license ecl_build --cmake-args -DBUILD_TESTING=OFF && \
    source /opt/ros2_ws/install/setup.sh

# Stage 2: Build ECL core packages
RUN . /opt/ros/humble/setup.sh && \
    source /opt/ros2_ws/install/setup.sh && \
    echo "Building ecl_core and ecl_lite packages..." && \
    colcon build --symlink-install --packages-up-to ecl_core ecl_lite --cmake-args -DBUILD_TESTING=OFF && \
    source /opt/ros2_ws/install/setup.sh

# Stage 3: Build kobuki interfaces
RUN . /opt/ros/humble/setup.sh && \
    source /opt/ros2_ws/install/setup.sh && \
    echo "Building kobuki_ros_interfaces..." && \
    colcon build --symlink-install --packages-select kobuki_ros_interfaces --cmake-args -DBUILD_TESTING=OFF && \
    source /opt/ros2_ws/install/setup.sh

# Stage 4: Build the rest of the packages
RUN . /opt/ros/humble/setup.sh && \
    source /opt/ros2_ws/install/setup.sh && \
    echo "Building remaining packages..." && \
    colcon build --symlink-install --packages-skip-up-to ecl_license ecl_build ecl_core ecl_lite kobuki_ros_interfaces --cmake-args -DBUILD_TESTING=OFF || true

# Final attempt to build everything
RUN . /opt/ros/humble/setup.sh && \
    source /opt/ros2_ws/install/setup.sh && \
    echo "Final build attempt for all packages..." && \
    colcon build --symlink-install --executor sequential --cmake-args -DBUILD_TESTING=OFF || true

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