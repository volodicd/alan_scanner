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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add ROS2 repository explicitly to ensure all packages are available
RUN echo "deb http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -

# Update package lists to include the new repository
RUN apt-get update

# Install kobuki velocity smoother and sophus as per instructions
RUN apt-get install -y \
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

# Clone the angles package from source (required for build)
RUN git clone https://github.com/ros/angles.git -b ros2

# Initialize rosdep
RUN apt-get update && \
    rosdep init || true && \
    rosdep update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies with rosdep
WORKDIR /opt/ros2_ws
RUN bash -c "source /opt/ros/humble/setup.bash && \
    rosdep install -i --from-path src --rosdistro humble -y || true"

# Modify cmd_vel_mux to disable testing if needed
RUN if [ -d "/opt/ros2_ws/src/cmd_vel_mux" ]; then \
    sed -i 's/find_package(ros_testing REQUIRED)/# find_package(ros_testing REQUIRED)/' /opt/ros2_ws/src/cmd_vel_mux/CMakeLists.txt && \
    sed -i '/add_executable(test_mux test\/test_mux.cpp)/,/target_link_libraries(test_mux/d' /opt/ros2_ws/src/cmd_vel_mux/CMakeLists.txt && \
    sed -i '/install(TARGETS test_mux/,/)/d' /opt/ros2_ws/src/cmd_vel_mux/CMakeLists.txt; \
    fi

# Build the workspace with a single command as per the guide
RUN bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --executor sequential --cmake-args -DBUILD_TESTING=OFF"

# List the built packages and find launch files
RUN bash -c "source /opt/ros/humble/setup.bash && \
    source /opt/ros2_ws/install/setup.bash && \
    echo 'Built packages:' && \
    ros2 pkg list | grep -i kobuki || echo 'No kobuki packages were built' && \
    echo 'Available launch files:' && \
    find /opt/ros2_ws/install -name '*.launch.py' 2>/dev/null || echo 'No launch.py files found'"

# Copy our API to the container
WORKDIR /app
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create entrypoint script line by line
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Source ROS2 setup files' >> /entrypoint.sh && \
    echo 'source /opt/ros/humble/setup.bash' >> /entrypoint.sh && \
    echo 'source /opt/ros2_ws/install/setup.bash || echo "Warning: Could not source workspace"' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Print ros2 package status' >> /entrypoint.sh && \
    echo 'echo "Checking ROS2 packages:"' >> /entrypoint.sh && \
    echo 'ros2 pkg list | grep -i kobuki || echo "No kobuki packages found"' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Check if running in WSL' >> /entrypoint.sh && \
    echo 'if grep -q WSL /proc/version; then' >> /entrypoint.sh && \
    echo '    echo "Running in WSL environment"' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Check if the robot device exists' >> /entrypoint.sh && \
    echo 'if [ -e /dev/kobuki ]; then' >> /entrypoint.sh && \
    echo '    echo "TurtleBot2/Kobuki device found at /dev/kobuki"' >> /entrypoint.sh && \
    echo '    ls -l /dev/kobuki' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '    echo "Warning: TurtleBot2/Kobuki device not found at /dev/kobuki"' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '    # Look for alternate device paths' >> /entrypoint.sh && \
    echo '    for device in /dev/ttyUSB* /dev/ttyACM*; do' >> /entrypoint.sh && \
    echo '        if [ -e "$device" ]; then' >> /entrypoint.sh && \
    echo '            echo "Found potential TurtleBot2 device at $device"' >> /entrypoint.sh && \
    echo '            echo "Consider creating a symlink: sudo ln -s $device /dev/kobuki"' >> /entrypoint.sh && \
    echo '        fi' >> /entrypoint.sh && \
    echo '    done' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '    # For testing without physical robot, create a mock device' >> /entrypoint.sh && \
    echo '    if [[ "$1" == "python3" ]]; then' >> /entrypoint.sh && \
    echo '        echo "Setting up virtual device for testing..."' >> /entrypoint.sh && \
    echo '        socat -d PTY,link=/dev/kobuki,raw,echo=0 PTY,link=/dev/virtualbot,raw,echo=0 &' >> /entrypoint.sh && \
    echo '        sleep 1' >> /entrypoint.sh && \
    echo '    fi' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Check network configuration' >> /entrypoint.sh && \
    echo 'echo "Network configuration:"' >> /entrypoint.sh && \
    echo 'ip addr show' >> /entrypoint.sh && \
    echo 'echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-Not set}"' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Execute the command' >> /entrypoint.sh && \
    echo 'echo "Running command: $@"' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "turtlebot2_example.py"]