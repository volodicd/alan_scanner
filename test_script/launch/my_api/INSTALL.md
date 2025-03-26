# TurtleBot2 API Installation Guide

## 1. ROS2 Humble Installation

First, install ROS2 Humble on Ubuntu 22.04:

```bash
# Set locale
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Setup sources
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 packages
sudo apt update
sudo apt install ros-humble-desktop

# Setup environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install development tools and ROS tools
sudo apt install -y python3-rosdep python3-colcon-common-extensions python3-vcstool
sudo rosdep init
rosdep update
```

## 2. TurtleBot2 ROS2 Stack Installation

Install the TurtleBot2 packages for ROS2:

```bash
# Create workspace
mkdir -p ~/turtlebot2_ws/src
cd ~/turtlebot2_ws/src

# Clone TurtleBot2 repos
git clone https://github.com/idorobotics/turtlebot2_ros2.git

# Install dependencies
sudo apt-get install ros-humble-kobuki-velocity-smoother ros-humble-sophus
sudo apt-get install ros-humble-teleop-twist-keyboard ros-humble-joy-teleop ros-humble-teleop-twist-joy

# Install any missing dependencies
cd ~/turtlebot2_ws
rosdep install -i --from-path src --rosdistro humble -y

# Build the workspace
colcon build --symlink-install --executor sequential

# Source the workspace
echo "source ~/turtlebot2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 3. Python Dependencies

Install the required Python packages:

```bash
cd ~/turtlebot2_ws/src/turtlebot2_ros2
pip install -r requirements.txt
```

## 4. Setup udev Rules

To access the TurtleBot2 without root privileges:

```bash
sudo cp ~/turtlebot2_ws/src/turtlebot2_ros2/kobuki_core/60-kobuki.rules /etc/udev/rules.d/
sudo service udev reload
sudo service udev restart
```

## 5. Running the TurtleBot2 API

1. Power on the TurtleBot2
2. Launch the kobuki node:
```bash
ros2 launch kobuki_node kobuki_node-launch.py
```
3. In a new terminal, run the example script:
```bash
cd ~/turtlebot2_ws/src/turtlebot2_ros2
python3 turtlebot2_example.py
```

## Troubleshooting

- If you get permission errors with the TurtleBot USB port, try:
```bash
sudo chmod a+rw /dev/kobuki
```

- If ROS2 nodes can't communicate, check that all terminals have sourced the workspace:
```bash
source ~/turtlebot2_ws/install/setup.bash
```

- Verify that the robot is properly connected:
```bash
ros2 topic list | grep kobuki
```