#!/bin/bash
# TurtleBot2 Diagnostic Script
# Run with: bash diagnostic.sh

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}TurtleBot2 ROS2 Diagnostic Tool${NC}"
echo "---------------------------------------"

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo -e "${GREEN}✓ Running inside Docker container${NC}"
else
    echo -e "${YELLOW}⚠ Not running in Docker container${NC}"
    echo "For best results, run this inside the container:"
    echo "  docker-compose run --rm debug bash diagnostic.sh"
    echo ""
fi

# Check APT repository configuration
echo -e "\n${BLUE}Checking ROS2 repository configuration...${NC}"
if grep -q "packages.ros.org/ros2" /etc/apt/sources.list.d/ros2-latest.list 2>/dev/null; then
    echo -e "${GREEN}✓ ROS2 repository is properly configured${NC}"
else
    echo -e "${RED}✗ ROS2 repository is not properly configured${NC}"
    echo "This may cause dependency issues. Run these commands to fix:"
    echo "  echo \"deb http://packages.ros.org/ros2/ubuntu \$(lsb_release -cs) main\" > /etc/apt/sources.list.d/ros2-latest.list"
    echo "  curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -"
    echo "  apt-get update"
fi

# Source ROS2 setup files
echo -e "\n${BLUE}Sourcing ROS2 setup files...${NC}"
source /opt/ros/humble/setup.bash || { echo -e "${RED}✗ Failed to source ROS2 installation${NC}"; exit 1; }
if [ -d /opt/ros2_ws/install ]; then
    source /opt/ros2_ws/install/setup.bash && echo -e "${GREEN}✓ Workspace setup sourced successfully${NC}" || echo -e "${YELLOW}⚠ Failed to source workspace setup${NC}"
fi

# Check for TurtleBot2 device
echo -e "\n${BLUE}Checking for TurtleBot2 device...${NC}"
if [ -e /dev/kobuki ]; then
    echo -e "${GREEN}✓ TurtleBot2 device found at /dev/kobuki${NC}"
    ls -l /dev/kobuki
else
    echo -e "${YELLOW}⚠ TurtleBot2 device not found at /dev/kobuki${NC}"
    echo "Checking for potential TurtleBot2 devices:"
    for device in /dev/ttyUSB* /dev/ttyACM*; do
        if [ -e "$device" ]; then
            echo "   - $device"
        fi
    done
fi

# Check for required source directories
echo -e "\n${BLUE}Checking for required source repositories...${NC}"
SRC_DIR="/opt/ros2_ws/src"

# Check for angles source
if [ -d "$SRC_DIR/angles" ]; then
    echo -e "${GREEN}✓ angles package found${NC}"
else
    echo -e "${RED}✗ angles package not found in workspace${NC}"
    echo "You should clone this repository: git clone https://github.com/ros/angles.git -b ros2"
fi

# Check for ecl_tools source
if [ -d "$SRC_DIR/ecl_tools" ]; then
    echo -e "${GREEN}✓ ecl_tools package found${NC}"
else
    echo -e "${RED}✗ ecl_tools package not found in workspace${NC}"
    echo "You should clone this repository: git clone https://github.com/stonier/ecl_tools.git"
fi

# Check for kobuki-related packages
echo -e "\n${BLUE}Checking for kobuki packages:${NC}"
KOBUKI_PKGS=$(ros2 pkg list 2>/dev/null | grep -i kobuki)
if [ -n "$KOBUKI_PKGS" ]; then
    echo -e "${GREEN}✓ Found kobuki packages:${NC}"
    echo "$KOBUKI_PKGS"
else
    echo -e "${RED}✗ No kobuki packages found!${NC}"
    echo "This suggests the build process failed or packages weren't installed correctly."
fi

# Check for diagnostic_updater package
echo -e "\n${BLUE}Checking for diagnostic_updater package:${NC}"
if ros2 pkg list 2>/dev/null | grep -q "diagnostic_updater"; then
    echo -e "${GREEN}✓ diagnostic_updater package found${NC}"
else
    echo -e "${RED}✗ diagnostic_updater package not found${NC}"
    echo "This is required for kobuki_node. Make sure your ROS repository is configured correctly."
fi

# Find all launch files
echo -e "\n${BLUE}Looking for kobuki launch files:${NC}"
LAUNCH_FILES=$(find /opt/ros2_ws/install -name "*.launch.py" 2>/dev/null | grep -i kobuki)
if [ -n "$LAUNCH_FILES" ]; then
    echo -e "${GREEN}✓ Found kobuki launch files:${NC}"
    echo "$LAUNCH_FILES"
else
    echo -e "${YELLOW}⚠ No kobuki launch files found. Looking for other launch files...${NC}"
    GENERIC_LAUNCH=$(find /opt/ros2_ws/install -path "*/launch/*" -name "*.py" 2>/dev/null)
    if [ -n "$GENERIC_LAUNCH" ]; then
        echo "$GENERIC_LAUNCH"
    else
        echo -e "${RED}✗ No launch files found at all!${NC}"
    fi
fi

# Check if ROS2 nodes are running
echo -e "\n${BLUE}Checking for running ROS2 nodes:${NC}"
NODES=$(ros2 node list 2>/dev/null)
if [ -n "$NODES" ]; then
    echo -e "${GREEN}✓ Found running ROS2 nodes:${NC}"
    echo "$NODES"
else
    echo -e "${YELLOW}⚠ No ROS2 nodes are currently running${NC}"
fi

# Check for ROS2 topics
echo -e "\n${BLUE}Checking for ROS2 topics:${NC}"
TOPICS=$(ros2 topic list 2>/dev/null)
if [ -n "$TOPICS" ]; then
    echo -e "${GREEN}✓ Found ROS2 topics:${NC}"
    echo "$TOPICS"

    # Check for kobuki-related topics
    KOBUKI_TOPICS=$(echo "$TOPICS" | grep -i kobuki)
    if [ -n "$KOBUKI_TOPICS" ]; then
        echo -e "${GREEN}✓ Found kobuki-related topics:${NC}"
        echo "$KOBUKI_TOPICS"
    else
        echo -e "${YELLOW}⚠ No kobuki-related topics found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No ROS2 topics found${NC}"
fi

# Generate launch commands
echo -e "\n${BLUE}Generated launch commands:${NC}"
if [ -n "$LAUNCH_FILES" ]; then
    KOBUKI_LAUNCH=$(echo "$LAUNCH_FILES" | head -1)
    PKG_NAME=$(echo "$KOBUKI_LAUNCH" | sed -E 's|.*/install/([^/]+)/.*|\1|')
    LAUNCH_FILE=$(basename "$KOBUKI_LAUNCH")
    echo -e "${GREEN}✓ For kobuki base:${NC}"
    echo "ros2 launch $PKG_NAME $LAUNCH_FILE"
else
    echo -e "${YELLOW}⚠ Cannot generate launch command - no kobuki launch files found${NC}"
fi

# Summary and recommendations
echo -e "\n${BLUE}Summary and Recommendations:${NC}"
if [ ! -e /dev/kobuki ]; then
    echo -e "1. ${YELLOW}Create a symlink to your TurtleBot2 device:${NC}"
    for device in /dev/ttyUSB* /dev/ttyACM*; do
        if [ -e "$device" ]; then
            echo "   sudo ln -s $device /dev/kobuki"
            break
        fi
    done
fi

if [ -z "$KOBUKI_PKGS" ]; then
    echo -e "2. ${YELLOW}Rebuild the workspace:${NC}"
    echo "   cd /opt/ros2_ws && colcon build --symlink-install"
fi

if [ -n "$LAUNCH_FILES" ]; then
    echo -e "3. ${GREEN}Use this launch command in docker-compose.yml:${NC}"
    KOBUKI_LAUNCH=$(echo "$LAUNCH_FILES" | head -1)
    PKG_NAME=$(echo "$KOBUKI_LAUNCH" | sed -E 's|.*/install/([^/]+)/.*|\1|')
    LAUNCH_FILE=$(basename "$KOBUKI_LAUNCH")
    echo "   command: bash -c \"source /opt/ros/humble/setup.bash && source /opt/ros2_ws/install/setup.bash && ros2 launch $PKG_NAME $LAUNCH_FILE\""
fi

echo -e "\n${GREEN}Diagnostic complete.${NC}"