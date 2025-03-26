# TurtleBot2 API Docker Setup

This repository contains a Python API for controlling TurtleBot2 robots using ROS2, packaged with Docker for easy deployment across platforms including macOS.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- A connected TurtleBot2 robot

## Project Structure

```
my_api/
├── Dockerfile                # Defines the Docker image
├── docker-compose.yml        # Defines the multi-container setup
├── entrypoint.sh             # Container entry script
├── requirements.txt          # Python dependencies
├── turtlebot2_api.py         # Main API implementation
├── turtlebot2_example.py     # Example script using the API
└── README.md                 # This readme file
```

## Quick Start

### 1. Configure Device Path

First, identify the device path for your TurtleBot2 on your host machine:

```bash
# On Linux/macOS, check for the device
ls -l /dev | grep -i turtlebot
# or
ls -l /dev | grep -i kobuki
```

Edit the `docker-compose.yml` file to use the correct device path. Modify this line:

```yaml
devices:
  - /dev/kobuki:/dev/kobuki  # Change the left side to match your system
```

### 2. Build the Docker Image

```bash
cd /path/to/my_api
docker-compose build
```

### 3. Run the TurtleBot2 Base Node

```bash
docker-compose up kobuki
```

### 4. Run the Safety Controller (in a new terminal)

```bash
docker-compose up safety
```

### 5. Run the API Example (in a new terminal)

```bash
docker-compose up api
```

## Development Workflow

### Interactive Shell

For development and debugging, you can start an interactive shell:

```bash
docker-compose run --rm shell
```

Inside the shell, you can run Python scripts directly:

```bash
# Inside the container
python3 turtlebot2_example.py
```

### Custom API Usage

To use the API in your own script:

1. Create a new Python file in the `my_api` directory
2. Import the API: `from turtlebot2_api import TurtleBot2, Direction`
3. Implement your robot control logic
4. Run it with Docker Compose:

```bash
docker-compose run --rm api python3 your_script.py
```

## macOS-Specific Notes

Since ROS2 doesn't natively support macOS, Docker is the recommended approach. Additional steps for macOS:

### USB Device Forwarding

On macOS, you'll need to install a tool to forward USB devices to Docker containers:

1. Install [Docker for Mac](https://docs.docker.com/desktop/install/mac-install/)
2. Install [VirtualHere](https://www.virtualhere.com/usb_client_software) USB client

Or use [usbmuxd](https://github.com/libimobiledevice/usbmuxd) for USB forwarding:

```bash
# Install with Homebrew
brew install usbmuxd
```

Configure the device in docker-compose.yml as:

```yaml
devices:
  - /var/run/usbmuxd:/var/run/usbmuxd
```

### Networking

The `network_mode: host` doesn't work the same on macOS. Use this alternative setup for development on macOS:

```bash
# Create a Docker network for ROS2
docker network create ros_net

# Edit docker-compose.yml to use this network instead of host networking
# Replace "network_mode: host" with:
#   networks:
#     - ros_net
```

## Troubleshooting

### Device Access Issues

If you get permission errors accessing the TurtleBot2:

```bash
# On Linux, add your user to the dialout group
sudo usermod -a -G dialout $USER
sudo chmod a+rw /dev/ttyUSB0  # or whatever your device is

# Restart the container
docker-compose restart kobuki
```

### ROS2 Communication Issues

If containers can't communicate:

```bash
# Check that all containers use the same ROS_DOMAIN_ID
docker-compose down
# Edit docker-compose.yml to ensure all services have the same ROS_DOMAIN_ID
docker-compose up -d
```

### Verify the Robot Connection

To check if the robot is connected and publishing data:

```bash
docker-compose run --rm shell
ros2 topic list | grep kobuki
ros2 topic echo /odom
```

## License

This project is released under the BSD license.