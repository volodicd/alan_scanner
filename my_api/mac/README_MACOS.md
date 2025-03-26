# Using TurtleBot2 API with macOS

This guide explains how to connect your TurtleBot2 to a macOS system using Docker and USB forwarding.

## USB Device Forwarding on macOS

Since Docker Desktop on macOS can't directly access USB devices from the host system, you'll need to use one of these methods to make your TurtleBot2 accessible to the Docker containers:

### Option 1: VirtualHere USB Server (Recommended)

[VirtualHere](https://www.virtualhere.com/) is a commercial USB over network solution that works well with Docker on macOS.

1. Install VirtualHere USB Client on your Mac:
   ```bash
   brew install --cask virtualhere-client
   ```

2. Install VirtualHere USB Server on a Linux machine (could be a Raspberry Pi) that's on the same network as your Mac.

3. Connect your TurtleBot2 to the Linux machine running the VirtualHere server.

4. On your Mac, open the VirtualHere client and share the TurtleBot's USB device.

5. Edit `docker-compose.mac.yml` to uncomment the VirtualHere volume mapping:
   ```yaml
   volumes:
     - /dev/virtualhere/kobuki:/dev/kobuki
   ```

### Option 2: Direct USB to Serial Connection with socat

If your TurtleBot connects as a serial device on macOS, you can use `socat` to forward it:

1. Install socat on your Mac:
   ```bash
   brew install socat
   ```

2. Find your TurtleBot's serial device:
   ```bash
   ls /dev/tty.usb*
   ```
   
   You'll see something like `/dev/tty.usbserial-XXXXXXXX`.

3. Set up a TCP-to-serial proxy:
   ```bash
   socat -d -d /dev/tty.usbserial-XXXXXXXX tcp-listen:4321,reuseaddr
   ```

4. Edit `docker-compose.mac.yml` to use the TCP port:
   ```yaml
   ports:
     - "4321:4321"
   ```

5. Also create a mock serial device configuration in your Dockerfile or via entrypoint script:
   ```bash
   socat -d -d tcp:host.docker.internal:4321 pty,raw,echo=0,link=/dev/ttyUSB0
   ```

## Running the TurtleBot2 API on macOS

1. **Build the Docker image**:
   ```bash
   cd my_api
   docker-compose -f docker-compose.mac.yml build
   ```

2. **Start the TurtleBot2 node**:
   ```bash
   docker-compose -f docker-compose.mac.yml up kobuki
   ```

3. **Check the logs** to see if the USB device is detected:
   ```bash
   docker-compose -f docker-compose.mac.yml logs kobuki
   ```

4. If the kobuki node is running properly, **start the safety controller** in another terminal:
   ```bash
   docker-compose -f docker-compose.mac.yml up safety
   ```

5. Finally, **run your API example**:
   ```bash
   docker-compose -f docker-compose.mac.yml up api
   ```

## Troubleshooting

### Check USB Device Connection

Run this inside the container to list USB devices:
```bash
docker-compose -f docker-compose.mac.yml exec kobuki bash -c "lsusb"
```

### Check if the serial device exists:
```bash
docker-compose -f docker-compose.mac.yml exec kobuki bash -c "ls -la /dev/ttyUSB*"
```

### View ROS topics:
```bash
docker-compose -f docker-compose.mac.yml exec kobuki bash -c "source /opt/ros/humble/setup.bash && ros2 topic list"
```

### Run an interactive shell:
```bash
docker-compose -f docker-compose.mac.yml run --rm shell
```

### Use the TurtleBot without actual hardware:

To test the API without hardware, you can create a simple simulator. Check with the project maintainers for simulator options compatible with this API.