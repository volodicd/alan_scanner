# Stereo Vision Project Commands and Guidelines

## Installation and Setup

```bash
# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test RAFT-Stereo implementation
python test.py
```

## Build/Run Commands

```bash
# Run the web application
python main.py
```

## Testing

```bash
# Basic camera test
python stereo_vision.py  # Then select option 1

# Test calibration
python stereo_vision.py  # Then select option 2 or 3

# Test stereo processing
python stereo_vision.py  # Then select option 4 (SGBM) or 5 (RAFT-Stereo)
```

## RAFT-Stereo Implementation

The project now integrates RAFT-Stereo for deep learning-based disparity estimation. Key components:

1. `models/raft_stereo_wrapper.py` - Wrapper for the RAFT-Stereo model
2. `models/utils.py` - Utilities for loading and using the model
3. `test.py` - Simple test script for the RAFT-Stereo implementation

## Code Style Guidelines

### Python Conventions
- Use PEP 8 style guide
- 4 spaces for indentation
- Maximum line length of 100 characters
- Clear, descriptive variable names
- Document functions with docstrings

### Error Handling
- Use try/except with specific exceptions
- Include detailed logging with error context
- Use the configured logger system (`logger.info`, `logger.error`, etc.)

### Project Structure
- Core vision code in `stereo_vision.py`
- Web interface in `main.py`
- Frontend modules in corresponding JS/CSS/HTML files

### Important Patterns
- Use OpenCV for all image processing
- Use Flask and SocketIO for web communication
- Keep computationally intensive operations in separate threads
- Log important events and errors with appropriate levels