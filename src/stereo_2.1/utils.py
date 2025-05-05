# utils.py
socketio = None

def init_socketio(io):
    global socketio
    socketio = io

def emit(event, data):
    global socketio
    if socketio:
        socketio.emit(event, data)
    else:
        print(f"WARNING: Attempted to emit '{event}' but socketio is not initialized!")