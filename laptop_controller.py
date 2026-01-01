import socket
from pynput import keyboard



PI_IP = "192.168.230.173"  # ⬅️ Change to your Raspberry Pi's local IP
PORT = 8490


s = socket.socket()
s.connect((PI_IP, PORT))
print("[CONNECTED] to Raspberry Pi for driving robot.")

key_map = {
    'up': 'UP',
    'down': 'DOWN',
    'left': 'LEFT',
    'right': 'RIGHT'
}

last_sent = None

def on_press(key):
    global last_sent
    try:
        if hasattr(key, 'name') and key.name in key_map:
            command = key_map[key.name]
            if command != last_sent:
                s.sendall(command.encode())
                last_sent = command
        elif key.char == 'q':
            s.sendall(b'QUIT')
            return False
    except AttributeError:
        pass

def on_release(key):
    global last_sent
    if hasattr(key, 'name') and key.name in key_map:
        s.sendall(b'STOP')
        last_sent = None

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

s.close()
