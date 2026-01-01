import socket
import threading
import cv2
import struct
import pickle
from pynput import keyboard

PI_IP = "192.168.230.173"  # Your Raspberry Pi's local IP
CONTROL_PORT = 8490
VIDEO_PORT = 8491

class RobotController:
    def __init__(self):
        # Create sockets for control and video
        self.control_socket = socket.socket()
        self.video_socket = socket.socket()
        
        self.key_map = {
            'up': 'UP',
            'down': 'DOWN',
            'left': 'LEFT',
            'right': 'RIGHT'
        }
        
        self.last_sent = None
        self.running = True
        
    def connect(self):
        try:
            # Connect control socket
            self.control_socket.connect((PI_IP, CONTROL_PORT))
            print("[CONNECTED] Control to Raspberry Pi")
            
            # Connect video socket
            self.video_socket.connect((PI_IP, VIDEO_PORT))
            print("[CONNECTED] Video from Raspberry Pi")
            
            return True
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False
    
    def video_receiver(self):
        """Receive and display video frames from robot"""
        try:
            data = b""
            payload_size = struct.calcsize("!L")  # Use network byte order
            
            while self.running:
                # Receive message size
                while len(data) < payload_size:
                    try:
                        packet = self.video_socket.recv(4096)
                        if not packet:
                            print("[INFO] Video connection closed by server")
                            return
                        data += packet
                    except socket.error:
                        print("[ERROR] Socket error while receiving size")
                        return
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("!L", packed_msg_size)[0]
                
                # Sanity check for message size
                if msg_size > 1000000:  # 1MB limit
                    print(f"[ERROR] Message size too large: {msg_size}")
                    data = b""
                    continue
                
                # Receive frame data
                while len(data) < msg_size:
                    try:
                        packet = self.video_socket.recv(min(4096, msg_size - len(data)))
                        if not packet:
                            print("[INFO] Video connection closed during frame receive")
                            return
                        data += packet
                    except socket.error:
                        print("[ERROR] Socket error while receiving frame")
                        return
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                try:
                    # Decode and display frame
                    frame_buffer = pickle.loads(frame_data)
                    frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        cv2.imshow('Robot Camera Feed', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("[WARNING] Failed to decode frame")
                except (pickle.PickleError, cv2.error) as e:
                    print(f"[ERROR] Frame decode error: {e}")
                    continue
                        
        except Exception as e:
            print(f"[ERROR] Video receiver: {e}")
        finally:
            cv2.destroyAllWindows()
    
    def on_press(self, key):
        try:
            if hasattr(key, 'name') and key.name in self.key_map:
                command = self.key_map[key.name]
                if command != self.last_sent:
                    self.control_socket.sendall(command.encode())
                    self.last_sent = command
                    print(f"[SENT] {command}")
            elif key.char == 'q':
                print("[QUIT] Stopping robot and closing connections...")
                self.control_socket.sendall(b'QUIT')
                self.running = False
                return False
        except AttributeError:
            pass
    
    def on_release(self, key):
        if hasattr(key, 'name') and key.name in self.key_map:
            self.control_socket.sendall(b'STOP')
            self.last_sent = None
            print("[SENT] STOP")
    
    def start_control(self):
        """Start keyboard control in a separate thread"""
        print("[INFO] Starting keyboard control...")
        print("Use arrow keys to control robot, 'q' to quit")
        
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()
    
    def run(self):
        if not self.connect():
            return
        
        # Start video receiver in a separate thread
        video_thread = threading.Thread(target=self.video_receiver, daemon=True)
        video_thread.start()
        
        # Start keyboard control (blocking)
        self.start_control()
        
        # Cleanup
        self.running = False
        self.control_socket.close()
        self.video_socket.close()
        print("[INFO] Connections closed.")

def main():
    controller = RobotController()
    controller.run()

if __name__ == "__main__":
    main()