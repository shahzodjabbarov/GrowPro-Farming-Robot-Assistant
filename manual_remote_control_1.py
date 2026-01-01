# keycontrol.py (Raspberry Pi)
import socket
import threading
import time
import cv2
import struct
import pickle
from hardware import MotorControl

class RobotServer:
    def __init__(self):
        self.motor = MotorControl()
        self.camera = cv2.VideoCapture(0)  # Use camera index 1 if you prefer
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
    def handle_key_input(self, conn):
        print("[INFO] Control connection received. Ready to drive.")
        
        try:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                key = data.decode().strip()

                if key == 'UP':
                    print("Moving forward...")
                    self.motor.move_forward()
                elif key == 'DOWN':
                    print("Moving backward...")
                    self.motor.move_backward()
                elif key == 'LEFT':
                    print("Turning left...")
                    self.motor.turn_full_left()
                elif key == 'RIGHT':
                    print("Turning right...")
                    self.motor.turn_full_right()
                elif key == 'STOP':
                    print("Stopping...")
                    self.motor.stop()
                elif key == 'QUIT':
                    print("Quitting...")
                    break
                else:
                    print("Unknown command received:", key)

        except Exception as e:
            print("[ERROR] Control handler:", e)
        finally:
            self.motor.stop()
            conn.close()
            print("[INFO] Control connection closed.")

    def handle_video_stream(self, conn):
        print("[INFO] Video connection received. Starting camera stream.")
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Encode frame as JPEG to reduce bandwidth
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                data = pickle.dumps(buffer)
                message_size = struct.pack("L", len(data))
                
                # Send frame size followed by frame data
                conn.sendall(message_size + data)
                
        except Exception as e:
            print("[ERROR] Video handler:", e)
        finally:
            conn.close()
            print("[INFO] Video connection closed.")

    def start_server(self):
        # Create sockets for control and video
        control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        control_socket.bind(('0.0.0.0', 8490))  # Control port
        video_socket.bind(('0.0.0.0', 8491))   # Video port
        
        control_socket.listen(1)
        video_socket.listen(1)
        
        print("[WAITING] for laptop connections...")
        print("Control port: 8490")
        print("Video port: 8491")
        
        # Accept control connection
        control_conn, control_addr = control_socket.accept()
        print("[CONNECTED] Control from", control_addr)
        
        # Accept video connection
        video_conn, video_addr = video_socket.accept()
        print("[CONNECTED] Video to", video_addr)
        
        # Start threads for both connections
        control_thread = threading.Thread(target=self.handle_key_input, args=(control_conn,))
        video_thread = threading.Thread(target=self.handle_video_stream, args=(video_conn,))
        
        control_thread.start()
        video_thread.start()
        
        # Wait for threads to complete
        control_thread.join()
        video_thread.join()
        
        # Cleanup
        self.camera.release()
        control_socket.close()
        video_socket.close()

def main():
    server = RobotServer()
    server.start_server()

if __name__ == "__main__":
    main()