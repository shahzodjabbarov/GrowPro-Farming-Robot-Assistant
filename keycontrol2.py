# keycontrol.py (Raspberry Pi)
import socket
import threading
import time
from hardware import MotorControl

def handle_key_input(conn):
    motor = MotorControl()
    print("[INFO] Connection received. Ready to drive.")

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            key = data.decode().strip()

            if key == 'UP':
                print("Moving forward...")
                motor.move_forward()
            elif key == 'DOWN':
                print("Moving backward...")
                motor.move_backward()
            elif key == 'LEFT':
                print("Turning left...")
                motor.turn_full_left()
            elif key == 'RIGHT':
                print("Turning right...")
                motor.turn_full_right()
            elif key == 'STOP':
                print("Stopping...")
                motor.stop()
            elif key == 'QUIT':
                print("Quitting...")
                break
            else:
                print("Unknown command received:", key)

    except Exception as e:
        print("[ERROR]", e)

    finally:
        motor.stop()
        conn.close()
        print("[INFO] Connection closed.")

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 8490))
    server_socket.listen(1)
    print("[WAITING] for laptop connection to control robot...")

    conn, addr = server_socket.accept()
    print("[CONNECTED] to", addr)
    handle_key_input(conn)

if __name__ == "__main__":
    main()
