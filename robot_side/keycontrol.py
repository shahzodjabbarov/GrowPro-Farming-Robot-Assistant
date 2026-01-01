import sys
import tty
import termios
import time
from hardware import MotorControl

def get_key():
    """Reads a single keypress from the terminal (including arrow keys)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(3)  # Read 3 characters for arrow keys
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def manual_drive_continuous():
    motor = MotorControl()
    print("\n[INFO] Press ↑ ↓ ← → to move continuously. Press 'q' to quit.\n")

    try:
        while True:
            key = get_key()

            if key == '\x1b[A':  # Up arrow (move forward)
                print("Moving forward...")
                motor.move_forward()
            
            elif key == '\x1b[B':  # Down arrow (move backward)
                print("Moving backward...")
                motor.move_backward()
            
            elif key == '\x1b[D':  # Left arrow (turn left)
                print("Turning left...")
                motor.turn_full_left()

            elif key == '\x1b[C':  # Right arrow (turn right)
                print("Turning right...")
                motor.turn_full_right()

            elif key == 'q':  # Quit
                print("Quitting.")
                break

            else:  # Stop if any other key is pressed or invalid
                print("Unknown key. Waiting...")
                motor.stop()

            # Stop movement when key is released
            while True:
                key_release = get_key()
                if key_release != key:  # If key is released, stop movement
                    motor.stop()
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")

    finally:
        motor.stop()
        print("[INFO] Motors stopped. Exiting.")

if __name__ == "__main__":
    manual_drive_continuous()
