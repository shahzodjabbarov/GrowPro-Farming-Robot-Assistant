import socket

PI_IP = "192.168.1.42"  # ⬅️ Change to your Raspberry Pi's local IP
PORT = 9000

def send_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((PI_IP, PORT))
        s.sendall(command.encode())
        response = s.recv(1024).decode()
        print("📨 Response:", response)

if __name__ == "__main__":
    while True:
        print("\nOptions:\n 1. Run script\n 2. Stop script\n 3. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            script = input("Enter script name to run (e.g., camera.py): ")
            send_command(f"RUN {script}")

        elif choice == "2":
            send_command("STOP")

        elif choice == "3":
            break

        else:
            print("❌ Invalid choice.")
