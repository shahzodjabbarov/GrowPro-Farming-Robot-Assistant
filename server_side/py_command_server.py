import socket
import subprocess
import threading
import os

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 9000

print("🟢 Command server started on Raspberry Pi...")

def handle_client(conn, addr):
    print(f"🔌 Connected by {addr}")
    running_process = None

    try:
        while True:
            data = conn.recv(1024).decode().strip()
            if not data:
                break

            print(f"📩 Received: {data}")

            if data.startswith("RUN"):
                script_name = data.split(" ", 1)[1]

                if not os.path.isfile(script_name):
                    conn.sendall(f"❌ File not found: {script_name}".encode())
                    continue

                # If something is already running, terminate it
                if running_process and running_process.poll() is None:
                    running_process.terminate()
                    running_process.wait()

                running_process = subprocess.Popen(["python3", script_name])
                conn.sendall(f"✅ Running: {script_name}".encode())

            elif data == "STOP":
                if running_process and running_process.poll() is None:
                    running_process.terminate()
                    running_process.wait()
                    conn.sendall("🛑 Script stopped.".encode())
                else:
                    conn.sendall("⚠️ No script running.".encode())

            else:
                conn.sendall("❓ Unknown command.".encode())

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        if running_process and running_process.poll() is None:
            running_process.terminate()
            running_process.wait()
        conn.close()
        print("🔌 Client disconnected.")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"📡 Waiting for connection on port {PORT}...")

    while True:
        conn, addr = s.accept()
        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.start()
