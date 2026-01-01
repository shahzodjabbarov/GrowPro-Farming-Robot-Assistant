# program_server.py (Raspberry Pi)
import socket
import subprocess
import threading

HOST = '0.0.0.0'
PORT = 8600

current_process = None

def run_script(script_name):
    global current_process
    if current_process is not None:
        current_process.terminate()
        current_process = None
    current_process = subprocess.Popen(['python3', script_name])

def stop_script():
    global current_process
    if current_process is not None:
        current_process.terminate()
        current_process = None

def handle_client(conn):
    global current_process
    while True:
        data = conn.recv(1024).decode().strip()
        print("[RECEIVED]", data)

        if data.startswith('RUN '):
            script = data.split(' ', 1)[1]
            stop_script()
            try:
                run_script(script)
                conn.sendall(f"[OK] Running {script}\n".encode())
            except Exception as e:
                conn.sendall(f"[ERROR] Failed to run {script}: {e}\n".encode())

        elif data == 'STOP':
            stop_script()
            conn.sendall(b"[OK] Script stopped\n")

        elif data == 'EXIT':
            stop_script()
            conn.sendall(b"[EXITING]\n")
            break

        else:
            conn.sendall(b"[ERROR] Unknown command\n")

    conn.close()
    print("[INFO] Connection closed.")

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"[LISTENING] on {HOST}:{PORT}...")

        conn, addr = server.accept()
        print(f"[CONNECTED] to {addr}")
        handle_client(conn)

if __name__ == "__main__":
    main()
