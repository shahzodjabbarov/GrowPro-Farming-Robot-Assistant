# keycontrol_receiver.py (RASPBERRY PI)
import socket
import threading

def handle_client(conn):
    print("✅ Laptop connected for key input.")
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            key = data.decode()
            print(f"⬅️ Received key: {key}")
            # TODO: Call your motor functions here based on key
            # e.g., if key == 'w': move_forward()
    except:
        print("❌ Connection lost.")
    finally:
        conn.close()

# --- Setup server socket ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 8486))
server_socket.listen(1)
print("📡 Waiting for key input connection...")

while True:
    conn, addr = server_socket.accept()
    threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
