import cv2
import socket
import struct
import pickle

# === Connect to Raspberry Pi ===
rasp_ip = '192.168.X.X'  # Replace with your Pi's IP address
port = 8490

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((rasp_ip, port))

data = b""
payload_size = struct.calcsize(">L")

try:
    while True:
        # === Receive message length first ===
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet:
                raise ConnectionError("Socket closed by server.")
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        # === Receive frame data ===
        while len(data) < msg_size:
            packet = client_socket.recv(4096)
            if not packet:
                raise ConnectionError("Socket closed by server.")
            data += packet

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # === Decode JPEG buffer ===
        buffer = pickle.loads(frame_data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        # === Resize display window (optional) ===
        resized = cv2.resize(frame, (800, 600))  # or use (1280, 720) if preferred
        cv2.imshow("📷 USB Camera from Pi", resized)

        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print("[ERROR]", e)
finally:
    client_socket.close()
    cv2.destroyAllWindows()

