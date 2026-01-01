'''
# ORIGIN camera on raspberry ---> sends data to the computer via socket
#
# This code is the socet for the datatransfer between the robot and the computer and it sends data to the model and analysesit
# Important !!! socket must be activated and saved so that the firewall allows data transfer
#
# activated port 8485
'''

import socket
import struct
import pickle
import cv2
from ultralytics import YOLO
import gc
import time

# === 1. YOLOv8 Modell laden ===
gc.collect()
model = YOLO("prediction_ssppss.pt")

print("\n✅ Modell geladen. Klassen:")
for cls_id, cls_name in model.names.items():
    print(f"  {cls_id}: {cls_name}")
print()

# === 2. Socket starten ===
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 8485))
server_socket.listen(1)
print("📡 Warte auf Verbindung vom Raspberry Pi...")

conn, addr = server_socket.accept()
print("🔌 Verbunden mit:", addr)

data = b""
payload_size = struct.calcsize(">L")

# Frame-Skip-Mechanismus: nur jeden 3. Frame analysieren
frame_counter = 0

try:
    while True:
        # --- 3. Header (Größe) lesen ---
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet
        if len(data) < payload_size:
            continue

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        # --- 4. Frame-Daten lesen ---
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # --- 5. JPEG entpacken und dekodieren ---
        try:
            jpg_data = pickle.loads(frame_data)
            frame = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)
            if frame is None:
                print("⚠️ Frame ist None – fehlerhaft dekodiert")
                continue

            # --- 6. Frame-Skip: nur jeden 3. Frame durch YOLO schicken ---
            frame_counter += 1
            if frame_counter % 3 != 0:
                # Zeige nur Rohbild, aber überspringe YOLO
                cv2.imshow("Live Feed (roh)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # --- 7. YOLO Detection ---
            results = model(frame, conf=0.65)
            annotated = results[0].plot()

            # --- 8. Anzeige ---
            cv2.imshow("YOLO Live Detection", annotated)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC zum Abbruch
                break

        except Exception as e:
            print("❌ Fehler beim Verarbeiten:", e)

except KeyboardInterrupt:
    print("\n🛑 Abbruch durch Benutzer.")

finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("✅ Receiver beendet und Ressourcen freigegeben.")
