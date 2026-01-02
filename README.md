# 🌱 GrowPro – Intelligent Farming Robot Assistant

GrowPro is an autonomous farming robot designed to assist farmers with **navigation, crop monitoring, and agricultural analysis** using cutting-edge computer vision, deep learning, and sensor-based systems.

![GrowPro](https://github.com/user-attachments/assets/494cc7c3-f093-4f34-acc4-8ee765b3c90c)

This project is a complete end-to-end robotic system, encompassing robot-side control, server-side AI processing, and user-facing applications.

---

## 🌟 Key Features

### What GrowPro Does
- 🚜 **Autonomous field navigation**
- 🎥 **Manual remote control with live video**
- 🌾 **Crop disease detection**
- 🍓 **Ripeness classification**
- 🌱 **Weed detection**
- 📊 **Crop counting (yield estimation)**
- 💧 **Soil dry-spot detection**
- 🧍 **Person-following mode**

---

## 🚦 Operating Modes

### 1. **Autonomous Mode**
- Camera-based lane navigation  
- AI-powered crop monitoring  

### 2. **Manual Mode**
- Laptop-based remote control  
- Live camera streaming  

### 3. **Follow Mode**
- Person tracking using YOLO  
- Identity consistency with the Hungarian Algorithm  

---

## 🛠️ System Architecture

```plaintext
User Laptop (GUI)
↑
Socket Communication
↓
Server-Side Processing (AI / CV)
↑
Video & Sensor Streams
↓
Robot-Side Control (Motors / Sensors)
```

---

## 🔩 Hardware Components

- **Raspberry Pi 4 Model B** (8GB RAM, active cooling)
- **Dual cameras** (navigation + crop analysis)
- **Ultrasonic distance sensors**
- **IR line-following sensors**
- **Soil humidity sensors**
- **4-wheel drive motor system**
- **Servo motors**
- **Yahboom Raspbot chassis** (prototype)

---

## 🖥️ Software Stack

- **Programming Language**: Python  
- **Computer Vision**: OpenCV  
- **AI / ML**: YOLOv8 (Ultralytics)  
- **GUI**: Pygame  
- **Communication**: Socket-based networking  
- **Operating System**: Raspberry Pi OS / Linux  
- **External API**: Weather API (farming alerts)

---

## 📂 Repository Structure

```plaintext
GROWPRO-FARMING-ROBOT-ASSISTANT/
│
├── robot_side/        # Code running on the robot
│   ├── motor_control/
│   ├── navigation/
│   └── remote_net/
│
├── server_side/       # AI, CV, and control logic
│   ├── control/
│   ├── cv_models/
│   └── detection/
│
├── user_side/         # User applications and GUI
    ├── application/
    ├── manual_control/
    └── navigation_app/
```

---

##  Computer Vision & AI

### **Navigation**
- Lane detection using camera input
- CLAHE contrast enhancement
- Adaptive Canny edge detection
- ROI masking (removes ~60% background)
- Smooth proportional steering correction

### **Detection (YOLOv8)**
- Disease detection
- Ripeness classification
- Weed detection
- Crop counting

**Supported Crops:**
- 🎃 Pumpkin  
- 🥬 Lettuce  
- 🍓 Strawberry  

---

## 🔗 Communication

- Live video streaming via sockets
- Command-based remote control
- Optimized for low latency
- Frame skipping for real-time performance

---

## 🖥️ User Interface

- **Pygame-based graphical interface**
  - Field and crop selection
  - Mode switching (Auto / Manual / Follow)
  - Live annotated camera feed
  - Visual farm map with issue markers
  - Weather-based farming alerts

---

## 🖼️ Project Poster

![GrowPro Poster](https://github.com/user-attachments/assets/82965fc6-d16a-405a-87fd-cc7d1c55b03b)
