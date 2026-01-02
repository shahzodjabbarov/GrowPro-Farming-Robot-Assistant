# 🌱 GrowPro – Intelligent Farming Robot Assistant

GrowPro is an autonomous farming robot built to assist farmers with **navigation, crop monitoring, and agricultural analysis** using computer vision, deep learning, and sensor-based systems.


https://github.com/user-attachments/assets/494cc7c3-f093-4f34-acc4-8ee765b3c90c



This project was developed as a complete end-to-end robotic system, covering robot-side control, server-side AI processing, and user-facing applications.
---

## What GrowPro Does

- Autonomous field navigation
- Manual remote control with live video
- Crop disease detection
- Ripeness classification
- Weed detection
- Crop counting (yield estimation)
- Soil dry-spot detection
- Person-following mode

---

## Operating Modes

### Autonomous Mode
- Camera-based lane navigation  
- AI-powered crop monitoring  

### Manual Mode
- Laptop-based remote control  
- Live camera streaming  

### Follow Mode
- Person tracking using YOLO  
- Identity consistency using Hungarian Algorithm  

---

## System Architecture

```

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

## Hardware

- Raspberry Pi 4 Model B (8GB RAM, active cooling)
- Dual cameras (navigation + crop analysis)
- Ultrasonic distance sensors
- IR line-following sensors
- Soil humidity sensors
- 4-wheel drive motor system
- Servo motors
- Yahboom Raspbot chassis (prototype)

---

## Software Stack

- **Language**: Python  
- **Computer Vision**: OpenCV  
- **AI / ML**: YOLOv8 (Ultralytics)  
- **GUI**: Pygame  
- **Communication**: Socket-based networking  
- **OS**: Raspberry Pi OS / Linux  
- **External API**: Weather API (farming alerts)

---

## Repository Structure

```

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
│   ├── application/
│   ├── manual_control/
│   └── navigation_app/
│
├── media/             # Posters, images, media links
│
└── README.md

```

---

## Computer Vision & AI

### Navigation
- Lane detection using camera input
- CLAHE contrast enhancement
- Adaptive Canny edge detection
- ROI masking (removes ~60% background)
- Smooth proportional steering correction

### Detection (YOLOv8)
- Disease detection
- Ripeness classification
- Weed detection
- Crop counting

**Supported crops**
- Pumpkin 
- Lettuce
- Strawberry 

---

## Communication

- Live video streaming via sockets
- Command-based remote control
- Optimized for low latency
- Frame skipping for real-time performance

---

## User Interface

- Pygame-based graphical interface
- Field and crop selection
- Mode switching (Auto / Manual / Follow)
- Live annotated camera feed
- Visual farm map with issue markers
- Weather-based farming alerts

## Poster
<img width="1900" height="2704" alt="Grow Pro Poster (1) (3)" src="https://github.com/user-attachments/assets/82965fc6-d16a-405a-87fd-cc7d1c55b03b" />

