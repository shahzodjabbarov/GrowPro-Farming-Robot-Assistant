🌱 GrowPro – Intelligent Farming Robot Assistant

GrowPro is an autonomous agricultural robot system designed to tackle modern farming challenges such as labor shortages, rising operational costs, and unpredictable environmental conditions.
The system integrates computer vision, deep learning, IoT sensors, and real-time user interfaces to deliver precision agriculture capabilities in both autonomous and manual modes.

🚜 Key Features

🤖 Multi-mode operation (Autonomous / Manual / Follow)

👁️ Real-time computer vision for navigation and crop analysis

🧠 YOLOv8-based AI detection for disease, ripeness, weeds, and crop counting

🛰️ Hybrid navigation system (camera-based + sensor-based)

🖥️ Farmer-friendly GUI with visual maps and live camera feeds

🌦️ Weather-aware decision support (API-integrated)

🧩 System Architecture Overview
User Interface (Laptop)
        ↑
   Socket Communication
        ↓
Server-Side Processing (AI / CV / Control)
        ↑
   Video & Sensor Streams
        ↓
Robot-Side Execution (Motors / Sensors / Navigation)

🧱 Hardware Components

Main Computer: Raspberry Pi 4 Model B (8GB RAM, active cooling)

Vision: Dual high-resolution cameras (navigation + crop analysis)

Sensors:

Ultrasonic distance sensors

IR line-following sensors

Soil humidity sensors

Actuation:

Four-wheel drive system

Servo motors (soil sampling)

Chassis:

Yahboom Raspbot (prototype)

Planned custom motor driver board (future)

💻 Software Stack

Language: Python

Computer Vision: OpenCV

AI / ML: YOLOv8 (Ultralytics)

UI Framework: Pygame

Communication: Socket-based video & command streaming

Operating System: Raspberry Pi OS / Linux

External APIs: WeatherAPI (farming condition alerts)

📂 Repository Structure
GROWPRO-FARMING-ROBOT-ASSISTANT/
│
├── robot_side/        # Code running on the robot (Raspberry Pi)
│   ├── motor_control/ # Motors, ultrasonic, IR sensors
│   ├── navigation/    # Lane detection & camera-based navigation
│   └── remote_net/    # Remote control & video streaming
│
├── server_side/       # High-level processing & AI
│   ├── control/       # Command orchestration
│   ├── cv_models/     # YOLO models, datasets (linked externally)
│   └── detection/    # Crop, fruit & disease detection logic
│
├── user_side/         # User-facing applications
│   ├── application/  # Main GrowPro app
│   ├── manual_control/
│   └── navigation_app/
│
├── media/             # Posters, diagrams, and media links
│
└── README.md

🧠 Core Capabilities
1️⃣ Multi-Mode Operation

Autonomous Mode

Pre-mapped navigation

AI-guided crop monitoring

Manual Mode

Real-time remote driving

Live video feedback

Follow Mode

Person tracking using YOLO + Hungarian Algorithm

Occlusion handling and ID consistency

2️⃣ Detection & Analysis

🌿 Crop disease detection

🍓 Ripeness classification

💧 Dry spot & soil moisture recognition

📊 Crop counting (yield estimation)

🌱 Weed detection

Supported crops:

Pumpkin (A / B)

Salad (A / B)

Strawberry (A / B)

3️⃣ Navigation Systems
Outdoor Navigation

Camera-based lane detection

CLAHE contrast enhancement

Adaptive Canny edge detection

ROI masking (removes ~60% irrelevant pixels)

Indoor / Greenhouse Navigation

IR line-following sensors

Ultrasonic obstacle avoidance

Smooth proportional steering control

🧪 Technical Implementation
Computer Vision Pipeline
Image Capture
 → CLAHE Enhancement
 → Gaussian Blur (5×5)
 → Grayscale Conversion
 → ROI Masking (60% background removed)
 → Adaptive Canny Edge Detection
 → Morphological Cleaning (3×3)
 → Lane Center Calculation
 → Motor Command Generation

YOLOv8 Detection System

Model: YOLOv8 (custom-trained)

Classes: 6 crop categories

Confidence Threshold: 0.3 – 0.65

Performance:

Frame skipping (YOLO every 3rd frame)

Resolution scaling

~30 FPS real-time processing

Context-aware inference:

Detection classes switch automatically based on selected crop

🌐 Communication Architecture

Video Streaming

JPEG compression

Socket + pickle transmission

Port: 8491

Control Commands

Socket-based protocol

Port: 8490

Data Flow

Robot → Server → UI feedback loop

Latency Optimization

Frame dropping

Caching strategies

🖥️ User Interface Design
Multi-Page UI Flow

Field selection

Mode selection (Auto / Manual / Follow)

Crop selection

Function selection:

Disease

Ripeness

Weed

Moisture

Info

Weather dashboard

Visual Map Interface

Interactive farm map

Real-time robot position

Disease markers (numbered)

Live camera feed with YOLO overlays

Connection status indicators

⚙️ Performance Optimizations

Multi-threaded architecture

Cached UI assets

Pre-rendered rotated sprites

Queue-based frame management

Selective inference execution

☁️ Data Management & Future Integration
Current

Local sensor logging

Image capture & annotation

Crop count records

Planned

Cloud analytics dashboard

Multi-farm aggregation

Predictive disease modeling

Automated irrigation integration

🏆 Impact & Benefits

⏱️ 70% reduction in manual inspection time

🩺 2–3 days earlier disease detection

💧 Reduced water waste through precise moisture analysis

📈 Data-driven farming decisions

📦 Modular & scalable system design

🔮 Future Roadmap
Hardware

Custom motor driver board

360° vision system

LiDAR integration

Larger battery capacity

Software

Mobile companion app

Multi-robot coordination

Advanced disease prediction

Cloud-based dashboards

📸 Media & Demonstrations

👉 See media/README.md
 for:

Build process videos

Demonstration videos

Seminar presentations

Posters and certifications
