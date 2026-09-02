# 🕸️ who?! you????? — AI Vision & Cybernetic Telemetry System

A high-performance, real-time web-based computer vision dashboard powered by **TensorFlow.js**, **MediaPipe**, and **face-api.js**. Featuring multi-model AI tracking, biometric analysis, hand gesture classification, interactive spiderweb particle effects, and live telemetry analytics.

---

## ✨ Features

### 👤 1. Face Analysis & Biometrics
- **Multi-Face Detection & Tracking**: Real-time identification and IoU-based persistent tracking (`Person #1`, `Person #2`, etc.).
- **68-Point Facial Landmarks**: High-precision landmark correlation overlaid on face meshes.
- **Demographic Analysis**: Real-time age estimation and gender detection.
- **Emotion Recognition Engine**: Classifies emotions into **Happy**, **Neutral**, **Surprised**, **Sad**, **Angry**, **Fearful**, and **Disgusted** with confidence scores.

### ✋ 2. Cybernetic Hand Tracking & Gesture Recognition
- **21 3D Hand Keypoints**: Uses MediaPipe HandLandmarker with GPU acceleration to track up to two hands simultaneously.
- **Real-Time Gesture Classifier**: Instantly identifies hand gestures:
  - 🕸️ **Web Shooter**: Triggers custom animated web-blast shockwaves!
  - 👌 **OK Sign**
  - 👍 **Thumbs Up** / 👎 **Thumbs Down**
  - ✌️ **Peace Sign**
  - ☝️ **Pointing**
  - 🖐️ **Open Palm**
  - ✊ **Closed Fist**
- **Joint Index Overlay**: Option to toggle 0–20 joint index numbers directly over hand landmarks.

### 📦 3. Object Detection & Vector Streams
- **COCO-SSD Integration**: Detects and tracks 80+ everyday object classes (people, chairs, laptops, cell phones, etc.).
- **Multi-Frame Smoothing & IoU Tracking**: Reduces bounding box jitter and maintains stable object IDs across video frames.
- **Confidence Filter**: Dynamic slider to filter object detections based on confidence score (20% – 90%).

### 🕸️ 4. Cyberpunk HUD & Interactive Visual FX
- **Spiderweb Hand Effect**: Animated glowing neon dashed skeleton connections and web geometry between hand joints.
- **Fingertip Particle Emitters**: Dynamic particle emitters trailing off fingertips in motion.
- **Web Blast Effect**: Shockwave rings and radial web patterns fired on gesture triggers.
- **Dual Hand Cyber Link**: When two hands approach each other, a high-energy plasma web and power core dynamically connect them.
- **Bio-Link & Tether System**:
  - **Left Hand Bio-Link**: Dynamic curved beam tethering left hand to the nearest detected face.
  - **Right Hand Tether**: Dynamic wave helix tethering right hand to the nearest detected object with live distance measurements in pixels.
- **Futuristic Text Badges**: Glowing high-tech HUD labels for targets, confidence scores, and distances.
- **HUD Toggle**: Instantly hide or show visual overlays for a clean camera feed.

### 📊 5. Analytics & Dashboard Control Node
- **Settings Tab**:
  - Camera device selector to switch between multiple webcams.
  - Toggles for Object Detection, Face Analysis, Cyber Hands, Gestures, Spider Effect, and Joint Numbers.
  - Confidence threshold slider with live percentage display.
- **Detections Tab**: Live breakdown of detected object vector streams with confidence meters.
- **Biometrics Registry Tab**: Cards for each tracked individual displaying age, gender, dominant emotion, and full emotion spectrum breakdown.
- **Telemetry Rollups Tab**: Aggregate session metrics including total unique identity count, mean chrono-age, and real-time emotion distribution progress bars.

### 📸 6. Snapshot & System Control
- **Telemetry Screenshot Export**: One-click capture (`📸` button) exporting high-resolution PNG snapshots with optional embedded HUD overlays.
- **Stream Pause / Resume**: Pause live inference engine to inspect frame data without stopping the camera feed.
- **Dark / Light Theme Switcher**: Full custom CSS design system with smooth theme transitions and local storage persistence.
- **Toast Notifications**: Floating real-time status notifications for model initialization, gesture activation, and camera events.

---

## 🛠️ Technology Stack

- **Core**: HTML5, Vanilla JavaScript (ES Modules), CSS3 (CSS Variables & Glassmorphic UI)
- **Bundler & Dev Server**: [Vite](https://vitejs.dev/)
- **Machine Learning & Vision Engines**:
  - [@tensorflow/tfjs](https://www.tensorflow.org/js) (TensorFlow WebGL backend)
  - [@tensorflow-models/coco-ssd](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd) (Object Detection)
  - [@mediapipe/tasks-vision](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) (MediaPipe Hand Landmarker)
  - [@vladmandic/face-api](https://github.com/vladmandic/face-api) (Face Detection, Landmarks, Expressions, Age & Gender)

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v16+ recommended)
- `npm` package manager
- Webcam / video input device

### Installation & Setup

1. **Clone or navigate to the repository directory**:
   ```bash
   cd "face detector"
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Download AI Model Weights**:
   Download required `face-api.js` model weights into `public/models/`:
   ```bash
   node downlaoder.js
   ```

4. **Start Development Server**:
   ```bash
   npm run dev
   ```
   Open your browser at `http://localhost:5173`.

5. **Build for Production**:
   ```bash
   npm run build
   ```

---

## 📁 Project Structure

```
├── downlaoder.js        # Script to download face-api.js model weights
├── index.html           # Main UI dashboard template & modal structures
├── package.json         # Dependencies and scripts configuration
├── public/              # Static assets & downloaded neural model weights
│   └── models/          # face-api.js weight shards and manifests
├── src/
│   ├── main.js          # Core AI inference pipeline, tracking, HUD & rendering loop
│   └── style.css        # Glassmorphic UI styling, dark/light theme, and animations
└── vite.config.js       # Vite configuration
```

---

## 🎮 Usage Guide

1. **Camera Permission**: Allow browser access to your webcam upon prompt.
2. **Model Loading**: Wait a few seconds while TensorFlow.js, COCO-SSD, MediaPipe, and face-api models initialize.
3. **Sidebar Tabs**:
   - Use **Settings** to adjust confidence levels and toggle features on/off.
   - View live target telemetry in **Detections**, **Biometrics**, or **Stats**.
4. **Try Gestures**: Flash a **Web Shooter** gesture (thumb, index, pinky extended) towards the camera to unleash cyber web blasts!
5. **Snapshot**: Click the camera icon in the HUD to download a screenshot.
