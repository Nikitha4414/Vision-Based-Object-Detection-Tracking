# Vision-Based Object Detection & Tracking Using YOLO

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-blueviolet.svg)
![License](https://img.shields.io/badge/Academic-Project-orange.svg)

---

## 📌 Project Overview

**Vision-Based Object Detection & Tracking Using YOLO** is an academic computer vision project that integrates **Ultralytics YOLOv8** real-time object detection with **ByteTrack / BoT-SORT** multi-object tracking algorithms. The system provides an interactive, dark-themed **Streamlit** dashboard capable of detecting multiple common object classes, assigning persistent unique tracking IDs across video frames, measuring processing FPS, and allowing recording/exporting of processed outputs.

---

## ✨ Key Features

1. **Multi-Input Support**:
   - 📹 **Live Webcam**: Stream real-time camera feed with persistent object tracking.
   - 🎥 **Upload Video**: Process video files (`.mp4`, `.avi`, `.mov`, `.mkv`) with start/stop control and stream recording.
   - 🖼️ **Upload Image**: Run instant detection & tracking on static images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).

2. **Real-Time Object Detection**:
   - Powered by YOLOv8 architecture (Nano, Small, Medium, Large, Extra Large).
   - Detects up to 80 COCO object classes (people, vehicles, animals, electronics, etc.).
   - Adjustable **Confidence Threshold** slider (0.05 – 1.00).
   - Adjustable **IoU Threshold** slider (0.10 – 1.00).

3. **Multi-Object Tracking (MOT)**:
   - Integrated **ByteTrack** and **BoT-SORT** tracker engines.
   - Assigns persistent **Tracking IDs** (e.g., `ID: 1`, `ID: 2`).
   - Maintains track continuity through temporary object occlusions.

4. **Performance & Statistics Dashboard**:
   - Real-time **FPS Counter** (smooth exponential moving average).
   - Active detected object count per frame.
   - Total session unique object track tally.
   - Processing device hardware detector (NVIDIA CUDA GPU vs Apple MPS vs CPU).
   - Class-wise detection frequency breakdown table.

5. **Visual Customization & Export**:
   - Customizable visual toggles (Bounding Boxes, Class Labels, Confidence %, Tracking IDs, HUD overlay).
   - 📸 **Save Screenshot** button for instant high-resolution frame exports.
   - 🎬 **Record Output Video** feature for exporting full processed streams.

---

## 📂 Project Structure

```text
c:\Users\parim\Documents\OBJECT DETECTION ACADEMIC PROJECT/
│
├── app.py                   # Streamlit dashboard interface & main loop
├── detector.py              # YOLOv8 object detector module
├── tracker.py               # ByteTrack / BoT-SORT multi-object tracker module
├── utils.py                 # Device detection, FPS counter, draw annotations & video export
├── requirements.txt         # Project Python dependencies
├── README.md                # Project documentation & user guide
├── ACADEMIC_DOCUMENTATION.md# Academic report, literature review, methodology & viva QA
│
├── models/                  # Directory for YOLO weight files (e.g., yolov8n.pt)
├── input/                   # Directory for sample & input files
└── output/                  # Directory for exported screenshots & recorded videos
```

---

## ⚙️ Installation Instructions

### Prerequisites
- **Python 3.10+** (Tested on Python 3.10 – 3.13)
- **Git** (optional)

### Step 1: Clone or Open Project Workspace
Navigate to the project root directory in your command prompt / terminal:
```bash
cd "c:\Users\parim\Documents\OBJECT DETECTION ACADEMIC PROJECT"
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Application

Launch the Streamlit web dashboard using the following command:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at: `http://localhost:8501`

---

## 🎮 How to Use the App

1. **Select Input Source**: Choose between **Webcam**, **Upload Video**, or **Upload Image** from the sidebar.
2. **Select Model**: Choose **YOLOv8n** for fast laptop inference, or **YOLOv8s/m** for higher accuracy.
3. **Adjust Thresholds**: Modify confidence and IoU sliders to control detection sensitivity.
4. **Choose Tracker**: Select **ByteTrack** (default, handles occlusion) or **BoT-SORT**.
5. **Class Filter**: Optionally filter specific object classes (e.g., track only `person` or `car`).
6. **Start Processing**: Click `▶️ Start Tracking` or `▶️ Start Webcam`.
7. **Export Results**: Click `📸 Save Shot` to capture image frame screenshots or enable `Record Processed Output Video` before starting.

---

## 🔧 Troubleshooting & Error Handling

- **Camera Permission Error**: Ensure your web camera is connected and no other application (like Zoom or Teams) is using it.
- **CPU Performance**: If FPS is low on CPU, select `YOLOv8n.pt` and reduce input video resolution.
- **Model Auto-Download**: On first run, YOLO model weights (`.pt` files) automatically download from official Ultralytics releases into the `models/` directory.

---

## 📜 License & Academic Usage

This project is created for academic research, coursework, and laboratory presentation purposes. Feel free to modify and extend for thesis work or academic demonstrations.
