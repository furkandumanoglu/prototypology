# Personal Experience of Art (PEA) - Babylon Installation

An interactive art installation optimized for **Apple M3** that bridges the gap between the viewer and the artwork through real-time body tracking and narrative audio.

## 🎭 System Overview

This project uses **MediaPipe Holistic** and **OpenCV** to track user poses in real-time. The installation creates a "living" version of the painting *Babylon*, which reacts to the viewer's physical presence and specific body movements.

### Core Technologies
- **MediaPipe Holistic**: Full body landmark tracking (Pose, Hands, Face).
- **OpenCV**: High-performance image processing and cinematic mask transitions.
- **Pygame (Mixer)**: Low-latency spatial audio and narrative triggers.
- **Mac M3 Optimization**: Vectorized operations and hardware acceleration for smooth 30+ FPS performance.

## 🖼️ How It Works

### 1. Visual Layers
- **The Ghost Layer**: Initially, the painting is seen in a "spirit state"—grayscale and heavily blurred. This represents the hidden stories of the art.
- **The Color Layer**: The original high-resolution masterpiece, hidden beneath the ghost layer.
- **The Hand Torch**: Your index finger acts as a dynamic light source, allowing you to "peek" through the blur at any time.

### 2. Sequential Revelation (The Story Path)
The installation follows a narrative state machine:
1. **Pose 1 (The Trigger)**: Strike the first reference pose. Once matched, the **Bottom Section** of the painting begins a smooth Left-to-Right reveal, accompanied by the first audio narrative.
2. **Pose 2 (The Journey)**: Strike the second pose to reveal the **Middle Section**.
3. **Pose 3 (The Epiphany)**: Strike the final pose to clear the **Top Section**, revealing the full story and colors of Babylon.

### 3. Pose Matching Logic
The system is designed for natural interaction:
- **Upper Body Focus**: The tracking prioritizes shoulders and elbows to ensure stability and ignore lower-body noise.
- **Adaptive Tolerance**: A 35-degree tolerance allows for varied body types and natural movement.
- **Hold for Action**: Poses must be held for **0.8 seconds** to prevent accidental triggers, providing a deliberate and ceremonial feel to the interaction.

## 📁 Project Structure
- `interactive_art.py`: The main installation script.
- `calibrate_poses.py`: A tool to capture and save your own reference poses.
- `reference_poses.json`: Stored joint data for the matching algorithm.
- `babylon.jpg`: The high-resolution art asset.

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe pygame numpy
   ```
2. **Calibrate (Optional)**:
   Use `calibrate_poses.py` to set your own reference points.
3. **Launch the Installation**:
   ```bash
   python3 interactive_art.py
   ```

---
*Created as an exploration of narrative art and computer vision.*
