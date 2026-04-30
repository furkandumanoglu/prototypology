import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
import pygame

# --- Apple Silicon (M3) Optimization ---
cv2.setUseOptimized(True)

# --- Configuration & Constants ---
IMAGE_PATH = "babylon.jpg"
REFERENCE_POSES_PATH = "reference_poses.json"

# Tolerance increased and hold time reduced for better UX
TOLERANCE = 35  # degrees
HOLD_TIME = 0.8  # seconds

TRANSITION_DURATION = 25.0  # seconds
FEATHER_SIZE = 150  # Increased for a softer edge
TORCH_RADIUS = 150  

# Ripple Effect Settings
WAVE_AMPLITUDE = 60.0  # Wave width
WAVE_FREQUENCY = 0.02  # Wave frequency
OVERLAP_OFFSET = 200   # How many pixels a section extends upwards

# --- Initialize MediaPipe ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Initialize Audio (M3 Fixes) ---
# Get absolute path to the script's directory to ensure sounds are found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(SCRIPT_DIR, "sounds")

# Buffer and frequency preset for Mac audio drivers
pygame.mixer.pre_init(44100, -16, 2, 4096)
pygame.mixer.init()

# 1. Background Music (Continuous Loop)
BGM_PATH = os.path.join(SOUNDS_DIR, "cont.wav")
if os.path.exists(BGM_PATH):
    pygame.mixer.music.load(BGM_PATH)
    pygame.mixer.music.play(-1)  
    pygame.mixer.music.set_volume(0.4) 
    print(f"✅ Background music loaded: {BGM_PATH}")
else:
    print(f"⚠️ Error: Background music not found at {BGM_PATH}")

def load_sound(file_name):
    """Loads a sound effect from the absolute sounds directory."""
    file_path = os.path.join(SOUNDS_DIR, file_name)
    if os.path.exists(file_path):
        return pygame.mixer.Sound(file_path)
    print(f"⚠️ Error: Sound effect not found at {file_path}")
    return None

sounds = {
    "pose_1": load_sound("music1.wav"), 
    "pose_2": load_sound("music2.wav"),
    "pose_3": load_sound("music3.wav")
}

# --- Pose Math ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# Matching logic focused only on the 4 core joints
CRITICAL_JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]

def match_pose(current_angles, reference_pose, pose_name):
    if not current_angles: return False
    for key in CRITICAL_JOINTS:
        if key in current_angles and key in reference_pose:
            diff = abs(current_angles[key] - reference_pose[key])
            if diff > TOLERANCE: 
                return False
    return True

# --- State Machine Setup ---
class State:
    WAITING_FOR_POSE_1 = 0
    REVEALING_SECTION_3 = 1  
    WAITING_FOR_POSE_2 = 2
    REVEALING_SECTION_2 = 3  
    WAITING_FOR_POSE_3 = 4
    REVEALING_SECTION_1 = 5  
    ALL_REVEALED = 6

# --- Load Assets ---
def setup_layers():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load {IMAGE_PATH}")
    color_layer = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ghost_base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ghost_layer = cv2.GaussianBlur(ghost_base, (99, 99), 0)
    return color_layer, ghost_layer

# --- Main Application ---
def main():
    try:
        color_layer, ghost_layer = setup_layers()
        with open(REFERENCE_POSES_PATH, 'r') as f:
            reference_data = json.load(f)
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    H, W, _ = color_layer.shape
    section_h = H // 3
    reveal_progress = [0.0, 0.0, 0.0]
    
    transition_start_time = None
    current_state = State.WAITING_FOR_POSE_1
    pose_hold_start = None
    
    cap = cv2.VideoCapture(1) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("🚀 Setup Complete. Performance Starting.")
    
    # Time-based ripple variable for animation
    animation_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        current_angles = {}
        torch_pos = None
        
        # Increment to make the wave move autonomously
        animation_time += 0.1 
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            def get_pt(idx): return [landmarks[idx].x, landmarks[idx].y]
            
            l_s, l_e, l_w, l_h, l_k = get_pt(11), get_pt(13), get_pt(15), get_pt(23), get_pt(25)
            r_s, r_e, r_w, r_h, r_k = get_pt(12), get_pt(14), get_pt(16), get_pt(24), get_pt(26)
            
            current_angles = {
                "left_shoulder": calculate_angle(l_h, l_s, l_e),
                "right_shoulder": calculate_angle(r_h, r_s, r_e),
                "left_elbow": calculate_angle(l_s, l_e, l_w),
                "right_elbow": calculate_angle(r_s, r_e, r_w)
            }
            
            if results.right_hand_landmarks:
                idx_tip = results.right_hand_landmarks.landmark[8]
                torch_pos = (int(idx_tip.x * W), int(idx_tip.y * H))
            elif results.left_hand_landmarks:
                idx_tip = results.left_hand_landmarks.landmark[8]
                torch_pos = (int(idx_tip.x * W), int(idx_tip.y * H))

        # --- State Machine Logic ---
        target_pose_key = None
        if current_state == State.WAITING_FOR_POSE_1: target_pose_key = "pose_1"
        elif current_state == State.WAITING_FOR_POSE_2: target_pose_key = "pose_2"
        elif current_state == State.WAITING_FOR_POSE_3: target_pose_key = "pose_3"
        
        if target_pose_key:
            if match_pose(current_angles, reference_data[target_pose_key], target_pose_key):
                if pose_hold_start is None: 
                    pose_hold_start = time.time()
                elif time.time() - pose_hold_start >= HOLD_TIME:
                    if sounds[target_pose_key]: 
                        sounds[target_pose_key].play()
                        print(f"🎵 Playing {target_pose_key} sound!")
                    else:
                        print(f"⚠️ {target_pose_key} sound could not be loaded, skipping playback.")
                    
                    current_state += 1
                    transition_start_time = time.time() 
                    pose_hold_start = None
            else:
                pose_hold_start = None

        # --- Time-Based Animation Sync ---
        current_time = time.time()
        if current_state in [State.REVEALING_SECTION_1, State.REVEALING_SECTION_2, State.REVEALING_SECTION_3]:
            elapsed = current_time - transition_start_time
            idx = 2 if current_state == State.REVEALING_SECTION_3 else (1 if current_state == State.REVEALING_SECTION_2 else 0)
            reveal_progress[idx] = min(elapsed / TRANSITION_DURATION, 1.0)
            
            if reveal_progress[idx] >= 1.0: 
                current_state += 1

        # --- Cinematic Rendering with Liquid Wave ---
        full_mask = np.zeros((H, W), dtype=np.float32)
        
        for i in range(3):
            start_y, end_y = i * section_h, (i + 1) * section_h
            
            if i == 2: 
                end_y = H 
            
            # Extend sections upwards to prevent hard horizontal boundaries
            if i > 0:
                start_y = max(0, start_y - OVERLAP_OFFSET) 
            
            p = reveal_progress[i]
            if p > 0:
                y_coords = np.arange(start_y, end_y)
                wave_offset = np.sin(y_coords * WAVE_FREQUENCY + animation_time) * WAVE_AMPLITUDE
                
                base_x = p * (W + WAVE_AMPLITUDE * 2 + FEATHER_SIZE) - FEATHER_SIZE
                reveal_x_per_row = base_x + wave_offset
                
                x_grid = np.arange(W)
                t = np.clip((x_grid - reveal_x_per_row[:, None]) / -FEATHER_SIZE + 1.0, 0, 1)
                
                # Non-linear Ease-In-Out for horizontal liquid feel
                row_mask = t * t * (3 - 2 * t)
                
                # Vertical Fade: Eliminates the sharp horizontal cut line at the top of the section
                y_fade = np.ones((len(y_coords), 1), dtype=np.float32)
                if i > 0:
                    # Create a gradient from 0.0 to 1.0 for the overlapping area
                    fade_length = min(OVERLAP_OFFSET, len(y_coords))
                    y_fade[:fade_length, 0] = np.linspace(0, 1, fade_length)
                
                # Multiply horizontal reveal with vertical fade
                final_section_mask = row_mask * y_fade
                
                # Merge seamlessly with other sections
                full_mask[start_y:end_y, :] = np.maximum(full_mask[start_y:end_y, :], final_section_mask)

        if torch_pos:
            torch_mask = np.zeros((H, W), dtype=np.float32)
            cv2.circle(torch_mask, torch_pos, TORCH_RADIUS, 1.0, -1)
            torch_mask = cv2.GaussianBlur(torch_mask, (51, 51), 0)
            full_mask = np.maximum(full_mask, torch_mask)

        # Final Cinematic Composition
        mask_3ch = cv2.merge([full_mask, full_mask, full_mask])
        display_img = (ghost_layer * (1.0 - mask_3ch) + color_layer * mask_3ch).astype(np.uint8)

        if pose_hold_start:
            progress_pct = min(int((time.time() - pose_hold_start) / HOLD_TIME * 100), 100)
            cv2.putText(display_img, f"MATCHING... {progress_pct}%", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 150), 3)

        preview = cv2.resize(display_img, (W // 2, H // 2))
        cv2.imshow("Lumina Artis - Interactive Canvas", preview)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()