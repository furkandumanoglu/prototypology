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
SHADOW_PATHS = ["shadow1.png", "shadow2.png", "shadow3.png"] # Ordered to match bottom-to-top reveal sequence

# Tolerance increased and hold time reduced for better UX
TOLERANCE = 35  # degrees
HOLD_TIME = 0.8  # seconds

TRANSITION_DURATION = 25.0  # seconds (Revelation)
SPARK_DURATION = 2.0        # seconds (Lightning Flicker)
GHOST_BLUR_SIZE = (201, 201)

FEATHER_SIZE = 300  # Increased for much softer transition

# Ripple Effect Settings
WAVE_AMPLITUDE = 60.0  # Wave width
WAVE_FREQUENCY = 0.02  # Wave frequency
OVERLAP_OFFSET = 200   # How many pixels a section extends upwards

GOLDEN_COLOR = (0, 215, 255) # BGR

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

# 2. Sound Effects
matching_sound = load_sound("matchingsound.WAV")

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
    SPARKING_SECTION_3 = 1
    REVEALING_SECTION_3 = 2
    WAITING_FOR_POSE_2 = 3
    SPARKING_SECTION_2 = 4
    REVEALING_SECTION_2 = 5
    WAITING_FOR_POSE_3 = 6
    SPARKING_SECTION_1 = 7
    REVEALING_SECTION_1 = 8
    ALL_REVEALED = 9

# --- Load Assets ---
def setup_layers():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load {IMAGE_PATH}")
    color_layer = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ghost_base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ghost_layer = cv2.GaussianBlur(ghost_base, GHOST_BLUR_SIZE, 0)
    
    shadow_layers = []
    dist_transforms = []
    
    for path in SHADOW_PATHS:
        # Load shadow (white lines on black)
        shadow_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if shadow_mask is None:
            print(f"⚠️ Warning: Could not load shadow {path}")
            shadow_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # 1. Create Golden Glow Layer
        # Create a colored image from the mask
        glow_layer = np.zeros_like(img)
        glow_layer[shadow_mask > 127] = GOLDEN_COLOR
        # Blur the colored lines to create a glow/lightning effect
        glow_layer = cv2.GaussianBlur(glow_layer, (21, 21), 0)
        shadow_layers.append(glow_layer)
        
        # 2. Calculate Distance Transform for reveal logic
        # Invert mask so lines are 0 (distance is from lines)
        inverted_mask = cv2.bitwise_not(shadow_mask)
        dist = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
        dist_transforms.append(dist)
        
    return color_layer, ghost_layer, shadow_layers, dist_transforms

# --- Main Application ---
def main():
    try:
        color_layer, ghost_layer, shadow_layers, dist_transforms = setup_layers()
        with open(REFERENCE_POSES_PATH, 'r') as f:
            reference_data = json.load(f)
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    H, W, _ = color_layer.shape
    section_h = H // 3
    reveal_progress = [0.0, 0.0, 0.0]
    
    # Pre-calculate max distances for normalization
    max_dists = [np.max(dt) for dt in dist_transforms]
    
    transition_start_time = None
    current_state = State.WAITING_FOR_POSE_1
    pose_hold_start = None
    
    cap = cv2.VideoCapture(1) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("🚀 Divine Spark Update Active. Performance Starting.")
    
    # Time-based ripple variable for animation
    animation_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        current_angles = {}
        
        # Increment to make the wave move autonomously
        animation_time += 0.15 
        
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
                    # MATCH! Stage A Triggered
                    if matching_sound: matching_sound.play()
                    if sounds[target_pose_key]: 
                        sounds[target_pose_key].play()
                        print(f"🎵 Sparking {target_pose_key}!")
                    
                    current_state += 1 # Transition to SPARKING
                    transition_start_time = time.time() 
                    pose_hold_start = None
            else:
                pose_hold_start = None

        # --- Transition Management ---
        current_time = time.time()
        
        # Sparking Duration (Stage A)
        if current_state in [State.SPARKING_SECTION_3, State.SPARKING_SECTION_2, State.SPARKING_SECTION_1]:
            if current_time - transition_start_time >= SPARK_DURATION:
                current_state += 1 # Transition to REVEALING
                transition_start_time = time.time() # Reset timer for Stage B
        
        # Revelation Progress (Stage B)
        elif current_state in [State.REVEALING_SECTION_3, State.REVEALING_SECTION_2, State.REVEALING_SECTION_1]:
            idx = 0 if current_state == State.REVEALING_SECTION_3 else (1 if current_state == State.REVEALING_SECTION_2 else 2)
            elapsed = current_time - transition_start_time
            reveal_progress[idx] = min(elapsed / TRANSITION_DURATION, 1.0)
            if reveal_progress[idx] >= 1.0: 
                current_state += 1

        # --- Rendering Logic ---
        full_mask = np.zeros((H, W), dtype=np.float32)
        active_spark_layer = np.zeros((H, W, 3), dtype=np.uint8)
        
        for i in range(3):
            # i=0 is Section 3 (Bottom), i=1 is Section 2 (Middle), i=2 is Section 1 (Top)
            # This matches SHADOW_PATHS = [shadow3, shadow2, shadow1]
            p = reveal_progress[i]
            
            # 1. Handle Sparking (Stage A)
            section_sparking_state = [State.SPARKING_SECTION_3, State.SPARKING_SECTION_2, State.SPARKING_SECTION_1][i]
            if current_state == section_sparking_state:
                # Flicker Logic: rapid pulsing like lightning
                flicker = (np.sin(current_time * 60) * 0.5 + 0.5) * (np.random.rand() * 0.3 + 0.7)
                spark_img = (shadow_layers[i] * flicker).astype(np.uint8)
                active_spark_layer = cv2.add(active_spark_layer, spark_img)

            # 2. Handle Revelation (Stage B)
            if p > 0:
                dist_transform = dist_transforms[i]
                max_dist = max_dists[i]
                
                # Dynamic "Dance" logic: Combine two waves with different frequencies
                y_coords = np.linspace(0, H, H)
                wave_slow = np.sin(y_coords * WAVE_FREQUENCY + animation_time) * WAVE_AMPLITUDE
                wave_fast = np.sin(y_coords * WAVE_FREQUENCY * 3.5 + animation_time * 2.2) * (WAVE_AMPLITUDE * 0.4)
                combined_wave = wave_slow + wave_fast
                
                # reveal_threshold moves from 0 to max_dist
                threshold = p * (max_dist + WAVE_AMPLITUDE + FEATHER_SIZE)
                row_thresholds = threshold + combined_wave[:, None]
                
                # Calculate mask with Smoothstep falloff for premium softness
                # dist_transform: 0 at lines, increasing away
                t = np.clip((row_thresholds - dist_transform) / FEATHER_SIZE, 0, 1)
                section_mask = t * t * (3 - 2 * t)
                
                # Vertical constraint with Feathered boundaries to avoid sharp section lines
                v_mask = np.zeros((H, 1), dtype=np.float32)
                
                # 3cm (approx 120px) upward shift for the first layer (i=0)
                # But ensure the bottom section still covers the very bottom (H)
                y_shift = 120 if i == 0 else 0
                
                start_y = (2 - i) * section_h - y_shift
                end_y = (3 - i) * section_h
                
                # Clamp to screen boundaries
                start_y = max(0, start_y)
                end_y = min(H, end_y)
                
                if i < 2: start_y = max(0, start_y - OVERLAP_OFFSET)
                
                v_mask[int(start_y):int(end_y)] = 1.0
                
                # Soft vertical fade at the boundary of the section
                if start_y > 0:
                    fade_h = min(200, section_h)
                    fade_ramp = np.linspace(0, 1, fade_h).reshape(-1, 1)
                    target_range = v_mask[int(start_y):int(start_y)+fade_h]
                    v_mask[int(start_y):int(start_y)+fade_h] = np.minimum(target_range, fade_ramp)
                
                section_mask *= v_mask
                full_mask = np.maximum(full_mask, section_mask)

        # Composition
        mask_3ch = cv2.merge([full_mask, full_mask, full_mask])
        base_img = (ghost_layer * (1.0 - mask_3ch) + color_layer * mask_3ch).astype(np.uint8)
        
        # Add the Golden Spark (Additive blending for intense glow)
        display_img = cv2.add(base_img, active_spark_layer)

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