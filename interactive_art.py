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
SHADOW_PATHS = ["shadow1.png", "shadow2.png", "shadow3.png"] 

# Tolerance increased and hold time reduced for better UX
TOLERANCE = 35  # degrees
HOLD_TIME = 0.8  # seconds

TRANSITION_DURATION = 25.0  # seconds (Revelation)
SPARK_DURATION = 2.0        # seconds (Lightning Flicker)
CLIMAX_DELAY = 7.0         # seconds (Delay after final reveal)
GHOST_BLUR_SIZE = (251, 251) # Enhanced for deeper mystery

FEATHER_SIZE = 300  # Increased for much softer transition

# Ripple Effect Settings
WAVE_AMPLITUDE = 60.0  # Wave width
WAVE_FREQUENCY = 0.02  # Wave frequency
OVERLAP_OFFSET = 200   # How many pixels a section extends upwards

GOLDEN_COLOR = (0, 215, 255) # BGR
WINDOW_NAME = "Lumina Artis - Babylon Installation"

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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(SCRIPT_DIR, "sounds")

pygame.mixer.pre_init(44100, -16, 2, 4096)
pygame.mixer.init()
pygame.mixer.set_num_channels(16)

# Dedicated Channels
CHAN_BGM = pygame.mixer.Channel(0)
CHAN_SFX = pygame.mixer.Channel(1)
CHAN_MUSIC = pygame.mixer.Channel(2) # For narrative music (Ducked)
CHAN_LAYER = pygame.mixer.Channel(3) # For instrumental layers (Hero)
CHAN_CLIMAX = pygame.mixer.Channel(4) # For final climax

def load_sound(file_name):
    file_path = os.path.join(SOUNDS_DIR, file_name)
    if os.path.exists(file_path):
        return pygame.mixer.Sound(file_path)
    print(f"⚠️ Error: Sound not found at {file_path}")
    return None

# 1. Background Music (Continuous Loop)
bgm = load_sound("cont.wav")
if bgm:
    CHAN_BGM.play(bgm, loops=-1)
    CHAN_BGM.set_volume(0.4)

# 2. Sound Effects & Layers
matching_sound = load_sound("matchingsound.WAV")
climax_sound = load_sound("flay.mp3")

narrative_music = {
    "pose_1": load_sound("music1.wav"), 
    "pose_2": load_sound("music2.wav"),
    "pose_3": load_sound("music3.wav")
}

instrumental_layers = {
    "pose_1": load_sound("Layer 1.mp3"),
    "pose_2": load_sound("Layer 2.mp3"),
    "pose_3": load_sound("Layer 3.mp3")
}

# --- Pose Math ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# Matching logic focused on the core joints (Shoulders, Elbows, Hips)
CRITICAL_JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_hip", "right_hip"]

def match_pose(current_angles, reference_pose, pose_name):
    if not current_angles: return False
    for key in CRITICAL_JOINTS:
        if key in current_angles and key in reference_pose:
            diff = abs(current_angles[key] - reference_pose[key])
            if diff > TOLERANCE: 
                return False
    return True

# --- Clap Detection ---
def detect_clap(results):
    """Detects a 'clap' gesture by checking the distance between palm centers."""
    if results.left_hand_landmarks and results.right_hand_landmarks:
        # Landmark 9 is the Middle Finger MCP (center of palm area)
        l_palm = results.left_hand_landmarks.landmark[9]
        r_palm = results.right_hand_landmarks.landmark[9]
        
        # Calculate Euclidean distance in normalized coordinates
        dist = np.sqrt((l_palm.x - r_palm.x)**2 + (l_palm.y - r_palm.y)**2)
        
        # Threshold: 0.08 is a good 'hands touching' distance
        return dist < 0.08
    return False

# --- State Machine Setup ---
class State:
    START_SCREEN = -1
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
    MEMORY_DISPLAY = 10

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

def capture_user_snapshot(frame):
    """Crops the raw frame to portrait aspect ratio and adds a subtle golden border."""
    h, w, _ = frame.shape
    target_w = int(h * 0.75) # 3:4 portrait
    start_x = (w - target_w) // 2
    # Ensure indices are integers and within bounds
    cropped = frame[:, max(0, start_x):min(w, start_x + target_w)].copy()
    
    # Professional golden border
    border_size = 12
    snapshot = cv2.copyMakeBorder(cropped, border_size, border_size, border_size, border_size, 
                                  cv2.BORDER_CONSTANT, value=GOLDEN_COLOR)
    return snapshot

def draw_memory_wall(canvas, snapshots, alpha, aw_rect):
    """Displays user snapshots on the right side of the artwork area with a cinematic fade."""
    if not snapshots: return canvas
    ax, ay, aw, ah = aw_rect
    ch, cw, _ = canvas.shape
    
    # Calculate vertical layout
    margin = 50
    total_margin = margin * (len(snapshots) + 1)
    snap_h = (ah - total_margin) // len(snapshots)
    
    overlay = canvas.copy()
    for i, snap in enumerate(snapshots):
        sh, sw, _ = snap.shape
        scale = snap_h / sh
        rw, rh = int(sw * scale), int(snap_h)
        res = cv2.resize(snap, (rw, rh))
        
        # Position: Right-aligned within or adjacent to the artwork
        px = ax + aw - rw - 60
        py = ay + margin + i * (snap_h + margin)
        
        if py + rh <= ch and px + rw <= cw:
            roi = overlay[py:py+rh, px:px+rw]
            overlay[py:py+rh, px:px+rw] = cv2.addWeighted(res, alpha, roi, 1.0 - alpha, 0)
            
    return overlay

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
    revelation_finish_time = None
    memory_start_time = None
    current_state = State.START_SCREEN
    pose_hold_start = None
    climax_triggered = False
    user_snapshots = []
    
    # Initialize Display Window in Normal Mode (not full screen)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, W // 2, H // 2)
    
    cap = cv2.VideoCapture(1) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("🚀 Divine Spark - Professional Audio & Display Update Active.")
    
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
                "right_elbow": calculate_angle(r_s, r_e, r_w),
                "left_hip": calculate_angle(l_s, l_h, l_k),
                "right_hip": calculate_angle(r_s, r_h, r_k)
            }

        # --- State Machine Logic ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        force_trigger = (key == ord(' '))
        
        if current_state == State.START_SCREEN:
            if detect_clap(results) or force_trigger:
                current_state = State.WAITING_FOR_POSE_1
                CHAN_SFX.play(matching_sound)
                print("👏 Clap Detected! Launching Installation...")
                # Reset animation timer for a fresh start
                animation_time = 0.0

        target_pose_key = None
        if current_state == State.WAITING_FOR_POSE_1: target_pose_key = "pose_1"
        elif current_state == State.WAITING_FOR_POSE_2: target_pose_key = "pose_2"
        elif current_state == State.WAITING_FOR_POSE_3: target_pose_key = "pose_3"
        
        if target_pose_key:
            is_matched = match_pose(current_angles, reference_data[target_pose_key], target_pose_key)
            
            if is_matched or force_trigger:
                if pose_hold_start is None: 
                    pose_hold_start = time.time()
                elif force_trigger or (time.time() - pose_hold_start >= HOLD_TIME):
                    # 0. Capture Snapshot immediately before spark
                    user_snapshots.append(capture_user_snapshot(frame))
                    
                    # MATCH! Trigger Audio Layering
                    CHAN_SFX.play(matching_sound)
                    
                    # 1. Narrative Music (Ducked to 30%)
                    if narrative_music[target_pose_key]:
                        CHAN_MUSIC.play(narrative_music[target_pose_key])
                        CHAN_MUSIC.set_volume(0.3)
                    
                    # 2. Instrumental Layer (Hero at 100%)
                    if instrumental_layers[target_pose_key]:
                        CHAN_LAYER.play(instrumental_layers[target_pose_key])
                        CHAN_LAYER.set_volume(1.0)
                    
                    print(f"🎵 Sparking {target_pose_key} (Manual: {force_trigger})")
                    
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
                if current_state == State.REVEALING_SECTION_1:
                    revelation_finish_time = current_time
                current_state += 1

        # Final Climax Trigger (Stage C)
        elif current_state == State.ALL_REVEALED:
            if not climax_triggered and revelation_finish_time:
                if current_time - revelation_finish_time >= CLIMAX_DELAY:
                    if climax_sound:
                        CHAN_CLIMAX.play(climax_sound)
                        climax_triggered = True
                        print("✨ Ritual Complete: Final Climax Triggered!")
            
            # Transition to Memory Wall after climax sound finishes
            if climax_triggered and not CHAN_CLIMAX.get_busy():
                # Swap first and third snapshots as requested for the final display
                if len(user_snapshots) >= 3:
                    user_snapshots[0], user_snapshots[2] = user_snapshots[2], user_snapshots[0]
                
                current_state = State.MEMORY_DISPLAY
                memory_start_time = time.time()
                print("📸 Displaying Memory Wall...")

        # --- Rendering Logic ---
        full_mask = np.zeros((H, W), dtype=np.float32)
        active_spark_layer = np.zeros((H, W, 3), dtype=np.uint8)
        
        for i in range(3):
            p = reveal_progress[i]
            
            # 1. Handle Sparking (Stage A)
            section_sparking_state = [State.SPARKING_SECTION_3, State.SPARKING_SECTION_2, State.SPARKING_SECTION_1][i]
            if current_state == section_sparking_state:
                flicker = (np.sin(current_time * 60) * 0.5 + 0.5) * (np.random.rand() * 0.3 + 0.7)
                spark_img = (shadow_layers[i] * flicker).astype(np.uint8)
                active_spark_layer = cv2.add(active_spark_layer, spark_img)

            # 2. Handle Revelation (Stage B)
            if p > 0:
                dist_transform = dist_transforms[i]
                max_dist = max_dists[i]
                
                y_coords = np.linspace(0, H, H)
                wave_slow = np.sin(y_coords * WAVE_FREQUENCY + animation_time) * WAVE_AMPLITUDE
                wave_fast = np.sin(y_coords * WAVE_FREQUENCY * 3.5 + animation_time * 2.2) * (WAVE_AMPLITUDE * 0.4)
                combined_wave = wave_slow + wave_fast
                
                threshold = p * (max_dist + WAVE_AMPLITUDE + FEATHER_SIZE)
                row_thresholds = threshold + combined_wave[:, None]
                
                t = np.clip((row_thresholds - dist_transform) / FEATHER_SIZE, 0, 1)
                section_mask = t * t * (3 - 2 * t)
                
                v_mask = np.zeros((H, 1), dtype=np.float32)
                y_shift = 120 if i == 0 else 0
                start_y = (2 - i) * section_h - y_shift
                end_y = (3 - i) * section_h
                start_y, end_y = max(0, start_y), min(H, end_y)
                
                if i < 2: start_y = max(0, start_y - OVERLAP_OFFSET)
                v_mask[int(start_y):int(end_y)] = 1.0
                
                if start_y > 0:
                    fade_h = min(200, section_h)
                    fade_ramp = np.linspace(0, 1, fade_h).reshape(-1, 1)
                    target_range = v_mask[int(start_y):int(start_y)+fade_h]
                    v_mask[int(start_y):int(start_y)+fade_h] = np.minimum(target_range, fade_ramp)
                
                section_mask *= v_mask
                full_mask = np.maximum(full_mask, section_mask)

        # Composition
        if current_state == State.START_SCREEN:
            # Render Start Screen
            display_img = np.zeros((H, W, 3), dtype=np.uint8)
            pulse = (np.sin(current_time * 3.0) * 0.5 + 0.5) * 0.5 + 0.5
            text = "CLAP TO START"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3.5
            thickness = 8
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (W - text_size[0]) // 2
            text_y = (H + text_size[1]) // 2
            
            # Golden pulsing color
            color = (int(GOLDEN_COLOR[0]*pulse), int(GOLDEN_COLOR[1]*pulse), int(GOLDEN_COLOR[2]*pulse))
            cv2.putText(display_img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        else:
            mask_3ch = cv2.merge([full_mask, full_mask, full_mask])
            base_img = (ghost_layer * (1.0 - mask_3ch) + color_layer * mask_3ch).astype(np.uint8)
            display_img = cv2.add(base_img, active_spark_layer)

        if pose_hold_start:
            progress_pct = min(int((time.time() - pose_hold_start) / HOLD_TIME * 100), 100)
            cv2.putText(display_img, f"MATCHING... {progress_pct}%", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 150), 3)

        # --- Cinematic Letterbox Scaling ---
        # Get Current Window Size
        rect = cv2.getWindowImageRect(WINDOW_NAME)
        screen_w, screen_h = rect[2], rect[3]
        
        # Fallback if window rect is invalid
        if screen_w <= 0 or screen_h <= 0:
            screen_w, screen_h = W // 2, H // 2

        # Calculate scaling to fit screen while maintaining aspect ratio
        scale = min(screen_w / W, screen_h / H)
        new_w, new_h = max(1, int(W * scale)), max(1, int(H * scale))
        
        # Resize artwork
        scaled_img = cv2.resize(display_img, (new_w, new_h))
        
        # Create black canvas and center the artwork
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        off_x = (screen_w - new_w) // 2
        off_y = (screen_h - new_h) // 2
        canvas[off_y:off_y+new_h, off_x:off_x+new_w] = scaled_img
        
        # --- Memory Wall Overlay ---
        if current_state == State.MEMORY_DISPLAY and memory_start_time:
            alpha = min((time.time() - memory_start_time) / 2.0, 1.0) # 2-second fade
            canvas = draw_memory_wall(canvas, user_snapshots, alpha, (off_x, off_y, new_w, new_h))

        cv2.imshow(WINDOW_NAME, canvas)
        # Key handling moved to start of loop

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()