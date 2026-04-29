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

# Değişiklik 1: Tolerans artırıldı ve süre düşürüldü
# Tolerance is increased to 35 degrees, hold time reduced to 0.8s for better UX
TOLERANCE = 35  # degrees (Önceden 20'ydi, esneklik sağlandı)
HOLD_TIME = 0.8  # seconds (Önceden 2.0'dı, anlık yakalamalar için kısaltıldı)

TRANSITION_SPEED = 0.005  
FEATHER_SIZE = 100  
TORCH_RADIUS = 150  

# --- Initialize MediaPipe ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Initialize Audio ---
pygame.mixer.init()
def load_sound(file_path):
    if os.path.exists(file_path):
        return pygame.mixer.Sound(file_path)
    return None

sounds = {
    "pose_1": load_sound("music_1.wav"),
    "pose_2": load_sound("music_2.wav"),
    "pose_3": load_sound("music_3.wav")
}

# --- Pose Math ---
def calculate_angle(a, b, c):
    """Calculates the 2D angle between three landmarks."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# Değişiklik 2: Sadece 4 temel ekleme odaklanan eşleştirme mantığı
# Focus only on upper body joints to ignore lower body noise/occlusion
CRITICAL_JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]

def match_pose(current_angles, reference_pose, pose_name):
    """Checks if critical angles match reference pose within tolerance."""
    if not current_angles: return False
    
    for key in CRITICAL_JOINTS:
        # JSON dosyasından ve anlık veriden sadece omuz ve dirsekleri alıyoruz
        if key in current_angles and key in reference_pose:
            diff = abs(current_angles[key] - reference_pose[key])
            if diff > TOLERANCE:
                # Terminale hangi eklemin uymadığını yazdır (Debugging için harika)
                # print(f"[{pose_name}] Failed on {key}. Diff: {diff:.1f} (Max: {TOLERANCE})")
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
        print(f"Error initializing assets: {e}")
        return

    H, W, _ = color_layer.shape
    section_h = H // 3
    reveal_progress = [0.0, 0.0, 0.0]
    
    current_state = State.WAITING_FOR_POSE_1
    pose_hold_start = None
    
    # Kamerayı başlat (Start camera)
    cap = cv2.VideoCapture(1) 
    
    print("🚀 Art Installation Started. Focused on Upper Body.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        current_angles = {}
        torch_pos = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            def get_pt(idx): return [landmarks[idx].x, landmarks[idx].y]
            
            l_s, l_e, l_w, l_h, l_k = get_pt(11), get_pt(13), get_pt(15), get_pt(23), get_pt(25)
            r_s, r_e, r_w, r_h, r_k = get_pt(12), get_pt(14), get_pt(16), get_pt(24), get_pt(26)
            
            # Kalça hesaplamalarını sistemden kaldırmıyoruz, sadece eşleşmede kullanmıyoruz.
            current_angles = {
                "left_shoulder": calculate_angle(l_h, l_s, l_e),
                "right_shoulder": calculate_angle(r_h, r_s, r_e),
                "left_elbow": calculate_angle(l_s, l_e, l_w),
                "right_elbow": calculate_angle(r_s, r_e, r_w)
            }
            
            # Hand Torch
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
            # Sadece kritik 4 eklemi kontrol et (Check only the 4 critical joints)
            if match_pose(current_angles, reference_data[target_pose_key], target_pose_key):
                if pose_hold_start is None: 
                    pose_hold_start = time.time()
                elif time.time() - pose_hold_start >= HOLD_TIME:
                    if sounds[target_pose_key]: sounds[target_pose_key].play()
                    current_state += 1
                    pose_hold_start = None
                    print(f"✅ Triggered {target_pose_key} revelation!")
            else:
                pose_hold_start = None

        # --- Animation Progress ---
        if current_state == State.REVEALING_SECTION_3:
            reveal_progress[2] += TRANSITION_SPEED
            if reveal_progress[2] >= 1.0: 
                reveal_progress[2] = 1.0
                current_state = State.WAITING_FOR_POSE_2
        elif current_state == State.REVEALING_SECTION_2:
            reveal_progress[1] += TRANSITION_SPEED
            if reveal_progress[1] >= 1.0: 
                reveal_progress[1] = 1.0
                current_state = State.WAITING_FOR_POSE_3
        elif current_state == State.REVEALING_SECTION_1:
            reveal_progress[0] += TRANSITION_SPEED
            if reveal_progress[0] >= 1.0: 
                reveal_progress[0] = 1.0
                current_state = State.ALL_REVEALED

        # --- Cinematic Rendering ---
        full_mask = np.zeros((H, W), dtype=np.float32)
        
        for i in range(3):
            start_y, end_y = i * section_h, (i + 1) * section_h
            if i == 2: end_y = H 
            
            p = reveal_progress[i]
            if p > 0:
                reveal_x = p * (W + FEATHER_SIZE) - FEATHER_SIZE
                row_mask = np.clip((np.arange(W) - reveal_x) / -FEATHER_SIZE + 1.0, 0, 1)
                full_mask[start_y:end_y, :] = row_mask

        if torch_pos:
            torch_mask = np.zeros((H, W), dtype=np.float32)
            cv2.circle(torch_mask, torch_pos, TORCH_RADIUS, 1.0, -1)
            torch_mask = cv2.GaussianBlur(torch_mask, (51, 51), 0)
            full_mask = np.maximum(full_mask, torch_mask)

        mask_3ch = cv2.merge([full_mask, full_mask, full_mask])
        display_img = (ghost_layer * (1 - mask_3ch) + color_layer * mask_3ch).astype(np.uint8)

        # UI Feedback
        if pose_hold_start:
            progress_pct = int((time.time() - pose_hold_start) / HOLD_TIME * 100)
            # Eğer değer 100'ü geçerse 100'de sabitle
            progress_pct = min(progress_pct, 100)
            cv2.putText(display_img, f"MATCHING... {progress_pct}%", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 150), 3)

        preview = cv2.resize(display_img, (W // 2, H // 2))
        cv2.imshow("Interactive Art Installation - Babylon", preview)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()