import cv2
import mediapipe as mp
import numpy as np
import json
import os

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    Calculates the 2D angle between three landmarks.
    a, b, c are [x, y] coordinates. 'b' is the vertex.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (vertex)
    c = np.array(c)  # End point
    
    # Calculate the angle in radians and convert to degrees
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def save_pose_data(reference_poses, pose_key, current_angles, frame, output_file):
    """
    Saves the pose angles to JSON and the current frame as an image.
    """
    # Save JSON data
    reference_poses[pose_key] = current_angles
    with open(output_file, 'w') as f:
        json.dump(reference_poses, f, indent=4)
    
    # Save Image
    pose_num = pose_key.split('_')[1]
    image_filename = f"pose{pose_num}.png"
    cv2.imwrite(image_filename, frame)
    
    print(f"Saved {pose_key} to {output_file}")
    print(f"Saved image to {image_filename}")

def main():
    # File to save reference poses
    output_file = "reference_poses.json"
    
    # Load existing poses if the file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                reference_poses = json.load(f)
            except json.JSONDecodeError:
                reference_poses = {}
    else:
        reference_poses = {}

    # Initialize camera (0 for Mac, 1 or 2 for Phone/Continuity Camera)
    cap = cv2.VideoCapture(1)
    
    # Set model complexity to 1 for M3 (balanced performance)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5, 
        model_complexity=1
    ) as holistic:
        
        print("--- Pose Calibration Tool ---")
        print("Press '1', '2', or '3' to save the current pose.")
        print("Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip the frame horizontally for a later selfie-view display
            # M3 optimization: work on RGB copy
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make Detections
            results = holistic.process(image)
            
            # Draw landmarks back on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Calculate angles if landmarks are detected
            current_angles = {}
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for relevant landmarks
                # Index map: 11:L_Shoulder, 13:L_Elbow, 15:L_Wrist, 23:L_Hip, 25:L_Knee
                # Index map: 12:R_Shoulder, 14:R_Elbow, 16:R_Wrist, 24:R_Hip, 26:R_Knee
                
                l_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
                l_hip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y]
                
                r_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
                r_hip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y]
                
                # Calculate required angles
                current_angles = {
                    "left_shoulder": calculate_angle(l_hip, l_shoulder, l_elbow),
                    "right_shoulder": calculate_angle(r_hip, r_shoulder, r_elbow),
                    "left_elbow": calculate_angle(l_shoulder, l_elbow, l_wrist),
                    "right_elbow": calculate_angle(r_shoulder, r_elbow, r_wrist),
                    "left_hip": calculate_angle(l_shoulder, l_hip, l_knee),
                    "right_hip": calculate_angle(r_shoulder, r_hip, r_knee)
                }
                
                # Draw Pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                # Display angles on screen
                y_offset = 30
                for key, value in current_angles.items():
                    cv2.putText(image, f"{key}: {int(value)}", (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    y_offset += 25

            cv2.imshow('Pose Calibration', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in [ord('1'), ord('2'), ord('3')]:
                pose_num = chr(key)
                pose_key = f"pose_{pose_num}"
                if current_angles:
                    # Capture the current frame with landmarks but before text if possible
                    # However, since we are in the same loop, 'image' currently has both
                    save_pose_data(reference_poses, pose_key, current_angles, image, output_file)
                else:
                    print(f"Error: No pose detected. Could not save {pose_key}.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
