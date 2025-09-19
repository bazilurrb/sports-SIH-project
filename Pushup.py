import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime

# Initialize Mediapipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Start video capture
cap = cv2.VideoCapture(0)

# Video recording setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('pushup_session.avi', fourcc, 20.0, (640,480))

# Session duration
duration = 60  # 1 minute
start_time = None  # will start when body detected

# Counters
correct_pushups = 0
bad_pushups = 0
stage = None
posture_good = False  
session_started = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        full_body_detected = False
        try:
            landmarks = results.pose_landmarks.landmark
            # Check if critical landmarks exist
            required_points = [
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_ELBOW.value,
                mp_pose.PoseLandmark.LEFT_WRIST.value,
                mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.LEFT_KNEE.value,
                mp_pose.PoseLandmark.LEFT_ANKLE.value
            ]
            full_body_detected = all(0 <= landmarks[p].visibility > 0.7 for p in required_points)
            
            if session_started:  
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                
                # Calculate angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                body_angle = calculate_angle(shoulder, hip, knee)
                
                # Posture check
                posture_good = body_angle >= 150
                
                # Pushup counter logic
                if elbow_angle > 160:
                    stage = "up"
                if elbow_angle < 70 and stage == "up":
                    stage = "down"
                    if posture_good:
                        correct_pushups += 1
                    else:
                        bad_pushups += 1

        except:
            pass
        
        # Start session timer only after full body detected
        if full_body_detected and not session_started:
            session_started = True
            start_time = time.time()
        
        # Draw counters if session started
        if session_started:
            elapsed = int(time.time() - start_time)
            remaining = max(0, duration - elapsed)

            cv2.rectangle(image, (0,0), (400,140), (245,117,16), -1)

            # Correct pushups
            cv2.putText(image, 'Correct', (15,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, str(correct_pushups), 
                        (150,45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

            # Bad pushups
            cv2.putText(image, 'Bad', (15,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(image, str(bad_pushups), 
                        (150,95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

            # Timer
            cv2.putText(image, f"Time: {remaining}s", (250,80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)

            # End after 60 sec
            if elapsed >= duration:
                print(f"Recording finished. Video saved as pushup_session.avi")
                print(f"Correct Pushups: {correct_pushups}, Bad Pushups: {bad_pushups}")
                
                # Save results to CSV
                filename = "pushup_log.csv"
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if file.tell() == 0:
                        writer.writerow(["Date-Time", "Correct Pushups", "Bad Pushups"])
                    writer.writerow([now, correct_pushups, bad_pushups])
                print(f"Results saved to {filename}")
                break
        else:
            # Show waiting message until user is detected
            cv2.putText(image, "Stand in frame to start workout", (50,200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        # Render pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               
        
        # Save video frame
        out.write(image)
        
        cv2.imshow('Pushup Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
