import cv2
import mediapipe as mp
import time, csv, os
from statistics import median

# ---------- Config ----------
TEST_DURATION = 60.0           # 1 minute test
CALIBRATION_SEC = 2.0
EMA_ALPHA = 0.25
MIN_AIR_MS = 120
ANKLE_ASYMMETRY_MAX = 0.035
LIFT_FRAC_OF_LOWER_LIMB = 0.12
MIN_LIFT_ABS = 0.02
# ----------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

# Output folder
os.makedirs("skipping_sessions", exist_ok=True)
session_id = time.strftime("%Y%m%d_%H%M%S")
video_path = f"skipping_sessions/skip_{session_id}.mp4"
csv_path   = f"skipping_sessions/skip_{session_id}.csv"

# Video writer
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0: fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

# State
waiting_for_person = True
calibrating = False
calib_start = None
calib_ank, calib_hip = [], []
baseline_ank, baseline_hip, lower_limb, lift_thresh, land_thresh = [None]*5
ankle_ema = None
in_air, air_start, jump_count, peak_lift = False, None, 0, 0.0
jump_log = []
test_start_time, end_time = None, None

def put_text(img, text, org, scale=0.9, thick=2, color=(0,0,255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    now = time.time()

    if res.pose_landmarks:
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = res.pose_landmarks.landmark
        la, ra = lm[27].y, lm[28].y
        lh, rh = lm[23].y, lm[24].y
        ankle_y = (la + ra) / 2.0
        hip_y   = (lh + rh) / 2.0

        if waiting_for_person:
            put_text(frame, "Person detected ✅ - Starting calibration", (20, 40), 0.9, 2, (0,255,0))
            waiting_for_person = False
            calibrating = True
            calib_start = now
            calib_ank, calib_hip = [], []
        elif calibrating:
            calib_ank.append(ankle_y)
            calib_hip.append(hip_y)
            put_text(frame, "Calibrating... stand still", (20, 40), 0.9, 2, (0,255,255))
            if now - calib_start >= CALIBRATION_SEC and len(calib_ank) >= 10:
                baseline_ank = median(calib_ank)
                baseline_hip = median(calib_hip)
                lower_limb = max(1e-3, baseline_ank - baseline_hip)
                lift_thresh = max(MIN_LIFT_ABS, LIFT_FRAC_OF_LOWER_LIMB * lower_limb)
                land_thresh = lift_thresh * 0.5
                calibrating = False
                test_start_time = now
                end_time = test_start_time + TEST_DURATION
        else:
            # -------- Jump Detection --------
            if ankle_ema is None:
                ankle_ema = ankle_y
            else:
                ankle_ema = EMA_ALPHA * ankle_y + (1.0 - EMA_ALPHA) * ankle_ema
            lift_now = max(0.0, baseline_ank - ankle_ema)
            ankles_level = abs(la - ra) <= ANKLE_ASYMMETRY_MAX

            if not in_air:
                if lift_now >= lift_thresh and ankles_level:
                    in_air = True
                    air_start = now
                    peak_lift = lift_now
            else:
                if lift_now > peak_lift:
                    peak_lift = lift_now
                if lift_now <= land_thresh:
                    airtime_ms = (now - air_start) * 1000.0
                    if airtime_ms >= MIN_AIR_MS:
                        jump_count += 1
                        jump_log.append((now - test_start_time, jump_count, round(airtime_ms, 1), round(peak_lift, 4)))
                    in_air = False
                    peak_lift = 0.0

            # -------- Timer Display --------
            if test_start_time:
                remaining = max(0, end_time - now)
                put_text(frame, f"Time Left: {int(remaining)}s", (20, 40), 1.0, 2, (255,255,0))
                put_text(frame, f"Jumps: {jump_count}", (20, 80), 1.2, 3, (0,0,255))
                if remaining <= 0:
                    break
    else:
        if waiting_for_person:
            put_text(frame, "Waiting for person...", (20, 40), 1.0, 2, (0,0,255))
        else:
            put_text(frame, "Pose lost!", (20, 40), 1.0, 2, (0,165,255))

    out.write(frame)
    cv2.imshow("Skipping Rope Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save only total jumps to CSV
with open(csv_path, "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["Total_Jumps"])
    wcsv.writerow([jump_count])

print(f"✅ Test finished (1 min). Video: {video_path}, CSV: {csv_path}, Jumps: {jump_count}")


print(f"✅ Test finished (1 min). Video: {video_path}, CSV: {csv_path}, Jumps: {jump_count}")
