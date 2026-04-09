from ultralytics import YOLO
import cv2
import time
import os
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

# ============================================
# SMTP CONFIG — Apni details yahan daalo
# ============================================
SMTP_HOST     = 'smtp.gmail.com'
SMTP_PORT     = 587
EMAIL_SENDER  = 'mohammadukkasha4@gmail.com'       # tumhara email
EMAIL_PASSWORD= 'ywxy rqym egov uhmf'     # Gmail App Password
EMAIL_RECEIVER= 'thompsonmaria316@gmail.com'   # jise email bhejna hai

# ============================================
# MODELS
# ============================================
pose_model  = YOLO('yolo11n-pose.pt')
phone_model = YOLO('yolo11n.pt')

MOBILE       = 67
ALERT_TIME   = 120    # 2 minutes
TOLERANCE    = 7      # 7 second
WRIST_DIST   = 150    # pixel distance

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30

LEFT_WRIST  = 9
RIGHT_WRIST = 10

detection_start = None
last_seen       = None
alert_triggered = False

os.makedirs('screenshots', exist_ok=True)

# ============================================
# EMAIL FUNCTION
# ============================================
def send_email(screenshot_path, elapsed_time):
    try:
        msg = MIMEMultipart()
        msg['From']    = EMAIL_SENDER
        msg['To']      = EMAIL_RECEIVER
        msg['Subject'] = '🚨 Distraction Alert — Mobile Use Detected!'

        # Email body
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"""
🚨 DISTRACTION ALERT!

━━━━━━━━━━━━━━━━━━━━━━
📅 Time     : {now_str}
⏱ Duration  : {int(elapsed_time)} seconds
📱 Status   : Person using mobile detected
━━━━━━━━━━━━━━━━━━━━━━

Screenshot attached.

-- Distraction Detection System
        """
        msg.attach(MIMEText(body, 'plain'))

        # Screenshot attach karo
        with open(screenshot_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment',
                           filename=os.path.basename(screenshot_path))
            msg.attach(img)

        # Email bhejo
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"✅ Email sent to {EMAIL_RECEIVER}!")

    except Exception as e:
        print(f"❌ Email error: {e}")

# ============================================
# HELPER FUNCTIONS
# ============================================
def get_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1-25), (x1+w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# ============================================
# MAIN DETECTION
# ============================================
def detect(frame):
    global detection_start, last_seen, alert_triggered

    pose_results  = pose_model(frame, conf=0.5)[0]
    phone_results = phone_model(frame, conf=0.4, iou=0.3)[0]

    # Phones
    phones = []
    for box in phone_results.boxes:
        if int(box.cls) == MOBILE:
            bbox = box.xyxy[0].tolist()
            conf = float(box.conf)
            phones.append({'box': bbox, 'conf': conf})
            draw_box(frame, bbox, f"Phone {conf:.2f}", (255, 165, 0))

    # Wrists
    wrists = []
    if pose_results.keypoints is not None:
        for person_kps in pose_results.keypoints.xy:
            for idx in [LEFT_WRIST, RIGHT_WRIST]:
                kp = person_kps[idx]
                x, y = float(kp[0]), float(kp[1])
                if x > 0 and y > 0:
                    wrists.append((x, y))
                    cv2.circle(frame, (int(x), int(y)), 8, (0,255,255), -1)

    # Person boxes
    for box in pose_results.boxes:
        draw_box(frame, box.xyxy[0].tolist(),
                 f"Person {float(box.conf):.2f}", (0, 255, 0))

    # Phone + Wrist check
    mobile_in_use = False
    for phone in phones:
        phone_center = get_center(phone['box'])
        for wrist in wrists:
            dist = distance(phone_center, wrist)
            if dist < WRIST_DIST:
                mobile_in_use = True
                cv2.line(frame,
                         (int(wrist[0]), int(wrist[1])),
                         (int(phone_center[0]), int(phone_center[1])),
                         (0, 0, 255), 2)
                draw_box(frame, phone['box'], "IN USE!", (0, 0, 255))

    # Timer logic
    now = time.time()

    if mobile_in_use:
        last_seen = now
        if detection_start is None:
            detection_start = now
            alert_triggered = False
            print("⏱ Timer shuru!")
    else:
        if last_seen is not None:
            if (now - last_seen) > TOLERANCE:
                if detection_start is not None:
                    print("🔄 Timer Reset!")
                detection_start = None
                last_seen       = None
                alert_triggered = False

    # Status + Alert
    if detection_start is not None and not alert_triggered:
        elapsed   = now - detection_start
        remaining = ALERT_TIME - elapsed

        progress = min(elapsed / ALERT_TIME, 1.0)
        bar_w    = int(frame.shape[1] * progress)
        cv2.rectangle(frame, (0, frame.shape[0]-15),
                      (bar_w, frame.shape[0]), (0, 0, 255), -1)

        mins = int(remaining) // 60
        secs = int(remaining) % 60
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (0, 0, 200), -1)
        cv2.putText(frame, f"DISTRACTED! Alert in: {mins:02d}:{secs:02d}",
                    (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)

        # 2 min complete → screenshot + email
        if elapsed >= ALERT_TIME:
            alert_triggered = True
            filename = f"screenshots/alert_{int(now)}.jpg"
            cv2.imwrite(filename, frame)
            print(f"🚨 ALERT! Screenshot: {filename}")

            # Email bhejo
            send_email(filename, elapsed)

            detection_start = None
            last_seen       = None
    else:
        status = "Phone on TABLE — OK" if phones else "Normal"
        color  = (0, 200, 200) if phones else (0, 150, 0)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), color, -1)
        cv2.putText(frame, status, (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    return frame

# ============================================
# CAMERA
# ============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

print(f"✅ Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"📧 Email alerts: {EMAIL_RECEIVER}")

fps_counter = 0
fps_display = 0
fps_time    = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = detect(frame)

    fps_counter += 1
    if time.time() - fps_time >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_time    = time.time()

    cv2.putText(frame, f"FPS: {fps_display}",
                (frame.shape[1]-120, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow("Distraction Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()