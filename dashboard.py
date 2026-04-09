import streamlit as st
import cv2
import time
import os
import numpy as np
import smtplib
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from ultralytics import YOLO
import threading

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Distraction Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0e1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-label {
        font-size: 13px;
        color: #8892a4;
        margin-top: 5px;
    }
    .alert-card {
        background: linear-gradient(135deg, #2d1515, #3d1a1a);
        border: 1px solid #ff4444;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }
    .normal-card {
        background: linear-gradient(135deg, #0d2d1a, #0f3320);
        border: 1px solid #00cc44;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }
    .status-distracted {
        background: linear-gradient(90deg, #ff4444, #cc0000);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        animation: pulse 1s infinite;
    }
    .status-normal {
        background: linear-gradient(90deg, #00cc44, #009933);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
    }
    .status-table {
        background: linear-gradient(90deg, #ffaa00, #cc8800);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
    }
    .sidebar-header {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .log-entry {
        background: #1a1d2e;
        border-left: 3px solid #ff4444;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
    }
    footer { display: none !important; }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INIT
# ============================================
if 'alert_log'       not in st.session_state: st.session_state.alert_log       = []
if 'email_log'       not in st.session_state: st.session_state.email_log       = []
if 'total_alerts'    not in st.session_state: st.session_state.total_alerts    = 0
if 'total_emails'    not in st.session_state: st.session_state.total_emails    = 0
if 'detection_start' not in st.session_state: st.session_state.detection_start = None
if 'last_seen'       not in st.session_state: st.session_state.last_seen       = None
if 'alert_triggered' not in st.session_state: st.session_state.alert_triggered = False
if 'camera_active'   not in st.session_state: st.session_state.camera_active   = False

os.makedirs('screenshots', exist_ok=True)

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    pose  = YOLO('yolo11n-pose.pt')
    phone = YOLO('yolo11n.pt')
    return pose, phone

pose_model, phone_model = load_models()

MOBILE      = 67
LEFT_WRIST  = 9
RIGHT_WRIST = 10

# ============================================
# SMTP EMAIL
# ============================================
def send_email(screenshot_path, elapsed, sender, password, receiver):
    try:
        msg            = MIMEMultipart()
        msg['From']    = sender
        msg['To']      = receiver
        msg['Subject'] = '🚨 Distraction Alert!'

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"""
🚨 DISTRACTION ALERT!

📅 Time     : {now_str}
⏱ Duration  : {int(elapsed)} seconds
📱 Status   : Person using mobile detected

Screenshot attached.
-- Distraction Detection Dashboard
        """
        msg.attach(MIMEText(body, 'plain'))

        with open(screenshot_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment',
                           filename=os.path.basename(screenshot_path))
            msg.attach(img)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent!"
    except Exception as e:
        return False, str(e)

# ============================================
# DETECTION HELPERS
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

def run_detection(frame, conf_thresh, wrist_dist):
    pose_res  = pose_model(frame, conf=conf_thresh)[0]
    phone_res = phone_model(frame, conf=conf_thresh-0.1, iou=0.3)[0]

    phones, wrists = [], []

    for box in phone_res.boxes:
        if int(box.cls) == MOBILE:
            bbox = box.xyxy[0].tolist()
            conf = float(box.conf)
            phones.append({'box': bbox, 'conf': conf})
            draw_box(frame, bbox, f"Phone {conf:.2f}", (255,165,0))

    if pose_res.keypoints is not None:
        for kps in pose_res.keypoints.xy:
            for idx in [LEFT_WRIST, RIGHT_WRIST]:
                kp = kps[idx]
                x, y = float(kp[0]), float(kp[1])
                if x > 0 and y > 0:
                    wrists.append((x, y))
                    cv2.circle(frame, (int(x), int(y)), 8, (0,255,255), -1)

    for box in pose_res.boxes:
        draw_box(frame, box.xyxy[0].tolist(),
                 f"Person {float(box.conf):.2f}", (0,255,0))

    mobile_in_use = False
    for phone in phones:
        pc = get_center(phone['box'])
        for wrist in wrists:
            if distance(pc, wrist) < wrist_dist:
                mobile_in_use = True
                cv2.line(frame,
                         (int(wrist[0]), int(wrist[1])),
                         (int(pc[0]), int(pc[1])),
                         (0,0,255), 2)
                draw_box(frame, phone['box'], "IN USE!", (0,0,255))

    return frame, mobile_in_use, len(phones), len(wrists)//2

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)

    st.subheader("🎯 Detection")
    conf_thresh = st.slider("Confidence",    0.1, 0.9, 0.5, 0.05)
    wrist_dist  = st.slider("Wrist Distance", 50, 300, 150, 10)
    alert_time  = st.slider("Alert Time (sec)", 10, 300, 120, 10)
    tolerance   = st.slider("Tolerance (sec)",   1,  30,   7,  1)

    st.divider()

    st.subheader("📧 Email Config")
    email_enabled  = st.toggle("Enable Email Alerts", value=False)
    email_sender   = st.text_input("Sender Email",   placeholder="your@gmail.com")
    email_password = st.text_input("App Password",   type="password")
    email_receiver = st.text_input("Receiver Email", placeholder="receiver@gmail.com")

    if st.button("🧪 Test Email"):
        if email_sender and email_password and email_receiver:
            with st.spinner("Sending..."):
                ok, msg = send_email(
                    list(filter(lambda f: f.endswith('.jpg'),
                         os.listdir('screenshots') or ['']))[:1][0]
                    if os.listdir('screenshots') else 'screenshots',
                    0, email_sender, email_password, email_receiver
                )
            st.success("✅ Sent!") if ok else st.error(f"❌ {msg}")
        else:
            st.warning("Email details fill karo!")

    st.divider()

    if st.button("🗑️ Clear All Logs", type="secondary"):
        st.session_state.alert_log    = []
        st.session_state.email_log    = []
        st.session_state.total_alerts = 0
        st.session_state.total_emails = 0
        st.rerun()

# ============================================
# MAIN DASHBOARD
# ============================================
st.title("🚨 Distraction Detection Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

# ---- TABS ----
tab1, tab2, tab3, tab4 = st.tabs([
    "📹 Live Detection",
    "📊 Statistics",
    "🖼️ Alert History",
    "📧 Email Logs"
])

# ============================================
# TAB 1 — LIVE DETECTION
# ============================================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Camera Feed")
        cam_source = st.text_input("Camera Source", value="0",
                                   help="0=webcam, ya CCTV URL")

        c1, c2 = st.columns(2)
        start_btn = c1.button("▶ Start Detection", type="primary",  use_container_width=True)
        stop_btn  = c2.button("⏹ Stop",            type="secondary", use_container_width=True)

        frame_placeholder  = st.empty()
        status_placeholder = st.empty()
        timer_placeholder  = st.empty()

    with col2:
        st.subheader("📊 Live Stats")
        m1 = st.empty()
        m2 = st.empty()
        m3 = st.empty()
        m4 = st.empty()

        st.divider()
        st.subheader("📋 Live Log")
        log_placeholder = st.empty()

    # Detection loop
    if start_btn:
        st.session_state.camera_active   = True
        st.session_state.detection_start = None
        st.session_state.last_seen       = None
        st.session_state.alert_triggered = False

    if stop_btn:
        st.session_state.camera_active = False

    if st.session_state.camera_active:
        src = int(cam_source) if cam_source.isdigit() else cam_source
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        fps_counter = 0
        fps_display = 0
        fps_time    = time.time()

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera nahi mila!")
                break

            frame = cv2.resize(frame, (640, 480))
            frame, mobile_in_use, phone_count, person_count = run_detection(
                frame, conf_thresh, wrist_dist
            )

            now = time.time()

            # Timer logic
            if mobile_in_use:
                st.session_state.last_seen = now
                if st.session_state.detection_start is None:
                    st.session_state.detection_start = now
                    st.session_state.alert_triggered = False
                    st.session_state.alert_log.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'event': 'Detection Started',
                        'duration': 0
                    })
            else:
                if st.session_state.last_seen is not None:
                    if (now - st.session_state.last_seen) > tolerance:
                        st.session_state.detection_start = None
                        st.session_state.last_seen       = None
                        st.session_state.alert_triggered = False

            # Alert check
            if (st.session_state.detection_start is not None
                    and not st.session_state.alert_triggered):
                elapsed   = now - st.session_state.detection_start
                remaining = alert_time - elapsed

                if elapsed >= alert_time:
                    st.session_state.alert_triggered = True
                    st.session_state.total_alerts   += 1

                    filename = f"screenshots/alert_{int(now)}.jpg"
                    cv2.imwrite(filename, frame)

                    st.session_state.alert_log.append({
                        'time':     datetime.now().strftime("%H:%M:%S"),
                        'event':    '🚨 ALERT TRIGGERED',
                        'duration': int(elapsed),
                        'file':     filename
                    })

                    if email_enabled and email_sender and email_password and email_receiver:
                        ok, msg = send_email(filename, elapsed,
                                             email_sender, email_password, email_receiver)
                        st.session_state.email_log.append({
                            'time':   datetime.now().strftime("%H:%M:%S"),
                            'to':     email_receiver,
                            'status': '✅ Sent' if ok else f'❌ {msg}',
                            'file':   os.path.basename(filename)
                        })
                        if ok: st.session_state.total_emails += 1

                    st.session_state.detection_start = None
                    st.session_state.last_seen       = None

                # Timer on frame
                mins = int(remaining) // 60
                secs = int(remaining) % 60
                cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (0,0,200), -1)
                cv2.putText(frame, f"DISTRACTED! Alert in: {mins:02d}:{secs:02d}",
                            (15,38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)

                progress = min(elapsed / alert_time, 1.0)
                bar_w    = int(frame.shape[1] * progress)
                cv2.rectangle(frame, (0, frame.shape[0]-15),
                              (bar_w, frame.shape[0]), (0,0,255), -1)

                status_placeholder.markdown(
                    '<div class="status-distracted">🚨 DISTRACTED!</div>',
                    unsafe_allow_html=True)
                timer_placeholder.progress(progress, text=f"Alert in {mins:02d}:{secs:02d}")

            elif mobile_in_use is False and phone_count > 0:
                cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (0,200,200), -1)
                cv2.putText(frame, "Phone on TABLE", (15,38),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                status_placeholder.markdown(
                    '<div class="status-table">📱 Phone on Table — OK</div>',
                    unsafe_allow_html=True)
                timer_placeholder.empty()
            else:
                cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (0,150,0), -1)
                cv2.putText(frame, "Normal", (15,38),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                status_placeholder.markdown(
                    '<div class="status-normal">✅ Normal</div>',
                    unsafe_allow_html=True)
                timer_placeholder.empty()

            # FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_time    = time.time()

            cv2.putText(frame, f"FPS: {fps_display}",
                        (frame.shape[1]-120, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            # Show frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Live stats
            m1.markdown(f'<div class="metric-card"><div class="metric-value">{person_count}</div><div class="metric-label">👤 Persons</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="metric-value">{phone_count}</div><div class="metric-label">📱 Phones</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_alerts}</div><div class="metric-label">🚨 Total Alerts</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_emails}</div><div class="metric-label">📧 Emails Sent</div></div>', unsafe_allow_html=True)

            # Live log
            if st.session_state.alert_log:
                log_html = ""
                for entry in st.session_state.alert_log[-5:][::-1]:
                    log_html += f'<div class="log-entry">⏰ {entry["time"]} — {entry["event"]}</div>'
                log_placeholder.markdown(log_html, unsafe_allow_html=True)

        cap.release()

# ============================================
# TAB 2 — STATISTICS
# ============================================
with tab2:
    st.subheader("📊 Detection Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_alerts}</div><div class="metric-label">🚨 Total Alerts</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_emails}</div><div class="metric-label">📧 Emails Sent</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{len(os.listdir("screenshots"))}</div><div class="metric-label">📸 Screenshots</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{len(st.session_state.alert_log)}</div><div class="metric-label">📋 Log Entries</div></div>', unsafe_allow_html=True)

    st.divider()

    if st.session_state.alert_log:
        df = pd.DataFrame(st.session_state.alert_log)

        col1, col2 = st.columns(2)

        with col1:
            events = df['event'].value_counts().reset_index()
            fig = px.pie(events, values='count', names='event',
                         title='Event Distribution',
                         color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            alerts_only = df[df['event'] == '🚨 ALERT TRIGGERED']
            if not alerts_only.empty:
                fig2 = px.bar(alerts_only, x='time', y='duration',
                              title='Alert Duration (seconds)',
                              color='duration',
                              color_continuous_scale='Reds')
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)',
                                   font_color='white')
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Abhi koi data nahi — detection start karo!")

# ============================================
# TAB 3 — ALERT HISTORY
# ============================================
with tab3:
    st.subheader("🖼️ Alert Screenshots")

    screenshots = sorted(
        [f for f in os.listdir('screenshots') if f.endswith('.jpg')],
        reverse=True
    )

    if screenshots:
        cols_per_row = 3
        for i in range(0, len(screenshots), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i+j < len(screenshots):
                    fname = screenshots[i+j]
                    fpath = os.path.join('screenshots', fname)
                    with col:
                        img = Image.open(fpath)
                        st.image(img, caption=fname, use_container_width=True)
                        ts = fname.replace('alert_','').replace('.jpg','')
                        try:
                            dt = datetime.fromtimestamp(int(ts))
                            st.caption(f"📅 {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                        except:
                            pass
                        with open(fpath, 'rb') as f:
                            col.download_button(
                                "⬇️ Download",
                                f.read(),
                                fname,
                                "image/jpeg",
                                use_container_width=True
                            )
    else:
        st.info("Koi screenshot nahi — detection shuru karo!")

# ============================================
# TAB 4 — EMAIL LOGS
# ============================================
with tab4:
    st.subheader("📧 Email Logs")

    if st.session_state.email_log:
        df_email = pd.DataFrame(st.session_state.email_log)
        st.dataframe(
            df_email,
            use_container_width=True,
            column_config={
                'time':   'Time',
                'to':     'Sent To',
                'status': 'Status',
                'file':   'Screenshot'
            }
        )

        sent   = len([e for e in st.session_state.email_log if '✅' in e['status']])
        failed = len([e for e in st.session_state.email_log if '❌' in e['status']])

        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#00cc44">{sent}</div><div class="metric-label">✅ Sent</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ff4444">{failed}</div><div class="metric-label">❌ Failed</div></div>', unsafe_allow_html=True)
    else:
        st.info("Koi email log nahi abhi!")