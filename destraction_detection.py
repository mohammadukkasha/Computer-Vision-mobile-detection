import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ============================================
# MODEL LOAD
# ============================================
model = YOLO('yolo11n.pt')
model.overrides['verbose'] = False  # logs band karo

# Warmup — pehli inference slow hoti hai
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
model(dummy, verbose=False)

PERSON = 0
MOBILE = 67

# ============================================
# HELPER FUNCTIONS
# ============================================
def is_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x2 > x1 and y2 > y1

def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # thinner font = faster
    return frame

# ============================================
# MAIN DETECTION
# ============================================
def detect_distraction(image, confidence_threshold):
    if image is None:
        return None, "❌ Koi image nahi mili"

    # PIL → numpy → resize (small = fast)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (416, 416))  # 640 ki jagah 416 — faster

    results = model(
        frame,
        conf=confidence_threshold,
        iou=0.4,
        imgsz=416,        # small size = fast
        half=False,       # CPU pe False rakhna
        verbose=False     # logs off
    )[0]

    persons, mobiles = [], []

    for box in results.boxes:
        cls  = int(box.cls)
        bbox = box.xyxy[0].tolist()
        conf = float(box.conf)
        if cls == PERSON: persons.append({'box': bbox, 'conf': conf})
        elif cls == MOBILE: mobiles.append({'box': bbox, 'conf': conf})

    for p in persons:
        draw_box(frame, p['box'], f"P:{p['conf']:.2f}", (0, 255, 0))
    for m in mobiles:
        draw_box(frame, m['box'], f"M:{m['conf']:.2f}", (255, 165, 0))

    distracted = False
    for person in persons:
        for mobile in mobiles:
            if is_overlap(mobile['box'], person['box']):
                distracted = True
                draw_box(frame, person['box'], "DISTRACTED!", (0, 0, 255))

    if distracted:
        status_text = "🚨 DISTRACTED!"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 200), -1)
        cv2.putText(frame, "DISTRACTED!", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    else:
        status_text = "✅ Normal"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 150, 0), -1)
        cv2.putText(frame, "Normal", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    stats = f"""
📊 Stats:
👤 Persons : {len(persons)}
📱 Mobiles : {len(mobiles)}
Status     : {status_text}
    """

    output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output), stats

# ============================================
# WEBCAM
# ============================================
def detect_from_webcam(frame, confidence_threshold):
    if frame is None:
        return None
    result_img, _ = detect_distraction(frame, confidence_threshold)
    return result_img

# ============================================
# GRADIO UI
# ============================================
with gr.Blocks(
    title="Distraction Detection",
    theme=gr.themes.Soft(primary_hue="red"),
    css="footer { display: none !important; }"
) as app:

    gr.Markdown("# 🚨 Distraction Detection System")

    with gr.Tabs():

        with gr.Tab("📷 Image Upload"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Image", type="pil", height=400)
                    conf_slider = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                            label="Confidence")
                    detect_btn  = gr.Button("🔍 Detect", variant="primary")
                with gr.Column():
                    output_image = gr.Image(label="Result", height=400)
                    output_stats = gr.Textbox(label="Stats", lines=6)

            detect_btn.click(
                fn=detect_distraction,
                inputs=[input_image, conf_slider],
                outputs=[output_image, output_stats]
            )

        with gr.Tab("📹 Live Webcam"):
            with gr.Row():
                with gr.Column():
                    webcam_conf  = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                             label="Confidence")
                    webcam_input = gr.Image(sources=["webcam"], streaming=True,
                                            label="Webcam", height=380)
                with gr.Column():
                    webcam_output = gr.Image(label="Output", height=380,
                                             streaming=True)

            webcam_input.stream(
                fn=detect_from_webcam,
                inputs=[webcam_input, webcam_conf],
                outputs=webcam_output,
                time_limit=30,
                stream_every=0.1   # 100ms = ~10 FPS — smooth aur fast
            )

if __name__ == "__main__":
    app.launch(share=True, server_port=7860, show_error=True)