
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog


# -----------------------------
# Model Files (must be in same folder)
# -----------------------------
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"

# Load models
try:
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
except cv2.error as e:
    print("Error loading model files. Make sure the model files are in the same directory as the script.")
    print(e)
    exit()

# Labels and Constants
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60+)']
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# -----------------------------
# Setup CustomTkinter UI
# -----------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("👤 Gender & Age Detector")
root.geometry("800x800") # Increased height for stats
root.minsize(450, 600)

# Header
header = ctk.CTkLabel(root, text="👤 Gender & Age Detector", font=("Segoe UI", 28, "bold"))
header.pack(pady=10)

# Image/Video display panel
panel = ctk.CTkLabel(root, text="Upload an image or start webcam", font=("Segoe UI", 18))
panel.pack(expand=True, fill="both", padx=20, pady=10)

# --- NEW: Live Statistics Display ---
stats_frame = ctk.CTkFrame(root, corner_radius=12)
stats_frame.pack(pady=(5, 10), padx=20, fill="x")
stats_label = ctk.CTkLabel(stats_frame, text="Statistics: N/A", font=("Segoe UI", 14))
stats_label.pack(padx=10, pady=10)
# --- END NEW ---

# Confidence Slider
slider_frame = ctk.CTkFrame(root, corner_radius=12)
slider_frame.pack(pady=10, padx=20, fill="x")
slider_frame.columnconfigure(1, weight=1)

slider_title_label = ctk.CTkLabel(slider_frame, text="Confidence Threshold:", font=("Segoe UI", 14))
slider_title_label.grid(row=0, column=0, padx=(15, 10), pady=10)

confidence_threshold = 0.7
original_image = None

def slider_event(value):
    global confidence_threshold
    confidence_threshold = value
    slider_value_label.configure(text=f"{int(value * 100)}%")
    if original_image is not None and not running:
        reprocess_image()

slider = ctk.CTkSlider(slider_frame, from_=0.3, to=0.95, number_of_steps=65, command=slider_event)
slider.set(confidence_threshold)
slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

slider_value_label = ctk.CTkLabel(slider_frame, text=f"{int(confidence_threshold * 100)}%", font=("Segoe UI", 14), width=40)
slider_value_label.grid(row=0, column=2, padx=(10, 15), pady=10)

# Button frame
btn_frame = ctk.CTkFrame(root, corner_radius=12)
btn_frame.pack(pady=10, padx=20, fill="x")
btn_frame.columnconfigure(0, weight=1)

cap = None
running = False

# -----------------------------
# Core Functions
# -----------------------------
def detect_and_display(frame):
    # Initialize counters for statistics
    face_count = 0
    male_count = 0
    female_count = 0
    
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            face_count += 1 # Increment total face count
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            if x2 > x1 and y2 > y1:
                face = frame[y1:y2, x1:x2]
                if face.size == 0: continue

                blob2 = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                gender_net.setInput(blob2)
                gender_preds = gender_net.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]

                # Increment gender-specific counts
                if gender == "Male":
                    male_count += 1
                else:
                    female_count += 1

                age_net.setInput(blob2)
                age_preds = age_net.forward()
                age = AGE_BUCKETS[age_preds[0].argmax()]

                label = f"{gender}, {age}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Return the processed frame AND the collected statistics
    return frame, face_count, male_count, female_count

def reprocess_image():
    if original_image is None: return
    
    # Get stats back from the detection function
    processed_image, f_count, m_count, fem_count = detect_and_display(original_image.copy())
    
    # Update the statistics label
    stats_text = f"Faces Detected: {f_count}  ({m_count} Male, {fem_count} Female)"
    stats_label.configure(text=stats_text)
    
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(processed_image)
    img_pil.thumbnail((panel.winfo_width(), panel.winfo_height()), Image.Resampling.LANCZOS)
    
    photo_image = ImageTk.PhotoImage(img_pil)
    panel.configure(image=photo_image, text="")
    panel.image = photo_image

def upload_image():
    global original_image
    stop_webcam()
    file_path = filedialog.askopenfilename()
    if not file_path: return
    
    original_image = cv2.imread(file_path)
    reprocess_image()
    
    update_ui_state('image_view')

def update_webcam():
    global cap, running
    if running:
        ret, frame = cap.read()
        if ret:
            # Get stats back from the detection function
            processed_frame, f_count, m_count, fem_count = detect_and_display(frame.copy())
            
            # Update the statistics label in real-time
            stats_text = f"Faces Detected: {f_count}  ({m_count} Male, {fem_count} Female)"
            stats_label.configure(text=stats_text)

            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(processed_frame)
            img_pil.thumbnail((panel.winfo_width(), panel.winfo_height()), Image.Resampling.LANCZOS)
            
            photo_image = ImageTk.PhotoImage(img_pil)
            panel.configure(image=photo_image, text="")
            panel.image = photo_image
            
        root.after(15, update_webcam)

def start_webcam():
    global cap, running, original_image
    if not running:
        original_image = None
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            stats_label.configure(text="Status: Error! Webcam not found.")
            return
        running = True
        update_webcam()
        update_ui_state('webcam_running')

def stop_webcam():
    global cap, running
    running = False
    if cap:
        cap.release()
        cap = None

def go_back():
    global original_image
    stop_webcam()
    original_image = None
    panel.configure(image="", text="Upload an image or start webcam")
    # Reset stats label
    stats_label.configure(text="Statistics: N/A")
    update_ui_state('idle')

def update_ui_state(state: str):
    for widget in btn_frame.winfo_children():
        widget.destroy()

    btn_height = 40
    btn_font = ("Segoe UI", 14)

    if state == 'idle':
        ctk.CTkButton(btn_frame, text="🖼️  Upload Image", command=upload_image, height=btn_height, font=btn_font).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        ctk.CTkButton(btn_frame, text="📹  Start Webcam", command=start_webcam, height=btn_height, font=btn_font).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(btn_frame, text="🚪  Exit", command=root.quit, fg_color="#c0392b", hover_color="#a93226", height=btn_height, font=btn_font).grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")
    elif state == 'webcam_running':
        ctk.CTkButton(btn_frame, text="⏹️  Stop Webcam", command=go_back, fg_color="#e67e22", hover_color="#d35400", height=btn_height, font=btn_font).grid(row=0, column=0, padx=10, pady=10, sticky="ew")
    elif state == 'image_view':
        ctk.CTkButton(btn_frame, text="⬅️  Back to Menu", command=go_back, fg_color="#7f8c8d", hover_color="#95a5a6", height=btn_height, font=btn_font).grid(row=0, column=0, padx=10, pady=10, sticky="ew")

# -----------------------------
# Initialize and Run UI
# -----------------------------
update_ui_state('idle')
root.mainloop()

if cap:
    cap.release()