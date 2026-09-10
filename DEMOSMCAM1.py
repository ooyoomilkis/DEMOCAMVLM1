import cv2
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# load the RynnBrain Model & Processor
model_id = "Alibaba-DAMO-Academy/RynnBrain-2B"  # Or your chosen variant
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

# initialize Camera
cap = cv2.VideoCapture(0)  # 0 for default webcam or RTSP stream link
FPS_INTERVAL = 2  # Process a frame every 2 seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert OpenCV BGR to RGB PIL Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # running detection on frame?
    detect_objects(pil_image)
    
    time.sleep(FPS_INTERVAL)

cap.release()
