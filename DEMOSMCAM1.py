import cv2
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq



# loading vlm nb
MODEL_PATH = "Alibaba-DAMO-Academy/RynnBrain1.1-2B"
model_id = "Alibaba-DAMO-Academy/RynnBrain-2B" # same thing as above
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

'''
print("Loading VLM... (first run downloads weights, can take a while)")
model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
'''
# would need reading over


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

def capture_frame():
    """Grab a single frame from the webcam and return it as a PIL Image."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read from webcam.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb), frame

