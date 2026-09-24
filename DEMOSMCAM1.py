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


# Working with Depth-Anything:
import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy


# baseline world model (find its strengths, weaknesses, this is what my capable)
# preexisting camera world model, why would I want to create my own protoyep, etc etc
# comparison across different systems (baseline spatial awareness) (plug that into diff algorithms)
# train vlm giving spatial awareness understanding -> use that for robotic model training (current model fine tuned to detect patterns of action)
# 