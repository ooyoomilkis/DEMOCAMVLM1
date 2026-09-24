---

# **README.md**

## **RynnBrain Webcam Demo (Object Detection / VLM Inference)**

This project is a minimal demo showing how to run **RynnBrain 2B** (Alibaba DAMO Academy) on live webcam frames using OpenCV and HuggingFace Transformers.  
It captures frames from a webcam, converts them to PIL images, and passes them into a Vision-Language Model (VLM) for inference.

---

## **Features**
- Live webcam capture using **OpenCV**
- Frame conversion from BGR → RGB → PIL
- Integration with **RynnBrain 1.1 / 2B** Vision-Language Model
- Placeholder function `detect_objects()` for running inference on each frame
- Adjustable frame processing interval (default: every 2 seconds)
- Utility function `capture_frame()` for grabbing a single still image

---

## **Requirements**
Install dependencies:

```bash
pip install opencv-python pillow transformers accelerate
```

You must also have a working webcam or RTSP stream.

---

## **Model Setup**
This demo uses the HuggingFace model:

```
Alibaba-DAMO-Academy/RynnBrain-2B
```

The script loads both the model and processor:

```python
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)
```

The first run may take time as weights download.

---

## **Usage**

### **Run the webcam loop**
The main loop captures frames continuously:

```python
cap = cv2.VideoCapture(0)
FPS_INTERVAL = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    detect_objects(pil_image)   # your inference function

    time.sleep(FPS_INTERVAL)

cap.release()
```

Replace `detect_objects()` with your own inference logic using the model + processor.

---

## **Single Frame Capture**
You can also grab one frame at a time:

```python
image_pil, image_cv2 = capture_frame()
```

Useful for debugging or running inference on demand.

---

## **Project Structure**
```
SMTCAMVLM/
│── main.py               # your script
│── README.md             # this file
│── requirements.txt      # optional
│── ...
```

---

## **Notes**
- The repository currently includes a placeholder `detect_objects()` call.  
  You must implement this function to run actual VLM inference.
- If using GPU, ensure PyTorch is installed with CUDA support.
- For RTSP cameras, replace `cv2.VideoCapture(0)` with your stream URL.

---

## **Future Improvements**
- Add real inference pipeline for RynnBrain
- Add bounding box visualization
- Add streaming output window
- Add async frame processing for higher FPS

---