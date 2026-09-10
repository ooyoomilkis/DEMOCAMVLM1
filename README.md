# DEMOCAMVLM1
Rynnbrain


# Smart Camera — Object Detection + Spatial Awareness

Early-stage prototype work exploring object detection and spatial
reasoning for a smart camera, branching off RynnBrain (Alibaba DAMO
Academy) cookbook notebooks.

## Status
- `smart_camera_starter.py` — working YOLOv8n object detection +
  basic left/right/near/far spatial estimation on webcam input.
- `vlm_smart_camera_starter.py` — in progress. Follows the RynnBrain
  notebook pattern (encode image + prompt → parse returned bounding
  box) for open-vocabulary object grounding instead of a fixed class
  list. Detection function still being completed.

## Next steps
- Finish the VLM detection call
- Test on Colab given RynnBrain's compute needs
