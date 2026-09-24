# Smart Camera — Object Detection + Spatial Awareness

Early-stage prototype work exploring object detection and spatial reasoning
for a smart camera, branching off RynnBrain (Alibaba DAMO Academy) cookbook
notebooks.

## Status

- `smart_camera_starter.py` — working YOLOv8n object detection + basic
  left/right/near/far spatial estimation on webcam input. This is the
  current primary track, following the lightweight-model-first direction
  suggested in lab meeting (embedded/real-time deployment is not
  realistic with large VLMs).
- `vlm_smart_camera_starter.py` — in progress. Follows the RynnBrain
  notebook pattern (encode image + prompt → parse returned bounding box)
  for open-vocabulary object grounding instead of a fixed class list.
  Detection function still being completed. Serves as a VLM baseline for
  comparison against the lightweight pipeline, not the primary deployment
  target. (NOT YET CREATED TOO BIG OF A WORLD MODEL)

## Next steps

- Integrate a monocular depth model (starting with DepthAnything) with
  YOLO output to replace the box-size proximity proxy with real
  distance estimation
- Finish the VLM grounding call for baseline comparison
- Evaluate smaller VLM options (PaliGemma 2) for the VLM baseline instead
  of RynnBrain, given local GPU constraints
- Test VLM track on Colab/GWU HPC given compute needs

## Research direction

Comparing lightweight object detection + depth estimation against VLM-based
spatial grounding, to evaluate whether lightweight models can match VLM
performance on spatial awareness tasks — relevant for real-time,
embedded deployment (e.g. Socially Assistive Robotics contexts) where
running large VLMs is not feasible.