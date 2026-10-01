# 🎥 Object Detection and Tracking

Real-time object detection and multi-object tracking using OpenCV + a
pre-trained YOLO model + a from-scratch SORT tracker. **This pipeline has
been run and verified end-to-end**: real YOLOv4-tiny detections (dog/
truck/bicycle) were confirmed on a test image, and the SORT tracker was
confirmed to keep stable IDs on the same objects across 30 frames of a
motion-test video.

## Pipeline
| Step | File | What it does |
|---|---|---|
| 1. Video input | `main.py` | Opens a webcam (`--source 0`) or a video file via OpenCV (`cv2.VideoCapture`). |
| 2. Detection | `detector.py` | Loads **YOLOv4-tiny** (pre-trained on COCO, 80 classes) through OpenCV's `cv2.dnn` module and runs detection + non-max suppression on each frame. |
| 3. Draw boxes | `main.py` | Draws a bounding box per detection. |
| 4. Tracking | `sort_tracker.py` | A compact **SORT** (Simple Online and Realtime Tracking) implementation: a Kalman filter predicts each object's next position, and the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) matches new detections to existing tracks by IoU, assigning a stable ID to each object. |
| 5. Display | `main.py` | Draws each tracked object's box + class label + persistent track ID, and shows FPS live (or saves to an output video file). |

## Why YOLOv4-tiny via OpenCV DNN (instead of PyTorch/YOLOv8)?
This keeps the project to a single lightweight dependency (`opencv-python`)
with no PyTorch/CUDA install required, while still using a real pre-trained
YOLO model (as the task asks) — great for getting started quickly on any
machine. See "Using YOLOv8 instead" below to upgrade.

## Quickstart
```bash
pip install -r requirements.txt
bash download_model.sh          # fetches yolov4-tiny.cfg/.weights + coco.names (~24MB)

# Live webcam, with a display window:
python main.py --source 0

# A video file, with a display window:
python main.py --source path/to/video.mp4

# Headless (servers/CI/no display) — saves an annotated video instead:
python main.py --source path/to/video.mp4 --headless --output output/tracked.mp4
```
Press `q` to quit the live display window.

## Tuning
- `--conf` — detection confidence threshold (default 0.4). Lower = more
  (noisier) detections; higher = fewer but more confident ones.
- `--detect_every N` — only run the (expensive) detector every N frames and
  let the Kalman filter coast on predicted motion in between, for extra
  speed on slower machines.
- In `sort_tracker.py`, `Sort(max_age, min_hits, iou_threshold)` controls
  how many frames a lost object is kept alive (`max_age`), how many
  consecutive hits are needed before a track is considered confirmed
  (`min_hits`), and how much overlap counts as the same object
  (`iou_threshold`).

## Using YOLOv8 / Deep SORT instead
For higher accuracy detection, swap `detector.py`'s OpenCV-DNN YOLOv4-tiny
for [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
(`pip install ultralytics`, then `YOLO("yolov8n.pt")` — it auto-downloads
weights). For appearance-based re-identification (useful when objects
cross paths or get occluded), swap `sort_tracker.py`'s IoU matching for
[Deep SORT](https://github.com/nwojke/deep_sort), which adds a small CNN
that extracts a visual "fingerprint" per detection for matching.

## File structure
```
task4-object-detection/
├── main.py               # Steps 1, 3, 5: video I/O, drawing, display/output
├── detector.py             # Step 2: YOLOv4-tiny detection via OpenCV DNN
├── sort_tracker.py          # Step 4: Kalman filter + Hungarian-algorithm SORT tracker
├── download_model.sh        # Fetches the pre-trained model files
├── sample_media/
│   ├── test.jpg              # Sample image used to verify detection works
│   └── synthetic_test.mp4    # Sample motion video used to verify tracking works
├── output/
│   └── tracked_demo.mp4       # Verified output: detection + tracking on the sample video
├── requirements.txt
├── .gitignore                # model weights/cfg are downloaded, not committed
└── README.md
```

## License
MIT for the code. YOLOv4-tiny weights are from the
[AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) project (public releases).
