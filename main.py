"""
main.py
-------
Steps 1, 3, 5: Set up real-time video input (webcam or file), run YOLO
detection on each frame, apply SORT tracking, and display the output with
labels and tracking IDs.

Usage:
    python main.py --source 0                     # webcam, live display
    python main.py --source video.mp4              # video file, live display
    python main.py --source video.mp4 --headless --output output/tracked.mp4
                                                     # no display (e.g. servers/CI),
                                                     # writes an annotated video file instead
"""
import argparse
import os
import time

import cv2
import numpy as np

from detector import YOLODetector
from sort_tracker import Sort

MODEL_DIR = "models"


def color_for_id(track_id):
    """Deterministic distinct color per track ID."""
    rng = np.random.RandomState(track_id)
    return tuple(int(c) for c in rng.randint(60, 255, size=3))


def draw_tracks(frame, tracks):
    for x, y, w, h, track_id, class_name in tracks:
        x, y, w, h = int(x), int(y), int(w), int(h)
        color = color_for_id(track_id)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"ID {track_id} | {class_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def main(source, headless, output_path, conf_threshold, detect_every):
    detector = YOLODetector(
        cfg_path=os.path.join(MODEL_DIR, "yolov4-tiny.cfg"),
        weights_path=os.path.join(MODEL_DIR, "yolov4-tiny.weights"),
        names_path=os.path.join(MODEL_DIR, "coco.names"),
        conf_threshold=conf_threshold,
    )
    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)

    # source "0" (a string) means webcam index 0
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    last_detections = []
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run the (relatively expensive) detector every N frames and let the
        # Kalman-filter tracker coast on predicted motion in between, for speed.
        if frame_idx % detect_every == 0:
            last_detections = detector.detect(frame)
        tracks = tracker.update(last_detections)

        draw_tracks(frame, tracks)

        now = time.time()
        fps_display = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps_display:.1f}  Objects: {len(tracks)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if writer:
            writer.write(frame)
        if not headless:
            cv2.imshow("Object Detection & Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to '{output_path}'")
    if not headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0",
                         help="Webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--headless", action="store_true",
                         help="Don't open a display window (needed on servers/CI); "
                              "use with --output to save the result instead")
    parser.add_argument("--output", type=str, default=None,
                         help="Optional path to save the annotated video, e.g. output/tracked.mp4")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--detect_every", type=int, default=1,
                         help="Run the detector every N frames (>1 speeds things up)")
    args = parser.parse_args()
    main(args.source, args.headless, args.output, args.conf, args.detect_every)
