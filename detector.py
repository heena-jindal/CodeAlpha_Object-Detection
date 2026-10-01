"""
detector.py
-----------
Step 2: Load a pre-trained YOLO model (YOLOv4-tiny) via OpenCV's DNN module
and run object detection on a single frame.

YOLOv4-tiny is used because it runs in real time on CPU (no GPU/PyTorch
required) via cv2.dnn, which keeps this project dependency-light
(just opencv-python). Swap in yolov4.weights/.cfg (full-size YOLOv4) or a
custom-trained model for higher accuracy at the cost of speed.
"""
import os
import cv2
import numpy as np


class YOLODetector:
    def __init__(self, cfg_path, weights_path, names_path,
                 conf_threshold=0.4, nms_threshold=0.4, input_size=416):
        if not (os.path.exists(cfg_path) and os.path.exists(weights_path)):
            raise FileNotFoundError(
                "Model files not found. Run `bash download_model.sh` first "
                "to fetch yolov4-tiny.cfg / .weights / coco.names."
            )

        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_layers = self.net.getUnconnectedOutLayersNames()

        with open(names_path) as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

    def detect(self, frame):
        """Returns a list of [x, y, w, h, confidence, class_id, class_name]."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > self.conf_threshold:
                    cx, cy, bw, bh = det[0] * w, det[1] * h, det[2] * w, det[3] * h
                    x, y = int(cx - bw / 2), int(cy - bh / 2)
                    boxes.append([x, y, int(bw), int(bh)])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        detections = []
        if len(idxs) > 0:
            for i in np.array(idxs).flatten():
                x, y, bw, bh = boxes[i]
                detections.append([x, y, bw, bh, confidences[i], class_ids[i],
                                    self.classes[class_ids[i]]])
        return detections
