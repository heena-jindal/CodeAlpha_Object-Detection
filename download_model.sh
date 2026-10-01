#!/usr/bin/env bash
# Downloads YOLOv4-tiny (pre-trained on COCO) so the project doesn't have to
# ship a 24MB weights file inside the git repo.
set -e
mkdir -p models
cd models

echo "Downloading yolov4-tiny.cfg ..."
curl -sL -o yolov4-tiny.cfg "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"

echo "Downloading yolov4-tiny.weights (~24MB) ..."
curl -sL -o yolov4-tiny.weights "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"

echo "Downloading coco.names ..."
curl -sL -o coco.names "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"

echo "Done. Model files saved in models/"
