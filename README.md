# 🧠 Object Detection and Segmentation Portfolio

This repository contains multiple computer vision projects focused on object detection and segmentation, using various deep learning approaches. Each subproject includes training code, evaluation scripts, and preprocessed datasets in COCO format.

## 📁 Project Structure

- `faster_rcnn/` — Fine-tuned Faster R-CNN for object detection using PyTorch and torchvision.
- `yolov8/` — YOLOv8-based segmentation and detection using Ultralytics' framework.
- `panoptic_seg/` — Panoptic segmentation project (in progress / with current implementation issues).
- `segmentation/` — YOLOv8 segmentation task with cleaner setup and test utilities.
- `.gitignore` — Excludes large folders, checkpoints, virtual environments from version control.

## 📊 Dataset

- All datasets are annotated in **COCO format**.
- Annotations were prepared using [CVAT](https://github.com/opencv/cvat) and/or Roboflow.
- Classes include:
  - Milo, Nescafe, Toothpaste, Toothbrush, Mug, Loreal Men Shower Gel, Nivea Cream.

## 🛠️ Environments

Each project has its own `venv` for better isolation and dependency management. You may use a unified environment later if desired.

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/elisha136/object-detection-segmentation.git
cd object-detection-segmentation

# Go into a project and run
cd yolov8
python3 train.py  # or your test/inference scripts
