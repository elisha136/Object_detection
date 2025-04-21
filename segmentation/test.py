import cv2
from ultralytics import YOLO
from pathlib import Path

# === CONFIG ===
MODEL_PATH = "/home/manulab/projects/seg/yolov8x_results/train_yolov8x_300ep_nostop/weights/best.pt"
IMAGE_DIR = Path("/home/manulab/projects/images")
SAVE_DIR = Path("/home/manulab/projects/seg/inference_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Load and sort image files
image_files = sorted(IMAGE_DIR.glob("*.[jp][pn]g"))[:10]  # jpg or png, first 10

# Inference loop
for img_path in image_files:
    result = model(img_path, save=False, show=True)[0]  # show in OpenCV window

    # Save result with drawn masks and boxes
    result.save(filename=SAVE_DIR / img_path.name)

print(f"✅ Inference done! Results saved in: {SAVE_DIR}")
