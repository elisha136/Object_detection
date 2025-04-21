from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
from glob import glob

# Function to visualize results
def show_image_with_boxes(image_path, results):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coords
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{results.names[cls_id]} {conf:.2f}"

        # Draw box + label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display image
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(os.path.basename(image_path))
    plt.show()

# Test function
def test_yolov8_on_images(model_path, image_dir, num_images=10):
    model = YOLO(model_path)
    image_paths = sorted(glob(os.path.join(image_dir, '*.*')))[:num_images]

    if not image_paths:
        print("❌ No images found in:", image_dir)
        return

    for img_path in image_paths:
        print(f"🔍 Inference on: {os.path.basename(img_path)}")
        results = model(img_path)[0]  # First result
        show_image_with_boxes(img_path, results)

# === Example usage ===
test_yolov8_on_images(
    model_path="/home/manulab/projects/computer vision portfolio/portfoliowork.v1i.yolov8/results/yolov8_training_exp_v1/weights/best.pt",
    image_dir="/home/manulab/projects/images",
    num_images=10
)
