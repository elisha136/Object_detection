# coco_to_yolov8_seg_full.py
import json
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = Path("/home/manulab/Downloads/coco_dataset")
SPLITS = ["train", "val"]

for split in SPLITS:
    IMG_DIR = BASE_DIR / f"images/{split}"
    ANN_PATH = BASE_DIR / f"annotations/instances_{split}.json"
    LABEL_DIR = BASE_DIR / f"labels/{split}"
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    # === LOAD COCO JSON ===
    with open(ANN_PATH) as f:
        coco = json.load(f)

    # Build image dictionary
    images = {
        img["id"]: {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"]
        }
        for img in coco["images"]
    }
    annotations = coco["annotations"]

    # Group annotations by image_id
    grouped = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if ann["category_id"] >= 7:
            continue  # Only allow classes 0-6
        if img_id not in grouped:
            grouped[img_id] = []
        grouped[img_id].append(ann)

    # === CONVERT TO YOLO FORMAT ===
    for img_id, anns in tqdm(grouped.items(), desc=f"Converting {split}"):
        image_info = images[img_id]
        img_name = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]
        label_file = LABEL_DIR / f"{Path(img_name).stem}.txt"

        with open(label_file, "w") as f:
            for ann in anns:
                cat = ann["category_id"]
                if "segmentation" not in ann or not ann["segmentation"]:
                    continue
                for seg in ann["segmentation"]:
                    if len(seg) < 6:
                        continue  # Needs at least 3 points (x,y)
                    norm_coords = []
                    for i in range(0, len(seg), 2):
                        x = seg[i] / width
                        y = seg[i + 1] / height
                        x = min(max(x, 0), 1)
                        y = min(max(y, 0), 1)
                        norm_coords.extend([f"{x:.6f}", f"{y:.6f}"])
                    f.write(f"{cat} " + " ".join(norm_coords) + "\n")

print("✅ YOLOv8 labels regenerated for both train and val splits!")
