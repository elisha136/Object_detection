import json
import random
from pathlib import Path
from shutil import move

# CONFIG
BASE_DIR = Path("/home/manulab/Downloads/coco_dataset")
IMG_DIR = BASE_DIR / "images"
ANN_INPUT_PATH = BASE_DIR / "annotations/instances_train.json"
TRAIN_DIR = IMG_DIR / "train"
VAL_DIR = IMG_DIR / "val"
TRAIN_ANN_PATH = BASE_DIR / "annotations/instances_train.json"
VAL_ANN_PATH = BASE_DIR / "annotations/instances_val.json"

VAL_SPLIT = 0.1  # 10% for validation

# Load original annotations
with open(ANN_INPUT_PATH) as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# Shuffle and split image list
random.seed(42)
random.shuffle(images)
val_count = int(len(images) * VAL_SPLIT)
val_images = images[:val_count]
train_images = images[val_count:]

# Build image ID sets
val_ids = set(img["id"] for img in val_images)
train_ids = set(img["id"] for img in train_images)

# Split annotations
train_anns = [ann for ann in annotations if ann["image_id"] in train_ids]
val_anns = [ann for ann in annotations if ann["image_id"] in val_ids]

# Save updated annotation files
with open(TRAIN_ANN_PATH, "w") as f:
    json.dump({"images": train_images, "annotations": train_anns, "categories": categories}, f)

with open(VAL_ANN_PATH, "w") as f:
    json.dump({"images": val_images, "annotations": val_anns, "categories": categories}, f)

# Ensure val folder exists
VAL_DIR.mkdir(parents=True, exist_ok=True)

# Move validation images
moved_count = 0
for img in val_images:
    img_file = Path(img["file_name"]).name  # Just the filename (drop subfolders if any)
    src = TRAIN_DIR / img_file
    dst = VAL_DIR / img_file
    if src.exists():
        move(str(src), str(dst))
        moved_count += 1
    else:
        print(f"⚠️ Image not found, skipping: {src}")

print("✅ Dataset split completed!")
print(f"→ Training images: {len(train_images)}")
print(f"→ Validation images: {len(val_images)}")
print(f"→ Images moved to val/: {moved_count}")
