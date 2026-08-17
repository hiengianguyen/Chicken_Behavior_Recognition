from pathlib import Path

import cv2
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = Path("models/best.pt")

IMAGE_DIR = Path("datasets/prelabel/images")

LABEL_DIR = Path("datasets/prelabel/labels")

PREVIEW_DIR = Path("datasets/prelabel/preview")

CONFIDENCE = 0.50

IMAGE_SIZE = 640


# =========================================================
# SETUP
# =========================================================

LABEL_DIR.mkdir(parents=True, exist_ok=True)

PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("ChickenCareAI - Auto Pre-Label")
print("=" * 60)

print(f"Model : {MODEL_PATH}")


if not MODEL_PATH.exists():

    raise FileNotFoundError(f"Cannot find model: {MODEL_PATH}")


model = YOLO(str(MODEL_PATH))

print("Model loaded!")


# =========================================================
# FIND IMAGES
# =========================================================

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


images = sorted([p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in extensions])


print(f"Images : {len(images)}")


if not images:

    print(f"No images found in {IMAGE_DIR}")

    raise SystemExit


# =========================================================
# STATISTICS
# =========================================================

total_images = 0

total_detections = 0

no_detection = 0


# =========================================================
# PROCESS
# =========================================================

for index, image_path in enumerate(images, start=1):

    print(f"\n[{index}/{len(images)}] " f"{image_path.name}")

    # -----------------------------------------------------
    # YOLO INFERENCE
    # -----------------------------------------------------

    results = model.predict(
        source=str(image_path), conf=CONFIDENCE, imgsz=IMAGE_SIZE, verbose=False
    )

    result = results[0]

    total_images += 1

    # -----------------------------------------------------
    # IMAGE SIZE
    # -----------------------------------------------------

    image_height, image_width = result.orig_shape

    # -----------------------------------------------------
    # NO DETECTION
    # -----------------------------------------------------

    if result.boxes is None or len(result.boxes) == 0:

        print("  No detection")

        no_detection += 1

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        label_path.touch()

        continue

    # -----------------------------------------------------
    # DETECTIONS
    # -----------------------------------------------------

    label_lines = []

    for box in result.boxes:

        class_id = int(box.cls[0].item())

        confidence = float(box.conf[0].item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # -------------------------------------------------
        # XYXY → YOLO
        # -------------------------------------------------

        center_x = ((x1 + x2) / 2) / image_width

        center_y = ((y1 + y2) / 2) / image_height

        width = (x2 - x1) / image_width

        height = (y2 - y1) / image_height

        label_lines.append(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    # -----------------------------------------------------
    # SAVE LABEL
    # -----------------------------------------------------

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    with open(label_path, "w", encoding="utf-8") as file:

        file.write("\n".join(label_lines))

    detected = len(label_lines)

    total_detections += detected

    print(f"  Detected : {detected}")


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)

print("PRE-LABEL COMPLETED")

print("=" * 60)


print(f"Images processed : {total_images}")


print(f"Total detections : {total_detections}")


print(f"No detection     : {no_detection}")


print(f"Labels saved     : {LABEL_DIR}")


print(f"Preview saved    : {PREVIEW_DIR}")


print("=" * 60)
