from pathlib import Path
import shutil
import yaml

# =========================================================
# CONFIG
# =========================================================

SOURCE_IMAGES = Path("datasets/prelabel/images")

SOURCE_LABELS = Path("datasets/prelabel/labels")

OUTPUT_DIR = Path("datasets/cvat_upload")

IMAGE_OUTPUT = OUTPUT_DIR / "images" / "train"

LABEL_OUTPUT = OUTPUT_DIR / "labels" / "train"


# =========================================================
# CREATE DIRECTORIES
# =========================================================

IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)

LABEL_OUTPUT.mkdir(parents=True, exist_ok=True)


# =========================================================
# COPY IMAGES + LABELS
# =========================================================

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


images = [p for p in SOURCE_IMAGES.iterdir() if p.suffix.lower() in image_extensions]


copied_images = 0
copied_labels = 0


for image_path in images:

    label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    shutil.copy2(image_path, IMAGE_OUTPUT / image_path.name)

    copied_images += 1

    # -----------------------------------------------------
    # LABEL
    # -----------------------------------------------------

    if label_path.exists():

        shutil.copy2(label_path, LABEL_OUTPUT / label_path.name)

        copied_labels += 1

    else:

        print(f"WARNING: Missing label " f"for {image_path.name}")


# =========================================================
# CREATE data.yaml
# =========================================================

data = {"path": "./", "train": "images/train", "names": {0: "chicken"}}


yaml_path = OUTPUT_DIR / "data.yaml"


with open(yaml_path, "w", encoding="utf-8") as file:

    yaml.dump(data, file, sort_keys=False)


# =========================================================
# SUMMARY
# =========================================================

print("=" * 60)

print("CVAT DATASET PREPARED")

print("=" * 60)

print(f"Images : {copied_images}")

print(f"Labels : {copied_labels}")

print(f"Output : {OUTPUT_DIR}")

print(f"YAML   : {yaml_path}")

print("=" * 60)
