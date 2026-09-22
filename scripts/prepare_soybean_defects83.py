from pathlib import Path
import shutil
import yaml

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "datasets" / "source" / "soybean_roboflow"
DST = ROOT / "datasets" / "generated" / "soybean_defects83"

MODEL_PATH = ROOT / "yolo26n.pt"

SPLITS = ["train", "valid", "test"]


# Original Roboflow classes:
# 0 = Broken
# 1 = Immature
# 2 = Spotted
#
# New SceneLens classes:
# 80 = soybean_broken
# 81 = soybean_immature
# 82 = soybean_spotted

CLASS_MAPPING = {
    0: 80,
    1: 81,
    2: 82,
}


def segmentation_to_bbox(line: str):
    """
    Convert YOLO segmentation annotation:

        class x1 y1 x2 y2 x3 y3 ...

    into YOLO detection annotation:

        class x_center y_center width height

    The original soybean defect classes are remapped to:

        80 = soybean_broken
        81 = soybean_immature
        82 = soybean_spotted
    """

    parts = line.strip().split()

    if not parts:
        return None

    source_class = int(parts[0])

    if source_class not in CLASS_MAPPING:
        raise ValueError(
            f"Unexpected source class ID: {source_class}\n"
            f"Annotation: {line}"
        )

    target_class = CLASS_MAPPING[source_class]

    coords = [float(v) for v in parts[1:]]

    # In case the source annotation already contains
    # a normal YOLO detection bounding box.
    if len(coords) == 4:
        x_center, y_center, width, height = coords

        return (
            f"{target_class} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    # Segmentation polygon must contain at least 3 points.
    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError(
            f"Invalid segmentation annotation:\n{line}"
        )

    xs = coords[0::2]
    ys = coords[1::2]

    x_min = max(0.0, min(xs))
    x_max = min(1.0, max(xs))

    y_min = max(0.0, min(ys))
    y_max = min(1.0, max(ys))

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return None

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    return (
        f"{target_class} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )


def find_image(image_dir: Path, stem: str):
    for extension in [".jpg", ".jpeg", ".png"]:
        candidate = image_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    return None


def convert_split(split: str):
    src_images = SRC / split / "images"
    src_labels = SRC / split / "labels"

    dst_images = DST / split / "images"
    dst_labels = DST / split / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    label_files = list(src_labels.glob("*.txt"))

    images_written = 0
    objects_written = 0
    skipped_objects = 0
    missing_images = 0

    for label_path in label_files:

        image_path = find_image(
            src_images,
            label_path.stem
        )

        if image_path is None:
            print(
                f"[WARNING] No image found for: "
                f"{label_path.name}"
            )
            missing_images += 1
            continue

        output_lines = []

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():

            if not line.strip():
                continue

            converted = segmentation_to_bbox(line)

            if converted is None:
                skipped_objects += 1
                continue

            output_lines.append(converted)
            objects_written += 1

        shutil.copy2(
            image_path,
            dst_images / image_path.name
        )

        output_label = dst_labels / label_path.name

        output_label.write_text(
            "\n".join(output_lines),
            encoding="utf-8"
        )

        images_written += 1

    print(
        f"{split:>5}: "
        f"{images_written} images | "
        f"{objects_written} defect regions | "
        f"{skipped_objects} skipped | "
        f"{missing_images} missing images"
    )


def create_yaml():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}\n"
            "Place yolo26n.pt in the SceneLens root."
        )

    model = YOLO(str(MODEL_PATH))

    coco_names = model.names

    # Preserve all original 80 COCO class names.
    names = {
        i: coco_names[i]
        for i in range(80)
    }

    # Add our three soybean-specific classes.
    names[80] = "soybean_broken"
    names[81] = "soybean_immature"
    names[82] = "soybean_spotted"

    data = {
        "path": "datasets/generated/soybean_defects83",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
    }

    yaml_path = DST / "data.yaml"

    with yaml_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"\nCreated: {yaml_path}")


def main():

    print("Creating soybean defect detection dataset")
    print(f"Source:      {SRC}")
    print(f"Destination: {DST}\n")

    if not SRC.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{SRC}"
        )

    for split in SPLITS:
        convert_split(split)

    create_yaml()

    print("\nDone.")
    print()
    print("Class mapping:")
    print("  Broken   -> 80 = soybean_broken")
    print("  Immature -> 81 = soybean_immature")
    print("  Spotted  -> 82 = soybean_spotted")


if __name__ == "__main__":
    main()