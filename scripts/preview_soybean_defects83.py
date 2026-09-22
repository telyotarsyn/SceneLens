from pathlib import Path
import random
import cv2


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "datasets" / "generated" / "soybean_defects83"
SPLIT = "train"

IMAGE_DIR = DATASET / SPLIT / "images"
LABEL_DIR = DATASET / SPLIT / "labels"

OUTPUT_DIR = ROOT / "preview" / "soybean_defects83"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_IMAGES = 18


CLASS_NAMES = {
    80: "soybean_broken",
    81: "soybean_immature",
    82: "soybean_spotted",
}


def find_label(image_path: Path):
    return LABEL_DIR / f"{image_path.stem}.txt"


def draw_boxes(image_path: Path, label_path: Path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read {image_path}")

    h, w = image.shape[:2]

    if not label_path.exists():
        return image

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            print(
                f"WARNING: unexpected label format "
                f"in {label_path.name}: {line}"
            )
            continue

        class_id = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)

        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))

        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            f"{class_name} [{class_id}]",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return image


def main():
    images = []

    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(IMAGE_DIR.glob(ext))

    if not images:
        raise RuntimeError(
            f"No images found in {IMAGE_DIR}"
        )

    random.seed(42)

    selected = random.sample(
        images,
        min(NUM_IMAGES, len(images))
    )

    print(f"Creating {len(selected)} previews...")

    for image_path in selected:

        label_path = find_label(image_path)

        annotated = draw_boxes(
            image_path,
            label_path,
        )

        output_path = OUTPUT_DIR / image_path.name

        cv2.imwrite(
            str(output_path),
            annotated,
        )

        print(f"Saved: {output_path.name}")

    print()
    print("Preview folder:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()