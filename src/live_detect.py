import argparse
import time
import cv2
from ultralytics import YOLO

DEFAULT_CLASSES = {
    "person",
    "bottle",
    "cup",
    "chair",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "book",
    "scissors",
}

def parse_args():
    p = argparse.ArgumentParser(description="SceneLens: real-time object detection")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--model", type=str, default="yolo26n.pt", help="Ultralytics model")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--all-classes", action="store_true", help="Show all model classes")
    return p.parse_args()

def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Run: python src/list_cameras.py"
        )

    # Lower capture resolution to reduce latency. Change later if needed.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_t = time.perf_counter()
    smooth_fps = None

    print("SceneLens is running.")
    print("Press Q to quit. Press S to save the current annotated frame.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame.")
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )
        result = results[0]
        annotated = frame.copy()

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                name = model.names[cls_id]

                if not args.all_classes and name not in DEFAULT_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = f"{name} {score:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        now = time.perf_counter()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        smooth_fps = fps if smooth_fps is None else 0.9 * smooth_fps + 0.1 * fps

        cv2.putText(
            annotated,
            f"FPS: {smooth_fps:.1f}",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("SceneLens", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            from pathlib import Path
            out_dir = Path("data/frames")
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = out_dir / f"frame_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(filename), annotated)
            print(f"Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
