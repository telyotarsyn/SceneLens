import argparse
import time
from pathlib import Path
import cv2

def main():
    p = argparse.ArgumentParser(description="Collect frames for a future custom dataset")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--every", type=int, default=10, help="Save every Nth frame when auto-save is on")
    args = p.parse_args()

    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    auto = False
    n = 0
    print("SPACE = save one frame | A = toggle auto-save | Q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        n += 1
        preview = frame.copy()
        status = "AUTO ON" if auto else "AUTO OFF"
        cv2.putText(preview, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow("SceneLens dataset collector", preview)

        should_save = auto and n % max(args.every, 1) == 0
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("a"):
            auto = not auto
        if key == 32:  # space
            should_save = True

        if should_save:
            path = out / f"img_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
