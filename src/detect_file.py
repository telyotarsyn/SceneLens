import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="Path to an image or video file")
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    args = p.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    model = YOLO(args.model)
    model.predict(
        source=str(source),
        conf=args.conf,
        save=True,
        show=False,
    )
    print("Done. Check the newest folder inside runs/detect/.")

if __name__ == "__main__":
    main()
