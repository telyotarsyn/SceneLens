from ultralytics import YOLO

MODEL = "yolo26n.pt"

model = YOLO(MODEL)
path = model.export(
    format="onnx",
    imgsz=640,
    dynamic=True,
    simplify=True,
)
print(f"Exported ONNX model: {path}")
