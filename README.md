# SceneLens

**Real-time everyday object detection from an iPhone camera**

SceneLens is a portfolio-oriented computer-vision project for real-time detection of everyday objects.  
The first version uses a pretrained YOLO26n model and an iPhone connected to Windows as a webcam.
Later versions can add a custom dataset (e.g. glasses, headphones, pen), fine-tuning, ONNX inference,
benchmarking, and a browser-based mobile UI.

## 1. Requirements

- Windows 10/11
- Python 3.11 recommended
- iPhone
- Camo Camera on iPhone + Camo Studio on Windows, or any other method that exposes the iPhone as a Windows webcam
- Optional: NVIDIA GPU

## 2. Create the environment

Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Connect the iPhone camera

1. Install Camo Camera on the iPhone.
2. Install and open Camo Studio on Windows.
3. Connect/pair the phone and confirm that you see the iPhone video feed in Camo Studio.
4. Keep Camo running.

Then discover the OpenCV camera index:

```powershell
python src/list_cameras.py
```

## 4. Run live detection

Example:

```powershell
python src/live_detect.py --camera 0
```

If Camo is index 1:

```powershell
python src/live_detect.py --camera 1
```

Controls:
- `Q` — quit
- `S` — save the current annotated frame

The model file is downloaded automatically on first launch.

## 5. Test on an iPhone video file first

Copy an iPhone video to your PC, for example to:

`data/test_video.mov`

Then:

```powershell
python src/detect_file.py data/test_video.mov
```

Results are saved under `runs/detect/`.

## 6. Collect your own future dataset

```powershell
python src/collect_frames.py --camera 0
```

Controls:
- `SPACE` — save one frame
- `A` — toggle auto-save
- `Q` — quit

Raw images go to `data/raw/`.

Do not collect company-confidential imagery or proprietary company data for the public portfolio repository.

## 7. Export to ONNX

```powershell
python src/export_onnx.py
```

This creates an ONNX model. A strong portfolio extension is to benchmark:
- PyTorch inference latency/FPS
- ONNX Runtime latency/FPS
- CPU vs GPU if available
- 320 / 480 / 640 input resolutions

## Suggested portfolio roadmap

### Milestone 1 — Baseline
- Live iPhone camera
- Pretrained model
- Bounding boxes, labels, confidence, FPS

### Milestone 2 — Custom classes
Add classes that COCO does not have:
- glasses
- headphones
- pen

Collect your own images in several rooms, lighting conditions, distances, orientations, and backgrounds.
Create a train/validation/test split by recording session, not random neighboring frames.

### Milestone 3 — ML experiment
- Fine-tune a small detector
- Report mAP50 and mAP50-95
- Per-class precision/recall
- Failure-case gallery
- Ablation: dataset size or augmentation
- Baseline vs fine-tuned model

### Milestone 4 — Deployment
- Export to ONNX
- Benchmark latency and FPS
- Add a simple web/API layer
- Add tests and Docker
- Record a 30–60 second demo video

## Licensing note

Ultralytics currently provides YOLO under AGPL-3.0 for open-source use and offers separate commercial/enterprise licensing.
For a public portfolio repository, check and follow the current license terms.
If a company later wants to use the project internally or in a closed-source product, review the license before integration
or replace the detector with a model/framework whose license matches the company's use case.
