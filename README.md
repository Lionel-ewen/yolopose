# YOLO Pose Webcam Demo

A small Python demo that runs YOLO pose detection on a webcam, draws the skeleton, reads COCO keypoints, analyzes simple actions, shows FPS, and saves screenshots.

## Install

Recommended Conda setup:

```powershell
conda create -n yolopose python=3.10 -y
conda activate yolopose
pip install -r requirements.txt
```

If `conda activate` does not work in PowerShell, initialize Conda once and reopen the terminal:

```powershell
conda init powershell
```

Alternative venv setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python webcam_pose.py
```

Useful options:

```powershell
python webcam_pose.py --camera 0 --model yolov8n-pose.pt --conf 0.35 --imgsz 640
```

Controls:

- `q` or `Esc`: quit
- `s`: save the current annotated frame to `outputs/`

## Action Analysis

The webcam overlay now shows:

- Detected action labels, such as raised arms, standing, squat down, and torso leaning
- Basic joint angles for knees and elbows
- A simple squat counter based on knee-angle transitions
- The largest detected person as the primary analyzed subject

Export frame-by-frame keypoints and metrics to CSV:

```powershell
python webcam_pose.py --log-csv
```

Custom CSV path:

```powershell
python webcam_pose.py --log-csv --csv-path outputs\my_pose_log.csv
```

The CSV includes all 17 COCO keypoints as `x`, `y`, and `conf` columns, plus action labels, squat count, squat phase, elbow angles, knee angles, and torso lean.
