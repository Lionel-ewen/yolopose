"""Realtime YOLO pose detection from a webcam.

Controls:
  q / Esc  Quit
  s        Save the current annotated frame to outputs/
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from pose_analyzer import (
    COCO_KEYPOINT_NAMES,
    PoseAnalyzer,
    draw_analysis_overlay,
    extract_primary_keypoints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO pose detection on a webcam.")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model path/name.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--save-dir", default="outputs", help="Directory for screenshots.")
    parser.add_argument("--window", default="YOLO Pose Webcam", help="Preview window title.")
    parser.add_argument("--min-kpt-conf", type=float, default=0.25, help="Minimum keypoint confidence.")
    parser.add_argument("--log-csv", action="store_true", help="Save keypoints and action metrics to CSV.")
    parser.add_argument("--csv-path", default=None, help="Optional CSV output path.")
    return parser.parse_args()


def draw_hud(frame, fps: float, people_count: int) -> None:
    text = f"FPS: {fps:5.1f} | People: {people_count} | q: quit | s: save"
    cv2.rectangle(frame, (10, 10), (590, 46), (20, 20, 20), thickness=-1)
    cv2.putText(
        frame,
        text,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 255, 120),
        2,
        cv2.LINE_AA,
    )


def csv_header() -> list[str]:
    header = [
        "frame",
        "timestamp",
        "people_count",
        "primary_person_index",
        "actions",
        "squat_count",
        "squat_phase",
        "left_knee",
        "right_knee",
        "left_elbow",
        "right_elbow",
        "torso_lean",
    ]
    for name in COCO_KEYPOINT_NAMES:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])
    return header


def write_csv_row(writer, frame_index: int, people_count: int, keypoints, analysis) -> None:
    row = {
        "frame": frame_index,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "people_count": people_count,
        "primary_person_index": analysis.primary_person_index,
        "actions": ";".join(analysis.actions),
        "squat_count": analysis.squat_count,
        "squat_phase": analysis.squat_phase,
    }
    for angle_name in ("left_knee", "right_knee", "left_elbow", "right_elbow", "torso_lean"):
        value = analysis.angles.get(angle_name)
        row[angle_name] = "" if value is None else round(value, 2)

    for index, name in enumerate(COCO_KEYPOINT_NAMES):
        if index < len(keypoints):
            row[f"{name}_x"] = round(keypoints[index].x, 2)
            row[f"{name}_y"] = round(keypoints[index].y, 2)
            row[f"{name}_conf"] = round(keypoints[index].conf, 4)
        else:
            row[f"{name}_x"] = ""
            row[f"{name}_y"] = ""
            row[f"{name}_conf"] = ""

    writer.writerow(row)


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    analyzer = PoseAnalyzer(min_conf=args.min_kpt_conf)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}.")
        return 1

    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_path = Path(args.csv_path) if args.csv_path else save_dir / f"pose_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_header())
        csv_writer.writeheader()
        print(f"Logging keypoints and metrics to: {csv_path}")

    prev_time = time.perf_counter()
    fps = 0.0
    frame_index = 0

    print("Webcam started. Press q/Esc to quit, s to save a screenshot.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            frame_index += 1

            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            result = results[0]
            annotated = result.plot()

            now = time.perf_counter()
            elapsed = now - prev_time
            prev_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else (0.9 * fps + 0.1 * instant_fps)

            keypoints, primary_index, people_count = extract_primary_keypoints(result)
            analysis = analyzer.analyze(keypoints, primary_index)
            draw_analysis_overlay(annotated, analysis)

            if csv_writer is not None:
                write_csv_row(csv_writer, frame_index, people_count, keypoints, analysis)

            draw_hud(annotated, fps, people_count)
            cv2.imshow(args.window, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = save_dir / f"pose_{timestamp}.jpg"
                cv2.imwrite(str(output_path), annotated)
                print(f"Saved screenshot: {output_path}")
    finally:
        cap.release()
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
