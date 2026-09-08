"""Realtime YOLO pose detection from a webcam.

Controls:
  q / Esc  Quit
  s        Save the current annotated frame to outputs/

Performance notes (Raspberry Pi 5):
  * Use the NCNN export for realtime work:
        yolo export model=yolov8n-pose.pt format=ncnn imgsz=320
        python webcam_pose.py --model yolov8n-pose_ncnn_model --width 640 --height 480 --mjpeg
    The input size of an NCNN model is fixed at export time; --imgsz is ignored by the
    backend, so export with the imgsz you actually want to run.
  * --no-display measures the pipeline alone (no plot / no imshow), which is the upper
    bound of what the model can sustain.
  * --display-every N draws and shows only every N-th frame; inference still runs on
    every frame. This is the cheapest way to buy back FPS on a desktop session.
  * The summary printed at exit reports mean / median / P95. Report all three: on this
    project the mean has cleared the 30 FPS target while P95 has not.
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
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model path/name (.pt or *_ncnn_model).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (ignored by NCNN).")
    parser.add_argument("--save-dir", default="outputs", help="Directory for screenshots.")
    parser.add_argument("--window", default="YOLO Pose Webcam", help="Preview window title.")
    parser.add_argument("--min-kpt-conf", type=float, default=0.25, help="Minimum keypoint confidence.")
    parser.add_argument("--log-csv", action="store_true", help="Save keypoints and action metrics to CSV.")
    parser.add_argument("--csv-path", default=None, help="Optional CSV output path.")

    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
    parser.add_argument("--mjpeg", action="store_true", help="Request MJPEG from the camera (cheaper decode).")
    parser.add_argument("--buffersize", type=int, default=1, help="Capture buffer size; 1 avoids latency buildup.")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up frames excluded from statistics.")
    parser.add_argument("--display-every", type=int, default=1, help="Draw+show only every N-th frame.")
    parser.add_argument("--no-display", action="store_true", help="Never draw or show (measure pipeline only).")
    parser.add_argument("--no-plot", action="store_true", help="Skip result.plot(); keep the analyzer overlay only.")
    return parser.parse_args()


def draw_hud(frame, fps: float, people_count: int, frame_ms: float) -> None:
    text = f"FPS: {fps:5.1f} | {frame_ms:5.1f} ms | People: {people_count} | q: quit | s: save"
    cv2.rectangle(frame, (10, 10), (610, 46), (20, 20, 20), thickness=-1)
    cv2.putText(
        frame,
        text,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
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


def summarize_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": sum(ordered) / len(ordered),
        "median": ordered[len(ordered) // 2],
        "p95": ordered[index],
    }


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        return cap
    if args.mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffersize)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Capture: requested {args.width}x{args.height}, actual {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
    if (actual_w, actual_h) != (args.width, args.height):
        print("  note: camera did not honour the request; drawing cost scales with actual resolution.")
    return cap


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    display_every = max(1, args.display_every)

    model = YOLO(args.model)
    analyzer = PoseAnalyzer(min_conf=args.min_kpt_conf)
    if "ncnn" in str(args.model).lower():
        print("NCNN backend: input size is fixed at export time, --imgsz is ignored.")

    cap = open_camera(args)
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

    for _ in range(max(0, args.warmup)):
        ok, frame = cap.read()
        if not ok:
            break
        model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
    if args.warmup:
        print(f"Warm-up done ({args.warmup} frames, excluded from statistics).")

    prev_time = time.perf_counter()
    fps = 0.0
    frame_ms = 0.0
    frame_index = 0
    samples: list[float] = []

    print("Webcam started. Press q/Esc to quit, s to save a screenshot.")
    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            frame_index += 1

            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            result = results[0]

            keypoints, primary_index, people_count = extract_primary_keypoints(result)
            analysis = analyzer.analyze(keypoints, primary_index)

            if csv_writer is not None:
                write_csv_row(csv_writer, frame_index, people_count, keypoints, analysis)

            show_frame = (not args.no_display) and (frame_index % display_every == 0)
            if show_frame:
                annotated = frame if args.no_plot else result.plot()
                draw_analysis_overlay(annotated, analysis)
                draw_hud(annotated, fps, people_count, frame_ms)
                cv2.imshow(args.window, annotated)

            now = time.perf_counter()
            elapsed = now - prev_time
            prev_time = now
            frame_ms = elapsed * 1000.0
            samples.append((now - loop_start) * 1000.0)
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else (0.9 * fps + 0.1 * instant_fps)

            if frame_index % 100 == 0:
                recent = samples[-100:]
                recent_ms = sum(recent) / len(recent)
                print(f"  frame {frame_index}: last 100 mean {recent_ms:.2f} ms ({1000.0 / recent_ms:.2f} FPS)")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and show_frame:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = save_dir / f"pose_{timestamp}.jpg"
                cv2.imwrite(str(output_path), annotated)
                print(f"Saved screenshot: {output_path}")
    finally:
        cap.release()
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()

    stats = summarize_ms(samples)
    print("\n=== End-to-end statistics (warm-up excluded) ===")
    print(f"frames measured : {len(samples)}")
    for label in ("mean", "median", "p95"):
        value = stats[label]
        shown_fps = (1000.0 / value) if value > 0 else 0.0
        print(f"{label:<15}: {value:7.2f} ms  ->  {shown_fps:6.2f} FPS")
    print("Report mean AND p95: p95 is the tail latency a real user actually feels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
