"""Measure squat-counting accuracy of the full pipeline on a fixed video clip.

Why this script exists
----------------------
``benchmark.py`` deliberately reports *timing only*, and ``webcam_pose.py`` can
only read a camera.  Neither can answer the question "how many squats does the
system count on the fixed test clip?", which is exactly the scenario
sufficiency (O5) metric: the clip contains a known number of squats, so the
system count compared against that number is the accuracy figure.

This script runs the same detection + analysis path over a *file* input and
reports the counted repetitions against a manually established ground truth.

Usage
-----
    python count_squats.py --source outputs/test_clip.mp4 --truth 13
    python count_squats.py --source outputs/test_clip.mp4 --model yolov8n-pose_ncnn_model \
        --imgsz 320 --truth 13 --out-csv results/squat_count_pi5_ncnn_320.csv

Notes
-----
- Keep the same ``--conf`` / ``--imgsz`` as the corresponding timing run, so the
  accuracy figure belongs to that exact configuration.
- The counter is a threshold state machine on the mean knee angle.  If the count
  is off, ``--out-csv`` gives the per-frame knee angles needed to see whether the
  thresholds themselves are miscalibrated rather than the detector missing poses.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from pose_analyzer import PoseAnalyzer, extract_primary_keypoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count squats in a fixed video clip and compare with ground truth."
    )
    parser.add_argument("--source", required=True, help="固定测试视频路径（与 benchmark 用同一段）。")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="模型路径：.pt / .onnx / *_ncnn_model")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入边长。")
    parser.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值。")
    parser.add_argument("--min-kpt-conf", type=float, default=0.25, help="关键点最低置信度。")
    parser.add_argument("--squat-down-angle", type=float, default=105.0, help="判定为「蹲下」的膝角阈值。")
    parser.add_argument("--squat-up-angle", type=float, default=160.0, help="判定为「站起」的膝角阈值。")
    parser.add_argument("--truth", type=int, default=None, help="人工真值动作次数；给出则计算准确率。")
    parser.add_argument("--frames", type=int, default=0, help="最多处理帧数，0 表示处理整段视频。")
    parser.add_argument("--out-csv", default=None, help="逐帧明细输出路径（膝角 / 相位 / 累计计数）。")
    parser.add_argument("--tag", default="", help="写入明细 CSV 的标签，便于区分配置。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source: {args.source}", file=sys.stderr)
        return 2

    model = YOLO(args.model)
    analyzer = PoseAnalyzer(
        min_conf=args.min_kpt_conf,
        squat_down_angle=args.squat_down_angle,
        squat_up_angle=args.squat_up_angle,
    )

    rows: list[dict[str, object]] = []
    frame_index = 0
    no_pose_frames = 0
    started = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.frames and frame_index >= args.frames:
            break

        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        keypoints, primary_index, people_count = extract_primary_keypoints(result)
        analysis = analyzer.analyze(keypoints, primary_index)
        if people_count == 0:
            no_pose_frames += 1

        rows.append(
            {
                "frame": frame_index,
                "tag": args.tag,
                "people": people_count,
                "squat_phase": analysis.squat_phase,
                "squat_count": analysis.squat_count,
                "left_knee_deg": round(analysis.angles.get("left_knee", float("nan")), 2),
                "right_knee_deg": round(analysis.angles.get("right_knee", float("nan")), 2),
                "actions": "|".join(analysis.actions),
            }
        )
        frame_index += 1

    cap.release()
    elapsed = time.perf_counter() - started

    analyzer.finalize()
    counted = analyzer.squat_count
    print("=" * 62)
    print(f"source        : {args.source}")
    print(f"model         : {args.model}   imgsz={args.imgsz}   conf={args.conf}")
    print(f"thresholds    : down<{args.squat_down_angle}  up>{args.squat_up_angle} (mean knee angle)")
    print(f"frames        : {frame_index}   ({no_pose_frames} frames with no pose detected)")
    print(f"wall time     : {elapsed:.1f} s   ({frame_index / elapsed:.2f} FPS, analysis only)")
    print(f"SQUAT COUNT   : {counted}")

    exit_code = 0
    if args.truth is not None:
        delta = counted - args.truth
        accuracy = 0.0 if args.truth == 0 else max(0.0, 1.0 - abs(delta) / args.truth)
        print(f"ground truth  : {args.truth}")
        print(f"difference    : {delta:+d}")
        print(f"accuracy      : {accuracy * 100:.1f}%")
        if delta != 0:
            print("               -> check --out-csv knee angles before blaming the detector")
            exit_code = 1
    print("=" * 62)

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"per-frame detail written to {out}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
