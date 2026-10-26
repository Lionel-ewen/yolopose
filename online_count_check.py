"""在线计数状态机验证（独立脚本，不改动任何现有实验代码）。

要回答的问题
------------
`PoseAnalyzer.finalize()` 里有一段收尾逻辑：

    if self.squat_phase == "down" and not self._counted_this_cycle:
        self.squat_count += 1

它的语义是「视频/片段结束时人还蹲着，把这一轮也算上」。
这对**离线批处理**是合理的（片段被截断在蹲底，真人确实蹲了），
但对**在线设备是错的** —— 在线系统没有「结束」这个事件，人蹲着就是没完成一次。

于是产生一个此前从未被量化的问题：

    P9 报的 10/13、11/13，到底有多少次是 finalize() 补出来的？
    如果补出来 1 次，在线形态实际只有 9/13，那么「计数 76.9%」这个数字就站不住。

本脚本把两种口径同时算出来，直接给出差额。

三种口径
--------
    online    逐帧实时输出，**不调 finalize()** —— 真实设备的行为
    offline   跑完再调 finalize() —— 现有 count_squats.py 的行为
    finalize  两者之差，即「收尾补计」的贡献

用法
----
    # 用固定视频（推荐：可复现，计数评测的纪律要求）
    python online_count_check.py --source outputs/test_clip.mp4 --truth 13 \
        --model yolov8n-pose_ncnn_model --imgsz 320 --tag online_test_clip_pi

    # 直接接摄像头（验证真实设备行为，但结果不可复现，只能作定性参考）
    python online_count_check.py --source camera --truth 10 \
        --model yolov8n-pose_ncnn_model --imgsz 320 --tag online_cam_trial

    # 调阈值做敏感性分析
    python online_count_check.py --source outputs/test_clip.mp4 --truth 13 \
        --down 105 --up 145 --tag online_up145

输出
----
    results/online_count_results.csv   每次运行一行（汇总）
    results/online_count_events_<tag>_<时间>.csv   每一次计数触发的明细
    results/online_count_trace_<tag>_<时间>.csv    逐帧膝角与相位（可选，--no-trace 关闭）

为什么计数要用固定视频
----------------------
摄像头受光照、曝光、自动增益影响，同一动作两次录制的角度曲线不同，
准确率不可复现。在线**行为**可以在摄像头上看，但准确率数字必须来自固定视频。
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2

from pose_analyzer import PoseAnalyzer, extract_primary_keypoints

SUMMARY_COLUMNS = [
    "tag",
    "timestamp",
    "device",
    "source",
    "model",
    "imgsz",
    "conf",
    "min_kpt_conf",
    "down_angle",
    "up_angle",
    "frames",
    "duration_s",
    "fps_e2e",
    "knee_available_pct",
    "online_count",
    "offline_count",
    "finalize_delta",
    "truth",
    "online_acc_pct",
    "offline_acc_pct",
    "first_event_frame",
    "last_event_frame",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比「在线实时计数」与「离线 finalize 补计」两种口径。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help='视频文件路径，或 "camera"/"0" 表示摄像头。')
    parser.add_argument("--model", default="yolov8n-pose.pt", help="模型路径：.pt / *_ncnn_model")
    parser.add_argument("--imgsz", type=int, default=320, help="推理输入边长（NCNN 固定）。")
    parser.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值。")
    parser.add_argument("--min-kpt-conf", type=float, default=0.05,
                        help="关键点最低置信度。P9-a 定的默认值，别随便改。")

    parser.add_argument("--down", type=float, default=105.0, help="进入下蹲态的膝角阈值。")
    parser.add_argument("--up", type=float, default=150.0, help="计数的站起膝角阈值。")
    parser.add_argument("--truth", type=int, default=0, help="人工真值次数，0 表示不评估准确率。")

    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数，0 表示全片。")
    parser.add_argument("--warmup", type=int, default=5, help="预热帧数，不计入统计。")

    parser.add_argument("--camera", type=int, default=0, help="--source camera 时的摄像头索引。")
    parser.add_argument("--width", type=int, default=640, help="摄像头采集宽度。")
    parser.add_argument("--height", type=int, default=480, help="摄像头采集高度。")
    parser.add_argument("--fourcc", default="MJPG", help="摄像头像素格式。")

    parser.add_argument("--tag", default="online_untagged", help="实验标签。")
    parser.add_argument("--notes", default="", help="备注。")
    parser.add_argument("--results-dir", default="results", help="结果输出目录。")
    parser.add_argument("--no-trace", action="store_true", help="不导出逐帧明细。")
    return parser.parse_args()


# --------------------------------------------------------------------------
# 输入
# --------------------------------------------------------------------------


def is_camera_source(source: str) -> bool:
    return source.strip().lower() in ("camera", "cam", "0")


def open_source(args: argparse.Namespace):
    """返回 (capture, kind, fps_hint)。kind 为 video / camera。"""
    if is_camera_source(args.source):
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            return cap, "camera", 0.0
        if args.fourcc and args.fourcc.lower() != "none":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, "camera", 0.0

    cap = cv2.VideoCapture(args.source)
    fps_hint = cap.get(cv2.CAP_PROP_FPS) or 0.0
    return cap, "video", fps_hint


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cap, kind, fps_hint = open_source(args)
    if not cap.isOpened():
        print(f"[ERROR] 打不开输入源: {args.source}")
        return 1

    model = YOLO(args.model)
    analyzer = PoseAnalyzer(
        min_conf=args.min_kpt_conf,
        squat_down_angle=args.down,
        squat_up_angle=args.up,
    )

    print(f"输入: {args.source} ({kind})   模型: {args.model}   imgsz={args.imgsz}")
    print(f"阈值: down<{args.down}° / up>{args.up}°   min_kpt_conf={args.min_kpt_conf}")
    print("=" * 66)

    # 预热：让 ncnn/ultralytics 完成图构建，否则首帧开销会污染计时
    for _ in range(max(0, args.warmup)):
        ok, frame = cap.read()
        if not ok:
            break
        model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

    trace: list[dict] = []
    events: list[dict] = []
    knee_available = 0
    prev_count = 0
    frames = 0
    started = time.perf_counter()
    wall = 0.0

    try:
        while True:
            if args.max_frames and frames >= args.max_frames:
                break

            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            t1 = time.perf_counter()

            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            t2 = time.perf_counter()

            keypoints, primary_index, _ = extract_primary_keypoints(results[0])
            analysis = analyzer.analyze(keypoints, primary_index)
            t3 = time.perf_counter()

            frames += 1
            knee = analysis.angles.get("left_knee")
            knee_r = analysis.angles.get("right_knee")
            knee_values = [v for v in (knee, knee_r) if v is not None]
            knee_avg = sum(knee_values) / len(knee_values) if knee_values else None
            if knee_avg is not None:
                knee_available += 1

            # 计数变化即「设备该给反馈的时刻」——这里的帧号就是反馈延迟的起点
            if analysis.squat_count > prev_count:
                events.append({
                    "event_no": analysis.squat_count,
                    "frame": frames,
                    "t_s": round(t1 - started, 4),
                    "video_time_s": round(frames / fps_hint, 3) if fps_hint > 0 else "",
                    "knee_avg_at_event": round(knee_avg, 2) if knee_avg is not None else "",
                    "phase_before": analysis.squat_phase,
                    "count_online": analysis.squat_count,
                })
                prev_count = analysis.squat_count

            trace.append({
                "frame": frames,
                "t_s": round(t1 - started, 4),
                "capture_ms": round((t1 - t0) * 1000.0, 3),
                "inference_ms": round((t2 - t1) * 1000.0, 3),
                "analyze_ms": round((t3 - t2) * 1000.0, 3),
                "knee_avg": round(knee_avg, 2) if knee_avg is not None else "",
                "phase": analysis.squat_phase,
                "count_online": analysis.squat_count,
                "actions": ";".join(analysis.actions),
            })
    finally:
        wall = time.perf_counter() - started
        cap.release()

    # ---- 关键对比：在线口径（不调 finalize）vs 离线口径（调 finalize）----
    online_count = analyzer.squat_count
    phase_at_end = analyzer.squat_phase
    counted_this_cycle = analyzer._counted_this_cycle
    analyzer.finalize()
    offline_count = analyzer.squat_count
    finalize_delta = offline_count - online_count

    fps_e2e = frames / wall if wall > 0 else 0.0
    knee_pct = knee_available / frames * 100.0 if frames else 0.0

    print(f"处理帧数      : {frames}   用时 {wall:.2f} s   → {fps_e2e:.2f} FPS")
    print(f"膝角可用率    : {knee_pct:.1f}%")
    print(f"计数触发次数  : {len(events)}")
    if events:
        print(f"  首次触发于第 {events[0]['frame']} 帧，末次于第 {events[-1]['frame']} 帧")
    print("")
    print(f"结束时状态    : phase={phase_at_end}  counted_this_cycle={counted_this_cycle}")
    print(f"【在线口径】  {online_count} 次（不调 finalize，真实设备行为）")
    print(f"【离线口径】  {offline_count} 次（调 finalize，现有 count_squats.py 行为）")
    print(f"【收尾补计】  {finalize_delta} 次")

    if finalize_delta > 0:
        print("")
        print("⚠️ finalize() 补计了 1 次 —— 说明录制结束时人处于「蹲着未站起」状态。")
        print("   在线设备上这一轮**不会被计入**（人没站起来 = 没完成一次，这是正确行为）。")
        print(f"   因此在线准确率应按 {online_count} 次算，不是 {offline_count} 次。")
    else:
        print("")
        print("✅ finalize() 未补计 —— 在线与离线口径一致，P9 的计数结论可直接用于在线形态。")

    acc_online = acc_offline = None
    if args.truth > 0:
        acc_online = online_count / args.truth * 100.0
        acc_offline = offline_count / args.truth * 100.0
        print("")
        print(f"真值 {args.truth} 次：在线 {online_count}/{args.truth} = {acc_online:.1f}%"
              f"   离线 {offline_count}/{args.truth} = {acc_offline:.1f}%")

    # ---- 落盘 ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    events_path = results_dir / f"online_count_events_{args.tag}_{stamp}.csv"
    if events:
        with events_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(events[0].keys()))
            writer.writeheader()
            writer.writerows(events)
        print(f"\n计数事件明细: {events_path}")

    if trace and not args.no_trace:
        trace_path = results_dir / f"online_count_trace_{args.tag}_{stamp}.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace[0].keys()))
            writer.writeheader()
            writer.writerows(trace)
        print(f"逐帧明细    : {trace_path}")

    row = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": _device_name(),
        "source": args.source,
        "model": args.model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "min_kpt_conf": args.min_kpt_conf,
        "down_angle": args.down,
        "up_angle": args.up,
        "frames": frames,
        "duration_s": round(wall, 3),
        "fps_e2e": round(fps_e2e, 2),
        "knee_available_pct": round(knee_pct, 1),
        "online_count": online_count,
        "offline_count": offline_count,
        "finalize_delta": finalize_delta,
        "truth": args.truth or "",
        "online_acc_pct": round(acc_online, 1) if acc_online is not None else "",
        "offline_acc_pct": round(acc_offline, 1) if acc_offline is not None else "",
        "first_event_frame": events[0]["frame"] if events else "",
        "last_event_frame": events[-1]["frame"] if events else "",
        "notes": args.notes,
    }
    summary_path = results_dir / "online_count_results.csv"
    exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})
    print(f"汇总结果    : {summary_path}")

    return 0


def _device_name() -> str:
    try:
        with open("/sys/firmware/devicetree/base/model", "r", encoding="utf-8", errors="ignore") as handle:
            return handle.readline().strip().strip("\x00") or "unknown"
    except OSError:
        import platform

        return f"{platform.system()} {platform.machine()}"


if __name__ == "__main__":
    raise SystemExit(main())
