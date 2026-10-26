"""P7 时序稀疏基准：跳帧推理 + 轻量跟踪插值（temporal sparsity）。

为什么需要这个脚本
------------------
P6 实测流水线开显示时瓶颈已转移到 **render 本身（38.27 ms，理论上限 26.13 FPS）**，
而推理（NCNN@320）每次约 20 ms。既然相邻帧的人体姿态高度连续，
就不必每帧都跑完整推理：**每 N 帧推理一次**（stride），中间帧用轻量跟踪把关键点
"搬"过来——推理负载降为 1/N，感知帧率保持满帧率。这是本毕设选题主轴里明确的
小贡献点（temporal sparsity，跳帧+跟踪插值）。

方法学要点（写报告前必读）
-------------------------
1. **因果性**：只实现"前向传播"的跟踪（hold / 光流），**不做事后线性插值**。
   事后插值要用到未来帧的推理结果，实时系统做不到——用它报精度会虚高。
   本脚本的精度数字是"实时系统能真实达到的下界"，这是有意识的保守设计。
2. **两个 track-mode 的消融**：
   - `hold`  ：跳帧直接冻结上一次推理的关键点（零成本下界）。
   - `flow`  ：金字塔 LK 光流逐帧跟踪关键点（每帧约零点几 ms，主方法）。
   - `none`  ：跳帧输出空（纯稀疏的伤害上界，用于消融对照）。
3. **精度怎么量**：`--dump-keypoints` 导出的是**输出流**（推理帧用实测关键点、
   跳帧用跟踪关键点），拿去和稠密参考基线（`ref_fp32_640.npz` 或 Pi 稠密 npz）
   跑 `compare_keypoints.py`，得到的 PCK 就是"用户感知流"的精度。
4. **计数怎么量**：`PoseAnalyzer` 在**满帧率**的输出流上运行（P9-a 状态机 +
   缺帧容忍正好适配这种流），脚本结尾直接报深蹲计数与准确率（`--truth`）。
   「跳帧后计数不掉」是场景够用性的核心论据。
5. **stride=1 是回归检查**：不跳帧时应与 `benchmark.py` 同配置结果一致。

计时口径
--------
- `t_inference` 只在**推理帧**有值（单次推理的真实开销）；
  CSV 里的 `inference_ms` 写**摊销值**（总推理时间 / 全部帧），与
  `benchmark.py` 的逐帧均值可比——两者都是"每帧预算"语义。
- `t_track` 是跟踪开销（灰度转换 + LK），只进 JSON 和 trace，不进 CSV 总表。
- 端到端 FPS 与 `benchmark.py` 同公式：1000 / 逐帧总耗时均值。

用法
----
    # 稀疏 stride=2 + 光流（主实验）
    python sparse_infer.py --source outputs/test_clip.mp4 \
        --model yolov8n-pose_ncnn_model --imgsz 320 --stride 2 --track-mode flow \
        --truth 13 --tag pi5_sparse_s2_flow_320 \
        --dump-keypoints results/kpts/pi5_sparse_s2_flow_320.npz

    # 消融：hold / none / stride=3,4
    python sparse_infer.py --source outputs/test_clip.mp4 \
        --model yolov8n-pose_ncnn_model --imgsz 320 --stride 3 --track-mode hold \
        --truth 13 --tag pi5_sparse_s3_hold_320

    # 回归检查：stride=1 应与 benchmark.py 同配置一致
    python sparse_infer.py --source outputs/test_clip.mp4 --imgsz 320 --stride 1 \
        --tag sanity_stride1
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from benchmark import (
    SUMMARY_COLUMNS,
    collect_system_info,
    detect_backend,
    fmt_ms,
    read_cpu_freq_mhz,
    read_cpu_temp_c,
    read_throttled_flag,
    summarize,
)
from pose_analyzer import PoseAnalyzer, Keypoint, extract_primary_keypoints

# COCO 17 关键点的骨架连边（渲染叠加用）
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P7 temporal-sparsity benchmark: stride inference + lightweight tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="固定测试视频路径（不要用摄像头）。")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="模型路径：.pt / .onnx / *_ncnn_model / .xml")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入边长。")
    parser.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值。")
    parser.add_argument("--threads", type=int, default=0, help="CPU 线程数，0 表示不干预。")
    parser.add_argument("--frames", type=int, default=200, help="测量帧数（含跳帧），0 表示不按帧数停止。")
    parser.add_argument("--duration", type=float, default=0.0, help="测量时长（秒），0 表示不按时长停止。")
    parser.add_argument("--warmup", type=int, default=10, help="预热帧数，不计入统计。")
    parser.add_argument("--loop", action="store_true", help="视频播完后循环，用于长时间热测试。")
    parser.add_argument("--stride", type=int, default=2, help="时序稀疏步长：每 N 帧推理一次。")
    parser.add_argument(
        "--track-mode",
        choices=("flow", "hold", "none"),
        default="flow",
        help="跳帧的关键点来源：flow=金字塔LK光流（主方法）；hold=冻结上次结果；none=输出空（消融下界）。",
    )
    parser.add_argument("--render", action="store_true", help="绘制骨架并显示窗口（默认 headless）。")
    parser.add_argument("--plot-only", action="store_true", help="只绘制不显示窗口，用于单独测量绘制开销。")
    parser.add_argument("--no-analyze", action="store_true", help="跳过 pose_analyzer 动作分析。")
    parser.add_argument("--truth", type=int, default=None, help="人工真值深蹲次数；给出则报计数准确率。")
    parser.add_argument("--sample-every", type=int, default=15, help="每隔多少帧采样一次温度/频率。")
    parser.add_argument("--tag", default="untagged", help="实验标签，会写入结果表。")
    parser.add_argument("--notes", default="", help="备注，写入结果表。")
    parser.add_argument("--results-dir", default="results", help="结果输出目录。")
    parser.add_argument("--dump-keypoints", default=None, help="导出输出流关键点为 .npz（感知流精度用）。")
    parser.add_argument("--no-trace", action="store_true", help="不导出 per-frame 明细 CSV。")
    return parser.parse_args()


# --------------------------------------------------------------------------
# 关键点容器转换
# --------------------------------------------------------------------------


def keypoints_to_buffer(keypoints) -> np.ndarray:
    """Keypoint 列表 -> (17, 3) float32 数组，缺位填 -1（与 benchmark.py 的 dump 格式一致）。"""
    buffer = np.full((17, 3), -1.0, dtype=np.float32)
    for slot, point in enumerate(keypoints[:17]):
        buffer[slot] = (point.x, point.y, point.conf)
    return buffer


def buffer_to_keypoints(buffer: np.ndarray):
    """(17, 3) 数组 -> Keypoint 列表，跳过无效位（x/y <= 0）。"""
    out = []
    for row in buffer:
        if row[0] > 0 and row[1] > 0:
            out.append(Keypoint(float(row[0]), float(row[1]), float(row[2])))
    return out


# --------------------------------------------------------------------------
# 跳帧跟踪器
# --------------------------------------------------------------------------


class SparseTracker:
    """跳帧期间的关键点来源。

    flow 模式：推理帧上 `seed()` 记录灰度图与关键点；之后每个跳帧 `update()`
    用金字塔 LK 光流把关键点从上一帧搬到当前帧（逐帧链式，漂移是真实成本）。
    单点跟踪失败（status=0）时该点退回 hold。丢失检测（无人）后自动置空。
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.prev_gray = None
        self.points = None  # (17, 3) float32，-1 表示无效
        self.last_primary_index = None

    def reset(self) -> None:
        self.prev_gray = None
        self.points = None

    def seed(self, gray, keypoints, primary_index) -> np.ndarray:
        """推理帧调用：以实测关键点作为跟踪起点。返回该帧的输出 buffer。"""
        self.points = keypoints_to_buffer(keypoints)
        self.prev_gray = gray
        self.last_primary_index = primary_index
        return self.points

    def update(self, gray) -> np.ndarray:
        """跳帧调用：把关键点搬到当前帧。返回输出 buffer（无状态时为空 buffer）。"""
        if self.mode == "none" or self.points is None or self.prev_gray is None:
            return np.full((17, 3), -1.0, dtype=np.float32)

        if self.mode == "hold":
            return self.points

        # flow：对有效点逐帧链式 LK
        valid = (self.points[:, 0] > 0) & (self.points[:, 1] > 0)
        if not valid.any():
            self.prev_gray = gray
            return self.points

        src = self.points[valid, :2].astype(np.float32).reshape(-1, 1, 2)
        dst, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, src, None,
            winSize=(21, 21), maxLevel=3, criteria=LK_CRITERIA,
        )
        moved = self.points.copy()
        ok = status.reshape(-1).astype(bool)
        slots = np.nonzero(valid)[0]
        for i, slot in enumerate(slots):
            if ok[i]:
                moved[slot, 0] = dst[i, 0, 0]
                moved[slot, 1] = dst[i, 0, 1]
            # status=0 的点保持原位（hold 退化）
        self.points = moved
        self.prev_gray = gray
        return moved


# --------------------------------------------------------------------------
# 渲染叠加（跳帧没有 result 对象，全部帧统一手绘，避免 plot() 的重开销混进来）
# --------------------------------------------------------------------------


def draw_overlay(frame: np.ndarray, buffer: np.ndarray, analysis) -> np.ndarray:
    annotated = frame
    for a, b in SKELETON_EDGES:
        pa, pb = buffer[a], buffer[b]
        if pa[0] > 0 and pb[0] > 0:
            cv2.line(
                annotated,
                (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                (80, 200, 80), 2, cv2.LINE_AA,
            )
    for row in buffer:
        if row[0] > 0:
            cv2.circle(annotated, (int(row[0]), int(row[1])), 4, (40, 40, 230), -1, cv2.LINE_AA)
    if analysis is not None:
        for i, line in enumerate(analysis.overlay_lines()):
            cv2.putText(annotated, line, (10, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
            cv2.putText(annotated, line, (10, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
    return annotated


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"[ERROR] 测试视频不存在: {source}")
        return 1
    if args.frames <= 0 and args.duration <= 0:
        print("[ERROR] --frames 和 --duration 至少要设一个大于 0。")
        return 1
    if args.stride < 1:
        print("[ERROR] --stride 至少为 1（1 即不跳帧，等价 benchmark.py）。")
        return 1

    if args.threads > 0:
        cv2.setNumThreads(args.threads)
        try:
            import torch

            torch.set_num_threads(args.threads)
        except Exception:
            pass

    results_dir = Path(args.results_dir)
    (results_dir / "runs").mkdir(parents=True, exist_ok=True)

    system_info = collect_system_info(args)
    backend = detect_backend(args.model)

    print("=" * 68)
    print(f"  实验标签 : {args.tag}")
    print(f"  设备     : {system_info['device']}")
    print(f"  后端     : {backend}   模型: {args.model}   imgsz: {args.imgsz}")
    print(f"  时序稀疏 : stride={args.stride}  track={args.track_mode}   渲染: {args.render or args.plot_only}")
    print("=" * 68)

    from ultralytics import YOLO

    model = YOLO(args.model)
    analyzer = None if args.no_analyze else PoseAnalyzer()
    tracker = SparseTracker(args.track_mode)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {source}")
        return 1

    # ---- 预热（稠密推理，与 benchmark.py 一致）----
    warmed = 0
    while warmed < args.warmup:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] 视频无法读取任何帧。")
                cap.release()
                return 1
        model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        warmed += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"预热完成 ({args.warmup} 帧)，开始测量...\n")

    t_capture: list[float] = []
    t_preprocess: list[float] = []
    t_inference: list[float] = []
    t_postprocess: list[float] = []
    t_track: list[float] = []
    t_analyze: list[float] = []
    t_render: list[float] = []
    t_total: list[float] = []

    trace_rows: list[dict] = []
    keypoint_dump: list[np.ndarray] = []

    temps: list[float] = []
    freqs: list[float] = []
    loop_count = 0
    detected_frames = 0        # 感知检出率：输出流上有有效关键点的帧
    inference_frames = 0       # 实际跑推理的帧
    timing_fallback_frames = 0

    render_enabled = args.render or args.plot_only
    window_name = f"sparse::{args.tag}"

    run_start = time.perf_counter()
    frame_index = 0
    interrupted = False
    last_primary_index = None

    try:
        while True:
            if args.frames > 0 and frame_index >= args.frames:
                break
            if args.duration > 0 and (time.perf_counter() - run_start) >= args.duration:
                break

            m0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    loop_count += 1
                    continue
                break
            m1 = time.perf_counter()

            is_inference_frame = frame_index % args.stride == 0

            if is_inference_frame:
                results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
                result = results[0]
                m2 = time.perf_counter()

                keypoints, primary_index, people_count = extract_primary_keypoints(result)
                if people_count > 0:
                    last_primary_index = primary_index
                    tracker.reset()  # 丢弃上一轮跟踪状态，以实测为锚

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if people_count > 0:
                    buffer = tracker.seed(gray, keypoints, primary_index)
                else:
                    tracker.reset()
                    buffer = np.full((17, 3), -1.0, dtype=np.float32)
                m3 = time.perf_counter()

                analyzer_input = keypoints
            else:
                m2 = m1  # 跳帧：无推理
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                buffer = tracker.update(gray)
                m3 = time.perf_counter()

                analyzer_input = buffer_to_keypoints(buffer)
                people_count = 1 if (buffer[:, 0] > 0).any() else 0

            analysis = None
            if analyzer is not None:
                analysis = analyzer.analyze(analyzer_input, last_primary_index)
            m4 = time.perf_counter()

            if render_enabled:
                annotated = draw_overlay(frame, buffer, analysis)
                if args.render:
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        interrupted = True
                        break
            m5 = time.perf_counter()

            # ---- 计时归集 ----
            cap_ms = (m1 - m0) * 1000.0
            track_ms = (m3 - m2) * 1000.0
            ana_ms = (m4 - m3) * 1000.0
            ren_ms = (m5 - m4) * 1000.0
            tot_ms = (m5 - m0) * 1000.0

            t_capture.append(cap_ms)
            t_track.append(track_ms)
            t_analyze.append(ana_ms)
            t_render.append(ren_ms)
            t_total.append(tot_ms)

            if people_count > 0:
                detected_frames += 1

            pre_ms = inf_ms = post_ms = 0.0
            if is_inference_frame:
                inference_frames += 1
                speed = getattr(result, "speed", {}) or {}
                pre_ms = float(speed.get("preprocess", 0.0) or 0.0)
                inf_ms = float(speed.get("inference", 0.0) or 0.0)
                post_ms = float(speed.get("postprocess", 0.0) or 0.0)
                fallback_used = False
                if inf_ms <= 0.0:
                    wall_predict_ms = (m2 - m1) * 1000.0
                    residual = wall_predict_ms - pre_ms - post_ms
                    inf_ms = residual if residual > 0 else wall_predict_ms
                    fallback_used = True
                    timing_fallback_frames += 1
                t_preprocess.append(pre_ms)
                t_inference.append(inf_ms)
                t_postprocess.append(post_ms)

            if args.dump_keypoints:
                keypoint_dump.append(buffer)

            temp_c = None
            freq_mhz = None
            if frame_index % max(1, args.sample_every) == 0:
                temp_c = read_cpu_temp_c()
                freq_mhz = read_cpu_freq_mhz()
                if temp_c is not None:
                    temps.append(temp_c)
                if freq_mhz is not None:
                    freqs.append(freq_mhz)

            if not args.no_trace:
                trace_rows.append(
                    {
                        "frame": frame_index,
                        "is_inference": int(is_inference_frame),
                        "age": frame_index % args.stride if args.stride > 0 else 0,
                        "elapsed_s": round(m5 - run_start, 4),
                        "capture_ms": round(cap_ms, 3),
                        "preprocess_ms": round(pre_ms, 3) if is_inference_frame else "",
                        "inference_ms": round(inf_ms, 3) if is_inference_frame else "",
                        "postprocess_ms": round(post_ms, 3) if is_inference_frame else "",
                        "track_ms": round(track_ms, 3),
                        "analyze_ms": round(ana_ms, 3),
                        "render_ms": round(ren_ms, 3),
                        "total_ms": round(tot_ms, 3),
                        "fps_instant": round(1000.0 / tot_ms, 2) if tot_ms > 0 else "",
                        "people": people_count,
                        "temp_c": "" if temp_c is None else round(temp_c, 1),
                        "freq_mhz": "" if freq_mhz is None else round(freq_mhz, 0),
                        "timing_fallback": int(timing_fallback_frames > 0 and is_inference_frame),
                    }
                )

            frame_index += 1
            if frame_index % 50 == 0:
                recent = t_total[-50:]
                avg_fps = 1000.0 / statistics.fmean(recent)
                temp_note = f"  {temps[-1]:.1f}C" if temps else ""
                print(f"  [{frame_index:5d}] 近 50 帧平均 {avg_fps:5.1f} FPS{temp_note}")

    except KeyboardInterrupt:
        interrupted = True
        print("\n[中断] 用户手动停止，已测数据仍会保存。")
    finally:
        cap.release()
        if args.render:
            cv2.destroyAllWindows()

    if not t_total:
        print("[ERROR] 没有测到任何帧。")
        return 1

    wall_seconds = time.perf_counter() - run_start
    frames_measured = len(t_total)

    # ---- 计数收尾（P9-a 状态机的 finalize 补计）----
    squat_count = None
    if analyzer is not None:
        analyzer.finalize()
        squat_count = analyzer.squat_count
    squat_accuracy = None
    if squat_count is not None and args.truth:
        squat_accuracy = max(0.0, 1.0 - abs(squat_count - args.truth) / args.truth)

    stages = {
        "capture": summarize(t_capture),
        "preprocess": summarize(t_preprocess),
        "inference": summarize(t_inference),
        "postprocess": summarize(t_postprocess),
        "track": summarize(t_track),
        "analyze": summarize(t_analyze),
        "render": summarize(t_render),
        "total": summarize(t_total),
    }

    total_stat = stages["total"]
    fps_mean = 1000.0 / total_stat["mean"]
    fps_worst5 = 1000.0 / total_stat["p95"]
    infer_stat = stages["inference"]
    fps_infer = 1000.0 / infer_stat["mean"] if infer_stat and infer_stat["mean"] > 0 else None

    # 摊销口径（CSV 里的 inference_ms 用它，与 benchmark.py 的逐帧均值同语义）
    amortized = {
        "preprocess_ms": sum(t_preprocess) / frames_measured if t_preprocess else 0.0,
        "inference_ms": sum(t_inference) / frames_measured if t_inference else 0.0,
        "postprocess_ms": sum(t_postprocess) / frames_measured if t_postprocess else 0.0,
    }

    timestamp = datetime.now()
    throttled_end = read_throttled_flag()

    record = {
        "tag": args.tag,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "notes": args.notes,
        "interrupted": interrupted,
        "config": {
            "source": str(source),
            "model": args.model,
            "backend": backend,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "threads": args.threads,
            "warmup": args.warmup,
            "render": args.render,
            "plot_only": args.plot_only,
            "analyze": not args.no_analyze,
            "loop": args.loop,
            "loops_completed": loop_count,
            "stride": args.stride,
            "track_mode": args.track_mode,
            "temporal_sparsity": True,
        },
        "system": system_info,
        "measurement": {
            "frames": frames_measured,
            "wall_seconds": round(wall_seconds, 3),
            "detect_rate": round(detected_frames / frames_measured, 4),
            "timing_fallback_frames": timing_fallback_frames,
        },
        "stages_ms": stages,  # inference 仅统计推理帧（单次真实开销）
        "sparse": {
            "stride": args.stride,
            "track_mode": args.track_mode,
            "inference_frames": inference_frames,
            "inference_ratio": round(inference_frames / frames_measured, 4),
            "inference_call_rate_fps": round(inference_frames / wall_seconds, 2),
            "inference_amortized_ms": round(amortized["inference_ms"], 3),
            "inference_per_call_ms": round(infer_stat["mean"], 3) if infer_stat else None,
            "track_ms": stages["track"],
            "squat_count": squat_count,
            "squat_truth": args.truth,
            "squat_accuracy": round(squat_accuracy, 4) if squat_accuracy is not None else None,
        },
        "fps": {
            "end_to_end_mean": round(fps_mean, 2),
            "end_to_end_worst5": round(fps_worst5, 2),
            "inference_only": round(fps_infer, 2) if fps_infer else None,
        },
        "thermal": {
            "temp_start_c": temps[0] if temps else None,
            "temp_end_c": temps[-1] if temps else None,
            "temp_max_c": max(temps) if temps else None,
            "freq_min_mhz": min(freqs) if freqs else None,
            "freq_max_mhz": max(freqs) if freqs else None,
            "throttled_end": throttled_end,
        },
    }

    # ---- 落盘 ----
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.tag)
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")

    json_path = results_dir / "runs" / f"{safe_tag}_{stamp}.json"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    trace_path = None
    if trace_rows:
        trace_path = results_dir / "runs" / f"{safe_tag}_{stamp}_trace.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trace_rows)

    kpt_path = None
    if args.dump_keypoints and keypoint_dump:
        kpt_path = Path(args.dump_keypoints)
        kpt_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            kpt_path,
            keypoints=np.stack(keypoint_dump),
            tag=args.tag,
            imgsz=args.imgsz,
            model=args.model,
        )

    summary_path = results_dir / "benchmark_results.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "tag": args.tag,
                "timestamp": record["timestamp"],
                "device": system_info["device"],
                "backend": backend,
                "model": args.model,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "threads": args.threads or "",
                "render": int(render_enabled),
                "analyze": int(not args.no_analyze),
                "frames": frames_measured,
                "fps_e2e_mean": round(fps_mean, 2),
                "fps_e2e_worst5": round(fps_worst5, 2),
                "fps_infer_only": round(fps_infer, 2) if fps_infer else "",
                "capture_ms": fmt_ms(stages["capture"]),
                "preprocess_ms": round(amortized["preprocess_ms"], 2),
                "inference_ms": round(amortized["inference_ms"], 2),
                "postprocess_ms": round(amortized["postprocess_ms"], 2),
                "analyze_ms": fmt_ms(stages["analyze"]),
                "render_ms": fmt_ms(stages["render"]),
                "total_ms": fmt_ms(stages["total"]),
                "temp_start_c": round(temps[0], 1) if temps else "",
                "temp_max_c": round(max(temps), 1) if temps else "",
                "throttled": throttled_end or "",
                "detect_rate": record["measurement"]["detect_rate"],
                "notes": args.notes,
            }
        )

    # ---- 控制台报告 ----
    print("\n" + "=" * 68)
    print(f"  耗时分解  ({frames_measured} 帧, {wall_seconds:.1f} 秒)")
    print("-" * 68)
    print(f"  {'阶段':<14}{'均值ms':>10}{'中位ms':>10}{'P95ms':>10}")
    print("-" * 68)
    for name, label in [
        ("capture", "采集"),
        ("preprocess", "预处理*"),
        ("inference", "推理*"),
        ("postprocess", "后处理*"),
        ("track", "跟踪"),
        ("analyze", "动作分析"),
        ("render", "渲染"),
    ]:
        stat = stages[name]
        if stat is None:
            continue
        print(f"  {label:<14}{stat['mean']:>10.2f}{stat['median']:>10.2f}{stat['p95']:>10.2f}")
    print("-" * 68)
    print(f"  {'端到端':<14}{total_stat['mean']:>10.2f}{total_stat['median']:>10.2f}{total_stat['p95']:>10.2f}")
    print("=" * 68)
    print(f"  端到端 FPS      : {fps_mean:.2f}   (最差 5%: {fps_worst5:.2f})   <- 感知帧率（满帧率输出）")
    print(f"  推理帧占比      : {inference_frames}/{frames_measured} = {100.0 * inference_frames / frames_measured:.1f}%")
    print(f"  推理调用率      : {inference_frames / wall_seconds:.2f} 次/秒（稠密时应等于端到端 FPS）")
    if infer_stat:
        print(f"  单次推理        : {infer_stat['mean']:.2f} ms   摊销到每帧: {amortized['inference_ms']:.2f} ms")
    if stages["track"]:
        print(f"  跟踪开销        : {stages['track']['mean']:.2f} ms/帧 (P95 {stages['track']['p95']:.2f})")
    if squat_count is not None:
        line = f"  深蹲计数        : {squat_count}"
        if args.truth:
            line += f"   真值 {args.truth}   准确率 {squat_accuracy * 100:.1f}%"
        print(line)
    print(f"  有人检出率      : {record['measurement']['detect_rate'] * 100:.1f}%  (感知口径)")
    if temps:
        print(f"  温度            : {temps[0]:.1f}C -> {temps[-1]:.1f}C  峰值 {max(temps):.1f}C")
    if timing_fallback_frames:
        print(f"  [!] 分项计时回退 : {timing_fallback_frames} 帧改用 predict() 墙钟时间")
    if throttled_end and throttled_end != "0x0":
        print(f"  [!] 降频标志    : {throttled_end}  <- 发生过降频，本次数据需在报告中标注")
    print("=" * 68)
    print("  * 预处理/推理/后处理仅在推理帧发生，表中为单次真实开销；")
    print("    CSV 总表里写的是摊销值（总推理时间 / 全部帧），与 benchmark.py 同语义。")
    print(f"  明细 JSON : {json_path}")
    if trace_path:
        print(f"  逐帧 trace: {trace_path}")
    if kpt_path:
        print(f"  关键点    : {kpt_path}  <- 输出流（感知流），供 compare_keypoints.py 用")
    print(f"  汇总总表  : {summary_path}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
