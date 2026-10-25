"""P6 多线程流水线基准：采集 / 推理 / 分析+渲染 三线程解耦。

为什么需要这个脚本
------------------
P4-c 实测发现一个关键现象：在**单线程**架构下，仅开启显示窗口（OpenCV HighGUI）
就把推理耗时从 23.51 ms 拖慢到 63.57 ms（×2.7）。同进程内 GUI 活动与 NCNN 多线程
推理争抢 CPU 核与内存带宽，渲染的真实账单 = 12.17 ms（自身）+ 40.06 ms（对推理的
拖慢）≈ 52.2 ms，占含渲染帧时间的 64%。

推论：单线程「推理+渲染同线程」架构在该场景到不了 20 FPS，**P6 多线程解耦从
「优化项」升级为「实时达标的必要条件」**。本脚本就是用来验证这个推论的。

与 benchmark.py 的关系
---------------------
- **参数、JSON 结构、CSV 列与 `benchmark.py` 完全一致**，结果可直接合并进
  `results/benchmark_results.csv` 横向对比。
- 系统探测、统计、汇总列定义直接 **import 复用** `benchmark.py`，避免两套口径漂移。
- 区别只在**执行模型**：benchmark.py 是串行逐帧；本脚本是三线程队列流水线。

计时口径（重要，读数据前必看）
------------------------------
流水线下「单帧总耗时」不再等于各阶段之和，因此本脚本同时报告两个口径：

1. **吞吐 FPS**（`fps.end_to_end_mean`）= 处理帧数 / 总墙钟时间。
   这是用户感知的帧率，也是流水线的真实产出能力。瓶颈由最慢的 stage 决定。
2. **单帧延迟**（`pipeline.latency_ms`）= 从该帧被采集到，到该帧渲染完成为止。
   流水线会**放大**单帧延迟（要在队列里等前一帧），但这不影响吞吐。

`fps_e2e_worst5` 仍按「最差 5% 帧」的语义，用**延迟 P95** 反推，与 benchmark.py 一致。
`fps_infer_only` 仍为推理 stage 均值的反推，与 benchmark.py 一致。

本脚本新增两个流水线特有指标（只进 JSON，不进 CSV，以免破坏总表列结构）：
- `pipeline.queue_wait_ms`：各线程阻塞等队列的耗时 —— 直接反映哪一级是瓶颈。
- `pipeline.latency_ms`：单帧端到端延迟分布。

用法
----
    # headless 流水线（应与 benchmark.py 的同配置对比）
    python async_pipeline.py --source outputs/test_clip.mp4 \
        --model yolov8n-pose_ncnn_model --imgsz 320 --tag pi5_ncnn_320_pipe

    # 开窗口的流水线 —— 这是 P4-c 问题的直接复现与修复验证
    python async_pipeline.py --source outputs/test_clip.mp4 \
        --model yolov8n-pose_ncnn_model --imgsz 320 --render \
        --tag pi5_ncnn_320_pipe_render --notes "P6 pipeline with render"

    # 热 soak
    python async_pipeline.py --source outputs/test_clip.mp4 --duration 600 --loop \
        --tag pi5_thermal_soak_pipe
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# 复用 benchmark.py 的系统探测与统计工具，保证口径一致。
# benchmark.py 的 main() 有 __name__ 保护，import 不会触发测量。
from benchmark import (
    SUMMARY_COLUMNS,
    collect_system_info,
    detect_backend,
    detect_device_name,
    fmt_ms,
    read_cpu_freq_mhz,
    read_cpu_temp_c,
    read_throttled_flag,
    summarize,
)
from pose_analyzer import PoseAnalyzer, extract_primary_keypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P6 multi-threaded pipeline benchmark (capture / inference / analyze+render).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="固定测试视频路径（不要用摄像头）。")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="模型路径：.pt / .onnx / *_ncnn_model / .xml")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入边长。")
    parser.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值。")
    parser.add_argument("--threads", type=int, default=0, help="CPU 线程数，0 表示不干预（用库的默认值）。")
    parser.add_argument("--frames", type=int, default=200, help="测量帧数，0 表示不按帧数停止。")
    parser.add_argument("--duration", type=float, default=0.0, help="测量时长（秒），0 表示不按时长停止。")
    parser.add_argument("--warmup", type=int, default=10, help="预热帧数，不计入统计。")
    parser.add_argument("--loop", action="store_true", help="视频播完后循环，用于长时间热测试。")
    parser.add_argument("--render", action="store_true", help="绘制骨架并显示窗口（默认 headless）。")
    parser.add_argument("--plot-only", action="store_true", help="只绘制不显示窗口，用于单独测量绘制开销。")
    parser.add_argument("--no-analyze", action="store_true", help="跳过 pose_analyzer 动作分析。")
    parser.add_argument("--queue-depth", type=int, default=2, help="队列最大长度。小值降延迟，大值抗抖动。")
    parser.add_argument("--sample-every", type=int, default=15, help="每隔多少帧采样一次温度/频率。")
    parser.add_argument("--tag", default="untagged", help="实验标签，会写入结果表，务必起有意义的名字。")
    parser.add_argument("--notes", default="", help="备注，写入结果表。")
    parser.add_argument("--results-dir", default="results", help="结果输出目录。")
    parser.add_argument("--dump-keypoints", default=None, help="导出逐帧关键点为 .npz，供精度对比使用。")
    parser.add_argument("--no-trace", action="store_true", help="不导出 per-frame 明细 CSV。")
    return parser.parse_args()


# --------------------------------------------------------------------------
# 共享状态
# --------------------------------------------------------------------------


class Shared:
    """线程间共享的计数器与计时缓冲。

    各 stage 的计时列表只由**对应的那一个线程**追加，因此不需要锁；
    只有 `done` / `loops` 这类跨线程读写的标量用锁保护。
    """

    def __init__(self) -> None:
        self.run_start = time.perf_counter()
        self.lock = threading.Lock()
        self.done = 0          # 已完成 analyze+render 的帧数
        self.loops = 0         # 视频循环播放次数
        self.detected = 0      # 有人检出的帧数
        self.interrupted = False

        # 各 stage 计时（毫秒），由各自线程追加
        self.t_capture: list[float] = []
        self.t_preprocess: list[float] = []
        self.t_inference: list[float] = []
        self.t_postprocess: list[float] = []
        self.t_analyze: list[float] = []
        self.t_render: list[float] = []
        self.t_latency: list[float] = []
        self.t_queue_capture: list[float] = []   # 推理线程等采集队列
        self.t_queue_render: list[float] = []    # 分析渲染线程等推理队列
        self.timing_fallback_frames = 0

        self.trace_rows: list[dict] = []
        self.keypoint_dump: list[np.ndarray] = []

    def bump_done(self) -> int:
        with self.lock:
            self.done += 1
            return self.done

    def snapshot_done(self) -> int:
        with self.lock:
            return self.done


# --------------------------------------------------------------------------
# 三个工作线程
# --------------------------------------------------------------------------


def capture_worker(cap, args, q_frames, stop, shared: Shared) -> None:
    """采集线程：只负责读帧，尽快把帧塞进队列。"""
    index = 0
    while not stop.is_set():
        if args.frames > 0 and index >= args.frames:
            break
        if args.duration > 0 and (time.perf_counter() - shared.run_start) >= args.duration:
            break

        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            if args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                with shared.lock:
                    shared.loops += 1
                continue
            break

        shared.t_capture.append((t1 - t0) * 1000.0)
        q_frames.put((index, frame, t0, (t1 - t0) * 1000.0))
        index += 1

    # 哨兵：通知下游没有更多帧了
    q_frames.put(None)


def infer_worker(model, args, q_frames, q_results, stop, shared: Shared) -> None:
    """推理线程：独占模型推理。这是流水线里最重的一级。"""
    while True:
        t_w0 = time.perf_counter()
        item = q_frames.get()
        t_w1 = time.perf_counter()
        queue_wait_infer = (t_w1 - t_w0) * 1000.0
        shared.t_queue_capture.append(queue_wait_infer)

        if item is None:
            q_results.put(None)
            break

        index, frame, t_captured, capture_ms = item

        t0 = time.perf_counter()
        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        result = results[0]
        t1 = time.perf_counter()

        q_results.put(
            (index, result, t_captured, (t1 - t0) * 1000.0, capture_ms, queue_wait_infer)
        )


def render_worker(analyzer, args, q_results, stop, shared: Shared, window_name) -> None:
    """分析+渲染线程：与 benchmark.py 的顺序一致，保证计数/关键点语义不变。"""
    render_enabled = args.render or args.plot_only

    while True:
        t_w0 = time.perf_counter()
        item = q_results.get()
        t_w1 = time.perf_counter()
        shared.t_queue_render.append((t_w1 - t_w0) * 1000.0)

        if item is None:
            break

        index, result, t_captured, predict_ms, capture_ms, queue_wait_infer = item

        # ---- 分析 ----
        t0 = time.perf_counter()
        keypoints, primary_index, people_count = extract_primary_keypoints(result)
        if analyzer is not None:
            analyzer.analyze(keypoints, primary_index)
        t1 = time.perf_counter()

        # ---- 渲染 ----
        if render_enabled:
            annotated = result.plot()
            if args.render:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    shared.interrupted = True
                    stop.set()
        t2 = time.perf_counter()

        # ---- ultralytics 内建三段计时，缺失时回退为 predict 墙钟 ----
        speed = getattr(result, "speed", {}) or {}
        pre_ms = float(speed.get("preprocess", 0.0) or 0.0)
        inf_ms = float(speed.get("inference", 0.0) or 0.0)
        post_ms = float(speed.get("postprocess", 0.0) or 0.0)
        fallback_used = False
        if inf_ms <= 0.0:
            residual = predict_ms - pre_ms - post_ms
            inf_ms = residual if residual > 0 else predict_ms
            fallback_used = True
            with shared.lock:
                shared.timing_fallback_frames += 1

        ana_ms = (t1 - t0) * 1000.0
        ren_ms = (t2 - t1) * 1000.0
        latency_ms = (t2 - t_captured) * 1000.0

        shared.t_preprocess.append(pre_ms)
        shared.t_inference.append(inf_ms)
        shared.t_postprocess.append(post_ms)
        shared.t_analyze.append(ana_ms)
        shared.t_render.append(ren_ms)
        shared.t_latency.append(latency_ms)

        if people_count > 0:
            with shared.lock:
                shared.detected += 1

        if args.dump_keypoints:
            buffer = np.full((17, 3), -1.0, dtype=np.float32)
            for slot, point in enumerate(keypoints[:17]):
                buffer[slot] = (point.x, point.y, point.conf)
            shared.keypoint_dump.append(buffer)

        if not args.no_trace:
            shared.trace_rows.append(
                {
                    "frame": index,
                    "elapsed_s": round(t2 - shared.run_start, 4),
                    "capture_ms": round(capture_ms, 3),
                    "preprocess_ms": round(pre_ms, 3),
                    "inference_ms": round(inf_ms, 3),
                    "postprocess_ms": round(post_ms, 3),
                    "analyze_ms": round(ana_ms, 3),
                    "render_ms": round(ren_ms, 3),
                    "total_ms": round(latency_ms, 3),
                    "fps_instant": round(1000.0 / latency_ms, 2) if latency_ms > 0 else "",
                    "queue_wait_infer_ms": round(queue_wait_infer, 3),
                    "queue_wait_render_ms": round(shared.t_queue_render[-1], 3),
                    "latency_ms": round(latency_ms, 3),
                    "people": people_count,
                    "temp_c": "",
                    "freq_mhz": "",
                    "timing_fallback": int(fallback_used),
                }
            )

        shared.bump_done()


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

    if args.queue_depth < 1:
        print("[ERROR] --queue-depth 至少为 1。")
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
    print(f"  线程     : {args.threads or '默认'}   渲染: {args.render or args.plot_only}")
    print(f"  流水线   : capture -> inference -> analyze+render   队列深度 {args.queue_depth}")
    print("=" * 68)
    print("  口径提醒：fps_e2e_mean 为吞吐口径；total_ms 为单帧延迟（会被队列放大）。")

    from ultralytics import YOLO

    model = YOLO(args.model)
    analyzer = None if args.no_analyze else PoseAnalyzer()

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {source}")
        return 1

    # ---- 预热：首帧含图构建/内存分配，必须排除 ----
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
    print(f"预热完成 ({args.warmup} 帧)，启动流水线...\n")

    shared = Shared()
    shared.run_start = time.perf_counter()
    stop = threading.Event()

    import queue as queue_module

    q_frames: Any = queue_module.Queue(maxsize=args.queue_depth)
    q_results: Any = queue_module.Queue(maxsize=args.queue_depth)
    window_name = f"pipeline::{args.tag}"

    threads = [
        threading.Thread(target=capture_worker, args=(cap, args, q_frames, stop, shared), name="capture", daemon=True),
        threading.Thread(target=infer_worker, args=(model, args, q_frames, q_results, stop, shared), name="inference", daemon=True),
        threading.Thread(target=render_worker, args=(analyzer, args, q_results, stop, shared, window_name), name="render", daemon=True),
    ]
    for thread in threads:
        thread.start()

    # ---- 主线程：采样温度/频率 + 打进度 ----
    temps: list[float] = []
    freqs: list[float] = []
    next_sample = args.sample_every
    last_reported = 0

    try:
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.05)
            done = shared.snapshot_done()

            while done >= next_sample:
                temp_c = read_cpu_temp_c()
                freq_mhz = read_cpu_freq_mhz()
                if temp_c is not None:
                    temps.append(temp_c)
                if freq_mhz is not None:
                    freqs.append(freq_mhz)
                next_sample += args.sample_every

            if done >= last_reported + 50:
                last_reported = done
                if shared.t_latency:
                    recent = shared.t_latency[-50:]
                    throughput = 50.0 / (sum(recent) / 1000.0) if sum(recent) > 0 else 0.0
                    temp_note = f"  {temps[-1]:.1f}C" if temps else ""
                    print(f"  [{done:5d}] 近 50 帧吞吐 {throughput:5.1f} FPS{temp_note}")

            if stop.is_set():
                break
    except KeyboardInterrupt:
        shared.interrupted = True
        print("\n[中断] 用户手动停止，已测数据仍会保存。")
        stop.set()

    # 等流水线排空（render 线程收到哨兵后自然退出）
    for thread in threads:
        thread.join(timeout=10.0)

    stop.set()
    cap.release()
    if args.render:
        cv2.destroyAllWindows()

    if not shared.t_latency:
        print("[ERROR] 没有测到任何帧。")
        return 1

    wall_seconds = time.perf_counter() - shared.run_start
    frames_measured = len(shared.t_latency)

    stages = {
        "capture": summarize(shared.t_capture),
        "preprocess": summarize(shared.t_preprocess),
        "inference": summarize(shared.t_inference),
        "postprocess": summarize(shared.t_postprocess),
        "analyze": summarize(shared.t_analyze),
        "render": summarize(shared.t_render),
        "total": summarize(shared.t_latency),
    }

    # 吞吐口径：真实产出能力
    fps_throughput = frames_measured / wall_seconds if wall_seconds > 0 else None
    # 最差 5% 帧：用延迟 P95 反推，与 benchmark.py 的 worst5 语义一致
    latency_stat = stages["total"]
    fps_worst5 = 1000.0 / latency_stat["p95"] if latency_stat and latency_stat["p95"] > 0 else None
    infer_stat = stages["inference"]
    fps_infer = 1000.0 / infer_stat["mean"] if infer_stat and infer_stat["mean"] > 0 else None

    timestamp = datetime.now()
    throttled_end = read_throttled_flag()

    record = {
        "tag": args.tag,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "notes": args.notes,
        "interrupted": shared.interrupted,
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
            "loops_completed": shared.loops,
            "queue_depth": args.queue_depth,
            "pipeline": True,
        },
        "system": system_info,
        "measurement": {
            "frames": frames_measured,
            "wall_seconds": round(wall_seconds, 3),
            "detect_rate": round(shared.detected / frames_measured, 4),
            "timing_fallback_frames": shared.timing_fallback_frames,
        },
        "stages_ms": stages,
        "pipeline": {
            "throughput_fps": round(fps_throughput, 2) if fps_throughput else None,
            "latency_ms": stages["total"],
            "queue_wait_infer_ms": summarize(shared.t_queue_capture),
            "queue_wait_render_ms": summarize(shared.t_queue_render),
            "bottleneck_stage": max(
                (("capture", stages["capture"]), ("inference", stages["inference"]), ("render", stages["render"])),
                key=lambda item: item[1]["mean"] if item[1] else 0.0,
            )[0],
        },
        "fps": {
            "end_to_end_mean": round(fps_throughput, 2) if fps_throughput else None,
            "end_to_end_worst5": round(fps_worst5, 2) if fps_worst5 else None,
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
    if shared.trace_rows:
        trace_path = results_dir / "runs" / f"{safe_tag}_{stamp}_trace.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(shared.trace_rows[0].keys()))
            writer.writeheader()
            writer.writerows(shared.trace_rows)

    kpt_path = None
    if args.dump_keypoints and shared.keypoint_dump:
        kpt_path = Path(args.dump_keypoints)
        kpt_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            kpt_path,
            keypoints=np.stack(shared.keypoint_dump),
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
                "render": int(args.render or args.plot_only),
                "analyze": int(not args.no_analyze),
                "frames": frames_measured,
                "fps_e2e_mean": round(fps_throughput, 2) if fps_throughput else "",
                "fps_e2e_worst5": round(fps_worst5, 2) if fps_worst5 else "",
                "fps_infer_only": round(fps_infer, 2) if fps_infer else "",
                "capture_ms": fmt_ms(stages["capture"]),
                "preprocess_ms": fmt_ms(stages["preprocess"]),
                "inference_ms": fmt_ms(stages["inference"]),
                "postprocess_ms": fmt_ms(stages["postprocess"]),
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
    print(f"  {'阶段':<14}{'均值ms':>10}{'中位ms':>10}{'P95ms':>10}{'占比*':>10}")
    print("-" * 68)
    latency_mean = latency_stat["mean"]
    for name, label in [
        ("capture", "采集"),
        ("preprocess", "预处理"),
        ("inference", "推理"),
        ("postprocess", "后处理"),
        ("analyze", "动作分析"),
        ("render", "渲染"),
    ]:
        stat = stages[name]
        if stat is None:
            continue
        share = stat["mean"] / latency_mean * 100 if latency_mean > 0 else 0
        print(
            f"  {label:<14}{stat['mean']:>10.2f}{stat['median']:>10.2f}"
            f"{stat['p95']:>10.2f}{share:>9.1f}%"
        )
    print("-" * 68)
    print(f"  {'单帧延迟':<14}{latency_mean:>10.2f}{latency_stat['median']:>10.2f}{latency_stat['p95']:>10.2f}")
    print("=" * 68)
    print(f"  吞吐 FPS        : {fps_throughput:.2f}   <- 流水线真实产出能力")
    print(f"  最差 5% 帧      : {fps_worst5:.2f}   (按延迟 P95 反推)")
    if fps_infer:
        print(f"  纯推理 FPS      : {fps_infer:.2f}   <- 报告里不要只写这个数")
    print(f"  瓶颈 stage      : {record['pipeline']['bottleneck_stage']}")
    for label, key in (("推理等采集队列", "queue_wait_infer_ms"), ("渲染等推理队列", "queue_wait_render_ms")):
        stat = record["pipeline"][key]
        if stat:
            print(f"  {label:<14}: {stat['mean']:.2f} ms (P95 {stat['p95']:.2f})")
    print(f"  有人检出率      : {record['measurement']['detect_rate'] * 100:.1f}%")
    if temps:
        print(f"  温度            : {temps[0]:.1f}C -> {temps[-1]:.1f}C  峰值 {max(temps):.1f}C")
    if freqs:
        print(f"  CPU 频率        : {min(freqs):.0f} - {max(freqs):.0f} MHz")
    if shared.timing_fallback_frames:
        print(
            f"  [!] 分项计时回退 : {shared.timing_fallback_frames} 帧未提供内建计时，"
            f"推理耗时改用 predict() 墙钟时间"
        )
    if throttled_end and throttled_end != "0x0":
        print(f"  [!] 降频标志    : {throttled_end}  <- 发生过降频，本次数据需在报告中标注")
    print("=" * 68)
    print("  * 占比 = 该 stage 均值 / 单帧延迟均值。流水线下各 stage 之和会大于单帧延迟，")
    print("    因为不同帧的不同 stage 在并行执行。判断瓶颈请看上面的「瓶颈 stage」。")
    print(f"  明细 JSON : {json_path}")
    if trace_path:
        print(f"  逐帧 trace: {trace_path}")
    if kpt_path:
        print(f"  关键点    : {kpt_path}")
    print(f"  汇总总表  : {summary_path}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
