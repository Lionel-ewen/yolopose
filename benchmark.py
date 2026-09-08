"""边缘设备姿态估计性能基准测试脚手架。

设计原则（毕设全部实验的地基，改动前请三思）：

1. 输入固定  —— 只接受视频文件，禁止用摄像头做基准测试。每次实验输入完全一致，
                帧率才具备可比性。摄像头受光照、曝光、自动增益影响，帧率不可复现。
2. 分项计时  —— capture / preprocess / inference / postprocess / analyze / render
                分别统计，输出耗时分解表。只报一个总 FPS 是没有分析价值的。
3. 结果累积  —— 每次运行追加一行到 results/benchmark_results.csv 总表，
                所有实验横向可比，直接可以画图。
4. 零 I/O 污染 —— per-frame 明细全部在内存缓冲，运行结束后一次性落盘，
                避免逐帧写文件干扰帧率测量。
5. 环境留痕  —— 自动记录设备型号、CPU 温度、频率、降频状态、线程数、后端。
                树莓派降频会让同一份代码在第 10 秒和第 300 秒跑出不同帧率。

用法示例：

    # PC 基线
    python benchmark.py --source outputs/test_clip.mp4 --tag pc_pytorch_640

    # 树莓派 NCNN + 320
    python benchmark.py --source outputs/test_clip.mp4 \
        --model yolov8n-pose_ncnn_model --imgsz 320 --tag pi5_ncnn_320

    # 热稳定性测试（跑 10 分钟，循环播放）
    python benchmark.py --source outputs/test_clip.mp4 \
        --duration 600 --loop --tag pi5_thermal_soak

    # 导出关键点，供 compare_keypoints.py 做精度对比
    python benchmark.py --source outputs/test_clip.mp4 --imgsz 320 \
        --dump-keypoints results/kpts/imgsz320.npz --tag imgsz_320
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pose_analyzer import PoseAnalyzer, extract_primary_keypoints

SUMMARY_COLUMNS = [
    "tag",
    "timestamp",
    "device",
    "backend",
    "model",
    "imgsz",
    "conf",
    "threads",
    "render",
    "analyze",
    "frames",
    "fps_e2e_mean",
    "fps_e2e_worst5",
    "fps_infer_only",
    "capture_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "analyze_ms",
    "render_ms",
    "total_ms",
    "temp_start_c",
    "temp_max_c",
    "throttled",
    "detect_rate",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible pose-estimation benchmark for edge devices.",
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
    parser.add_argument("--sample-every", type=int, default=15, help="每隔多少帧采样一次温度/频率。")
    parser.add_argument("--tag", default="untagged", help="实验标签，会写入结果表，务必起有意义的名字。")
    parser.add_argument("--notes", default="", help="备注，写入结果表。")
    parser.add_argument("--results-dir", default="results", help="结果输出目录。")
    parser.add_argument("--dump-keypoints", default=None, help="导出逐帧关键点为 .npz，供精度对比使用。")
    parser.add_argument("--no-trace", action="store_true", help="不导出 per-frame 明细 CSV。")
    return parser.parse_args()


# --------------------------------------------------------------------------
# 系统环境探测（树莓派上有值，Windows/macOS 上优雅降级为 None）
# --------------------------------------------------------------------------


def _read_first_line(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.readline().strip().strip("\x00")
    except OSError:
        return None


def read_cpu_temp_c() -> float | None:
    raw = _read_first_line("/sys/class/thermal/thermal_zone0/temp")
    if raw is None:
        return None
    try:
        return int(raw) / 1000.0
    except ValueError:
        return None


def read_cpu_freq_mhz() -> float | None:
    raw = _read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if raw is None:
        return None
    try:
        return int(raw) / 1000.0
    except ValueError:
        return None


def read_throttled_flag() -> str | None:
    """树莓派降频状态。0x0 表示从未降频；非 0 说明电压不足或过热降频。"""
    try:
        output = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = output.stdout.strip()
    return text.split("=", 1)[1] if "=" in text else (text or None)


def detect_device_name() -> str:
    pi_model = _read_first_line("/sys/firmware/devicetree/base/model")
    if pi_model:
        return pi_model
    return f"{platform.system()} {platform.machine()}"


def detect_backend(model_path: str) -> str:
    lowered = model_path.lower()
    if lowered.endswith("_ncnn_model") or "ncnn" in lowered:
        return "ncnn"
    if lowered.endswith(".onnx"):
        return "onnx"
    if lowered.endswith(".xml"):
        return "openvino"
    if lowered.endswith(".tflite"):
        return "tflite"
    if lowered.endswith(".engine"):
        return "tensorrt"
    return "pytorch"


def collect_system_info(args: argparse.Namespace) -> dict:
    info = {
        "device": detect_device_name(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "cpu_count": None,
        "torch": None,
        "ultralytics": None,
        "cpu_freq_mhz_start": read_cpu_freq_mhz(),
        "temp_c_start": read_cpu_temp_c(),
        "throttled_start": read_throttled_flag(),
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------


def summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    count = len(ordered)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[min(count - 1, int(0.95 * (count - 1)))],
        "min": ordered[0],
        "max": ordered[-1],
        "count": count,
    }


def fmt_ms(stat: dict | None) -> str:
    if stat is None:
        return "-"
    return f"{stat['mean']:.2f}"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"[ERROR] 测试视频不存在: {source}")
        print("        先用 record_clip.py 录一段固定测试片段，或指定已有视频。")
        return 1

    if args.frames <= 0 and args.duration <= 0:
        print("[ERROR] --frames 和 --duration 至少要设一个大于 0。")
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
    print("=" * 68)

    from ultralytics import YOLO

    model = YOLO(args.model)
    analyzer = None if args.no_analyze else PoseAnalyzer()

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {source}")
        return 1

    # ---- 预热：首次推理包含图构建/内存分配开销，必须排除 ----
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

    # ---- 计时容器（全部内存缓冲，结束后统一落盘）----
    t_capture: list[float] = []
    t_preprocess: list[float] = []
    t_inference: list[float] = []
    t_postprocess: list[float] = []
    t_analyze: list[float] = []
    t_render: list[float] = []
    t_total: list[float] = []

    trace_rows: list[dict] = []
    keypoint_dump: list[np.ndarray] = []

    temps: list[float] = []
    freqs: list[float] = []
    detected_frames = 0
    loop_count = 0
    timing_fallback_frames = 0

    render_enabled = args.render or args.plot_only
    window_name = f"benchmark::{args.tag}"

    run_start = time.perf_counter()
    frame_index = 0
    interrupted = False

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

            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            result = results[0]
            m2 = time.perf_counter()

            keypoints, primary_index, people_count = extract_primary_keypoints(result)
            if analyzer is not None:
                analyzer.analyze(keypoints, primary_index)
            m3 = time.perf_counter()

            if render_enabled:
                annotated = result.plot()
                if args.render:
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        interrupted = True
                        break
            m4 = time.perf_counter()

            # ultralytics 自带的三段计时（毫秒），比自己拆更准确
            speed = getattr(result, "speed", {}) or {}
            pre_ms = float(speed.get("preprocess", 0.0) or 0.0)
            inf_ms = float(speed.get("inference", 0.0) or 0.0)
            post_ms = float(speed.get("postprocess", 0.0) or 0.0)

            # 部分后端（如 NCNN）可能不填充内建三段计时，导致推理耗时记为 0、
            # 耗时分解失真。此时回退为 predict() 调用的墙钟时间。
            # 注意：端到端 FPS 由 tot_ms 独立计算，不受此回退影响。
            fallback_used = False
            if inf_ms <= 0.0:
                wall_predict_ms = (m2 - m1) * 1000.0
                residual = wall_predict_ms - pre_ms - post_ms
                inf_ms = residual if residual > 0 else wall_predict_ms
                fallback_used = True
                timing_fallback_frames += 1

            cap_ms = (m1 - m0) * 1000.0
            ana_ms = (m3 - m2) * 1000.0
            ren_ms = (m4 - m3) * 1000.0
            tot_ms = (m4 - m0) * 1000.0

            t_capture.append(cap_ms)
            t_preprocess.append(pre_ms)
            t_inference.append(inf_ms)
            t_postprocess.append(post_ms)
            t_analyze.append(ana_ms)
            t_render.append(ren_ms)
            t_total.append(tot_ms)

            if people_count > 0:
                detected_frames += 1

            if args.dump_keypoints:
                buffer = np.full((17, 3), -1.0, dtype=np.float32)
                for slot, point in enumerate(keypoints[:17]):
                    buffer[slot] = (point.x, point.y, point.conf)
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
                        "elapsed_s": round(m4 - run_start, 4),
                        "capture_ms": round(cap_ms, 3),
                        "preprocess_ms": round(pre_ms, 3),
                        "inference_ms": round(inf_ms, 3),
                        "postprocess_ms": round(post_ms, 3),
                        "analyze_ms": round(ana_ms, 3),
                        "render_ms": round(ren_ms, 3),
                        "total_ms": round(tot_ms, 3),
                        "fps_instant": round(1000.0 / tot_ms, 2) if tot_ms > 0 else "",
                        "people": people_count,
                        "temp_c": "" if temp_c is None else round(temp_c, 1),
                        "freq_mhz": "" if freq_mhz is None else round(freq_mhz, 0),
                        "timing_fallback": int(fallback_used),
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

    stages = {
        "capture": summarize(t_capture),
        "preprocess": summarize(t_preprocess),
        "inference": summarize(t_inference),
        "postprocess": summarize(t_postprocess),
        "analyze": summarize(t_analyze),
        "render": summarize(t_render),
        "total": summarize(t_total),
    }

    total_stat = stages["total"]
    infer_stat = stages["inference"]
    fps_mean = 1000.0 / total_stat["mean"]
    fps_worst5 = 1000.0 / total_stat["p95"]
    fps_infer = 1000.0 / infer_stat["mean"] if infer_stat and infer_stat["mean"] > 0 else None

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
        },
        "system": system_info,
        "measurement": {
            "frames": len(t_total),
            "wall_seconds": round(wall_seconds, 3),
            "detect_rate": round(detected_frames / len(t_total), 4),
            "timing_fallback_frames": timing_fallback_frames,
        },
        "stages_ms": stages,
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

    if trace_rows:
        trace_path = results_dir / "runs" / f"{safe_tag}_{stamp}_trace.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trace_rows)
    else:
        trace_path = None

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
    else:
        kpt_path = None

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
                "frames": len(t_total),
                "fps_e2e_mean": round(fps_mean, 2),
                "fps_e2e_worst5": round(fps_worst5, 2),
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
    print(f"  耗时分解  ({len(t_total)} 帧, {wall_seconds:.1f} 秒)")
    print("-" * 68)
    print(f"  {'阶段':<14}{'均值ms':>10}{'中位ms':>10}{'P95ms':>10}{'占比':>10}")
    print("-" * 68)
    total_mean = total_stat["mean"]
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
        share = stat["mean"] / total_mean * 100 if total_mean > 0 else 0
        print(
            f"  {label:<14}{stat['mean']:>10.2f}{stat['median']:>10.2f}"
            f"{stat['p95']:>10.2f}{share:>9.1f}%"
        )
    print("-" * 68)
    print(f"  {'端到端':<14}{total_mean:>10.2f}{total_stat['median']:>10.2f}{total_stat['p95']:>10.2f}")
    print("=" * 68)
    print(f"  端到端 FPS      : {fps_mean:.2f}   (最差 5%: {fps_worst5:.2f})")
    if fps_infer:
        print(f"  纯推理 FPS      : {fps_infer:.2f}   <- 报告里不要只写这个数")
    print(f"  有人检出率      : {record['measurement']['detect_rate'] * 100:.1f}%")
    if temps:
        print(f"  温度            : {temps[0]:.1f}C -> {temps[-1]:.1f}C  峰值 {max(temps):.1f}C")
    if freqs:
        print(f"  CPU 频率        : {min(freqs):.0f} - {max(freqs):.0f} MHz")
    if timing_fallback_frames:
        print(
            f"  [!] 分项计时回退 : {timing_fallback_frames} 帧未提供内建计时，"
            f"推理耗时改用 predict() 墙钟时间（端到端 FPS 不受影响）"
        )
    if throttled_end and throttled_end != "0x0":
        print(f"  [!] 降频标志    : {throttled_end}  <- 发生过降频，本次数据需在报告中标注")
    print("=" * 68)
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
