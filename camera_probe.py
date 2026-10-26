"""摄像头链路实测脚手架（独立脚本）。

⚠️ 本脚本与 benchmark.py 的分工，是一条不可越过的界线：

    benchmark.py      —— 固定视频文件的基准测试，结果写 results/benchmark_results.csv。
                         那份 CSV 是报告全部性能数据的唯一权威来源，**摄像头数据绝不可混入**。
    camera_probe.py   —— 本文。只测摄像头链路，结果写 results/camera_probe_results.csv。

之所以必须先有这一步：项目 P0–P8 的全部数据都是「读固定视频文件 + 逐帧处理」的吞吐，
**不含采集**。设备化的第一个未知就是摄像头本身能给多少帧、read() 会阻塞多久、
缓冲会不会积压延迟。这些在视频文件上永远测不出来。

三种模式（一次运行先测采集再测端到端，自动给出瓶颈判断）：

    probe   只 read 不推理 —— 回答「相机到底能给多少帧」，这是上限。
    e2e     采集 + 推理     —— 回答「整条链路能不能实时」。
    both    两段连着跑（默认）—— 用同一台相机、同一次上电，两组数字才可比。

为什么必须连着跑而不是分两次：相机的自动曝光、自动增益在打开后几秒内一直在调整，
分两次运行等于换了实验条件。

典型用法：

    # 树莓派：先问相机能给多少帧，再看整条链路
    python camera_probe.py --probe-frames 300 --frames 300 --tag pi5_cam_640x480

    # 不确定该用什么格式/分辨率时，先扫一遍
    python camera_probe.py --mode matrix --tag pi5_cam_matrix

    # 拿到 NCNN 模型后的标准测法
    python camera_probe.py --model yolov8n-pose_ncnn_model \\
        --width 640 --height 480 --probe-frames 300 --frames 300 --tag pi5_cam_ncnn_640

    # 想验证「不渲染」到底省多少（注意：窗口一出现就会污染推理计时，见 P7/P8 结论）
    python camera_probe.py --model yolov8n-pose_ncnn_model --render --frames 300 --tag pi5_cam_render

关键输出字段说明：

    capture_share_pct  read() 耗时占整个循环的百分比。
                       > 60% 说明大部分时间在「等相机」，算力有余；
                       < 30% 说明相机在「等我们」，处理能力是瓶颈。
    fps_e2e / fps_probe 的比值才是判断实时的真依据，单独看任一个都会误判。

已知的坑（写在这里省得重踩）：

    1. OpenCV 的 cap.read() 耗时包含「等下一帧」的阻塞时间，它不是一个纯解码耗时。
       处理越快，read 阻塞越久 —— 所以 capture_ms 高不等于采集慢。
    2. CAP_PROP_BUFFERSIZE 必须设成 1。默认缓冲会让 read 拿到几百毫秒前的旧帧，
       表现为「帧率正常但动作延迟明显」。--buffersize 可对比。
    3. FOURCC 必须在设分辨率之前设，否则部分 UVC 相机不生效。
    4. 相机报的 CAP_PROP_FPS 经常是它「声称」的值，不是实测值。以 fps_probe 为准。
    5. 低光照下 MJPG 相机会自动降帧并重复上一帧。--detect-dup 可检出，
       这类重复帧不改变关节角度，对计数无害，但会让「帧率」虚高。
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2

# 复用 benchmark.py 的环境探测与统计，避免同一套逻辑写两遍走偏。
# 只 import 工具函数，不触发它的 argparse / main。
from benchmark import (
    detect_device_name,
    read_cpu_freq_mhz,
    read_cpu_temp_c,
    read_throttled_flag,
    summarize,
)

CAMERA_COLUMNS = [
    "tag",
    "timestamp",
    "device",
    "mode",
    "backend",
    "model",
    "imgsz",
    "conf",
    "width",
    "height",
    "fourcc",
    "buffersize",
    "api",
    "cam_fps_reported",
    "frames",
    "duration_s",
    "fps_probe",
    "fps_e2e",
    "fps_e2e_worst5",
    "capture_ms_mean",
    "capture_ms_median",
    "capture_ms_p95",
    "inference_ms_mean",
    "inference_ms_p95",
    "render_ms_mean",
    "analyze_ms_mean",
    "total_ms_mean",
    "total_ms_p95",
    "capture_share_pct",
    "dup_frame_pct",
    "detect_rate",
    "temp_start_c",
    "temp_max_c",
    "freq_min_mhz",
    "throttled_end",
    "verdict",
    "notes",
]

FOURCC_CHOICES = ["MJPG", "YUYV", "YUY2", "none"]


# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="摄像头链路实测：采集上限 / 端到端延迟 / 瓶颈归属。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["probe", "e2e", "both", "matrix"], default="both",
                        help="probe=只采集；e2e=采集+推理；both=先采集再端到端；matrix=扫格式×分辨率。")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引。")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="模型路径：.pt / *_ncnn_model")
    parser.add_argument("--imgsz", type=int, default=320, help="推理输入边长（NCNN 固定为导出时尺寸）。")
    parser.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值。")
    parser.add_argument("--threads", type=int, default=0, help="CPU 线程数，0 表示不干预。")

    parser.add_argument("--width", type=int, default=640, help="请求采集宽度。")
    parser.add_argument("--height", type=int, default=480, help="请求采集高度。")
    parser.add_argument("--fourcc", default="MJPG", choices=FOURCC_CHOICES,
                        help="请求的像素格式。MJPG 解码便宜，YUYV 带宽大但有相机不支持。")
    parser.add_argument("--buffersize", type=int, default=1, help="采集缓冲深度，1 表示不积压延迟。")
    parser.add_argument("--api", default="auto",
                        choices=["auto", "v4l2", "dshow", "msmf", "any"],
                        help="OpenCV 后端。Linux 用 v4l2，Windows 用 dshow。")

    parser.add_argument("--probe-frames", type=int, default=300, help="采集段测量帧数。")
    parser.add_argument("--frames", type=int, default=300, help="端到端段测量帧数。")
    parser.add_argument("--duration", type=float, default=0.0, help="按时长停止（秒），0 表示按帧数。")
    parser.add_argument("--warmup", type=int, default=30, help="预热帧数：等自动曝光收敛，不计入统计。")
    parser.add_argument("--gap", type=int, default=5, help="两段之间丢弃的过渡帧数。")

    parser.add_argument("--render", action="store_true",
                        help="绘制并显示窗口。⚠️ 会严重污染推理计时（P7/P8 已证明），默认关闭。")
    parser.add_argument("--display-every", type=int, default=1, help="每 N 帧绘制一次（仅 --render 时有效）。")
    parser.add_argument("--no-analyze", action="store_true", help="跳过 pose_analyzer 动作分析。")
    parser.add_argument("--detect-dup", action="store_true",
                        help="检测重复帧（推断掉帧/自动降帧）。会额外消耗约 0.3 ms/帧。")

    parser.add_argument("--sample-every", type=int, default=15, help="每隔多少帧采样温度/频率。")
    parser.add_argument("--tag", default="cam_untagged", help="实验标签，务必起有意义的名字。")
    parser.add_argument("--notes", default="", help="备注，写入结果表。")
    parser.add_argument("--results-dir", default="results", help="结果输出目录。")
    parser.add_argument("--no-trace", action="store_true", help="不导出逐帧明细。")
    return parser.parse_args()


# --------------------------------------------------------------------------
# 相机工具
# --------------------------------------------------------------------------


def resolve_api(name: str) -> int:
    """把 --api 映射成 OpenCV 常量。取不到就退回 CAP_ANY。"""
    table = {
        "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        "any": cv2.CAP_ANY,
    }
    if name != "auto":
        return table[name]
    system = platform.system()
    if system == "Linux":
        return table["v4l2"]
    if system == "Windows":
        return table["dshow"]
    return cv2.CAP_ANY


def fourcc_to_str(value: float) -> str:
    code = int(value or 0)
    if code <= 0:
        return "----"
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in chars)


def fourcc_matches(requested: str, actual: str | None) -> bool:
    """相机经常不认请求的格式并静默回落（实测 PC 上请求 MJPG 回读成 YUY2）。
    YUYV / YUY2 是同一格式的两种写法，视为等价。"""
    if requested == "none" or not actual:
        return True
    alias = {"YUY2": "YUYV"}
    return alias.get(actual, actual) == alias.get(requested, requested)


def open_camera(args: argparse.Namespace, fourcc: str | None = None) -> tuple[cv2.VideoCapture, dict]:
    """打开相机并回读真实参数。fourcc 允许覆盖命令行值（matrix 模式用）。"""
    api = resolve_api(args.api)
    cap = cv2.VideoCapture(args.camera, api)
    info = {"api": args.api if args.api != "auto" else _api_name(api)}

    if not cap.isOpened():
        return cap, info

    wanted_fourcc = args.fourcc if fourcc is None else fourcc
    # 坑 3：FOURCC 要在设分辨率之前设，否则部分 UVC 相机不生效。
    if wanted_fourcc != "none":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*wanted_fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffersize)

    info.update(
        {
            "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "cam_fps_reported": round(cap.get(cv2.CAP_PROP_FPS), 2),
        }
    )
    return cap, info


def _api_name(api: int) -> str:
    for name in ("CAP_V4L2", "CAP_DSHOW", "CAP_MSMF", "CAP_GSTREAMER", "CAP_ANY"):
        if getattr(cv2, name, -1) == api:
            return name.replace("CAP_", "").lower()
    return str(api)


def probe_device_summary(args: argparse.Namespace, fourcc: str | None = None) -> dict:
    """只回答一个问题：这台相机、这个配置下，到底能给多少帧。不加载模型。"""
    cap, info = open_camera(args, fourcc)
    if not cap.isOpened():
        return {**info, "ok": False, "fps_probe": 0.0, "frames": 0}

    # 预热：自动曝光/自动增益收敛需要 1-2 秒，不预热测出来的帧率偏高且不稳。
    for _ in range(max(0, args.warmup)):
        if not cap.read()[0]:
            break

    intervals: list[float] = []
    capture_ms: list[float] = []
    dup_flags: list[bool] = []
    prev_small = None
    started = time.perf_counter()
    wall = 0.0
    frames = 0

    try:
        while args.probe_frames == 0 or frames < args.probe_frames:
            if args.duration > 0 and (time.perf_counter() - started) >= args.duration:
                break
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                break
            frames += 1
            capture_ms.append((t1 - t0) * 1000.0)
            intervals.append(t1 - t0)
            if args.detect_dup:
                small = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
                if prev_small is not None:
                    dup_flags.append(float(cv2.absdiff(prev_small, small).mean()) < 0.5)
                prev_small = small
    finally:
        # 坑：cap.release() 在 UVC 相机上是慢操作（几十到上百毫秒），
        # 必须在 release 之前掐表，否则 fps 会被系统性低估。
        wall = time.perf_counter() - started
        cap.release()

    fps_wall = frames / wall if wall > 0 else 0.0
    stat = summarize(capture_ms)
    dup_pct = (sum(dup_flags) / len(dup_flags) * 100.0) if dup_flags else 0.0

    return {
        **info,
        "ok": frames > 0,
        "frames": frames,
        "fps_probe": round(fps_wall, 2),
        # 这里用帧间隔的中位数倒数作交叉校验：wall 计算受首尾边界影响，中位数更稳。
        "fps_probe_median": round(1000.0 / statistics.median(capture_ms), 2) if capture_ms else 0.0,
        "capture_ms_mean": round(stat["mean"], 2) if stat else 0.0,
        "capture_ms_median": round(stat["median"], 2) if stat else 0.0,
        "capture_ms_p95": round(stat["p95"], 2) if stat else 0.0,
        "dup_frame_pct": round(dup_pct, 2),
    }


# --------------------------------------------------------------------------
# 端到端
# --------------------------------------------------------------------------


def run_e2e(args: argparse.Namespace) -> dict:
    """采集 + 推理（+可选渲染）。分项计时，口径与 benchmark.py 对齐。"""
    from pose_analyzer import PoseAnalyzer, extract_primary_keypoints

    model = None
    analyzer = None
    if not args.no_analyze:
        analyzer = PoseAnalyzer(min_conf=0.05)

    from ultralytics import YOLO

    model = YOLO(args.model)
    if "ncnn" in str(args.model).lower():
        print("  [提示] NCNN 输入尺寸在导出时固定，--imgsz 对它无效。")

    cap, info = open_camera(args)
    if not cap.isOpened():
        return {**info, "ok": False, "fps_e2e": 0.0, "frames": 0}

    temp_max = read_cpu_temp_c()
    freq_min = read_cpu_freq_mhz()
    throttled = None

    # 预热同时跑推理：让 ultralytics / ncnn 完成首次图构建，否则第一帧会计入统计。
    for _ in range(max(0, args.warmup)):
        ok, frame = cap.read()
        if not ok:
            break
        model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

    frames = 0
    started = time.perf_counter()
    wall = 0.0
    samples: list[dict] = []
    detect_hits = 0

    try:
        while args.frames == 0 or frames < args.frames:
            if args.duration > 0 and (time.perf_counter() - started) >= args.duration:
                break

            loop_start = time.perf_counter()

            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                print("  [中断] 读取摄像头失败。")
                break
            capture_ms = (t1 - t0) * 1000.0

            t2 = time.perf_counter()
            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            t3 = time.perf_counter()
            inference_ms = (t3 - t2) * 1000.0
            result = results[0]

            analyze_ms = 0.0
            analysis = None
            if analyzer is not None:
                t4 = time.perf_counter()
                keypoints, primary_index, _ = extract_primary_keypoints(result)
                analysis = analyzer.analyze(keypoints, primary_index)
                analyze_ms = (time.perf_counter() - t4) * 1000.0

            if len(result.boxes) > 0:
                detect_hits += 1

            render_ms = 0.0
            if args.render and frames % max(1, args.display_every) == 0:
                from pose_analyzer import draw_analysis_overlay

                t5 = time.perf_counter()
                annotated = result.plot()
                if analysis is not None:
                    draw_analysis_overlay(annotated, analysis)
                cv2.imshow("camera_probe", annotated)
                cv2.waitKey(1)
                render_ms = (time.perf_counter() - t5) * 1000.0

            loop_end = time.perf_counter()
            frames += 1
            samples.append(
                {
                    "frame": frames,
                    "t_rel_s": round(loop_start - started, 4),
                    "capture_ms": round(capture_ms, 3),
                    "inference_ms": round(inference_ms, 3),
                    "analyze_ms": round(analyze_ms, 3),
                    "render_ms": round(render_ms, 3),
                    "total_ms": round((loop_end - loop_start) * 1000.0, 3),
                }
            )

            if args.sample_every > 0 and frames % args.sample_every == 0:
                temp = read_cpu_temp_c()
                freq = read_cpu_freq_mhz()
                if temp is not None:
                    temp_max = temp if temp_max is None else max(temp_max, temp)
                if freq is not None:
                    freq_min = freq if freq_min is None else min(freq_min, freq)

            if frames % 100 == 0:
                recent = samples[-100:]
                mean_ms = sum(s["total_ms"] for s in recent) / len(recent)
                print(f"  帧 {frames}: 近 100 帧均值 {mean_ms:.2f} ms ({1000.0 / mean_ms:.2f} FPS)")
    finally:
        # 同 probe：release 之前掐表，否则帧率被系统性低估。
        wall = time.perf_counter() - started
        cap.release()
        if args.render:
            cv2.destroyAllWindows()
        throttled = read_throttled_flag()

    total_stats = summarize([s["total_ms"] for s in samples])
    capture_stats = summarize([s["capture_ms"] for s in samples])
    infer_stats = summarize([s["inference_ms"] for s in samples])
    analyze_stats = summarize([s["analyze_ms"] for s in samples])
    render_stats = summarize([s["render_ms"] for s in samples])
    fps_e2e = frames / wall if wall > 0 else 0.0

    # worst5：最差的 5% 帧对应的帧率。报告必须报这一项 —— 用户感受到的是尾延迟。
    ordered = sorted([s["total_ms"] for s in samples], reverse=True)
    worst_count = max(1, int(round(0.05 * len(ordered)))) if ordered else 0
    worst_ms = sum(ordered[:worst_count]) / worst_count if worst_count else 0.0

    capture_total = sum(s["capture_ms"] for s in samples)
    loop_total = sum(s["total_ms"] for s in samples)
    capture_share = (capture_total / loop_total * 100.0) if loop_total > 0 else 0.0

    return {
        **info,
        "ok": frames > 0,
        "frames": frames,
        "duration_s": round(wall, 3),
        "fps_e2e": round(fps_e2e, 2),
        "fps_e2e_worst5": round(1000.0 / worst_ms, 2) if worst_ms > 0 else 0.0,
        "inference_ms_mean": round(infer_stats["mean"], 2) if infer_stats else 0.0,
        "inference_ms_p95": round(infer_stats["p95"], 2) if infer_stats else 0.0,
        "render_ms_mean": round(render_stats["mean"], 2) if render_stats else 0.0,
        "analyze_ms_mean": round(analyze_stats["mean"], 2) if analyze_stats else 0.0,
        "total_ms_mean": round(total_stats["mean"], 2) if total_stats else 0.0,
        "total_ms_p95": round(total_stats["p95"], 2) if total_stats else 0.0,
        "capture_ms_mean": round(capture_stats["mean"], 2) if capture_stats else 0.0,
        "capture_ms_median": round(capture_stats["median"], 2) if capture_stats else 0.0,
        "capture_ms_p95": round(capture_stats["p95"], 2) if capture_stats else 0.0,
        "capture_share_pct": round(capture_share, 1),
        "detect_rate": round(detect_hits / frames, 4) if frames else 0.0,
        "temp_max_c": temp_max,
        "freq_min_mhz": freq_min,
        "throttled_end": throttled,
        "_samples": samples,
    }


# --------------------------------------------------------------------------
# 判断（把数字翻成人话，跑完直接看结论）
# --------------------------------------------------------------------------


def verdict_lines(fps_probe: float, fps_e2e: float | None, process_ms: float | None = None,
                  cam_reported: float | None = None) -> tuple[str, list[str]]:
    """返回 (单行结论, 详细说明)。

    判断口径的演进（这一条踩过坑，记下来）：

      第一版用 capture_share_pct > 60% 判「算力有余」。错了 —— 相机本身慢的时候，
      read() 一样会阻塞很久，capture 占比照样高，但结论正好反了。

      第二版用 e2e_fps / probe_fps 的比值。也不可靠 —— 实测同一颗 PC 摄像头在
      probe 段和 e2e 段之间帧率从 13.4 漂到 9.8 FPS（自动曝光 + USB 带宽竞争），
      比值算出来的「瓶颈」其实是相机漂移，不是处理能力。

      现在这版：拿「不含采集的处理耗时」直接对标「相机帧周期」。
      这两个量各自独立测量、互不牵连，才是真正的物理判据：
          处理耗时 < 相机周期  →  每次都抢在下一帧到达前算完，算力有余
          处理耗时 > 相机周期  →  必然积压，处理就是瓶颈
    比值和 capture_share 仍保留，但只作旁证，不单独下结论。
    """
    lines: list[str] = []
    if fps_probe <= 0:
        return "相机未采到任何帧", ["检查 --camera 索引、--api（Linux 用 v4l2 / Windows 用 dshow）、是否被别的进程占用"]

    if cam_reported and cam_reported > 0:
        lines.append(f"相机自报 {cam_reported:.1f} FPS，实测 {fps_probe:.1f} FPS"
                     f"（差 {fps_probe - cam_reported:+.1f}）")
    else:
        lines.append("相机未上报 FPS（CAP_PROP_FPS = -1），只能以实测值为准")

    if fps_e2e is None:
        head = f"仅采集：相机上限 {fps_probe:.1f} FPS"
        return head, lines

    cam_period = 1000.0 / fps_probe
    ratio = fps_e2e / fps_probe if fps_probe > 0 else 0.0

    if process_ms and process_ms > 0:
        headroom = cam_period / process_ms
        lines.append(f"相机帧周期 {cam_period:.1f} ms vs 处理耗时 {process_ms:.1f} ms"
                     f"（余量 {headroom:.2f}×）")
        if headroom >= 1.15:
            head = f"瓶颈在相机：处理仅需 {process_ms:.1f} ms，相机 {cam_period:.1f} ms 才给一帧"
            lines.append("→ 算力有余。可以上调推理分辨率/换更大模型，或降低采集分辨率省带宽。")
        elif headroom >= 0.85:
            head = f"接近打平：处理 {process_ms:.1f} ms ≈ 相机周期 {cam_period:.1f} ms"
            lines.append("→ 没有余量。任何抖动都会丢帧，建议留 30% 以上余量再上线。")
        else:
            head = f"瓶颈在处理：处理需 {process_ms:.1f} ms，相机 {cam_period:.1f} ms 就给一帧"
            lines.append("→ 处理能力不足。首选降 imgsz 或换 NCNN 后端。")
    else:
        if ratio >= 0.95:
            head = f"瓶颈在相机：端到端 {fps_e2e:.1f} FPS 已追平采集上限 {fps_probe:.1f} FPS"
        elif ratio >= 0.75:
            head = f"基本持平：端到端 {fps_e2e:.1f} FPS / 采集上限 {fps_probe:.1f} FPS（{ratio * 100:.0f}%）"
        else:
            head = f"端到端 {fps_e2e:.1f} FPS 仅为采集上限 {fps_probe:.1f} FPS 的 {ratio * 100:.0f}%"

    if 0.0 < ratio < 1.05:
        lines.append(f"旁证：e2e/probe 比值 {ratio * 100:.0f}%。⚠️ 两段测量的相机状态可能不同"
                     "（自动曝光/带宽竞争会让相机自己漂），比值异常时先怀疑相机而非处理。")

    if fps_e2e < 30:
        lines.append(f"⚠️ 端到端 {fps_e2e:.1f} FPS 未达 30 FPS。但实时计数不需要 30 FPS —— "
                     "角度曲线采样够密，15–20 FPS 也能判对峰谷，代价是反馈延迟变明显。")
    return head, lines


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------


def append_row(path: Path, row: dict) -> None:
    """追加到独立结果表。绝不写 benchmark_results.csv。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAMERA_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CAMERA_COLUMNS})


def dump_trace(results_dir: Path, tag: str, samples: list[dict]) -> Path | None:
    if not samples:
        return None
    trace_dir = results_dir / "camera_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{tag}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)
    return path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    # --frames 0 / --probe-frames 0 表示「不按帧数停止」，只按时长停。
    # 两者都为 0 会变成无限循环，所以这里兜回默认值并提示。
    if args.duration <= 0:
        if args.probe_frames == 0:
            print("⚠️ --probe-frames 0 需要配合 --duration 使用，已回退为 300 帧。")
            args.probe_frames = 300
        if args.frames == 0:
            print("⚠️ --frames 0 需要配合 --duration 使用，已回退为 300 帧。")
            args.frames = 300

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "camera_probe_results.csv"
    device = detect_device_name()
    temp_start = read_cpu_temp_c()

    if args.mode == "matrix":
        return run_matrix(args, csv_path, device, temp_start)

    print(f"设备: {device}")
    print(f"模式: {args.mode}  相机索引: {args.camera}  API: {args.api}")
    print("=" * 64)

    probe: dict = {}
    if args.mode in ("probe", "both"):
        print(f"[1/2] 采集段：只 read 不推理，{args.probe_frames} 帧"
              f"（预热 {args.warmup} 帧用于自动曝光收敛）")
        probe = probe_device_summary(args)
        if not probe.get("ok"):
            print("  ❌ 相机打不开或采不到帧。")
            print(probe)
            return 1
        print(f"  实际配置: {probe.get('width')}x{probe.get('height')} "
              f"FOURCC={probe.get('fourcc')} 缓冲={args.buffersize}")
        if not fourcc_matches(args.fourcc, probe.get("fourcc")):
            print(f"  ⚠️ 请求 FOURCC={args.fourcc}，相机实际回落到 {probe.get('fourcc')}。"
                  "格式不生效会连带改变带宽与解码开销，报告里必须写明实际格式。")
        print(f"  实测采集 {probe['fps_probe']:.2f} FPS "
              f"（中位数口径 {probe.get('fps_probe_median')}）")
        print(f"  read() {probe['capture_ms_mean']:.2f} ms 均值 / "
              f"{probe['capture_ms_p95']:.2f} ms P95")
        if args.detect_dup:
            print(f"  重复帧占比 {probe.get('dup_frame_pct')}%")
        if args.mode == "both":
            # 两段之间丢几帧，避免上一段的缓冲状态影响下一段。
            time.sleep(0.1)

    e2e: dict = {}
    if args.mode in ("e2e", "both"):
        label = "[2/2]" if args.mode == "both" else "[1/1]"
        print(f"{label} 端到端段：采集 + 推理，{args.frames} 帧"
              f"（渲染{'开' if args.render else '关'}）")
        e2e = run_e2e(args)
        if not e2e.get("ok"):
            print("  ❌ 端到端段未采到帧。")
            return 1
        print(f"  端到端 {e2e['fps_e2e']:.2f} FPS（最差 5% 帧 {e2e['fps_e2e_worst5']} FPS）")
        print(f"  capture {e2e['capture_ms_mean']:.2f} ms / 推理 {e2e['inference_ms_mean']:.2f} ms / "
              f"分析 {e2e['analyze_ms_mean']:.2f} ms / 渲染 {e2e['render_ms_mean']:.2f} ms")
        process_ms_preview = (e2e["inference_ms_mean"] + e2e["analyze_ms_mean"] + e2e["render_ms_mean"])
        print(f"  处理耗时（不含采集）{process_ms_preview:.2f} ms，capture 占循环 {e2e['capture_share_pct']}%")

    fps_probe = probe.get("fps_probe", 0.0)
    fps_e2e = e2e.get("fps_e2e") if e2e else None
    process_ms = None
    if e2e:
        process_ms = e2e["inference_ms_mean"] + e2e["analyze_ms_mean"] + e2e["render_ms_mean"]
    head, detail = verdict_lines(
        fps_probe=fps_probe,
        fps_e2e=fps_e2e,
        process_ms=process_ms,
        cam_reported=probe.get("cam_fps_reported") or e2e.get("cam_fps_reported"),
    )

    print("=" * 64)
    print(f"结论: {head}")
    for line in detail:
        print(f"  {line}")

    row = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "mode": args.mode,
        "backend": "ncnn" if "ncnn" in str(args.model).lower() else "pytorch",
        "model": args.model if args.mode in ("e2e", "both") else "none",
        "imgsz": args.imgsz,
        "conf": args.conf,
        "width": probe.get("width") or e2e.get("width"),
        "height": probe.get("height") or e2e.get("height"),
        "fourcc": probe.get("fourcc") or e2e.get("fourcc"),
        "buffersize": args.buffersize,
        "api": probe.get("api") or e2e.get("api"),
        "cam_fps_reported": probe.get("cam_fps_reported") or e2e.get("cam_fps_reported"),
        "frames": e2e.get("frames") or probe.get("frames"),
        "duration_s": e2e.get("duration_s"),
        "fps_probe": round(fps_probe, 2),
        "fps_e2e": e2e.get("fps_e2e"),
        "fps_e2e_worst5": e2e.get("fps_e2e_worst5"),
        "capture_ms_mean": e2e.get("capture_ms_mean") if e2e else probe.get("capture_ms_mean"),
        "capture_ms_median": e2e.get("capture_ms_median") if e2e else probe.get("capture_ms_median"),
        "capture_ms_p95": e2e.get("capture_ms_p95") if e2e else probe.get("capture_ms_p95"),
        "inference_ms_mean": e2e.get("inference_ms_mean"),
        "inference_ms_p95": e2e.get("inference_ms_p95"),
        "render_ms_mean": e2e.get("render_ms_mean"),
        "analyze_ms_mean": e2e.get("analyze_ms_mean"),
        "total_ms_mean": e2e.get("total_ms_mean"),
        "total_ms_p95": e2e.get("total_ms_p95"),
        "capture_share_pct": e2e.get("capture_share_pct"),
        "dup_frame_pct": probe.get("dup_frame_pct"),
        "detect_rate": e2e.get("detect_rate"),
        "temp_start_c": temp_start,
        "temp_max_c": e2e.get("temp_max_c"),
        "freq_min_mhz": e2e.get("freq_min_mhz"),
        "throttled_end": e2e.get("throttled_end"),
        "verdict": head,
        "notes": args.notes,
    }
    append_row(csv_path, row)
    print(f"\n结果已追加: {csv_path}")

    if e2e.get("_samples") and not args.no_trace:
        trace = dump_trace(results_dir, args.tag, e2e["_samples"])
        if trace:
            print(f"逐帧明细: {trace}")

    print("\n报告写法提醒：")
    print("  1. 摄像头数据只能写进 camera_probe_results.csv，不可混入 benchmark_results.csv。")
    print("  2. 报 e2e 必须同时报 worst5 —— 均值达标尾延迟不达标是常态。")
    print("  3. 相机自报 FPS 不可信，以 fps_probe 为准。")
    return 0


def run_matrix(args: argparse.Namespace, csv_path: Path, device: str, temp_start: float | None) -> int:
    """扫 格式 × 分辨率，找出这台相机实际支持且最快的组合。"""
    resolutions = [(640, 480), (480, 360), (320, 240), (1280, 720)]
    fourcc_options = [args.fourcc] + [f for f in FOURCC_CHOICES if f != args.fourcc and f != "none"]

    print(f"设备: {device}")
    print("矩阵扫描：只测采集（不加载模型），每格 100 帧 + 20 帧预热")
    print("=" * 64)
    print(f"{'FOURCC':<8}{'分辨率':<12}{'实际':<12}{'实测FPS':<10}{'read均值':<10}{'备注'}")
    print("-" * 64)

    rows: list[dict] = []
    for fourcc in fourcc_options:
        for width, height in resolutions:
            probe_args = argparse.Namespace(**vars(args))
            probe_args.width = width
            probe_args.height = height
            probe_args.probe_frames = 100
            probe_args.warmup = 20
            probe_args.duration = 0.0
            probe_args.detect_dup = False
            result = probe_device_summary(probe_args, fourcc=fourcc)

            if not result.get("ok") or result.get("width", 0) == 0:
                print(f"{fourcc:<8}{f'{width}x{height}':<12}{'打不开':<12}{'-':<10}{'-':<10}")
                continue

            got = f"{result['width']}x{result['height']}"
            flags = []
            if (result["width"], result["height"]) != (width, height):
                flags.append("分辨率未采纳")
            if not fourcc_matches(fourcc, result.get("fourcc")):
                flags.append("格式回落")
            honoured = " / ".join(flags)
            print(f"{fourcc:<8}{f'{width}x{height}':<12}{got:<12}"
                  f"{result['fps_probe']:<10.1f}{result['capture_ms_mean']:<10.2f}{honoured}")

            rows.append({
                "tag": f"{args.tag}_{fourcc}_{width}x{height}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "device": device,
                "mode": "matrix",
                "backend": "none",
                "model": "none",
                "imgsz": args.imgsz,
                "conf": args.conf,
                "width": result["width"],
                "height": result["height"],
                "fourcc": result.get("fourcc"),
                "buffersize": args.buffersize,
                "api": result.get("api"),
                "cam_fps_reported": result.get("cam_fps_reported"),
                "frames": result["frames"],
                "fps_probe": result["fps_probe"],
                "capture_ms_mean": result.get("capture_ms_mean"),
                "capture_ms_median": result.get("capture_ms_median"),
                "capture_ms_p95": result.get("capture_ms_p95"),
                "temp_start_c": temp_start,
                "temp_max_c": read_cpu_temp_c(),
                "throttled_end": read_throttled_flag(),
                "verdict": honoured or "ok",
                "notes": args.notes,
            })

    for row in rows:
        append_row(csv_path, row)

    print("-" * 64)
    if rows:
        best = max(rows, key=lambda r: r["fps_probe"])
        print(f"最快组合: {best['fourcc']} {best['width']}x{best['height']} "
              f"→ {best['fps_probe']:.1f} FPS")
        print(f"建议下一步: python camera_probe.py --fourcc {best['fourcc']} "
              f"--width {best['width']} --height {best['height']} "
              f"--model yolov8n-pose_ncnn_model --probe-frames 300 --frames 300 --tag <你的标签>")
    print(f"\n矩阵结果已追加: {csv_path}  （共 {len(rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
