"""多距离扫描分析：一把跑完四段录制，直接回答「148° 广角对姿态估计有多大影响」。

为什么不逐段手工跑
------------------
手工跑 `online_count_check.py` 四次，每次的阈值、窗宽、统计口径都容易漂；
跨距离的差异本来就只有几个百分点，口径不一致会直接淹没结论。本脚本对每段用
**完全相同的处理链路**，只让「距离」这一个变量变化。

它不只需要「计数准不准」——更重要的是下面三件事，这三件**不需要人工真值就能出结论**：

1. **人体框像素高度随距离的变化** —— 把 148° 广角的视场推算（1 m→98 px、2 m→49 px）
   从「推算」变成「实测」。这是判断有效工作距离的硬依据。
2. **膝角峰谷幅值是否随距离压缩** —— 若幅值随距离变小，说明关键点定位在退化，
   阈值余量会被吃掉，计数和纠正都会变差。
3. **关键点可用率与置信度** —— 膝/踝/髋三组关节分别在什么距离下开始掉。

用法
----
`--clip` 用 `路径:真值次数` 的格式，可重复。真值记不清就写 0，脚本会跳过准确率。

    python distance_sweep.py \
      --clip "outputs/clip_dist_1.0m.mp4:12" \
      --clip "outputs/clip_dist_1.5m.mp4:10" \
      --clip "outputs/clip_dist_2.0m.mp4:11" \
      --clip "outputs/clip_dist_2.5m.mp4:9" \
      --model yolov8n-pose_ncnn_model --imgsz 320 \
      --out results/distance_sweep.csv

输出
----
终端一张跨距离对比表 + `results/distance_sweep.csv`（汇总）+
`results/distance_sweep_detail_<时间>.csv`（每段每个谷底的明细）。
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

# 判据与 PoseAnalyzer 的默认值保持一致，保证「deep 谷底」与「实际被计数」是同一口径
DOWN_ANGLE = 105.0
UP_ANGLE = 150.0

SUMMARY_COLUMNS = [
    "distance_m",
    "clip",
    "timestamp",
    "device",
    "model",
    "imgsz",
    "frames",
    "fps_file",
    "fps_e2e",
    "duration_s",
    "knee_available_pct",
    "kpt_conf_median",
    "bbox_h_px_median",
    "bbox_h_ratio_median",
    "bbox_h_px_in320",
    "knee_range_deg",
    "valley_total",
    "valley_deep",
    "valley_shallow",
    "valley_noise",
    "online_count",
    "offline_count",
    "truth",
    "acc_pct",
    "notes",
]

DETAIL_COLUMNS = ["distance_m", "clip", "valley_no", "frame", "video_time_s", "knee_deg", "class"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="跨距离扫描：人体尺寸 / 膝角幅值 / 可用率 / 计数准确率。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clip", action="append", required=True,
                        help='格式 "视频路径:真值次数"，可重复。真值未知写 0。')
    parser.add_argument("--model", default="yolov8n-pose_ncnn_model", help="模型路径。")
    parser.add_argument("--imgsz", type=int, default=320, help="推理输入边长。")
    parser.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值。")
    parser.add_argument("--min-kpt-conf", type=float, default=0.05, help="关键点最低置信度。")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数，0 表示全片。")
    parser.add_argument("--out", default="results/distance_sweep.csv", help="汇总 CSV 路径。")
    parser.add_argument("--no-detail", action="store_true", help="不导出谷底明细。")
    return parser.parse_args()


def parse_clip_spec(spec: str) -> tuple[Path, int, float]:
    """解析 "路径:真值" 或 "路径:真值:距离"。路径里的冒号（如 D:\\）用 rsplit 保护。"""
    parts = spec.rsplit(":", 2)
    if len(parts) == 3 and _is_number(parts[2]):
        return Path(parts[0]), int(parts[1] or 0), float(parts[2])
    if len(parts) >= 2 and _is_number(parts[1]):
        path = Path(parts[0])
        return path, int(parts[1] or 0), _guess_distance(path)
    return Path(spec), 0, _guess_distance(Path(spec))


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _guess_distance(path: Path) -> float:
    """从文件名猜距离，如 clip_dist_1.5m → 1.5。猜不到返回 0。"""
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*m", path.stem)
    return float(match.group(1)) if match else 0.0


def find_valleys(knee_series: list[tuple[int, float]], window: int = 3,
                 min_gap: int = 25) -> list[tuple[int, float]]:
    """局部极小值检测。window=3 表示前后各 3 帧共 7 帧窗口。

    min_gap 用于抑制同一次蹲底的抖动：距离上一个谷底不足 25 帧就不算新的一次。
    """
    valleys: list[tuple[int, float]] = []
    for i in range(window, len(knee_series) - window):
        frame, value = knee_series[i]
        neighbours = [knee_series[j][1] for j in range(i - window, i + window + 1)]
        if value != min(neighbours):
            continue
        if not valleys or frame - valleys[-1][0] > min_gap:
            valleys.append((frame, value))
        elif value < valleys[-1][1]:
            valleys[-1] = (frame, value)
    return valleys


def analyze_clip(path: Path, truth: int, distance: float, model, args) -> tuple[dict, list[dict]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": f"打不开 {path}"}, []

    fps_file = cap.get(cv2.CAP_PROP_FPS) or 0.0
    analyzer = PoseAnalyzer(min_conf=args.min_kpt_conf,
                            squat_down_angle=DOWN_ANGLE,
                            squat_up_angle=UP_ANGLE)

    knees: list[tuple[int, float]] = []
    confs: list[float] = []
    bbox_heights: list[float] = []
    bbox_ratios: list[float] = []
    knee_available = 0
    frames = 0
    started = time.perf_counter()
    wall = 0.0
    frame_h = 0

    try:
        while True:
            if args.max_frames and frames >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            if frame_h == 0:
                frame_h = frame.shape[0]

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            keypoints, primary_index, _ = extract_primary_keypoints(result)
            analysis = analyzer.analyze(keypoints, primary_index)

            values = [analysis.angles[k] for k in ("left_knee", "right_knee") if k in analysis.angles]
            if values:
                knee_available += 1
                knees.append((frames, sum(values) / len(values)))

            # 关键点置信度：只看膝/踝/髋六点，它们决定膝角能不能算
            if keypoints:
                for name in ("left_hip", "right_hip", "left_knee", "right_knee",
                             "left_ankle", "right_ankle"):
                    index = _KPT_INDEX.get(name)
                    if index is not None and index < len(keypoints):
                        confs.append(keypoints[index].conf)

            # 人体框高度：用来把 148° 广角的视场推算变成实测
            if primary_index is not None and result.boxes is not None and len(result.boxes) > primary_index:
                try:
                    box = result.boxes[primary_index].xyxy[0].tolist()
                    h = float(box[3] - box[1])
                    if h > 0:
                        bbox_heights.append(h)
                        if frame_h > 0:
                            bbox_ratios.append(h / frame_h)
                except (AttributeError, IndexError, ValueError):
                    pass
    finally:
        wall = time.perf_counter() - started
        cap.release()

    if frames == 0:
        return {"error": f"{path} 没有读到帧"}, []

    online_count = analyzer.squat_count
    analyzer.finalize()
    offline_count = analyzer.squat_count

    valleys = find_valleys(knees)
    deep = [(f, v) for f, v in valleys if v < DOWN_ANGLE]
    shallow = [(f, v) for f, v in valleys if DOWN_ANGLE <= v < UP_ANGLE]
    noise = [(f, v) for f, v in valleys if v >= UP_ANGLE]

    knee_values = [v for _, v in knees]
    knee_range = (max(knee_values) - min(knee_values)) if knee_values else 0.0

    bbox_median = statistics.median(bbox_heights) if bbox_heights else 0.0
    ratio_median = statistics.median(bbox_ratios) if bbox_ratios else 0.0

    detail = []
    for idx, (frame, value) in enumerate(valleys, start=1):
        label = "deep" if value < DOWN_ANGLE else ("shallow" if value < UP_ANGLE else "noise")
        detail.append({
            "distance_m": distance,
            "clip": path.name,
            "valley_no": idx,
            "frame": frame,
            "video_time_s": round(frame / fps_file, 2) if fps_file > 0 else "",
            "knee_deg": round(value, 1),
            "class": label,
        })

    summary = {
        "distance_m": distance or "",
        "clip": path.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": _device_name(),
        "model": args.model,
        "imgsz": args.imgsz,
        "frames": frames,
        "fps_file": round(fps_file, 2),
        "fps_e2e": round(frames / wall, 2) if wall > 0 else 0.0,
        "duration_s": round(wall, 2),
        "knee_available_pct": round(knee_available / frames * 100.0, 1),
        "kpt_conf_median": round(statistics.median(confs), 3) if confs else "",
        "bbox_h_px_median": round(bbox_median, 1),
        "bbox_h_ratio_median": round(ratio_median, 4),
        # 640×480 采集帧 → 320×320 输入是等比缩放到长边 320，即 × 0.5（不是 × 0.667）
        "bbox_h_px_in320": round(bbox_median * 0.5, 1),
        "knee_range_deg": round(knee_range, 1),
        "valley_total": len(valleys),
        "valley_deep": len(deep),
        "valley_shallow": len(shallow),
        "valley_noise": len(noise),
        "online_count": online_count,
        "offline_count": offline_count,
        "truth": truth or "",
        "acc_pct": round(online_count / truth * 100.0, 1) if truth > 0 else "",
        "notes": f"finalize_delta={offline_count - online_count}",
    }
    return summary, detail


# COCO 关键点索引（与 pose_analyzer.KEYPOINT_INDEX 一致，这里只取需要的几个）
_KPT_INDEX = {
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


def _device_name() -> str:
    try:
        with open("/sys/firmware/devicetree/base/model", "r", encoding="utf-8", errors="ignore") as handle:
            return handle.readline().strip().strip("\x00") or "unknown"
    except OSError:
        import platform

        return f"{platform.system()} {platform.machine()}"


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    specs = [parse_clip_spec(spec) for spec in args.clip]
    model = YOLO(args.model)

    print(f"模型: {args.model}   imgsz={args.imgsz}   共 {len(specs)} 段")
    print("=" * 96)

    summaries: list[dict] = []
    details: list[dict] = []

    for path, truth, distance in specs:
        if not path.exists():
            print(f"[跳过] 文件不存在: {path}")
            continue
        label = f"{distance:g} m" if distance else path.stem
        print(f"处理 {label}  ({path.name})  真值={truth or '未填'}")
        summary, detail = analyze_clip(path, truth, distance, model, args)
        if "error" in summary:
            print(f"  [错误] {summary['error']}")
            continue
        print(f"  {summary['frames']} 帧 / {summary['duration_s']} s → {summary['fps_e2e']} FPS  "
              f"膝角可用 {summary['knee_available_pct']}%  人体 {summary['bbox_h_px_median']} px")
        summaries.append(summary)
        details.extend(detail)

    if not summaries:
        print("没有任何结果。")
        return 1

    summaries.sort(key=lambda r: (r["distance_m"] if isinstance(r["distance_m"], float) else 999))

    # ---------------- 跨距离对比表 ----------------
    print("\n" + "=" * 96)
    print("跨距离对比（人体高度为采集帧内像素，末列为换算到 320 输入后的像素）")
    print("-" * 96)
    header = (f"{'距离':<7}{'帧数':<7}{'人体高':<9}{'320内':<8}{'膝角幅值':<10}"
              f"{'可用率':<8}{'置信度':<8}{'峰谷':<7}{'深/浅':<8}{'系统':<6}{'真值':<6}{'准确率':<8}")
    print(header)
    print("-" * 96)
    for r in summaries:
        dist = f"{r['distance_m']:g} m" if isinstance(r["distance_m"], float) and r["distance_m"] else "?"
        acc = f"{r['acc_pct']}%" if r["acc_pct"] != "" else "—"
        print(f"{dist:<7}{r['frames']:<7}{r['bbox_h_px_median']:<9}{r['bbox_h_px_in320']:<8}"
              f"{r['knee_range_deg']:<10}{r['knee_available_pct']:<8}{r['kpt_conf_median']:<8}"
              f"{r['valley_total']:<7}{str(r['valley_deep']) + '/' + str(r['valley_shallow']):<8}"
              f"{r['online_count']:<6}{r['truth'] or '-':<6}{acc:<8}")

    # ---------------- 自动判读 ----------------
    print("\n" + "=" * 96)
    print("自动判读")
    print("-" * 96)

    if len(summaries) >= 2:
        near, far = summaries[0], summaries[-1]
        if near["bbox_h_px_median"] > 0 and far["bbox_h_px_median"] > 0:
            drop = (1 - far["bbox_h_px_median"] / near["bbox_h_px_median"]) * 100
            print(f"· 人体像素高度：{near['distance_m']:g} m 时 {near['bbox_h_px_median']:.0f} px，"
                  f"{far['distance_m']:g} m 时 {far['bbox_h_px_median']:.0f} px，衰减 {drop:.0f}%")
            if far["bbox_h_px_in320"] < 50:
                print(f"  ⚠️ {far['distance_m']:g} m 处换算到 320 输入只有 {far['bbox_h_px_in320']:.0f} px，"
                      "已接近小目标区间，关键点定位不可靠。")

        if near["knee_range_deg"] > 0:
            compress = (1 - far["knee_range_deg"] / near["knee_range_deg"]) * 100
            print(f"· 膝角峰谷幅值：{near['distance_m']:g} m 时 {near['knee_range_deg']:.0f}°，"
                  f"{far['distance_m']:g} m 时 {far['knee_range_deg']:.0f}°，变化 {compress:+.0f}%")
            if abs(compress) < 10:
                print("  → 幅值基本不压缩，说明角度对距离不敏感（与 P9-c 的结论一致），阈值余量保持。")
            else:
                print("  → 幅值随距离明显变化，阈值余量被吃掉，需要按距离重新标定。")

        avail = [r["knee_available_pct"] for r in summaries]
        print(f"· 膝角可用率：{min(avail):.1f}% – {max(avail):.1f}%")
        if min(avail) < 90:
            worst = min(summaries, key=lambda r: r["knee_available_pct"])
            print(f"  ⚠️ {worst['distance_m']:g} m 处跌到 {worst['knee_available_pct']:.1f}%，"
                  "深蹲时膝/踝被遮挡或出画。")

    print("\n· 谷底构成（deep=膝角<105° 会被计数 / shallow=105–150° 半蹲不会计数 / noise=站立波动）")
    for r in summaries:
        dist = f"{r['distance_m']:g} m" if isinstance(r["distance_m"], float) and r["distance_m"] else "?"
        print(f"  {dist:<7} deep={r['valley_deep']:<3} shallow={r['valley_shallow']:<3} "
              f"noise={r['valley_noise']:<3}  系统计数={r['online_count']}")

    # ---------------- 落盘 ----------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        if not exists:
            writer.writeheader()
        for r in summaries:
            writer.writerow({k: r.get(k, "") for k in SUMMARY_COLUMNS})
    print(f"\n汇总: {out}")

    if details and not args.no_detail:
        detail_path = out.with_name(f"{out.stem}_detail_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with detail_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS)
            writer.writeheader()
            writer.writerows(details)
        print(f"谷底明细: {detail_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
