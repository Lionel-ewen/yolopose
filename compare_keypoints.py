"""关键点精度对比：量化「为了提速损失了多少精度」。

思路
----
毕设自己录的测试片段没有人工标注的 ground truth，直接算 COCO mAP 不现实。
实用且站得住脚的做法是设一个**参考基线（reference）**：

    在 PC 上用 PyTorch fp32 + imgsz=640 跑出的关键点，视为伪真值。

之后每个优化档位（imgsz=320、NCNN、int8 量化……）的输出都与它对比，
得到「相对参考基线的精度保持率」。报告里配合帧率数据，就是那条
**帧率 vs 精度权衡曲线** —— 整篇论文含金量最高的图之一。

指标说明
--------
PCK@t (Percentage of Correct Keypoints)
    以人体尺度归一化后，位移误差小于阈值 t 的关键点占比。
    归一化尺度取躯干对角线（肩中点到髋中点距离），对远近距离不敏感。
    这是姿态估计领域的标准做法，答辩时经得起问。

MPE (Mean Pixel Error)
    平均像素误差，直观但受分辨率影响，作为辅助指标。

用法
----
    # 1. 先跑参考基线并导出关键点
    python benchmark.py --source outputs/test_clip.mp4 --imgsz 640 \
        --dump-keypoints results/kpts/ref_fp32_640.npz --tag ref_fp32_640

    # 2. 跑各个优化档位
    python benchmark.py --source outputs/test_clip.mp4 --imgsz 320 \
        --dump-keypoints results/kpts/imgsz_320.npz --tag imgsz_320

    # 3. 对比
    python compare_keypoints.py --ref results/kpts/ref_fp32_640.npz \
        --test results/kpts/imgsz_320.npz results/kpts/imgsz_256.npz
"""

from __future__ import annotations

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pose_analyzer import COCO_KEYPOINT_NAMES

LEFT_SHOULDER = COCO_KEYPOINT_NAMES.index("left_shoulder")
RIGHT_SHOULDER = COCO_KEYPOINT_NAMES.index("right_shoulder")
LEFT_HIP = COCO_KEYPOINT_NAMES.index("left_hip")
RIGHT_HIP = COCO_KEYPOINT_NAMES.index("right_hip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare keypoint accuracy against a reference run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ref", required=True, help="参考基线 .npz（建议 PC fp32 imgsz=640）。")
    parser.add_argument("--test", nargs="+", required=True, help="待比较的一个或多个 .npz。")
    parser.add_argument("--min-conf", type=float, default=0.3, help="参与比较的关键点最低置信度。")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2],
        help="PCK 阈值（相对躯干尺度的比例）。",
    )
    parser.add_argument("--out", default="results/accuracy_comparison.csv", help="对比结果 CSV 输出路径。")
    return parser.parse_args()


def load_keypoints(path: str) -> tuple[np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    keypoints = data["keypoints"].astype(np.float32)
    meta = {
        "tag": str(data["tag"]) if "tag" in data else Path(path).stem,
        "imgsz": int(data["imgsz"]) if "imgsz" in data else -1,
        "model": str(data["model"]) if "model" in data else "?",
    }
    return keypoints, meta


def torso_scale(person: np.ndarray) -> float | None:
    """躯干对角线长度，作为归一化尺度。person 形状 (17, 3)。"""
    shoulders = person[[LEFT_SHOULDER, RIGHT_SHOULDER], :2]
    hips = person[[LEFT_HIP, RIGHT_HIP], :2]
    if np.any(shoulders <= 0) or np.any(hips <= 0):
        return None
    shoulder_mid = shoulders.mean(axis=0)
    hip_mid = hips.mean(axis=0)
    scale = float(np.linalg.norm(shoulder_mid - hip_mid))
    return scale if scale > 1.0 else None


def compare(
    reference: np.ndarray,
    candidate: np.ndarray,
    min_conf: float,
    thresholds: list[float],
) -> dict:
    frames = min(len(reference), len(candidate))
    if frames == 0:
        raise ValueError("参考或待测数据为空。")

    errors: list[float] = []
    normalized: list[float] = []
    per_joint_error: dict[int, list[float]] = {index: [] for index in range(17)}

    ref_detected = 0
    both_detected = 0
    missed = 0

    for index in range(frames):
        ref_person = reference[index]
        test_person = candidate[index]

        ref_valid = ref_person[:, 2] >= min_conf
        test_valid = test_person[:, 2] >= min_conf

        if not ref_valid.any():
            continue
        ref_detected += 1

        if not test_valid.any():
            missed += 1
            continue
        both_detected += 1

        scale = torso_scale(ref_person)
        pair_valid = ref_valid & test_valid

        for joint in np.nonzero(pair_valid)[0]:
            delta = float(np.linalg.norm(ref_person[joint, :2] - test_person[joint, :2]))
            errors.append(delta)
            per_joint_error[int(joint)].append(delta)
            if scale:
                normalized.append(delta / scale)

    if not errors:
        raise ValueError("没有可比较的关键点，检查 --min-conf 或输入数据。")

    result = {
        "frames_compared": frames,
        "ref_detected_frames": ref_detected,
        "both_detected_frames": both_detected,
        "miss_rate": round(missed / ref_detected, 4) if ref_detected else None,
        "mpe_px": round(float(np.mean(errors)), 3),
        "mpe_px_median": round(float(np.median(errors)), 3),
        "mpe_px_p95": round(float(np.percentile(errors, 95)), 3),
        "keypoints_compared": len(errors),
    }

    normalized_array = np.array(normalized) if normalized else None
    for threshold in thresholds:
        key = f"pck@{threshold}"
        if normalized_array is not None and normalized_array.size:
            result[key] = round(float((normalized_array < threshold).mean()), 4)
        else:
            result[key] = None

    worst = sorted(
        ((name, float(np.mean(per_joint_error[i]))) for i, name in enumerate(COCO_KEYPOINT_NAMES) if per_joint_error[i]),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    result["worst_joints"] = "; ".join(f"{name} {value:.1f}px" for name, value in worst)
    return result


def main() -> int:
    args = parse_args()

    reference, ref_meta = load_keypoints(args.ref)
    print(f"参考基线: {ref_meta['tag']}  (imgsz={ref_meta['imgsz']}, {len(reference)} 帧)")
    print()

    rows = []
    for test_path in args.test:
        candidate, meta = load_keypoints(test_path)
        try:
            stats = compare(reference, candidate, args.min_conf, args.thresholds)
        except ValueError as error:
            print(f"[跳过] {test_path}: {error}")
            continue

        row = {"tag": meta["tag"], "imgsz": meta["imgsz"], "model": meta["model"], **stats}
        rows.append(row)

        print(f"--- {meta['tag']} (imgsz={meta['imgsz']}) ---")
        print(f"  可比帧数        : {stats['both_detected_frames']} / {stats['ref_detected_frames']}")
        print(f"  漏检率          : {stats['miss_rate']}")
        print(f"  平均像素误差    : {stats['mpe_px']} px  (中位 {stats['mpe_px_median']}, P95 {stats['mpe_px_p95']})")
        for threshold in args.thresholds:
            print(f"  PCK@{threshold:<5}       : {stats[f'pck@{threshold}']}")
        print(f"  误差最大的关节  : {stats['worst_joints']}")
        print()

    if not rows:
        print("[ERROR] 没有产生任何对比结果。")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"对比结果已写入: {out_path}")
    print("把这张表和 results/benchmark_results.csv 的 fps 列拼起来，就是帧率-精度权衡曲线。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
