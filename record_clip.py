"""录制一段固定的测试片段，作为所有性能实验的统一输入。

为什么必须用固定视频而不是摄像头做基准测试：
  - 摄像头自动曝光/自动增益会随光照变化，导致每次实验的图像内容不同；
  - 摄像头驱动本身的帧率抖动会混入测量结果；
  - 换设备（PC vs 树莓派）时摄像头型号不同，输入根本不可比。

建议录制内容（针对健身动作计数场景）：
  - 时长 20-30 秒，包含 8-10 个完整深蹲；
  - 单人、全身入镜、背景简洁；
  - 中途保留 1-2 秒无人画面，用于测试空场景下的性能与误检；
  - 记录下真实动作次数，作为计数准确率的 ground truth。

用法：
    python record_clip.py --seconds 25 --output outputs/test_clip.mp4
    python record_clip.py --seconds 30 --width 1280 --height 720 --fps 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a fixed benchmark clip from the webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引。")
    parser.add_argument("--output", default="outputs/test_clip.mp4", help="输出视频路径。")
    parser.add_argument("--seconds", type=float, default=25.0, help="录制时长（秒）。")
    parser.add_argument("--width", type=int, default=640, help="采集宽度。")
    parser.add_argument("--height", type=int, default=480, help="采集高度。")
    parser.add_argument("--fps", type=float, default=30.0, help="目标帧率。")
    parser.add_argument("--countdown", type=int, default=3, help="开始前的倒计时秒数。")
    parser.add_argument("--no-preview", action="store_true", help="不显示预览窗口。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {args.camera}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    # 缓冲区设为 1，避免采集积压造成的画面延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps

    print(f"采集分辨率: {actual_width}x{actual_height} @ {actual_fps:.1f} fps")

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (actual_width, actual_height),
    )
    if not writer.isOpened():
        print(f"[ERROR] 无法创建视频文件: {output}")
        cap.release()
        return 1

    # 倒计时
    for remaining in range(args.countdown, 0, -1):
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            if not ok:
                continue
            if not args.no_preview:
                preview = frame.copy()
                cv2.putText(
                    preview,
                    str(remaining),
                    (actual_width // 2 - 30, actual_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3.0,
                    (0, 220, 255),
                    6,
                    cv2.LINE_AA,
                )
                cv2.imshow("record_clip", preview)
                cv2.waitKey(1)
        print(f"  {remaining}...")

    print(f"开始录制 {args.seconds:.0f} 秒，按 q 提前结束。")

    frame_count = 0
    start = time.perf_counter()
    try:
        while (time.perf_counter() - start) < args.seconds:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] 丢帧")
                continue
            writer.write(frame)
            frame_count += 1

            if not args.no_preview:
                elapsed = time.perf_counter() - start
                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"REC {elapsed:4.1f}s / {args.seconds:.0f}s  frames={frame_count}",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (60, 60, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("record_clip", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        elapsed = time.perf_counter() - start
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print()
    print(f"已保存: {output}")
    print(f"  帧数    : {frame_count}")
    print(f"  实际时长: {elapsed:.1f} 秒  (实测采集帧率 {frame_count / elapsed:.1f} fps)")
    print()
    print("下一步：数清楚这段视频里的真实动作次数，记到 docs/EXPERIMENT_LOG.md，")
    print("       它是后续计数准确率的 ground truth。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
