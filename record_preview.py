"""带实时预览的录制工具（独立脚本，不改动任何现有实验代码）。

为什么需要它
------------
`record_clip.py` 已经能预览，但它有两个问题会让录制数据不可用：

1. **不设 FOURCC** → 在这颗相机上走 YUYV，只有 18 FPS（MJPG 640×480 实测 51.18）。
2. **写盘帧率用命令行请求值**（默认 30），而不是实测值。采集 18 FPS 却按 30 FPS 写头，
   播放器会以 1.67 倍速播放，**角度变化率、动作节奏的时间分析全部失真**。

本脚本的解法：
- 启动时先采 30 帧实测帧率（取帧间隔中位数），**用实测值开 VideoWriter**；
- 默认 MJPG@640×480，与 P10 测出的最优采集配置一致；
- 预览画面叠加倒计时、录制时长、实时帧率，方便自己站位；
- **写盘的是干净帧**，HUD 只画在预览副本上，避免叠加文字影响后续姿态检测。

用法
----
    # 最基本：预览 + 录 45 秒
    python record_preview.py --output outputs/clip_dist_1.5m.avi

    # 按距离录一组（站好后回车开始）
    python record_preview.py --output outputs/clip_dist_1.0m.avi --seconds 40
    python record_preview.py --output outputs/clip_dist_1.5m.avi --seconds 40

    # 预览窗口太大/卡顿时缩小
    python record_preview.py --output outputs/clip.avi --preview-scale 0.6

    # 没有图形界面时（纯采集，不预览）
    python record_preview.py --output outputs/clip.avi --no-preview

按键
----
    空格 / s   提前结束录制并保存
    q / Esc    提前结束并保存（不会丢已录内容）

输出说明
--------
默认编码 mp4v（兼容性好，但编码要花 CPU）。若预览时帧率明显掉，改用 MJPG：

    python record_preview.py --output outputs/clip.avi --codec MJPG

MJPG 几乎零编码成本（相机本来就是 MJPG，直接存字节流），代价是文件更大。
两种格式 `count_squats.py` 和 `online_count_check.py` 都能直接读。
"""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="带实时预览的摄像头录制，用实测帧率写文件头。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default=None,
                        help="输出路径。省略则自动生成 outputs/clip_<时间戳>.<ext>")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引。")
    parser.add_argument("--width", type=int, default=640, help="采集宽度。")
    parser.add_argument("--height", type=int, default=480, help="采集高度。")
    parser.add_argument("--fourcc", default="MJPG",
                        choices=["MJPG", "YUYV", "YUY2", "none"],
                        help="采集像素格式。MJPG 是这颗相机的最优配置。")
    parser.add_argument("--buffersize", type=int, default=1, help="采集缓冲，1 避免延迟积压。")

    parser.add_argument("--seconds", type=float, default=45.0, help="录制时长（秒）。")
    parser.add_argument("--countdown", type=int, default=3, help="开始前倒计时秒数。")
    parser.add_argument("--probe-frames", type=int, default=60,
                        help="探测帧率用的帧数（不计入录制）。")
    parser.add_argument("--drain", type=int, default=0,
                        help="冲掉缓冲后额外再丢弃的帧数。0 表示只冲到「read 出现阻塞」为止（推荐）。")

    parser.add_argument("--codec", default="mp4v", choices=["mp4v", "MJPG", "XVID"],
                        help="写盘编码。mp4v 兼容好；MJPG 几乎不耗 CPU。")
    parser.add_argument("--preview", dest="preview", action="store_true", default=True,
                        help="显示预览窗口（默认开启）。")
    parser.add_argument("--no-preview", dest="preview", action="store_false",
                        help="关闭预览。")
    parser.add_argument("--preview-scale", type=float, default=0.8,
                        help="预览窗口缩放比例。1.0 为原始尺寸，小于 1 可降低合成开销。")
    return parser.parse_args()


def fourcc_to_str(value: float) -> str:
    code = int(value or 0)
    if code <= 0:
        return "----"
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    # 部分后端在相机不认请求格式时返回不可打印的垃圾值（实测显示成 "????"）
    if sum(1 for c in chars if 32 <= ord(c) < 127) < 3:
        return "无效"
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in chars)


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        return cap
    # FOURCC 要在设分辨率之前设，否则部分 UVC 相机不静默忽略
    if args.fourcc != "none":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffersize)
    return cap


def drain_until_blocking(cap: cv2.VideoCapture, max_seconds: float = 1.5,
                         block_ms: float = 6.0) -> tuple[int, bool]:
    """连续读帧，直到某一次 read() 明显变慢 —— 那说明缓冲空了，开始等相机出新帧。

    这是探测帧率唯一可靠的前置条件。相机一打开就在后台连续采集，驱动缓冲里积压着
    一批"过去的帧"，此时连续 read() 会瞬间返回。

    ⚠️ 实测教训：按「帧数」冲缓冲不管用。在 Windows/DSHOW 后端上，丢弃 25 帧、
    甚至 150 帧，仍然能测出 400+ FPS 的荒谬值 —— 因为 read() 只要 2 ms，
    150 帧总共才 0.3 秒，深缓冲根本撑不到耗尽。**必须按时间冲**。

    block_ms 取 6 ms：普通 USB 相机帧周期 20–35 ms，缓冲空时等待必然超过这个值。
    返回 (丢弃帧数, 是否成功追上实时)。第二项为 False 时测出的帧率不可信。
    """
    started = time.perf_counter()
    count = 0
    while (time.perf_counter() - started) < max_seconds:
        t0 = time.perf_counter()
        ok, _ = cap.read()
        duration_ms = (time.perf_counter() - t0) * 1000.0
        count += 1
        if not ok:
            return count, False
        if duration_ms >= block_ms:
            return count, True
    return count, False


def measure_fps(cap: cv2.VideoCapture, frames: int, extra_drain: int = 0) -> tuple[float, tuple[int, int], str]:
    """冲到出现阻塞后，再采若干帧，用帧间隔中位数反推真实帧率。

    为什么用中位数：自动曝光调整期会夹进个别超长间隔，均值会被拖偏。
    为什么不用 cap.get(CAP_PROP_FPS)：这颗相机自报 120.1，实测 51.18。
    """
    drained, reached_realtime = drain_until_blocking(cap, max_seconds=1.5)
    for _ in range(max(0, extra_drain)):
        cap.read()

    intervals: list[float] = []
    size = (0, 0)
    for _ in range(max(20, frames)):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        intervals.append((time.perf_counter() - t0) * 1000.0)
        if frame is not None:
            size = (frame.shape[1], frame.shape[0])

    if not intervals:
        return 0.0, size, "----"

    # 只取后 1/3：缓冲抽干是渐进的，前面的间隔偏小（还在吃积压帧），
    # 越往后越接近真实帧周期。用全段中位数会系统性偏高估帧率。
    tail = intervals[len(intervals) * 2 // 3:]
    median_ms = statistics.median(tail)
    fps = 1000.0 / median_ms if median_ms > 0 else 0.0

    # 合理性检查：USB 摄像头不会超过 150 FPS。超了说明缓冲没冲干净。
    if fps > 150:
        print(f"  ⚠️ 测出 {fps:.0f} FPS 不合理（缓冲未冲净），按 30 FPS 保守写入。")
        if not reached_realtime:
            print(f"     缓冲在 {drained} 帧内未冲空，可尝试加大 --drain。")
        fps = 30.0

    return fps, size, fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))


def draw_hud(view, elapsed: float, frames: int, avg_fps: float, target_s: float,
             recording: bool, blocked: bool = False) -> None:
    """只在预览副本上叠加信息，写盘用的是干净帧。

    avg_fps 用「帧数 / 已用时」而不是瞬时值 —— 瞬时值在缓冲未冲净时会显示 160+ FPS，
    对判断「预览有没有拖慢采集」毫无帮助。
    """
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, 0), (w, 34), (24, 24, 24), thickness=-1)
    if recording:
        cv2.circle(view, (16, 17), 7, (60, 60, 235), thickness=-1)
        text = f"REC  {elapsed:5.1f} / {target_s:.0f}s   {frames} 帧   均 {avg_fps:4.1f} FPS"
        color = (90, 240, 140)
        if blocked:
            text += "   [缓冲中]"
            color = (90, 200, 240)
    else:
        text = "准备中…"
        color = (230, 230, 230)
    cv2.putText(view, text, (32, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    cv2.rectangle(view, (0, h - 30), (w, h), (24, 24, 24), thickness=-1)
    tip = "整个人进画面 · 空格=提前结束并保存 · q=退出"
    cv2.putText(view, tip, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def show(view, scale: float, window: str) -> bool:
    """显示预览。返回 False 表示窗口不可用（无图形界面等），调用方应降级。"""
    try:
        if scale != 1.0:
            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(window, view)
        return True
    except cv2.error:
        return False


def fix_framerate(path, target_fps: float) -> tuple[bool, str]:
    """用 ffmpeg 重封装修正视频帧率（`-c copy`，不重新编码，通常 1 秒内完成）。

    为什么需要它：`cap.read()` 的耗时在部分后端（实测 Windows/DSHOW）跟相机真实速率脱钩 ——
    它按 60 FPS 的内部时钟轮询，相机只有 28 FPS 时每两次轮询返回一次同一帧，
    于是 read 稳定在 16 ms，怎么测都测不出 35 ms。这种平台上探测值不可救药，
    唯一可靠的办法是**录完用「帧数 / 实际用时」回头修正文件头**。

    在 Linux/V4L2（树莓派）上 buffersize=1 生效，read 会真实阻塞，探测值本身就是准的，
    此时偏差不超过 5%，这里会自动跳过。
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return False, "未找到 ffmpeg"

    tmp = path.with_name(f"{path.stem}_fix{path.suffix}")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(path),
        "-c", "copy",
        "-r", f"{target_fps:.3f}",
        str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)

    if proc.returncode != 0 or not tmp.exists():
        stderr = (proc.stderr or "ffmpeg 失败").strip().splitlines()
        return False, stderr[-1][:160] if stderr else "ffmpeg 失败"

    path.unlink()
    tmp.replace(path)
    return True, f"{target_fps:.2f} FPS"


def main() -> int:
    args = parse_args()

    ext = ".avi" if args.codec == "MJPG" else ".mp4"
    if args.output is None:
        output = Path("outputs") / f"clip_{datetime.now():%Y%m%d_%H%M%S}{ext}"
    else:
        output = Path(args.output)

    cap = open_camera(args)
    if not cap.isOpened():
        print(f"[ERROR] 打不开摄像头 {args.camera}")
        return 1

    print("正在探测真实帧率…")
    fps, (fw, fh), actual_fourcc = measure_fps(cap, args.probe_frames, args.drain)
    if fps <= 0:
        print("[ERROR] 采不到帧。")
        cap.release()
        return 1

    print(f"  实际配置: {fw}x{fh}  FOURCC={actual_fourcc}")
    print(f"  实测帧率: {fps:.2f} FPS（将以此写入文件头，不是标称值）")
    if args.fourcc != "none" and actual_fourcc not in (args.fourcc, "YUY2"):
        print(f"  ⚠️ 请求 {args.fourcc} 但实际是 {actual_fourcc} —— 相机不认该格式，帧率会低于预期。")
    if fps < 20:
        print(f"  ⚠️ 帧率偏低（{fps:.1f}），深蹲分析仍可用但谷底采样点会变少。")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*args.codec), fps, (fw, fh))
    if not writer.isOpened():
        print(f"[ERROR] 无法创建视频文件: {output}（试试 --codec MJPG --output xxx.avi）")
        cap.release()
        return 1

    window = "录制预览"
    preview_ok = args.preview
    if preview_ok:
        # 先试一次，无图形界面时优雅降级
        ok, frame = cap.read()
        if ok:
            preview_ok = show(frame, args.preview_scale, window)
        if not preview_ok:
            print("  ⚠️ 预览窗口不可用（无图形界面？），已自动改为纯录制。")

    # 倒计时：每帧都显示，让他有机会站好位置
    if preview_ok and args.countdown > 0:
        for remaining in range(args.countdown, 0, -1):
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline:
                ok, frame = cap.read()
                if not ok:
                    continue
                view = frame.copy()
                h, w = view.shape[:2]
                cv2.putText(view, str(remaining), (w // 2 - 20, h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (60, 60, 235), 6, cv2.LINE_AA)
                draw_hud(view, 0.0, 0, fps, args.seconds, False)
                show(view, args.preview_scale, window)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    print("已取消，未写入任何内容。")
                    cap.release()
                    writer.release()
                    cv2.destroyAllWindows()
                    output.unlink(missing_ok=True)
                    return 0

    print(f"\n开始录制 → {output}")
    print(f"  时长 {args.seconds:.0f} s，编码 {args.codec}。空格提前结束，q 退出。")

    frames = 0
    started = time.perf_counter()
    stopped_early = False
    window_ok = preview_ok

    try:
        while True:
            elapsed = time.perf_counter() - started
            if elapsed >= args.seconds:
                break

            ok, frame = cap.read()
            if not ok:
                print("  [中断] 读帧失败。")
                break

            # 写盘用干净帧
            writer.write(frame)
            frames += 1

            avg_fps = frames / elapsed if elapsed > 0 else 0.0
            # 缓冲未冲净时 read 会瞬间返回，看起来像 100+ FPS。用均值和写入值对比来识别。
            buffering = avg_fps > fps * 1.6

            if window_ok:
                view = frame.copy()
                draw_hud(view, elapsed, frames, avg_fps, args.seconds, True, buffering)
                window_ok = show(view, args.preview_scale, window)
                if not window_ok:
                    print("  ⚠️ 预览窗口已失效，继续纯录制。")
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27, ord(" "), ord("s")):
                    stopped_early = True
                    break

            if frames % 100 == 0:
                print(f"  {frames} 帧 / {elapsed:.1f}s  均值 {avg_fps:.1f} FPS")
    finally:
        wall = time.perf_counter() - started
        cap.release()
        writer.release()
        if window_ok:
            cv2.destroyAllWindows()

    if frames == 0:
        print("[ERROR] 没有录到任何帧。")
        output.unlink(missing_ok=True)
        return 1

    file_mb = output.stat().st_size / (1024 * 1024)
    actual_fps = frames / wall if wall > 0 else fps

    # 用「帧数 / 实际用时」回头校正文件头。探测值只是初始占位：
    # 部分后端（DSHOW）根本测不准，而实际平均帧率永远是可信的。
    print("\n" + "=" * 58)
    drift = abs(actual_fps - fps) / max(fps, 1e-6)
    if drift > 0.05:
        print(f"文件头帧率 {fps:.2f} ≠ 实测 {actual_fps:.2f}（偏差 {drift * 100:.1f}%），正在修正…")
        ok, info = fix_framerate(output, actual_fps)
        if ok:
            print(f"  ✅ 已重封装修正为 {info}")
            fps = actual_fps
        else:
            print(f"  ⚠️ 自动修正失败（{info}）")
            print(f"     影响：播放速度会偏 {drift * 100:.0f}%，但计数按帧处理，不受影响。")
            print(f"     手动修正：ffmpeg -y -i {output} -c copy -r {actual_fps:.2f} fixed{output.suffix}")
    else:
        print(f"文件头帧率与实测一致（{actual_fps:.2f} FPS）")
        fps = actual_fps

    print(f"已保存: {output}")
    print(f"  帧数      : {frames}")
    print(f"  实际时长  : {wall:.2f} s")
    print(f"  写入帧率  : {fps:.2f} FPS")
    print(f"  平均采集  : {wall / frames * 1000:.2f} ms/帧 → {frames / wall:.2f} FPS")
    print(f"  文件大小  : {file_mb:.1f} MB")
    if stopped_early:
        print("  （提前结束，已录内容全部保留）")
    print("")
    print("下一步：")
    print(f"  python online_count_check.py --source {output} --truth <你数出来的次数> \\")
    print("      --model yolov8n-pose_ncnn_model --imgsz 320 --tag <标签>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
