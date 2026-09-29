"""Pose keypoint utilities and simple action analysis for YOLO pose output."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

KEYPOINT_INDEX = {name: index for index, name in enumerate(COCO_KEYPOINT_NAMES)}


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    conf: float = 1.0


@dataclass(frozen=True)
class PoseAnalysis:
    actions: list[str]
    angles: dict[str, float]
    squat_count: int
    squat_phase: str
    primary_person_index: int | None

    def overlay_lines(self) -> list[str]:
        action_text = ", ".join(self.actions) if self.actions else "No clear action"
        lines = [
            f"Action: {action_text}",
            f"Squats: {self.squat_count} | Phase: {self.squat_phase}",
        ]
        for name in ("left_knee", "right_knee", "left_elbow", "right_elbow", "torso_lean"):
            if name in self.angles:
                label = name.replace("_", " ").title()
                lines.append(f"{label}: {self.angles[name]:.0f} deg")
        return lines


def _to_plain_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def extract_primary_keypoints(result) -> tuple[list[Keypoint], int | None, int]:
    """Return keypoints for the largest detected person in one YOLO result."""

    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or getattr(keypoints, "xy", None) is None:
        return [], None, 0

    xy_values = _to_plain_list(keypoints.xy)
    if not xy_values:
        return [], None, 0

    people_count = len(xy_values)
    primary_index = _largest_box_index(result, people_count)
    conf_values = _to_plain_list(getattr(keypoints, "conf", None))

    person_xy = xy_values[primary_index]
    person_conf = conf_values[primary_index] if conf_values else None

    extracted = []
    for index, xy in enumerate(person_xy):
        conf = person_conf[index] if person_conf and index < len(person_conf) else 1.0
        extracted.append(Keypoint(float(xy[0]), float(xy[1]), float(conf)))

    return extracted, primary_index, people_count


def _largest_box_index(result, people_count: int) -> int:
    boxes = getattr(result, "boxes", None)
    xyxy_values = _to_plain_list(getattr(boxes, "xyxy", None)) if boxes is not None else None
    if not xyxy_values or len(xyxy_values) != people_count:
        return 0

    areas = []
    for box in xyxy_values:
        width = max(0.0, float(box[2]) - float(box[0]))
        height = max(0.0, float(box[3]) - float(box[1]))
        areas.append(width * height)
    return max(range(len(areas)), key=areas.__getitem__)


class PoseAnalyzer:
    """Stateful action analyzer.

    The squat counter uses a simple up/down state machine, so keep one analyzer
    instance alive across frames.
    """

    def __init__(
        self,
        min_conf: float = 0.25,
        squat_down_angle: float = 105.0,
        squat_up_angle: float = 160.0,
    ) -> None:
        self.min_conf = min_conf
        self.squat_down_angle = squat_down_angle
        self.squat_up_angle = squat_up_angle
        self.squat_count = 0
        self.squat_phase = "unknown"
        self._counted_this_cycle = True  # 本轮站起是否已计数（去抖/缺帧容忍）

    def analyze(self, keypoints: list[Keypoint], primary_person_index: int | None) -> PoseAnalysis:
        if not keypoints:
            return PoseAnalysis(
                actions=["No visible pose"],
                angles={},
                squat_count=self.squat_count,
                squat_phase=self.squat_phase,
                primary_person_index=primary_person_index,
            )

        angles: dict[str, float] = {}
        self._add_joint_angle(angles, keypoints, "left_elbow", "left_shoulder", "left_elbow", "left_wrist")
        self._add_joint_angle(angles, keypoints, "right_elbow", "right_shoulder", "right_elbow", "right_wrist")
        self._add_joint_angle(angles, keypoints, "left_knee", "left_hip", "left_knee", "left_ankle")
        self._add_joint_angle(angles, keypoints, "right_knee", "right_hip", "right_knee", "right_ankle")

        torso_lean = self._torso_lean_degrees(keypoints)
        if torso_lean is not None:
            angles["torso_lean"] = torso_lean

        actions = self._detect_actions(keypoints, angles)
        return PoseAnalysis(actions, angles, self.squat_count, self.squat_phase, primary_person_index)

    def _add_joint_angle(
        self,
        angles: dict[str, float],
        keypoints: list[Keypoint],
        name: str,
        point_a: str,
        point_b: str,
        point_c: str,
    ) -> None:
        a = self._point(keypoints, point_a)
        b = self._point(keypoints, point_b)
        c = self._point(keypoints, point_c)
        if a and b and c:
            angles[name] = angle_degrees(a, b, c)

    def _detect_actions(self, keypoints: list[Keypoint], angles: dict[str, float]) -> list[str]:
        actions = []

        left_arm_up = self._is_wrist_above_shoulder(keypoints, "left")
        right_arm_up = self._is_wrist_above_shoulder(keypoints, "right")
        if left_arm_up and right_arm_up:
            actions.append("Both arms raised")
        elif left_arm_up:
            actions.append("Left arm raised")
        elif right_arm_up:
            actions.append("Right arm raised")

        knee_angles = [angles[name] for name in ("left_knee", "right_knee") if name in angles]
        if knee_angles:
            avg_knee_angle = sum(knee_angles) / len(knee_angles)
            if avg_knee_angle < self.squat_down_angle:
                # 下降边沿（仅从 up/unknown 转入 down 时触发一次，蹲底停顿不重复）
                if self.squat_phase in ("up", "unknown"):
                    self.squat_phase = "down"
                    self._counted_this_cycle = False
                actions.append("Squat down")
            elif avg_knee_angle > self.squat_up_angle:
                # 上升边沿：完成一次有效蹲起配对，本轮只计一次
                if self.squat_phase == "down" and not self._counted_this_cycle:
                    self.squat_count += 1
                    self._counted_this_cycle = True
                self.squat_phase = "up"
                actions.append("Standing")
            else:
                # 过渡区：保持 squat_phase 不变（缺帧容忍，不重置本轮状态）
                actions.append("Squat transition")

        torso_lean = angles.get("torso_lean")
        if torso_lean is not None and torso_lean > 25:
            actions.append("Torso leaning")

        return actions

    def _is_wrist_above_shoulder(self, keypoints: list[Keypoint], side: str) -> bool:
        wrist = self._point(keypoints, f"{side}_wrist")
        shoulder = self._point(keypoints, f"{side}_shoulder")
        return bool(wrist and shoulder and wrist.y < shoulder.y)

    def _torso_lean_degrees(self, keypoints: list[Keypoint]) -> float | None:
        left_shoulder = self._point(keypoints, "left_shoulder")
        right_shoulder = self._point(keypoints, "right_shoulder")
        left_hip = self._point(keypoints, "left_hip")
        right_hip = self._point(keypoints, "right_hip")
        if not (left_shoulder and right_shoulder and left_hip and right_hip):
            return None

        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        dx = hip_mid.x - shoulder_mid.x
        dy = hip_mid.y - shoulder_mid.y
        if abs(dy) < 1e-6:
            return 90.0
        return abs(math.degrees(math.atan2(dx, dy)))

    def _point(self, keypoints: list[Keypoint], name: str) -> Keypoint | None:
        index = KEYPOINT_INDEX[name]
        if index >= len(keypoints):
            return None
        point = keypoints[index]
        if point.conf < self.min_conf or point.x <= 0 or point.y <= 0:
            return None
        return point

    def finalize(self) -> None:
        """视频结束收尾：若最后一轮下蹲后站起帧缺失，补计一次。"""
        if self.squat_phase == "down" and not self._counted_this_cycle:
            self.squat_count += 1
            self._counted_this_cycle = True


def angle_degrees(a: Keypoint, b: Keypoint, c: Keypoint) -> float:
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    denominator = math.hypot(*ab) * math.hypot(*cb)
    if denominator <= 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, (ab[0] * cb[0] + ab[1] * cb[1]) / denominator))
    return math.degrees(math.acos(cosine))


def midpoint(a: Keypoint, b: Keypoint) -> Keypoint:
    return Keypoint((a.x + b.x) / 2, (a.y + b.y) / 2, min(a.conf, b.conf))


def draw_analysis_overlay(frame, analysis: PoseAnalysis, origin: tuple[int, int] = (10, 56)) -> None:
    lines = analysis.overlay_lines()
    line_height = 24
    width = 430
    height = 18 + line_height * len(lines)
    x, y = origin

    frame_height, frame_width = frame.shape[:2]
    x1 = min(max(x, 0), frame_width)
    y1 = min(max(y, 0), frame_height)
    x2 = min(x + width, frame_width)
    y2 = min(y + height, frame_height)
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        panel = roi.copy()
        cv2.rectangle(panel, (0, 0), (x2 - x1, y2 - y1), (20, 20, 20), thickness=-1)
        cv2.addWeighted(panel, 0.62, roi, 0.38, 0, roi)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 10, y + 24 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
