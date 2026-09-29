"""Generate the updated project Gantt chart for the 876H1Z interim report.

Two charts are produced so the Chinese draft and the English translation can
each embed a figure with the correct language:

    outputs/fig_gantt_zh.png   -- Chinese labels
    outputs/fig_gantt_en.png   -- English labels

Completed phases (P0-P4c) are drawn solid; remaining phases are drawn in a
lighter tone.  A "today" line marks the interim-report writing window, and the
three formal deadlines are drawn as milestone markers.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

TODAY = dt.date(2026, 9, 23)

DONE_FACE = "#85B7EB"
DONE_EDGE = "#185FA5"
PLAN_FACE = "#E6F1FB"
PLAN_EDGE = "#85B7EB"
WRITE_FACE = "#FAC775"
WRITE_EDGE = "#854F0B"
TEXT_DONE = "#042C53"
TEXT_PLAN = "#185FA5"
TEXT_WRITE = "#633806"

# (label_zh, label_en, start, end, kind)
#   kind: "done" | "plan" | "write"
TASKS: list[tuple[str, str, dt.date, dt.date, str]] = [
    (
        "P0 工具链地基 + 文献调研",
        "P0 Measurement scaffold + literature survey",
        dt.date(2026, 8, 1),
        dt.date(2026, 8, 25),
        "done",
    ),
    (
        "P1 PC 参考基线（PyTorch@640）",
        "P1 PC reference baseline (PyTorch@640)",
        dt.date(2026, 8, 24),
        dt.date(2026, 8, 26),
        "done",
    ),
    (
        "P2 树莓派基线（PyTorch@640）",
        "P2 Raspberry Pi baseline (PyTorch@640)",
        dt.date(2026, 9, 7),
        dt.date(2026, 9, 9),
        "done",
    ),
    (
        "P3 输入分辨率扫描 512→256",
        "P3 Input-resolution sweep 512-256",
        dt.date(2026, 9, 8),
        dt.date(2026, 9, 10),
        "done",
    ),
    (
        "P4 推理后端迁移（ONNX/NCNN）",
        "P4 Inference-backend migration (ONNX/NCNN)",
        dt.date(2026, 9, 8),
        dt.date(2026, 9, 12),
        "done",
    ),
    (
        "中期报告撰写与提交",
        "Interim report writing and submission",
        dt.date(2026, 9, 16),
        dt.date(2026, 9, 30),
        "write",
    ),
    (
        "P5 量化（fp16 / INT8）",
        "P5 Quantisation (fp16 / INT8)",
        dt.date(2026, 10, 1),
        dt.date(2026, 10, 14),
        "plan",
    ),
    (
        "P6 多线程流水线解耦",
        "P6 Multi-threaded pipeline decoupling",
        dt.date(2026, 10, 8),
        dt.date(2026, 10, 21),
        "plan",
    ),
    (
        "P7 时序稀疏（跳帧 + 跟踪插值）",
        "P7 Temporal sparsity (frame skipping + tracking)",
        dt.date(2026, 10, 15),
        dt.date(2026, 10, 31),
        "plan",
    ),
    (
        "P8 系统级（散热 / 热 soak）",
        "P8 System level (thermal / soak test)",
        dt.date(2026, 11, 2),
        dt.date(2026, 11, 14),
        "plan",
    ),
    (
        "P9 瓶颈归因 + 场景够用性验证",
        "P9 Bottleneck attribution + scenario sufficiency",
        dt.date(2026, 11, 5),
        dt.date(2026, 11, 20),
        "plan",
    ),
    (
        "补充实验与返工余量",
        "Contingency margin for repeat experiments",
        dt.date(2026, 11, 16),
        dt.date(2026, 11, 30),
        "plan",
    ),
    (
        "论文撰写（初稿 → 修订）",
        "Dissertation writing (draft to revision)",
        dt.date(2026, 11, 10),
        dt.date(2026, 12, 25),
        "write",
    ),
    (
        "答辩准备与演练",
        "Presentation preparation and rehearsal",
        dt.date(2026, 12, 1),
        dt.date(2026, 12, 10),
        "write",
    ),
    (
        "论文定稿 + logbook 整理提交",
        "Final submission + logbook consolidation",
        dt.date(2026, 12, 16),
        dt.date(2027, 1, 7),
        "write",
    ),
]

# (date, label_zh, label_en)
MILESTONES = [
    (dt.date(2026, 9, 30), "中期报告\n9 月 30 日", "Interim report\n30 Sep"),
    (dt.date(2026, 12, 11), "答辩\n12 月 11 日", "Colloquium\n11 Dec"),
    (dt.date(2027, 1, 7), "论文提交\n1 月 7 日", "Dissertation\n7 Jan"),
]

FACE = {"done": DONE_FACE, "plan": PLAN_FACE, "write": WRITE_FACE}
EDGE = {"done": DONE_EDGE, "plan": PLAN_EDGE, "write": WRITE_EDGE}
TEXT = {"done": TEXT_DONE, "plan": TEXT_PLAN, "write": TEXT_WRITE}


def build(lang: str) -> Path:
    idx = 0 if lang == "zh" else 1
    labels = [t[idx] for t in TASKS]
    kinds = [t[4] for t in TASKS]

    fig, ax = plt.subplots(figsize=(11.6, 6.4), dpi=200)
    ypos = list(range(len(TASKS)))[::-1]

    for y, (_zh, _en, start, end, kind) in zip(ypos, TASKS):
        left = mdates.date2num(start)
        width = max(mdates.date2num(end) - left, 1.0)
        ax.barh(
            y,
            width,
            left=left,
            height=0.56,
            facecolor=FACE[kind],
            edgecolor=EDGE[kind],
            linewidth=0.9,
            zorder=3,
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9.5)
    for tick, kind in zip(ax.get_yticklabels(), kinds):
        tick.set_color(TEXT[kind])

    ax.set_ylim(-0.8, len(TASKS) - 0.2)
    ax.set_xlim(mdates.date2num(dt.date(2026, 7, 25)), mdates.date2num(dt.date(2027, 1, 14)))

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", color="#D3D1C7", linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D3D1C7")

    today_x = mdates.date2num(TODAY)
    ax.axvline(today_x, color="#A32D2D", linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
    ax.annotate(
        "today" if lang == "en" else "今日",
        xy=(today_x, -0.72),
        xytext=(3, 0),
        textcoords="offset points",
        fontsize=8.5,
        color="#A32D2D",
        va="center",
        zorder=6,
    )

    for when, label_zh, label_en in MILESTONES:
        x = mdates.date2num(when)
        ax.plot(
            x,
            len(TASKS) - 0.5,
            marker="D",
            markersize=6,
            color="#A32D2D",
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=5,
            clip_on=False,
        )
        ax.annotate(
            label_zh if lang == "zh" else label_en,
            xy=(x, len(TASKS) - 0.5),
            xytext=(0, -9),
            textcoords="offset points",
            fontsize=7.6,
            color="#A32D2D",
            ha="center",
            va="top",
            linespacing=1.35,
        )

    handles = (
        [
            Patch(facecolor=DONE_FACE, edgecolor=DONE_EDGE, label="Completed (P0-P4c)"),
            Patch(facecolor=PLAN_FACE, edgecolor=PLAN_EDGE, label="Planned technical work"),
            Patch(facecolor=WRITE_FACE, edgecolor=WRITE_EDGE, label="Reporting / presentation"),
            Patch(facecolor="none", edgecolor="#A32D2D", label="Formal deadline"),
        ]
        if lang == "en"
        else [
            Patch(facecolor=DONE_FACE, edgecolor=DONE_EDGE, label="已完成（P0–P4c）"),
            Patch(facecolor=PLAN_FACE, edgecolor=PLAN_EDGE, label="计划中的技术工作"),
            Patch(facecolor=WRITE_FACE, edgecolor=WRITE_EDGE, label="报告与答辩"),
            Patch(facecolor="none", edgecolor="#A32D2D", label="正式截止日期"),
        ]
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        fontsize=8.5,
        frameon=True,
        facecolor="white",
        edgecolor="#D3D1C7",
        framealpha=1.0,
    )

    ax.set_title(
        "Updated project plan after the interim report"
        if lang == "en"
        else "中期报告后的更新版项目计划",
        fontsize=12.5,
        color="#2C2C2A",
        pad=26,
        loc="left",
    )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fig_gantt_{lang}.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for lang in ("zh", "en"):
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        print("wrote", build(lang))
