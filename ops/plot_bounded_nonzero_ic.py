#!/usr/bin/env python3
"""Plot bounded-excitation/nonzero-IC trials without commanding hardware."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "log"
OUTPUT = LOG_DIR / "bounded_nonzero_ic_5deg_comparison.png"
PHASE_COLORS = {
    "excite": "#dbeafe",
    "id_hold": "#fef3c7",
    "wait_peak": "#fee2e2",
    "arm_profile": "#ffedd5",
    "armed_profile": "#e0e7ff",
    "maneuver": "#dcfce7",
    "residual": "#f3e8ff",
}
PHASE_ORDER = (
    "excite",
    "id_hold",
    "wait_peak",
    "arm_profile",
    "armed_profile",
    "maneuver",
    "residual",
)


def load_trials() -> list[tuple[pd.DataFrame, str, str]]:
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")
    paths = sorted(LOG_DIR.glob("bounded_nonzero_ic_5deg_rep*.csv"))
    if not paths:
        raise SystemExit(f"No bounded nonzero-IC logs found in {LOG_DIR}")
    return [
        (pd.read_csv(path), f"Rep {index}", colors[(index - 1) % len(colors)])
        for index, path in enumerate(paths, start=1)
    ]


def shade_phases(ax, data: pd.DataFrame) -> None:
    ranges = []
    for phase in PHASE_ORDER:
        rows = data[data["phase"] == phase]
        if not rows.empty:
            ranges.append((phase, rows["run_time_sec"].min(), rows["run_time_sec"].max(), False))

    # Rep 1/2 predate wait_peak logging. Mark their missing interval explicitly
    # rather than drawing a fictitious straight connector through it.
    if data[data["phase"] == "wait_peak"].empty:
        hold = data[data["phase"] == "id_hold"]
        maneuver = data[data["phase"] == "maneuver"]
        if not hold.empty and not maneuver.empty:
            ranges.append((
                "wait_peak",
                hold["run_time_sec"].max(),
                maneuver["run_time_sec"].min(),
                True,
            ))

    peak_arm_ranges = []
    for phase, start, end, missing in sorted(ranges, key=lambda item: item[1]):
        ax.axvspan(start, end, color=PHASE_COLORS[phase], alpha=0.55, linewidth=0)
        if phase in ("wait_peak", "arm_profile", "armed_profile") and not missing:
            peak_arm_ranges.append((start, end))
            continue
        label = phase.replace("_", " ") + ("\n(no CSV rows)" if missing else "")
        ax.text(
            0.5 * (start + end),
            0.985,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )
    if peak_arm_ranges:
        start = min(item[0] for item in peak_arm_ranges)
        end = max(item[1] for item in peak_arm_ranges)
        ax.text(
            0.5 * (start + end),
            0.985,
            "peak + arm",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            rotation=90 if end - start < 0.8 else 0,
        )


def angle_column(data: pd.DataFrame) -> str:
    if "swing_axis_angle_deg" in data and data["swing_axis_angle_deg"].notna().any():
        return "swing_axis_angle_deg"
    return "swing_pitch_deg"


def main() -> None:
    trials = load_trials()
    detail, detail_label, _ = trials[-1]

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), constrained_layout=True)
    fig.suptitle("Adaptive nonzero-IC: bounded 5° automatic excitation", fontsize=15)

    ax = axes[0]
    ax.plot(detail["run_time_sec"], detail["cart_q_mm"], color="#1d4ed8", lw=1.5)
    ax.set_ylabel("Trolley X (mm)")
    ax.grid(True, alpha=0.25)
    shade_phases(ax, detail)
    ax.set_title(f"{detail_label} trolley motion")

    ax = axes[1]
    ax.step(detail["run_time_sec"], detail["cmd_vx_mm_s"], where="post",
            color="#dc2626", lw=1.2, label="Commanded velocity")
    ax.plot(detail["run_time_sec"], detail["cart_vx_mm_s"],
            color="#1d4ed8", lw=1.2, label="Measured trolley velocity")
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.5)
    ax.set_ylabel("X velocity (mm/s)")
    ax.grid(True, alpha=0.25)
    shade_phases(ax, detail)
    ax.legend(loc="upper left")
    ax.set_title(f"{detail_label} commanded and measured trolley velocity")

    ax = axes[2]
    column = angle_column(detail)
    angle = detail[column].copy()
    angle.loc[detail["run_time_sec"].diff().gt(0.10)] = np.nan
    ax.plot(detail["run_time_sec"], angle, color="#7c3aed", lw=1.3)
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.5)
    ax.set_ylabel("Payload swing (deg)")
    ax.grid(True, alpha=0.25)
    shade_phases(ax, detail)
    ax.set_title(f"{detail_label} payload swing angle")

    ax = axes[3]
    last5_p2p = []
    for data, label, color in trials:
        residual = data[data["phase"] == "residual"].copy()
        residual["residual_time_s"] = residual["run_time_sec"] - residual["run_time_sec"].iloc[0]
        column = angle_column(residual)
        last5 = residual[
            residual["residual_time_s"] >= residual["residual_time_s"].max() - 5.0
        ][column].dropna()
        last5_p2p.append(f"{label} = {last5.max() - last5.min():.2f}°")
        ax.plot(residual["residual_time_s"], residual[column],
                color=color, lw=1.3, label=label)
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual swing (deg)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_title("Residual comparison — final 5 s p2p: " + ", ".join(last5_p2p))

    for axis in axes[:3]:
        axis.set_xlabel("Run time (s)")

    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
