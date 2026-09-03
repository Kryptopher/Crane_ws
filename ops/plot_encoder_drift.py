#!/usr/bin/env python3
"""
Plot payload encoder pitch/roll over time to check for drift.

Reads a logger_sync_pose_*.csv (from experiment_logger's logger_node,
~/payload_logs by default) and plots pitch_deg/roll_deg (and the raw
pitch_count/roll_count) vs time_sec. Fits a straight line to each and
prints the slope (deg/min) — a non-zero slope while the payload is at
rest is drift.

Usage:
  # Plot the most recent run in ~/payload_logs
  python3 plot_encoder_drift.py

  # Plot a specific run
  python3 plot_encoder_drift.py --csv ~/payload_logs/logger_sync_pose_20260710_152523.csv

  # Only look at a stationary window (e.g. skip the first 5s of motion)
  python3 plot_encoder_drift.py --start-s 5 --end-s 60
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LOG_DIR = Path("~/payload_logs").expanduser()


def find_latest_csv(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("logger_sync_pose_*.csv"))
    if not candidates:
        raise SystemExit(f"No logger_sync_pose_*.csv files found in {log_dir}")
    return candidates[-1]


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path} has no data rows")
    fields = reader.fieldnames or []
    out = {}
    for field in fields:
        try:
            out[field] = np.array([float(r[field]) for r in rows])
        except (TypeError, ValueError):
            continue
    return out


def fit_slope(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares line fit. Returns (slope_per_s, intercept)."""
    if len(t) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    slope, intercept = np.polyfit(t, y, 1)
    return float(slope), float(intercept)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None,
                         help="path to logger_sync_pose_*.csv (default: latest in --log-dir)")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--start-s", type=float, default=None,
                         help="ignore samples before this time_sec (e.g. skip motion transient)")
    parser.add_argument("--end-s", type=float, default=None,
                         help="ignore samples after this time_sec")
    parser.add_argument("--out", type=Path, default=None,
                         help="output PNG path (default: alongside the CSV)")
    args = parser.parse_args()

    csv_path = args.csv.expanduser() if args.csv else find_latest_csv(args.log_dir.expanduser())
    print(f"Reading {csv_path}")
    data = load_csv(csv_path)

    required = ("time_sec", "pitch_deg", "roll_deg", "pitch_count", "roll_count")
    missing = [f for f in required if f not in data]
    if missing:
        raise SystemExit(
            f"{csv_path} is missing columns {missing} — was encoder_serial_node "
            "running (start_encoder:=true) during this log?"
        )

    t = data["time_sec"]
    mask = np.ones_like(t, dtype=bool)
    if args.start_s is not None:
        mask &= t >= args.start_s
    if args.end_s is not None:
        mask &= t <= args.end_s
    if not mask.any():
        raise SystemExit("--start-s/--end-s window excludes all samples")

    t = t[mask]
    t0 = t[0]
    t_rel = t - t0

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ("pitch_deg", "Pitch [deg]", axes[0, 0]),
        ("roll_deg", "Roll [deg]", axes[0, 1]),
        ("pitch_count", "Pitch encoder [counts]", axes[1, 0]),
        ("roll_count", "Roll encoder [counts]", axes[1, 1]),
    ]
    for field, ylabel, ax in panels:
        y = data[field][mask]
        slope_per_s, intercept = fit_slope(t_rel, y)
        slope_per_min = slope_per_s * 60.0
        fit_line = intercept + slope_per_s * t_rel
        ax.plot(t_rel, y, lw=0.8, color="#2c7fb8", label="measured")
        ax.plot(t_rel, fit_line, lw=1.2, ls="--", color="#d95f02",
                 label=f"drift fit: {slope_per_min:+.4g}/min")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        unit = "deg/min" if "deg" in field else "counts/min"
        print(f"{field}: drift = {slope_per_min:+.4g} {unit} "
              f"over {t_rel[-1]:.1f}s ({len(y)} samples)")

    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    fig.suptitle(f"Encoder drift — {csv_path.name}")
    fig.tight_layout()

    out_path = args.out.expanduser() if args.out else csv_path.with_suffix(".drift.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
