#!/usr/bin/env python3
"""
Compute and plot residual swing energy per unit mass for paper single tests.

The energy model is the small-angle pendulum energy per payload mass:

    E / m = 0.5 * x_dot**2 + 0.5 * (g / L) * x**2

where x is the encoder-derived swing displacement in meters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path("~/crane_ws/log/paper_single_tests").expanduser()
DEFAULT_OUT = DEFAULT_ROOT / "residual_energy_analysis"
METHOD_ORDER = ("pulse", "robust", "is2")
METHOD_LABELS = {
    "pulse": "Pulse",
    "robust": "Robust ZVD",
    "is2": "IS2 / ISA2",
}
METHOD_COLORS = {
    "pulse": "#6b6b6b",
    "robust": "#2ca25f",
    "is2": "#d95f02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="paper_single_tests root")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="analysis output folder")
    parser.add_argument("--gravity", type=float, default=9.80665, help="gravity [m/s^2]")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="centered moving-average window for residual displacement samples",
    )
    parser.add_argument(
        "--min-residual-samples",
        type=int,
        default=20,
        help="minimum post-stop samples required for a valid run",
    )
    parser.add_argument(
        "--include-method",
        action="append",
        choices=METHOD_ORDER,
        help="method to include; may be repeated. Default: pulse, robust, is2",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    out[key] = math.nan
                    continue
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
            rows.append(out)
    return rows


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(newline="") as f:
        try:
            return next(csv.DictReader(f))
        except StopIteration:
            return {}


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def discover_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    run_dirs = []
    for group in sorted(root.glob("encoder_L*_v*_x*")):
        if not group.is_dir():
            continue
        for candidate in sorted(group.iterdir()):
            if candidate.is_dir():
                run_dirs.append(candidate)
    return run_dirs


def repeat_from_run_id(run_id: str) -> int | str:
    match = re.search(r"_rep(\d+)(?:_|$)", run_id)
    if match:
        return int(match.group(1))
    return ""


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1 or values.size < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def residual_window(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    cmd = [
        abs(finite_float(row.get("cmd_vx_mm_s"), 0.0))
        + abs(finite_float(row.get("cmd_vy_mm_s"), 0.0))
        for row in rows
    ]
    times = [finite_float(row.get("move_time_sec")) for row in rows]
    last_motion_idx = max(
        (i for i, value in enumerate(cmd) if math.isfinite(value) and value > 1.0e-6),
        default=len(rows) - 1,
    )
    stop_time = times[last_motion_idx] if last_motion_idx < len(times) else math.nan
    residual = [
        row
        for row in rows
        if math.isfinite(finite_float(row.get("move_time_sec")))
        and math.isfinite(finite_float(row.get("swing_mm")))
        and finite_float(row.get("move_time_sec")) >= stop_time
    ]
    return residual, stop_time


def energy_for_run(
    run_dir: Path,
    *,
    gravity: float,
    smooth_window: int,
    min_residual_samples: int,
    allowed_methods: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    if "ABORTED" in run_dir.name:
        return None, f"excluded aborted run: {run_dir.name}"

    metadata_path = run_dir / "metadata.json"
    raw_path = run_dir / "raw.csv"
    if not metadata_path.exists():
        return None, f"missing metadata.json: {run_dir.name}"
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return None, f"missing or empty raw.csv: {run_dir.name}"

    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"bad metadata.json in {run_dir.name}: {exc}"

    run = metadata.get("run", {})
    method = str(run.get("method", ""))
    if method not in allowed_methods:
        return None, f"excluded method {method or 'unknown'}: {run_dir.name}"

    rope_length_m = finite_float(run.get("rope_length_m"))
    if not math.isfinite(rope_length_m) or rope_length_m <= 0.0:
        return None, f"bad rope length in {run_dir.name}: {run.get('rope_length_m')!r}"

    rows = read_csv_rows(raw_path)
    if not rows:
        return None, f"empty raw.csv rows: {run_dir.name}"
    residual, stop_time = residual_window(rows)
    if len(residual) < min_residual_samples:
        return None, (
            f"too few residual samples in {run_dir.name}: "
            f"{len(residual)} < {min_residual_samples}"
        )

    t = np.asarray([finite_float(row.get("move_time_sec")) for row in residual], dtype=np.float64)
    x = np.asarray([finite_float(row.get("swing_mm")) / 1000.0 for row in residual], dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(x)
    t = t[valid]
    x = x[valid]
    if t.size < min_residual_samples:
        return None, f"too few finite residual samples in {run_dir.name}: {t.size}"
    if np.any(np.diff(t) <= 0.0):
        order = np.argsort(t)
        t = t[order]
        x = x[order]
        unique = np.concatenate(([True], np.diff(t) > 0.0))
        t = t[unique]
        x = x[unique]
    if t.size < min_residual_samples:
        return None, f"too few unique residual samples in {run_dir.name}: {t.size}"

    x_zeroed = x - float(np.mean(x))
    x_smooth = centered_moving_average(x_zeroed, smooth_window)
    x_dot = np.gradient(x_smooth, t)
    energy = 0.5 * x_dot * x_dot + 0.5 * (float(gravity) / rope_length_m) * x_smooth * x_smooth
    finite_energy = energy[np.isfinite(energy)]
    if finite_energy.size == 0:
        return None, f"no finite energy values in {run_dir.name}"

    summary = read_summary(run_dir / "summary_metrics.csv")
    run_id = str(run.get("run_id") or run_dir.name)
    target_mm = finite_float(run.get("target_mm"))
    vmax_mm_s = finite_float(run.get("vmax_mm_s"))
    out: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "repeat": repeat_from_run_id(run_id),
        "rope_length_m": rope_length_m,
        "target_mm": target_mm,
        "vmax_mm_s": vmax_mm_s,
        "command_stop_time_s": stop_time,
        "residual_samples": int(finite_energy.size),
        "residual_duration_s": float(t[-1] - t[0]) if t.size >= 2 else math.nan,
        "smooth_window": int(smooth_window if smooth_window % 2 == 1 else smooth_window + 1),
        "energy_mean_j_per_kg": float(np.mean(finite_energy)),
        "energy_median_j_per_kg": float(np.median(finite_energy)),
        "energy_max_j_per_kg": float(np.max(finite_energy)),
        "energy_std_j_per_kg": float(np.std(finite_energy, ddof=1))
        if finite_energy.size >= 2
        else 0.0,
        "swing_mean_removed_m": float(np.mean(x)),
        "swing_rms_m": float(np.sqrt(np.mean(x_zeroed * x_zeroed))),
        "residual_rms_mm": finite_float(summary.get("residual_rms_mm")),
        "residual_p2p_mm": finite_float(summary.get("residual_p2p_mm")),
        "travel_error_mm": finite_float(summary.get("travel_error_mm")),
    }
    return out, None


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return math.nan
    return float(np.quantile(finite, q))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["rope_length_m"]), str(row["method"]))].append(row)

    summary = []
    for (rope, method), group in sorted(groups.items()):
        vals = [finite_float(row.get("energy_mean_j_per_kg")) for row in group]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        summary.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "rope_length_m": rope,
                "n": int(arr.size),
                "energy_mean_j_per_kg_mean": float(np.mean(arr)),
                "energy_mean_j_per_kg_std": float(np.std(arr, ddof=1))
                if arr.size >= 2
                else 0.0,
                "energy_mean_j_per_kg_median": float(np.median(arr)),
                "energy_mean_j_per_kg_q1": quantile(vals, 0.25),
                "energy_mean_j_per_kg_q3": quantile(vals, 0.75),
                "energy_mean_j_per_kg_min": float(np.min(arr)),
                "energy_mean_j_per_kg_max": float(np.max(arr)),
            }
        )
    return summary


def consistent_title_suffix(rows: list[dict[str, Any]]) -> str:
    targets = {
        int(round(finite_float(row.get("target_mm"))))
        for row in rows
        if math.isfinite(finite_float(row.get("target_mm")))
    }
    speeds = {
        int(round(abs(finite_float(row.get("vmax_mm_s")))))
        for row in rows
        if math.isfinite(finite_float(row.get("vmax_mm_s")))
    }
    parts = []
    if len(targets) == 1:
        parts.append(f"x{next(iter(targets))}")
    if len(speeds) == 1:
        parts.append(f"v{next(iter(speeds))}")
    return ", ".join(parts)


def plot_boxplot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    rope_lengths = sorted({float(row["rope_length_m"]) for row in rows})
    offsets = {"pulse": -0.22, "robust": 0.0, "is2": 0.22}
    width = 0.16 if len(rope_lengths) > 1 else 0.12

    fig, ax = plt.subplots(figsize=(11, 6))
    legend_handles = []
    for method in METHOD_ORDER:
        color = METHOD_COLORS[method]
        legend_handles.append(
            plt.Line2D([0], [0], color=color, lw=8, label=METHOD_LABELS[method])
        )
        for i, rope in enumerate(rope_lengths):
            vals = [
                finite_float(row.get("energy_mean_j_per_kg"))
                for row in rows
                if row["method"] == method and abs(float(row["rope_length_m"]) - rope) < 1.0e-9
            ]
            vals = [v for v in vals if math.isfinite(v)]
            if not vals:
                continue
            pos = i + offsets[method]
            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=width,
                patch_artist=True,
                showmeans=True,
                manage_ticks=False,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": color,
                    "markersize": 5,
                },
                medianprops={"color": "black", "linewidth": 1.4},
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.55},
                whiskerprops={"color": color, "linewidth": 1.2},
                capprops={"color": color, "linewidth": 1.2},
                flierprops={
                    "marker": "o",
                    "markerfacecolor": color,
                    "markeredgecolor": color,
                    "alpha": 0.45,
                    "markersize": 4,
                },
            )
            # Keep mypy/linters happy when matplotlib changes return shape.
            _ = bp
            jitter = np.linspace(-width * 0.22, width * 0.22, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                color=color,
                edgecolor="black",
                linewidth=0.35,
                s=28,
                zorder=3,
                alpha=0.9,
            )

    title = "Residual Energy per Unit Mass"
    suffix = consistent_title_suffix(rows)
    if suffix:
        title += f" ({suffix})"
    ax.set_title(title)
    ax.set_ylabel("V_res / m [J/kg]")
    ax.set_xlabel("Rope length [m]")
    ax.set_xticks(range(len(rope_lengths)))
    ax.set_xticklabels([f"{rope:.2f}" for rope in rope_lengths])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_energy_boxplot.png", dpi=220)
    fig.savefig(out_dir / "residual_energy_boxplot.pdf")
    plt.close(fig)


def plot_mean_trend(summary: list[dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in METHOD_ORDER:
        rows = [row for row in summary if row["method"] == method]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: float(row["rope_length_m"]))
        x = np.asarray([float(row["rope_length_m"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["energy_mean_j_per_kg_mean"]) for row in rows], dtype=np.float64)
        e = np.asarray([float(row["energy_mean_j_per_kg_std"]) for row in rows], dtype=np.float64)
        ax.errorbar(
            x,
            y,
            yerr=e,
            marker="o",
            capsize=4,
            linewidth=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_title("Mean Residual Energy Trend")
    ax.set_xlabel("Rope length [m]")
    ax.set_ylabel("Mean V_res / m [J/kg]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_energy_mean_trend.png", dpi=220)
    plt.close(fig)


def print_report(rows: list[dict[str, Any]], warnings: list[str]) -> None:
    counts: dict[tuple[str, float], int] = defaultdict(int)
    for row in rows:
        counts[(str(row["method"]), float(row["rope_length_m"]))] += 1
    print("[energy] Valid runs:")
    for rope in sorted({key[1] for key in counts}):
        parts = []
        for method in METHOD_ORDER:
            n = counts.get((method, rope), 0)
            if n:
                parts.append(f"{method}={n}")
        print(f"[energy]   L={rope:.2f} m: " + ", ".join(parts))
    if warnings:
        print("[energy] Warnings/exclusions:")
        for warning in warnings:
            print(f"[energy]   - {warning}")


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_methods = set(args.include_method or METHOD_ORDER)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs(root):
        row, warning = energy_for_run(
            run_dir,
            gravity=args.gravity,
            smooth_window=args.smooth_window,
            min_residual_samples=args.min_residual_samples,
            allowed_methods=allowed_methods,
        )
        if row is not None:
            rows.append(row)
        if warning is not None:
            warnings.append(warning)

    if not rows:
        print(f"[energy] No valid runs found under {root}", file=sys.stderr)
        for warning in warnings:
            print(f"[energy]   - {warning}", file=sys.stderr)
        return 2

    rows.sort(key=lambda row: (float(row["rope_length_m"]), METHOD_ORDER.index(str(row["method"])), row["run_id"]))
    summary = summarize(rows)
    write_table(out_dir / "residual_energy_runs.csv", rows)
    write_table(out_dir / "residual_energy_summary.csv", summary)
    plot_boxplot(rows, out_dir)
    plot_mean_trend(summary, out_dir)
    print_report(rows, warnings)
    print(f"[energy] Wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
