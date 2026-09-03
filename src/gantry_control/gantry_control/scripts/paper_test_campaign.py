#!/usr/bin/env python3
"""
paper_test_campaign.py

Generate and run repeatable paper-quality gantry input-shaping experiments.

The campaign compares:
  - pulse baseline
  - IS1 adaptive paper TDF
  - IS2 colleague paper closed-form TDF
  - model-based robust ZVD

All runs use the gimbal encoder payload topic by default:
  /payload/pose_e_rel
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKSPACE = Path("~/crane_ws").expanduser()
DEFAULT_ROOT = WORKSPACE / "log"
PLAYER = "adaptive_paper_tdf_player.py"


@dataclass
class Method:
    key: str
    label: str
    profile: str
    extra_args: list[str]


@dataclass
class RunSpec:
    run_id: str
    method_key: str
    method_label: str
    profile: str
    repeat: int
    rope_length_m: float
    target_mm: float
    vmax_mm_s: float
    max_travel_mm: float
    residual_window_s: float
    command: list[str]
    csv_path: str
    terminal_log_path: str


def default_methods(args) -> list[Method]:
    common_adaptive = [
        "--a0",
        str(args.k),
        "--id-lock-mode",
        "best-cond",
        "--min-id-duration",
        str(args.min_id_duration),
        "--switch-margin",
        str(args.switch_margin),
        "--estimate-deadline",
        str(args.estimate_deadline),
        "--no-fallback",
        "--zv-t-min",
        str(args.zv_t_min),
        "--zv-t-max",
        str(args.zv_t_max),
        "--id-zeta-min",
        str(args.id_zeta_min),
        "--zeta-max",
        str(args.zeta_max),
        "--accept-valid-count",
        "1",
        "--stability-count",
        "1",
    ]
    return [
        Method("pulse", "Pulse baseline", "pulse", []),
        Method("is1", f"IS1 adaptive TDF, K={args.k:g}", "adaptive", common_adaptive),
        Method(
            "is2",
            f"IS2 paper closed, K={args.k:g}, tau={args.tau:g}s",
            "colleague-paper-closed",
            common_adaptive + ["--tau", str(args.tau)],
        ),
        Method(
            "robust",
            "Robust ZVD from rope length",
            "robust",
            [
                "--robust-zeta",
                str(args.robust_zeta),
                "--robust-t-scale",
                str(args.robust_t_scale),
            ],
        ),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-name", default="")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--axis", default="x", choices=["x", "y"])
    parser.add_argument("--payload-topic", default="/payload/pose_e_rel")
    parser.add_argument("--target-distance-mm", type=float, default=750.0)
    parser.add_argument("--vmax-mm-s", type=float, default=150.0)
    parser.add_argument("--max-travel-mm", type=float, default=800.0)
    parser.add_argument("--residual-window", type=float, default=10.0)
    parser.add_argument("--rope-lengths-m", nargs="+", type=float, default=[1.0, 1.2, 1.4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--k", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--min-id-duration", type=float, default=0.45)
    parser.add_argument("--switch-margin", type=float, default=0.05)
    parser.add_argument("--estimate-deadline", type=float, default=1.10)
    parser.add_argument("--zv-t-min", type=float, default=0.75)
    parser.add_argument("--zv-t-max", type=float, default=1.25)
    parser.add_argument("--id-zeta-min", type=float, default=-0.25)
    parser.add_argument("--zeta-max", type=float, default=0.0)
    parser.add_argument("--robust-zeta", type=float, default=0.0)
    parser.add_argument("--robust-t-scale", type=float, default=1.0)
    parser.add_argument("--stream-rate-hz", type=float, default=100.0)
    parser.add_argument("--print-period", type=float, default=0.25)
    parser.add_argument("--execute", action="store_true", help="Actually move the gantry.")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--no-prompts", action="store_true")
    parser.add_argument(
        "--set-rope-param",
        action="store_true",
        help="Try to set encoder rope length params before each rope-length block.",
    )
    parser.add_argument("--zip", action="store_true", default=True)
    return parser.parse_args()


def make_session_dir(args) -> Path:
    root = Path(args.root).expanduser()
    name = args.session_name or "paper_campaign_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    session = root / name
    for sub in ("raw", "terminal", "plots", "tables"):
        (session / sub).mkdir(parents=True, exist_ok=True)
    return session


def shell_quote(parts: list[str]) -> str:
    return " ".join(shlex_quote(p) for p in parts)


def shlex_quote(value: str) -> str:
    if value == "":
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=./:"
    if all(c in safe for c in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_run_specs(args, session: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    methods = default_methods(args)
    order = 1
    for rope in args.rope_lengths_m:
        rope_tag = f"L{int(round(rope * 100)):03d}"
        for repeat in range(1, args.repeats + 1):
            for method in methods:
                run_id = (
                    f"{order:03d}_{method.key}_{rope_tag}_"
                    f"x{int(args.target_distance_mm)}_v{int(abs(args.vmax_mm_s))}_rep{repeat:02d}"
                )
                csv_path = session / "raw" / f"{run_id}.csv"
                log_path = session / "terminal" / f"{run_id}.log"
                cmd = [
                    "ros2",
                    "run",
                    "gantry_control",
                    PLAYER,
                    "--axis",
                    args.axis,
                    "--profile",
                    method.profile,
                    "--payload-topic",
                    args.payload_topic,
                    "--target-distance-mm",
                    f"{args.target_distance_mm:g}",
                    "--vmax-mm-s",
                    f"{args.vmax_mm_s:g}",
                    "--residual-window",
                    f"{args.residual_window:g}",
                    "--max-travel-mm",
                    f"{args.max_travel_mm:g}",
                    "--stream-rate-hz",
                    f"{args.stream_rate_hz:g}",
                    "--print-period",
                    f"{args.print_period:g}",
                    "--log-csv",
                    str(csv_path),
                ]
                cmd.extend(method.extra_args)
                if method.profile == "robust":
                    cmd.extend(["--robust-rope-length-m", f"{rope:g}"])
                specs.append(
                    RunSpec(
                        run_id=run_id,
                        method_key=method.key,
                        method_label=method.label,
                        profile=method.profile,
                        repeat=repeat,
                        rope_length_m=rope,
                        target_mm=args.target_distance_mm,
                        vmax_mm_s=args.vmax_mm_s,
                        max_travel_mm=args.max_travel_mm,
                        residual_window_s=args.residual_window,
                        command=cmd,
                        csv_path=str(csv_path),
                        terminal_log_path=str(log_path),
                    )
                )
                order += 1
    return specs


def write_manifest(session: Path, args, specs: list[RunSpec]):
    data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(WORKSPACE),
        "encoder_payload_topic": args.payload_topic,
        "operator_notes": "All runs use gimbal encoder payload position.",
        "parameters": vars(args),
        "runs": [asdict(s) for s in specs],
    }
    (session / "manifest.json").write_text(json.dumps(data, indent=2))
    with (session / "manifest.csv").open("w", newline="") as f:
        fieldnames = list(asdict(specs[0]).keys()) if specs else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in specs:
            row = asdict(spec)
            row["command"] = shell_quote(row["command"])
            writer.writerow(row)


def write_run_commands(session: Path, specs: list[RunSpec]):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd ~/crane_ws",
        "source install/setup.bash",
        "",
    ]
    for spec in specs:
        lines.append(f"# {spec.run_id}: {spec.method_label}, rope L={spec.rope_length_m:.2f} m")
        lines.append(shell_quote(spec.command))
        lines.append("")
    path = session / "RUN_COMMANDS.sh"
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def write_test_plan(session: Path, args, specs: list[RunSpec]):
    plan = f"""# Adaptive Input Shaping Paper Test Plan

## Objective
Quantify residual payload swing and travel accuracy for four controllers using the gimbal encoder payload measurement:

- Pulse baseline
- IS1 adaptive TDF
- IS2 paper closed-form TDF
- Robust model-based ZVD

## Fixed Test Conditions
- Axis: {args.axis}
- Payload topic: `{args.payload_topic}`
- Target travel: {args.target_distance_mm:g} mm
- Command vmax: {args.vmax_mm_s:g} mm/s
- Max allowed travel: {args.max_travel_mm:g} mm
- Residual window: {args.residual_window:g} s
- Repeats per method/rope length: {args.repeats}
- Rope lengths: {", ".join(f"{v:.2f} m" for v in args.rope_lengths_m)}

## Method Definitions
- Pulse: constant velocity until `tf = distance / vmax`, then zero.
- IS1: online adaptive TDF with switches at `T, 2T, tf, tf+T, tf+2T`.
- IS2: paper closed-form offset shaper with switches at `tau+T, tau+2T, tf, tf+tau+T, tf+tau+2T`.
- Robust: fixed ZVD from `omega_n = sqrt(g/L)`, `T = pi / omega_n`, amplitudes `[1, 2K, K^2]/(1+2K+K^2)` where `K=exp(-zeta*pi/sqrt(1-zeta^2))`.

## Pre-Run Checklist
- Verify gantry is homed and workspace is clear.
- Verify gimbal encoder node is publishing `{args.payload_topic}`.
- Calibrate/reset payload origin from Mission Planner at the start pose.
- Set physical rope length to the block being tested.
- Confirm max travel has enough margin for the selected target.
- Keep payload mass, box geometry, start position, and camera/lighting unchanged.

## Per-Run Procedure
1. Move the cart to the same start position.
2. Wait for payload swing to settle.
3. Run the next command in `RUN_COMMANDS.sh` or use `--execute`.
4. Do not touch the payload during the residual window.
5. Record any visual anomaly in `operator_notes.md`.

## Required Paper Outputs
- Raw CSV per run in `raw/`.
- Terminal log per run in `terminal/`.
- `tables/summary_metrics.csv` for every run.
- `tables/summary_by_condition.csv` with mean/std by method and rope length.
- Plots in `plots/`: command/travel/swing overlays, residual comparison, travel error, schedule T, and residual boxplot.

## Primary Metrics
- Residual swing p2p, max absolute, and RMS over the configured residual window.
- Cart travel error.
- Online ID lock time, estimated `T`, damping, and conditioning number when available.
- Abort rate and no-estimate failures.

## Statistical Treatment
For each method and rope length, report mean and standard deviation across repeats. The paper comparison should use matched rope length, target, speed, payload, and repeat count.
"""
    (session / "TEST_PLAN.md").write_text(plan)
    (session / "operator_notes.md").write_text("# Operator Notes\n\n")


def read_csv_rows(path: Path) -> list[dict[str, float | str]]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            converted = {}
            for key, value in row.items():
                if value == "" or value is None:
                    converted[key] = math.nan
                else:
                    try:
                        converted[key] = float(value)
                    except ValueError:
                        converted[key] = value
            rows.append(converted)
    return rows


def finite(values):
    return [float(v) for v in values if isinstance(v, (float, int)) and math.isfinite(float(v))]


def last_finite(rows, key: str) -> float:
    for row in reversed(rows):
        value = row.get(key, math.nan)
        if isinstance(value, (float, int)) and math.isfinite(value):
            return float(value)
    return math.nan


def first_nonempty(rows, key: str) -> str:
    for row in rows:
        value = row.get(key, "")
        if isinstance(value, str) and value:
            return value
    return ""


def first_finite_time(rows, value_key: str) -> float:
    for row in rows:
        value = row.get(value_key, math.nan)
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            return float(row.get("move_time_sec", math.nan))
    return math.nan


def metrics_for_run(spec: RunSpec) -> dict[str, float | str]:
    path = Path(spec.csv_path)
    base: dict[str, float | str] = {
        "run_id": spec.run_id,
        "method": spec.method_key,
        "method_label": spec.method_label,
        "profile": spec.profile,
        "repeat": spec.repeat,
        "rope_length_m": spec.rope_length_m,
        "target_mm": spec.target_mm,
        "vmax_mm_s": spec.vmax_mm_s,
        "status": "missing",
    }
    if not path.exists() or path.stat().st_size == 0:
        return base
    rows = read_csv_rows(path)
    if not rows:
        base["status"] = "empty"
        return base

    cmd = [abs(float(r.get("cmd_vx_mm_s", 0.0) or 0.0)) + abs(float(r.get("cmd_vy_mm_s", 0.0) or 0.0)) for r in rows]
    times = [float(r.get("move_time_sec", math.nan)) for r in rows]
    last_motion_idx = max((i for i, v in enumerate(cmd) if math.isfinite(v) and v > 1.0e-6), default=len(rows) - 1)
    stop_time = times[last_motion_idx] if last_motion_idx < len(times) else math.nan
    residual = [
        float(r["swing_mm"])
        for r in rows
        if isinstance(r.get("swing_mm"), (float, int))
        and math.isfinite(float(r["swing_mm"]))
        and isinstance(r.get("move_time_sec"), (float, int))
        and float(r["move_time_sec"]) >= stop_time
    ]
    travel = last_finite(rows, "traveled_mm")
    schedule_source = first_nonempty(rows, "schedule_source")
    if not schedule_source and spec.profile == "pulse":
        schedule_source = "pulse"
    base.update(
        {
            "status": "ok",
            "travel_actual_mm": travel,
            "travel_error_mm": travel - spec.target_mm if math.isfinite(travel) else math.nan,
            "command_stop_time_s": stop_time,
            "run_duration_s": last_finite(rows, "move_time_sec"),
            "residual_samples": len(residual),
            "residual_p2p_mm": max(residual) - min(residual) if residual else math.nan,
            "residual_max_abs_mm": max(abs(v) for v in residual) if residual else math.nan,
            "residual_rms_mm": math.sqrt(sum(v * v for v in residual) / len(residual)) if residual else math.nan,
            "id_first_valid_time_s": first_finite_time(rows, "T_sec"),
            "schedule_source": schedule_source,
            "schedule_lock_time_s": first_finite_time(rows, "A0"),
            "schedule_T_s": last_finite(rows, "T_sec") if spec.profile != "pulse" else math.nan,
            "schedule_A0": last_finite(rows, "A0"),
            "schedule_A1": last_finite(rows, "A1"),
            "schedule_A2": last_finite(rows, "A2"),
            "id_omega_n_rad_s": last_finite(rows, "omega_n_rad_s"),
            "id_zeta": last_finite(rows, "id_zeta"),
            "id_cond_b": last_finite(rows, "cond_b"),
        }
    )
    return base


def write_table(path: Path, rows: list[dict[str, float | str]]):
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


def summarize_by_condition(metrics: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, float], list[dict[str, float | str]]] = {}
    for row in metrics:
        if row.get("status") != "ok":
            continue
        groups.setdefault((str(row["method"]), float(row["rope_length_m"])), []).append(row)
    summary = []
    metric_names = [
        "residual_p2p_mm",
        "residual_max_abs_mm",
        "residual_rms_mm",
        "travel_error_mm",
        "schedule_T_s",
        "id_first_valid_time_s",
        "schedule_lock_time_s",
    ]
    for (method, rope), rows in sorted(groups.items()):
        out: dict[str, float | str] = {
            "method": method,
            "rope_length_m": rope,
            "n": len(rows),
        }
        for name in metric_names:
            vals = finite([r.get(name, math.nan) for r in rows])
            out[f"{name}_mean"] = mean(vals) if vals else math.nan
            out[f"{name}_std"] = stdev(vals) if len(vals) >= 2 else 0.0 if vals else math.nan
        summary.append(out)
    return summary


def plot_campaign(session: Path, specs: list[RunSpec], metrics: list[dict[str, float | str]]):
    plots = session / "plots"
    colors = {
        "pulse": "#525252",
        "is1": "#1f77b4",
        "is2": "#d62728",
        "robust": "#2ca02c",
    }
    labels = {s.method_key: s.method_label for s in specs}

    for rope in sorted({s.rope_length_m for s in specs}):
        rope_specs = [s for s in specs if abs(s.rope_length_m - rope) < 1.0e-9]
        fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=False)
        for spec in rope_specs:
            path = Path(spec.csv_path)
            if not path.exists() or path.stat().st_size == 0:
                continue
            rows = read_csv_rows(path)
            if not rows:
                continue
            t = [float(r.get("move_time_sec", math.nan)) for r in rows]
            vx = [float(r.get("cmd_vx_mm_s", 0.0) or 0.0) for r in rows]
            travel = [float(r.get("traveled_mm", math.nan)) for r in rows]
            swing = [float(r.get("swing_mm", math.nan)) for r in rows]
            alpha = 0.45 if spec.repeat > 1 else 0.9
            label = f"{spec.method_key} rep{spec.repeat}"
            axes[0].plot(t, vx, color=colors.get(spec.method_key, "black"), alpha=alpha, label=label)
            axes[1].plot(t, travel, color=colors.get(spec.method_key, "black"), alpha=alpha)
            axes[2].plot(t, swing, color=colors.get(spec.method_key, "black"), alpha=alpha)
            stop_time = metrics_lookup(metrics, spec.run_id, "command_stop_time_s")
            if math.isfinite(stop_time):
                rt = [ti - stop_time for ti in t if ti >= stop_time]
                rs = [swing[i] for i, ti in enumerate(t) if ti >= stop_time]
                axes[3].plot(rt, rs, color=colors.get(spec.method_key, "black"), alpha=alpha)
        axes[0].set_title(f"Command and response overlay, rope length {rope:.2f} m")
        axes[0].set_ylabel("cmd vx [mm/s]")
        axes[1].set_ylabel("travel [mm]")
        axes[2].set_ylabel("swing [mm]")
        axes[3].set_ylabel("residual swing [mm]")
        axes[3].set_xlabel("time after command zero [s]")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="best", ncol=2, fontsize=8)
        save_fig(plots / f"overlay_L{int(round(rope * 100)):03d}.png")

    ok = [m for m in metrics if m.get("status") == "ok"]
    for metric, ylabel, filename in [
        ("residual_rms_mm", "residual RMS [mm]", "residual_rms_by_condition.png"),
        ("residual_p2p_mm", "residual p2p [mm]", "residual_p2p_by_condition.png"),
        ("travel_error_mm", "travel error [mm]", "travel_error_by_condition.png"),
        ("schedule_T_s", "schedule/ID T [s]", "schedule_T_by_condition.png"),
    ]:
        plt.figure(figsize=(11, 5.5))
        x_labels = []
        values = []
        bar_colors = []
        for rope in sorted({float(m["rope_length_m"]) for m in ok}):
            for method in ("pulse", "is1", "is2", "robust"):
                vals = finite([m.get(metric, math.nan) for m in ok if m["method"] == method and abs(float(m["rope_length_m"]) - rope) < 1.0e-9])
                if not vals:
                    continue
                x_labels.append(f"{method}\nL={rope:.2f}")
                values.append(mean(vals))
                bar_colors.append(colors.get(method, "#777777"))
        plt.bar(range(len(values)), values, color=bar_colors)
        plt.xticks(range(len(values)), x_labels, rotation=0)
        plt.ylabel(ylabel)
        plt.grid(True, axis="y", alpha=0.3)
        plt.title(ylabel)
        save_fig(plots / filename)

    plt.figure(figsize=(10, 6))
    data = []
    names = []
    for method in ("pulse", "is1", "is2", "robust"):
        vals = finite([m.get("residual_rms_mm", math.nan) for m in ok if m["method"] == method])
        if vals:
            data.append(vals)
            names.append(labels.get(method, method))
    if data:
        plt.boxplot(data, labels=names, showmeans=True)
        plt.ylabel("residual RMS [mm]")
        plt.title("Residual RMS distribution across all rope lengths")
        plt.grid(True, axis="y", alpha=0.3)
        save_fig(plots / "residual_rms_boxplot.png")
    else:
        plt.close()


def metrics_lookup(metrics: list[dict[str, float | str]], run_id: str, key: str) -> float:
    for row in metrics:
        if row.get("run_id") == run_id:
            value = row.get(key, math.nan)
            return float(value) if isinstance(value, (float, int)) else math.nan
    return math.nan


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_process(spec: RunSpec) -> int:
    env = os.environ.copy()
    log_path = Path(spec.terminal_log_path)
    with log_path.open("w") as log:
        log.write("$ " + shell_quote(spec.command) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            spec.command,
            cwd=str(WORKSPACE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def try_set_rope_length(rope: float):
    for node in ("/encoder_serial_node", "/encoder_node"):
        cmd = ["ros2", "param", "set", node, "rope_length_m", f"{rope:g}"]
        result = subprocess.run(cmd, cwd=str(WORKSPACE), text=True, capture_output=True)
        if result.returncode == 0:
            print(f"[campaign] Set {node} rope_length_m={rope:g}")
            return
    print("[campaign] Rope-length param was not set automatically; set/check it manually.")


def execute_campaign(args, specs: list[RunSpec]):
    current_rope = None
    for spec in specs:
        if current_rope != spec.rope_length_m:
            current_rope = spec.rope_length_m
            if args.set_rope_param:
                try_set_rope_length(current_rope)
            if not args.no_prompts:
                input(
                    f"\nSet rope length to {current_rope:.2f} m, reset payload origin, "
                    "return gantry to start, then press Enter..."
                )
        if not args.no_prompts:
            ans = input(f"Run {spec.run_id} ({spec.method_label})? [Enter=yes, s=skip] ").strip().lower()
            if ans == "s":
                continue
        print(f"[campaign] Running {spec.run_id}")
        code = run_process(spec)
        if code != 0:
            print(f"[campaign] WARNING: {spec.run_id} exited with code {code}")
        if not args.no_prompts:
            input("Let payload settle and return to start if needed, then press Enter...")


def analyze(session: Path, specs: list[RunSpec]):
    metrics = [metrics_for_run(spec) for spec in specs]
    summary = summarize_by_condition(metrics)
    write_table(session / "tables" / "summary_metrics.csv", metrics)
    write_table(session / "tables" / "summary_by_condition.csv", summary)
    plot_campaign(session, specs, metrics)
    write_analysis_notes(session, metrics, summary)
    return metrics, summary


def write_analysis_notes(session: Path, metrics, summary):
    lines = ["# Campaign Summary", ""]
    lines.append("## Condition Means")
    for row in summary:
        lines.append(
            f"- {row['method']} L={row['rope_length_m']:.2f}m n={row['n']}: "
            f"rms={row['residual_rms_mm_mean']:.2f} +/- {row['residual_rms_mm_std']:.2f} mm, "
            f"p2p={row['residual_p2p_mm_mean']:.2f} +/- {row['residual_p2p_mm_std']:.2f} mm, "
            f"travel_error={row['travel_error_mm_mean']:+.2f} +/- {row['travel_error_mm_std']:.2f} mm"
        )
    lines.append("")
    lines.append("## Interpretation Checklist")
    lines.append("- Compare methods only within the same rope length, target, speed, payload, and repeat count.")
    lines.append("- IS1 should switch earliest; it usually gives the strongest cancellation when ID is early enough.")
    lines.append("- IS2 delays cancellation by tau, so it may be more tolerant of later ID but can leave more residual p2p.")
    lines.append("- Robust ZVD is the fixed model-based baseline; degradation with rope-length mismatch shows sensitivity.")
    lines.append("- Pulse provides the uncontrolled residual reference.")
    (session / "SUMMARY.md").write_text("\n".join(lines))


def make_zip(session: Path):
    zip_base = str(session)
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=session.parent, base_dir=session.name)
    print(f"[campaign] ZIP ready: {zip_path}")


def load_existing_specs(session: Path) -> list[RunSpec]:
    manifest = json.loads((session / "manifest.json").read_text())
    return [RunSpec(**run) for run in manifest["runs"]]


def main():
    args = parse_args()
    session = make_session_dir(args)
    if args.analyze_only:
        specs = load_existing_specs(session)
    else:
        specs = build_run_specs(args, session)
        write_manifest(session, args, specs)
        write_run_commands(session, specs)
        write_test_plan(session, args, specs)

    print(f"[campaign] Session: {session}")
    print(f"[campaign] Runs: {len(specs)}")
    if args.execute:
        execute_campaign(args, specs)
    else:
        print("[campaign] Dry run only. Use --execute to move the gantry.")
    analyze(session, specs)
    if args.zip:
        make_zip(session)
    print("[campaign] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
