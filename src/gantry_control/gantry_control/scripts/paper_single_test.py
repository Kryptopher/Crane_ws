#!/usr/bin/env python3
"""
paper_single_test.py

Run one paper-quality gantry input-shaping test and save a self-contained
folder for that individual experiment. This is meant for operator-led testing:
choose one method, one rope length, one trial label, run it, then compare later.
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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKSPACE = Path("~/crane_ws").expanduser()
DEFAULT_ROOT = WORKSPACE / "log" / "paper_single_tests"
PLAYER = "adaptive_paper_tdf_player.py"


@dataclass
class SingleRun:
    run_id: str
    method: str
    profile: str
    rope_length_m: float
    target_mm: float
    vmax_mm_s: float
    k: float
    tau_s: float
    residual_window_s: float
    payload_topic: str
    csv_path: str
    terminal_log_path: str
    command: list[str]
    id_period_s: float = 4.0


METHODS = {
    "pulse": "pulse",
    "is1": "adaptive",
    "is2": "colleague-paper-closed",
    "robust": "robust",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--trial", default="", help="Human-readable trial label, e.g. rep01 or calm_start.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--axis", default="x", choices=["x", "y"])
    parser.add_argument("--payload-topic", default="/payload/pose_e_rel")
    parser.add_argument("--rope-length-m", type=float, default=1.20)
    parser.add_argument("--target-distance-mm", type=float, default=750.0)
    parser.add_argument("--vmax-mm-s", type=float, default=150.0)
    parser.add_argument("--max-travel-mm", type=float, default=800.0)
    parser.add_argument("--residual-window", type=float, default=3.0)
    parser.add_argument("--k", type=float, default=0.4)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--min-id-duration", type=float, default=0.45)
    parser.add_argument("--id-period", type=float, default=4.0)
    parser.add_argument("--switch-margin", type=float, default=0.05)
    parser.add_argument("--estimate-deadline", type=float, default=1.90)
    parser.add_argument("--zv-t-min", type=float, default=0.75)
    parser.add_argument("--zv-t-max", type=float, default=1.25)
    parser.add_argument("--id-zeta-min", type=float, default=-0.25)
    parser.add_argument("--zeta-max", type=float, default=0.05)
    parser.add_argument("--robust-zeta", type=float, default=0.0)
    parser.add_argument("--robust-t-scale", type=float, default=1.0)
    parser.add_argument("--stream-rate-hz", type=float, default=100.0)
    parser.add_argument("--print-period", type=float, default=0.25)
    parser.add_argument("--execute", action="store_true", help="Actually move the gantry.")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--operator", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--zip", action="store_true", default=True)
    return parser.parse_args()


def safe_tag(text: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in text.strip())
    return cleaned.strip("_")


def make_run_id(args) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rope = f"L{int(round(args.rope_length_m * 100)):03d}"
    base = (
        f"{stamp}_{args.method}_{rope}_"
        f"x{int(args.target_distance_mm)}_v{int(abs(args.vmax_mm_s))}"
    )
    if args.method in ("is1", "is2"):
        base += f"_k{str(args.k).replace('.', 'p')}"
    if args.method == "is2":
        base += f"_tau{str(args.tau).replace('.', 'p')}"
    if args.trial:
        base += "_" + safe_tag(args.trial)
    return base


def build_command(args, csv_path: Path) -> list[str]:
    profile = METHODS[args.method]
    cmd = [
        "ros2",
        "run",
        "gantry_control",
        PLAYER,
        "--axis",
        args.axis,
        "--profile",
        profile,
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
    if args.method in ("is1", "is2"):
        cmd.extend(
            [
                "--a0",
                f"{args.k:g}",
                "--id-lock-mode",
                "best-cond",
                "--min-id-duration",
                f"{args.min_id_duration:g}",
                "--id-period",
                f"{args.id_period:g}",
                "--switch-margin",
                f"{args.switch_margin:g}",
                "--estimate-deadline",
                f"{args.estimate_deadline:g}",
                "--no-fallback",
                "--zv-t-min",
                f"{args.zv_t_min:g}",
                "--zv-t-max",
                f"{args.zv_t_max:g}",
                "--id-zeta-min",
                f"{args.id_zeta_min:g}",
                "--zeta-max",
                f"{args.zeta_max:g}",
                "--accept-valid-count",
                "1",
                "--stability-count",
                "1",
            ]
        )
    if args.method == "is2":
        cmd.extend(["--tau", f"{args.tau:g}"])
    if args.method == "robust":
        cmd.extend(
            [
                "--robust-rope-length-m",
                f"{args.rope_length_m:g}",
                "--robust-zeta",
                f"{args.robust_zeta:g}",
                "--robust-t-scale",
                f"{args.robust_t_scale:g}",
            ]
        )
    return cmd


def shell_quote(parts: list[str]) -> str:
    return " ".join(shlex_quote(p) for p in parts)


def shlex_quote(value: str) -> str:
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=./:"
    if value and all(c in safe for c in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_metadata(args, run: SingleRun, run_dir: Path):
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "operator": args.operator,
        "notes": args.notes,
        "run": asdict(run),
        "parameters": vars(args),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "RUN_COMMAND.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd ~/crane_ws\n"
        "source install/setup.bash\n\n"
        + shell_quote(run.command)
        + "\n"
    )
    (run_dir / "RUN_COMMAND.sh").chmod(0o755)
    (run_dir / "operator_notes.md").write_text(
        "# Operator Notes\n\n"
        f"- Operator: {args.operator or 'not specified'}\n"
        f"- Notes: {args.notes or 'none'}\n\n"
        "Add observations here: start quality, payload settling, visible anomalies, aborts, etc.\n"
    )
    (run_dir / "README.md").write_text(
        f"# Single Test Run: {run.run_id}\n\n"
        f"- Method: `{run.method}` / profile `{run.profile}`\n"
        f"- Rope length: {run.rope_length_m:.3f} m\n"
        f"- Target: {run.target_mm:.1f} mm\n"
        f"- Vmax: {run.vmax_mm_s:.1f} mm/s\n"
        f"- Payload topic: `{run.payload_topic}`\n\n"
        "Outputs after execution:\n"
        "- `raw.csv`\n"
        "- `terminal.log`\n"
        "- `summary_metrics.csv`\n"
        "- `summary.txt`\n"
        "- `plots/*.png`\n"
    )


def run_process(run: SingleRun) -> int:
    log_path = Path(run.terminal_log_path)
    with log_path.open("w") as log:
        log.write("$ " + shell_quote(run.command) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            run.command,
            cwd=str(WORKSPACE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def read_rows(path: Path) -> list[dict[str, float | str]]:
    rows = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out = {}
            for key, value in row.items():
                if value == "" or value is None:
                    out[key] = math.nan
                else:
                    try:
                        out[key] = float(value)
                    except ValueError:
                        out[key] = value
            rows.append(out)
    return rows


def last_finite(rows, key: str) -> float:
    for row in reversed(rows):
        value = row.get(key, math.nan)
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            return float(value)
    return math.nan


def first_finite_time(rows, key: str) -> float:
    for row in rows:
        value = row.get(key, math.nan)
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            return float(row.get("move_time_sec", math.nan))
    return math.nan


def first_finite_row(rows, key: str) -> dict[str, float | str] | None:
    for row in rows:
        value = row.get(key, math.nan)
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            return row
    return None


def last_finite_row(rows, key: str) -> dict[str, float | str] | None:
    for row in reversed(rows):
        value = row.get(key, math.nan)
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            return row
    return None


def last_finite_row_until(
    rows,
    key: str,
    *,
    time_key: str,
    max_time_s: float,
) -> dict[str, float | str] | None:
    for row in reversed(rows):
        time_value = row.get(time_key, math.nan)
        value = row.get(key, math.nan)
        if (
            isinstance(time_value, (float, int))
            and math.isfinite(float(time_value))
            and float(time_value) <= max_time_s
            and isinstance(value, (float, int))
            and math.isfinite(float(value))
        ):
            return row
    return None


def finite_from_row(row: dict[str, float | str] | None, key: str) -> float:
    if row is None:
        return math.nan
    value = row.get(key, math.nan)
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return float(value)
    return math.nan


def first_source(rows) -> str:
    for row in rows:
        value = row.get("schedule_source", "")
        if isinstance(value, str) and value:
            return value
    return ""


def analyze(run: SingleRun, run_dir: Path):
    rows = read_rows(Path(run.csv_path))
    summary = {
        "run_id": run.run_id,
        "method": run.method,
        "profile": run.profile,
        "rope_length_m": run.rope_length_m,
        "target_mm": run.target_mm,
        "vmax_mm_s": run.vmax_mm_s,
        "status": "ok" if rows else "missing_or_empty",
    }
    if rows:
        cmd = [
            abs(float(r.get("cmd_vx_mm_s", 0.0) or 0.0))
            + abs(float(r.get("cmd_vy_mm_s", 0.0) or 0.0))
            for r in rows
        ]
        times = [float(r.get("move_time_sec", math.nan)) for r in rows]
        last_motion_idx = max((i for i, v in enumerate(cmd) if math.isfinite(v) and v > 1.0e-6), default=len(rows) - 1)
        stop_time = times[last_motion_idx]
        residual = [
            float(r["swing_mm"])
            for r in rows
            if isinstance(r.get("swing_mm"), (float, int))
            and math.isfinite(float(r["swing_mm"]))
            and isinstance(r.get("move_time_sec"), (float, int))
            and float(r["move_time_sec"]) >= stop_time
        ]
        travel = last_finite(rows, "traveled_mm")
        first_id_row = first_finite_row(rows, "T_sec")
        last_id_row = last_finite_row_until(
            rows,
            "T_sec",
            time_key="move_time_sec",
            max_time_s=run.id_period_s,
        )
        chosen_row = last_finite_row(rows, "schedule_id_time_s")
        schedule_T = last_finite(rows, "schedule_T_sec")
        if not math.isfinite(schedule_T):
            schedule_T = last_finite(rows, "T_sec") if run.method != "pulse" else math.nan
        chosen_id_T = last_finite(rows, "schedule_id_T_sec")
        if not math.isfinite(chosen_id_T):
            chosen_id_T = schedule_T
        summary.update(
            {
                "travel_actual_mm": travel,
                "travel_error_mm": travel - run.target_mm if math.isfinite(travel) else math.nan,
                "command_stop_time_s": stop_time,
                "run_duration_s": last_finite(rows, "move_time_sec"),
                "residual_samples": len(residual),
                "residual_p2p_mm": max(residual) - min(residual) if residual else math.nan,
                "residual_max_abs_mm": max(abs(v) for v in residual) if residual else math.nan,
                "residual_rms_mm": math.sqrt(sum(v * v for v in residual) / len(residual)) if residual else math.nan,
                "id_first_valid_time_s": first_finite_time(rows, "T_sec"),
                "id_first_valid_T_s": finite_from_row(first_id_row, "T_sec"),
                "id_period_s": run.id_period_s,
                "id_last_valid_time_s": finite_from_row(last_id_row, "move_time_sec"),
                "id_last_valid_T_s": finite_from_row(last_id_row, "T_sec"),
                "schedule_lock_time_s": first_finite_time(rows, "A0"),
                "schedule_source": first_source(rows) or ("pulse" if run.method == "pulse" else ""),
                "schedule_T_s": schedule_T,
                "schedule_locked_at_s": last_finite(rows, "schedule_locked_at_s"),
                "chosen_id_time_s": finite_from_row(chosen_row, "schedule_id_time_s"),
                "chosen_id_T_s": chosen_id_T,
                "schedule_A0": last_finite(rows, "A0"),
                "schedule_A1": last_finite(rows, "A1"),
                "schedule_A2": last_finite(rows, "A2"),
                "id_omega_n_rad_s": last_finite(rows, "omega_n_rad_s"),
                "id_zeta": last_finite(rows, "id_zeta"),
                "id_cond_b": last_finite(rows, "cond_b"),
            }
        )
        plot_single(rows, run, run_dir)

    with (run_dir / "summary_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    lines = ["Single Test Summary", ""]
    for key, value in summary.items():
        lines.append(f"{key}: {value}")
    (run_dir / "summary.txt").write_text("\n".join(lines))


def plot_single(rows, run: SingleRun, run_dir: Path):
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    t = [float(r.get("move_time_sec", math.nan)) for r in rows]
    vx = [float(r.get("cmd_vx_mm_s", 0.0) or 0.0) for r in rows]
    travel = [float(r.get("traveled_mm", math.nan)) for r in rows]
    swing = [float(r.get("swing_mm", math.nan)) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, vx, linewidth=2)
    axes[0].set_ylabel("cmd vx [mm/s]")
    axes[1].plot(t, travel, linewidth=2)
    axes[1].axhline(run.target_mm, color="black", linestyle="--", alpha=0.4)
    axes[1].set_ylabel("travel [mm]")
    axes[2].plot(t, swing, linewidth=1.8)
    axes[2].set_ylabel("swing [mm]")
    axes[2].set_xlabel("move time [s]")
    axes[0].set_title(run.run_id)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_fig(plots / "run_stack.png")

    cmd_abs = [abs(v) for v in vx]
    last_motion_idx = max((i for i, v in enumerate(cmd_abs) if math.isfinite(v) and v > 1.0e-6), default=len(rows) - 1)
    stop_time = t[last_motion_idx]
    rt = [ti - stop_time for ti in t if ti >= stop_time]
    rs = [swing[i] for i, ti in enumerate(t) if ti >= stop_time]
    plt.figure(figsize=(10, 4))
    plt.plot(rt, rs, linewidth=2)
    plt.xlabel("time after command zero [s]")
    plt.ylabel("residual swing [mm]")
    plt.title("Residual swing")
    plt.grid(True, alpha=0.3)
    save_fig(plots / "residual_swing.png")

    if run.method != "pulse":
        T_vals = [float(r.get("T_sec", math.nan)) for r in rows]
        cond_vals = [float(r.get("cond_b", math.nan)) for r in rows]
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, T_vals, linewidth=1.8)
        axes[0].set_ylabel("T estimate [s]")
        axes[1].plot(t, cond_vals, linewidth=1.8)
        axes[1].set_ylabel("condB")
        axes[1].set_xlabel("move time [s]")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        axes[0].set_title("Online ID trace")
        save_fig(plots / "id_trace.png")


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_zip(run_dir: Path):
    zip_path = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir.parent, base_dir=run_dir.name)
    print(f"[single-test] ZIP ready: {zip_path}")


def main():
    args = parse_args()
    run_id = make_run_id(args)
    run_dir = Path(args.root).expanduser() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    csv_path = run_dir / "raw.csv"
    terminal_log_path = run_dir / "terminal.log"
    command = build_command(args, csv_path)
    run = SingleRun(
        run_id=run_id,
        method=args.method,
        profile=METHODS[args.method],
        rope_length_m=args.rope_length_m,
        target_mm=args.target_distance_mm,
        vmax_mm_s=args.vmax_mm_s,
        k=args.k,
        tau_s=args.tau,
        residual_window_s=args.residual_window,
        payload_topic=args.payload_topic,
        csv_path=str(csv_path),
        terminal_log_path=str(terminal_log_path),
        command=command,
        id_period_s=args.id_period,
    )
    write_metadata(args, run, run_dir)
    print(f"[single-test] Run folder: {run_dir}")
    print("[single-test] Command:")
    print(shell_quote(command))
    if args.execute:
        if not args.no_prompt:
            input("Confirm rope length, reset payload origin, clear workspace, then press Enter...")
        code = run_process(run)
        if code != 0:
            print(f"[single-test] WARNING: player exited with code {code}")
    else:
        print("[single-test] Dry run only. Add --execute to move the gantry.")
    analyze(run, run_dir)
    if args.zip:
        make_zip(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
