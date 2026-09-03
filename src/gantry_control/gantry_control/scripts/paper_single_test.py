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
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    id_method: str = "integral"
    id_lowpass_hz: float = 0.0
    is2_selection_window_s: float = 0.5


METHODS = {
    "idonly": "pulse",
    "pulse": "pulse",
    "is1": "adaptive",
    "is2": "colleague-paper-closed",
    "nonrobust": "zv",
    "robust": "robust",
    "zv": "zv",
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
    parser.add_argument("--residual-window", type=float, default=10.0)
    parser.add_argument("--k", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--min-id-duration", type=float, default=0.45)
    parser.add_argument("--id-period", type=float, default=4.0)
    parser.add_argument(
        "--id-method",
        choices=("integral", "paper2-step", "zero-zeta-ls", "freq-bank"),
        default="integral",
    )
    parser.add_argument("--eq21-input-model", choices=("measured_cart", "ideal_k"), default="measured_cart")
    parser.add_argument("--final-id-source", choices=("active_id", "zero_zeta"), default="active_id")
    parser.add_argument("--id-lowpass-hz", type=float, default=0.0)
    parser.add_argument("--integral-id-window-s", type=float, default=1.6)
    parser.add_argument("--id-assume-zero-zeta", action="store_true")
    parser.add_argument("--mode-fit-window-s", type=float, default=1.00)
    parser.add_argument("--mode-fit-history-s", type=float, default=2.00)
    parser.add_argument("--mode-fit-after-s", type=float, default=0.80)
    parser.add_argument("--mode-fit-min-samples", type=int, default=40)
    parser.add_argument("--mode-fit-t-min", type=float, default=0.60)
    parser.add_argument("--mode-fit-t-max", type=float, default=1.25)
    parser.add_argument("--mode-fit-grid-count", type=int, default=160)
    parser.add_argument("--mode-fit-min-p2p-mm", type=float, default=8.0)
    parser.add_argument("--mode-fit-min-amp-mm", type=float, default=3.0)
    parser.add_argument("--mode-fit-max-norm-rmse", type=float, default=0.85)
    parser.add_argument("--mode-fit-cond-weight", type=float, default=0.01)
    parser.add_argument("--mode-fit-edge-margin-s", type=float, default=0.02)
    parser.add_argument("--zero-zeta-window-s", type=float, default=0.0)
    parser.add_argument("--zero-zeta-history-s", type=float, default=4.0)
    parser.add_argument("--zero-zeta-after-s", type=float, default=0.8)
    parser.add_argument("--zero-zeta-update-period-s", type=float, default=0.10)
    parser.add_argument("--zero-zeta-min-samples", type=int, default=80)
    parser.add_argument("--zero-zeta-t-min", type=float, default=0.45)
    parser.add_argument("--zero-zeta-t-max", type=float, default=1.45)
    parser.add_argument("--zero-zeta-grid-count", type=int, default=240)
    parser.add_argument("--zero-zeta-min-p2p-mm", type=float, default=4.0)
    parser.add_argument("--zero-zeta-min-amp-mm", type=float, default=1.0)
    parser.add_argument("--zero-zeta-max-norm-rmse", type=float, default=1.5)
    parser.add_argument("--two-mode-window-s", type=float, default=1.20)
    parser.add_argument("--two-mode-history-s", type=float, default=2.00)
    parser.add_argument("--two-mode-after-s", type=float, default=0.90)
    parser.add_argument("--two-mode-update-period-s", type=float, default=0.05)
    parser.add_argument("--two-mode-min-samples", type=int, default=55)
    parser.add_argument("--two-mode-t1-min", type=float, default=0.50)
    parser.add_argument("--two-mode-t1-max", type=float, default=1.35)
    parser.add_argument("--two-mode-t1-grid-count", type=int, default=80)
    parser.add_argument("--two-mode-t2-min", type=float, default=0.12)
    parser.add_argument("--two-mode-t2-max", type=float, default=0.80)
    parser.add_argument("--two-mode-t2-grid-count", type=int, default=55)
    parser.add_argument("--two-mode-max-t2-t1-ratio", type=float, default=0.75)
    parser.add_argument("--two-mode-min-amp1-mm", type=float, default=4.0)
    parser.add_argument("--two-mode-max-norm-rmse", type=float, default=0.75)
    parser.add_argument("--two-mode-cond-weight", type=float, default=0.01)
    parser.add_argument("--two-mode-amp2-weight", type=float, default=0.0)
    parser.add_argument("--paper2-window-s", type=float, default=1.60)
    parser.add_argument("--paper2-history-s", type=float, default=2.50)
    parser.add_argument("--paper2-after-s", type=float, default=0.80)
    parser.add_argument("--paper2-min-samples", type=int, default=40)
    parser.add_argument("--paper2-min-p2p-mm", type=float, default=8.0)
    parser.add_argument("--paper2-min-peak-mm", type=float, default=3.0)
    parser.add_argument("--paper2-min-peak-dt", type=float, default=0.20)
    parser.add_argument("--paper2-min-extrema", type=int, default=3)
    parser.add_argument("--paper2-smooth-radius", type=int, default=2)
    parser.add_argument("--is2-selection-window-s", type=float, default=0.5)
    parser.add_argument(
        "--is2-selection-mode",
        choices=("median", "recent-median", "stable-window", "stable", "latest"),
        default="recent-median",
    )
    parser.add_argument("--is2-stable-window-s", type=float, default=0.50)
    parser.add_argument("--is2-stable-step-s", type=float, default=0.05)
    parser.add_argument("--is2-stable-after-s", type=float, default=1.00)
    parser.add_argument("--is2-stable-min-count", type=int, default=12)
    parser.add_argument("--is2-stable-slope-weight", type=float, default=1.0)
    parser.add_argument("--is2-stable-max-range-s", type=float, default=0.45)
    parser.add_argument("--is2-id-t-min", type=float, default=0.0)
    parser.add_argument("--is2-id-t-max", type=float, default=0.0)
    parser.add_argument("--no-is2-schedule-filter", action="store_true")
    parser.add_argument("--is2-schedule-margin-s", type=float, default=0.50)
    parser.add_argument(
        "--fixed-id-t-sec",
        type=float,
        default=0.0,
        help="Bypass online ID and inject this shaper ID period T [s] at tau. <=0 disables.",
    )
    parser.add_argument(
        "--fixed-id-rope-length-m",
        type=float,
        default=0.0,
        help="Bypass online ID using omega=sqrt(g/L) for this rope length. <=0 disables.",
    )
    parser.add_argument("--fixed-id-zeta", type=float, default=0.0)
    parser.add_argument("--switch-margin", type=float, default=0.05)
    parser.add_argument(
        "--estimate-deadline",
        type=float,
        default=None,
        help="Adaptive estimate deadline [s]. Default: tau-0.1 for IS2/idonly, 1.9 otherwise.",
    )
    parser.add_argument(
        "--zv-t-min",
        type=float,
        default=None,
        help="Minimum accepted schedule T. Default: 0.75 for IS1, 0.50 for IS2, 0.20 otherwise.",
    )
    parser.add_argument("--zv-t-max", type=float, default=1.25)
    parser.add_argument("--id-zeta-min", type=float, default=-0.25)
    parser.add_argument("--zeta-max", type=float, default=0.0)
    parser.add_argument("--robust-zeta", type=float, default=0.0)
    parser.add_argument("--robust-t-scale", type=float, default=1.0)
    parser.add_argument("--stream-rate-hz", type=float, default=100.0)
    parser.add_argument("--print-period", type=float, default=0.25)
    parser.add_argument("--execute", action="store_true", help="Actually move the gantry.")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument(
        "--no-reset-origin",
        action="store_true",
        help="Skip the encoder origin reset that normally runs just before the move.",
    )
    parser.add_argument("--operator", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--zip", action="store_true", default=True)
    args = parser.parse_args()
    if args.is2_selection_mode == "stable":
        args.is2_selection_mode = "stable-window"
    if args.estimate_deadline is None:
        if args.method in ("idonly", "is2"):
            args.estimate_deadline = max(0.05, args.tau - 0.1)
        else:
            args.estimate_deadline = 1.90
    return args


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
    if args.method in ("idonly", "is1", "is2"):
        base += f"_k{str(args.k).replace('.', 'p')}"
    if args.method in ("idonly", "is2"):
        base += f"_tau{str(args.tau).replace('.', 'p')}"
        if args.id_method == "paper2-step":
            base += "_paper2step"
        elif args.id_method == "zero-zeta-ls":
            base += "_zerozeta"
        elif args.id_method == "freq-bank":
            base += "_freqbank"
        elif args.integral_id_window_s > 0.0:
            base += f"_iw{str(args.integral_id_window_s).replace('.', 'p')}"
        if args.is2_selection_mode == "stable-window":
            base += f"_stable{str(args.is2_stable_window_s).replace('.', 'p')}"
    if args.method in ("idonly", "is1", "is2"):
        if args.fixed_id_t_sec > 0.0:
            base += f"_fixedT{str(args.fixed_id_t_sec).replace('.', 'p')}"
        elif args.fixed_id_rope_length_m > 0.0:
            base += f"_fixedL{int(round(args.fixed_id_rope_length_m * 100)):03d}"
    if args.method in ("idonly", "is1", "is2") and args.id_lowpass_hz > 0.0:
        base += f"_lpf{str(args.id_lowpass_hz).replace('.', 'p')}hz"
    if args.trial:
        base += "_" + safe_tag(args.trial)
    return base


def build_command(args, csv_path: Path) -> list[str]:
    profile = METHODS[args.method]
    if args.zv_t_min is not None:
        zv_t_min = args.zv_t_min
    elif args.method == "is1":
        zv_t_min = 0.75
    elif args.method == "is2":
        zv_t_min = 0.20
    else:
        zv_t_min = 0.20
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
    if args.method in ("idonly", "is1", "is2"):
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
                "--id-method",
                args.id_method,
                "--eq21-input-model",
                args.eq21_input_model,
                "--final-id-source",
                args.final_id_source,
                "--id-lowpass-hz",
                f"{args.id_lowpass_hz:g}",
                "--integral-id-window-s",
                f"{args.integral_id_window_s:g}",
                "--paper2-window-s",
                f"{args.paper2_window_s:g}",
                "--paper2-history-s",
                f"{args.paper2_history_s:g}",
                "--paper2-after-s",
                f"{args.paper2_after_s:g}",
                "--paper2-min-samples",
                f"{args.paper2_min_samples:d}",
                "--paper2-min-p2p-mm",
                f"{args.paper2_min_p2p_mm:g}",
                "--paper2-min-peak-mm",
                f"{args.paper2_min_peak_mm:g}",
                "--paper2-min-peak-dt",
                f"{args.paper2_min_peak_dt:g}",
                "--paper2-min-extrema",
                f"{args.paper2_min_extrema:d}",
                "--paper2-smooth-radius",
                f"{args.paper2_smooth_radius:d}",
                "--zero-zeta-window-s",
                f"{args.zero_zeta_window_s:g}",
                "--zero-zeta-history-s",
                f"{args.zero_zeta_history_s:g}",
                "--zero-zeta-after-s",
                f"{args.zero_zeta_after_s:g}",
                "--zero-zeta-update-period-s",
                f"{args.zero_zeta_update_period_s:g}",
                "--zero-zeta-min-samples",
                f"{args.zero_zeta_min_samples:d}",
                "--zero-zeta-t-min",
                f"{args.zero_zeta_t_min:g}",
                "--zero-zeta-t-max",
                f"{args.zero_zeta_t_max:g}",
                "--zero-zeta-grid-count",
                f"{args.zero_zeta_grid_count:d}",
                "--zero-zeta-min-p2p-mm",
                f"{args.zero_zeta_min_p2p_mm:g}",
                "--zero-zeta-min-amp-mm",
                f"{args.zero_zeta_min_amp_mm:g}",
                "--zero-zeta-max-norm-rmse",
                f"{args.zero_zeta_max_norm_rmse:g}",
                "--is2-selection-window-s",
                f"{args.is2_selection_window_s:g}",
                "--is2-selection-mode",
                args.is2_selection_mode,
                "--is2-stable-window-s",
                f"{args.is2_stable_window_s:g}",
                "--is2-stable-step-s",
                f"{args.is2_stable_step_s:g}",
                "--is2-stable-after-s",
                f"{args.is2_stable_after_s:g}",
                "--is2-stable-min-count",
                f"{args.is2_stable_min_count:d}",
                "--is2-stable-slope-weight",
                f"{args.is2_stable_slope_weight:g}",
                "--is2-stable-max-range-s",
                f"{args.is2_stable_max_range_s:g}",
                "--is2-id-t-min",
                f"{args.is2_id_t_min:g}",
                "--is2-id-t-max",
                f"{args.is2_id_t_max:g}",
                "--is2-schedule-margin-s",
                f"{args.is2_schedule_margin_s:g}",
                "--fixed-id-t-sec",
                f"{args.fixed_id_t_sec:g}",
                "--fixed-id-rope-length-m",
                f"{args.fixed_id_rope_length_m:g}",
                "--fixed-id-zeta",
                f"{args.fixed_id_zeta:g}",
                "--switch-margin",
                f"{args.switch_margin:g}",
                "--estimate-deadline",
                f"{args.estimate_deadline:g}",
                "--no-fallback",
                "--zv-t-min",
                f"{zv_t_min:g}",
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
        if args.no_is2_schedule_filter:
            cmd.append("--no-is2-schedule-filter")
        if args.id_assume_zero_zeta:
            cmd.append("--id-assume-zero-zeta")
    if args.method == "is2":
        cmd.extend(["--tau", f"{args.tau:g}"])
    if args.method in ("robust", "zv", "nonrobust"):
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


def reset_encoder_origin() -> bool:
    """Re-zero the payload encoder just before the move.

    encoder_serial_node latches its origin once at node startup and never again,
    so by the time a run starts the payload's rest position has usually shifted
    from wherever that origin was captured. The resulting offset is a fixed
    fraction of a degree, which is invisible under an unshaped pulse (+/-5 deg)
    but dominates a shaped run (+/-0.02 deg). Zeroing here means every run's
    swing is measured from that run's own rest position.
    """
    try:
        result = subprocess.run(
            ["ros2", "service", "call",
             "/payload/encoder/reset_origin", "std_srvs/srv/Trigger"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"[single-test] WARNING: encoder origin reset failed: {exc}")
        return False
    if "success=True" not in result.stdout.replace(" ", ""):
        detail = result.stdout.strip() or result.stderr.strip()
        print(f"[single-test] WARNING: encoder origin reset not confirmed: {detail}")
        return False
    print("[single-test] Encoder origin reset.")
    return True


def run_process(run: SingleRun) -> int:
    def player_pids_for_run() -> list[int]:
        try:
            result = subprocess.run(
                ["pgrep", "-af", PLAYER],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception:
            return []

        pids: list[int] = []
        for line in result.stdout.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            pid_text, cmdline = parts
            if run.csv_path not in cmdline:
                continue
            try:
                pids.append(int(pid_text))
            except ValueError:
                continue
        return pids

    def stop_detached_player(sig: int) -> None:
        for pid in player_pids_for_run():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except Exception:
                pass

    def stop_process_tree(proc: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        except Exception:
            if sig == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()

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
            start_new_session=True,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
                if "Paper TDF move complete" in line or "Paper TDF move aborted" in line:
                    deadline = time.monotonic() + 2.0
                    while proc.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.05)
                    if proc.poll() is None:
                        print("[single-test] Player printed final report but did not exit; terminating so plots can be generated.")
                        log.write("[single-test] Player printed final report but did not exit; terminating so plots can be generated.\n")
                        log.flush()
                        stop_process_tree(proc, signal.SIGTERM)
                        stop_detached_player(signal.SIGTERM)
                    break
            try:
                remaining, _ = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                stop_process_tree(proc, signal.SIGKILL)
                stop_detached_player(signal.SIGKILL)
                try:
                    remaining, _ = proc.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    remaining = ""
            if remaining:
                sys.stdout.write(remaining)
                log.write(remaining)
                log.flush()
            return proc.returncode if proc.returncode is not None else proc.wait()
        except KeyboardInterrupt:
            print("[single-test] Interrupted; stopping player and analyzing any completed CSV data.")
            log.write("[single-test] Interrupted; stopping player and analyzing any completed CSV data.\n")
            log.flush()
            stop_process_tree(proc, signal.SIGTERM)
            stop_detached_player(signal.SIGTERM)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                stop_process_tree(proc, signal.SIGKILL)
                stop_detached_player(signal.SIGKILL)
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
            return 130


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


def id_candidate_points(rows) -> list[dict[str, float | str]]:
    points: list[dict[str, float | str]] = []
    seen: set[float] = set()
    for row in rows:
        cand_t = row.get("id_candidate_time_s", math.nan)
        cond_b = row.get("id_candidate_cond_b", math.nan)
        if not isinstance(cand_t, (float, int)) or not math.isfinite(float(cand_t)):
            continue
        if not isinstance(cond_b, (float, int)) or not math.isfinite(float(cond_b)):
            continue
        key = round(float(cand_t), 6)
        if key in seen:
            continue
        seen.add(key)
        points.append(row)
    return points


def first_source(rows) -> str:
    for row in rows:
        value = row.get("schedule_source", "")
        if isinstance(value, str) and value:
            return value
    return ""


RESIDUAL_GUARDS_S = (0.0, 0.2, 0.5, 1.0)


def guard_tag(guard_s: float) -> str:
    return str(guard_s).replace(".", "p")


def row_float(row: dict[str, float | str], key: str, default: float = math.nan) -> float:
    value = row.get(key, default)
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return float(value)
    return default


def column_series(rows, key: str, default: float = math.nan) -> np.ndarray:
    return np.array([row_float(row, key, default) for row in rows], dtype=float)


def command_axis_columns(run: SingleRun) -> tuple[str, str]:
    axis = "x"
    for i, part in enumerate(run.command):
        if part == "--axis" and i + 1 < len(run.command):
            axis = run.command[i + 1]
            break
    return f"cmd_v{axis}_mm_s", f"cart_v{axis}_mm_s"


def swing_angle_series(rows, run: SingleRun) -> tuple[np.ndarray, str]:
    """Return encoder angle samples for the commanded axis, never mm-derived."""
    axis = "x"
    for i, part in enumerate(run.command):
        if part == "--axis" and i + 1 < len(run.command):
            axis = run.command[i + 1]
            break
    candidates = ["swing_axis_angle_deg"]
    if axis == "x":
        candidates.extend(["swing_pitch_deg", "enc_pitch_deg"])
    else:
        candidates.extend(["swing_roll_deg", "enc_roll_deg"])
    for key in candidates:
        values = column_series(rows, key)
        if np.isfinite(values).any():
            return values, key
    return np.full(len(rows), np.nan), candidates[0]


def command_event_data(rows, run: SingleRun) -> dict[str, object]:
    t = column_series(rows, "move_time_sec")
    cmd_col, act_col = command_axis_columns(run)
    cmd = column_series(rows, cmd_col, 0.0)
    act = column_series(rows, act_col)
    finite = np.isfinite(t)
    t = t[finite]
    cmd = cmd[finite]
    act = act[finite] if len(act) == len(finite) else act

    switches: list[float] = []
    if len(t) and len(cmd):
        dcmd = np.r_[0.0, np.abs(np.diff(cmd))]
        for ts in t[dcmd > 1.0e-6]:
            if not switches or abs(float(ts) - switches[-1]) > 0.03:
                switches.append(float(ts))

    residual_start = math.nan
    nonzero_seen = False
    for ts, cv in zip(t, cmd):
        if abs(float(cv)) > 1.0e-6:
            nonzero_seen = True
        elif nonzero_seen:
            residual_start = float(ts)
            break
    if not math.isfinite(residual_start):
        nonzero = np.where(np.isfinite(cmd) & (np.abs(cmd) > 1.0e-6))[0]
        if len(nonzero):
            residual_start = float(t[nonzero[-1]])
        elif len(t):
            residual_start = float(t[-1])

    tf = run.target_mm / abs(run.vmax_mm_s) if abs(run.vmax_mm_s) > 1.0e-9 else math.nan
    lock_t = last_finite(rows, "schedule_locked_at_s")
    if not math.isfinite(lock_t) and run.method in ("idonly", "is1", "is2"):
        lock_t = run.tau_s
    chosen_id_t = last_finite(rows, "schedule_id_time_s")
    return {
        "t": t,
        "cmd": cmd,
        "act": act,
        "cmd_col": cmd_col,
        "act_col": act_col,
        "switches": switches,
        "residual_start": residual_start,
        "tf": tf,
        "lock_t": lock_t,
        "chosen_id_t": chosen_id_t,
    }


def residual_metrics_from_series(
    t: np.ndarray,
    values: np.ndarray,
    residual_start_s: float,
    *,
    prefix: str,
    unit: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if not math.isfinite(residual_start_s):
        residual_start_s = math.nan
    for guard in RESIDUAL_GUARDS_S:
        tag = guard_tag(guard)
        start = residual_start_s + guard if math.isfinite(residual_start_s) else math.nan
        mask = np.isfinite(t) & np.isfinite(values)
        if math.isfinite(start):
            mask &= t >= start
        vals = values[mask]
        base = f"{prefix}_guard_{tag}"
        out[f"{base}_samples"] = float(len(vals))
        if len(vals) == 0:
            out[f"{base}_p2p_{unit}"] = math.nan
            out[f"{base}_max_abs_{unit}"] = math.nan
            out[f"{base}_rms_{unit}"] = math.nan
            out[f"{base}_rms_demeaned_{unit}"] = math.nan
            out[f"{base}_mean_{unit}"] = math.nan
            continue
        centered = vals - float(np.mean(vals))
        out[f"{base}_p2p_{unit}"] = float(np.max(vals) - np.min(vals))
        out[f"{base}_max_abs_{unit}"] = float(np.max(np.abs(vals)))
        out[f"{base}_rms_{unit}"] = float(np.sqrt(np.mean(vals * vals)))
        out[f"{base}_rms_demeaned_{unit}"] = float(np.sqrt(np.mean(centered * centered)))
        out[f"{base}_mean_{unit}"] = float(np.mean(vals))
    return out


def write_residual_report(run_dir: Path, summary: dict[str, object]) -> None:
    lines = [
        "Residual Guard Sensitivity",
        "",
        "Metrics start at command-zero plus guard. RMS demeaned removes static offset.",
        "",
        "guard_s,p2p_mm,rms_demeaned_mm,p2p_deg,rms_demeaned_deg",
    ]
    for guard in RESIDUAL_GUARDS_S:
        tag = guard_tag(guard)
        lines.append(
            f"{guard:g},"
            f"{summary.get(f'residual_guard_{tag}_p2p_mm', math.nan)},"
            f"{summary.get(f'residual_guard_{tag}_rms_demeaned_mm', math.nan)},"
            f"{summary.get(f'residual_angle_guard_{tag}_p2p_deg', math.nan)},"
            f"{summary.get(f'residual_angle_guard_{tag}_rms_demeaned_deg', math.nan)}"
        )
    (run_dir / "residual_report.txt").write_text("\n".join(lines))


def add_event_lines(
    ax,
    events: dict[str, object],
    *,
    include_residual: bool = False,
    include_guard: bool = False,
    annotate_switches: bool = True,
    label_prefix: str = "",
) -> None:
    lock_t = float(events.get("lock_t", math.nan))
    tf = float(events.get("tf", math.nan))
    residual_start = float(events.get("residual_start", math.nan))
    switches = events.get("switches", [])

    if math.isfinite(lock_t):
        ax.axvline(
            lock_t,
            color="black",
            linestyle="-.",
            linewidth=1.6,
            alpha=0.85,
            label=f"{label_prefix}ID lock",
        )
    if math.isfinite(tf):
        ax.axvline(
            tf,
            color="tab:purple",
            linestyle=":",
            linewidth=1.8,
            alpha=0.9,
            label=f"{label_prefix}tf",
        )
    if include_residual and math.isfinite(residual_start):
        ax.axvline(
            residual_start,
            color="tab:red",
            linestyle="-",
            linewidth=1.5,
            alpha=0.75,
            label=f"{label_prefix}command zero",
        )
        if include_guard:
            for guard in (0.2, 0.5, 1.0):
                ax.axvline(
                    residual_start + guard,
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.25 if guard != 0.5 else 0.55,
                    label=f"{label_prefix}+{guard:g}s guard" if guard == 0.5 else None,
                )

    ymin, ymax = ax.get_ylim()
    y_text = ymin + 0.92 * (ymax - ymin)
    for i, ts in enumerate(switches if isinstance(switches, list) else []):
        if not math.isfinite(float(ts)):
            continue
        if math.isfinite(lock_t) and abs(float(ts) - lock_t) < 0.04:
            continue
        if math.isfinite(tf) and abs(float(ts) - tf) < 0.04:
            continue
        if include_residual and math.isfinite(residual_start) and abs(float(ts) - residual_start) < 0.04:
            continue
        ax.axvline(float(ts), color="0.35", linestyle="--", linewidth=0.9, alpha=0.28)
        if annotate_switches:
            ax.text(float(ts), y_text, f"s{i}", rotation=90, va="top", ha="right", fontsize=8)


def plot_velocity_with_events(
    rows: list[dict[str, float | str]],
    run: SingleRun,
    plots: Path,
    events: dict[str, object],
) -> None:
    t = np.asarray(events["t"], dtype=float)
    cmd = np.asarray(events["cmd"], dtype=float)
    act = np.asarray(events["act"], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(t, cmd, label=f"commanded ({events['cmd_col']})", linewidth=2.0)
    if len(act) == len(t) and np.isfinite(act).any():
        ax.plot(t, act, label=f"actual ({events['act_col']})", linewidth=1.4, alpha=0.9)
    ax.set_xlabel("move time [s]")
    ax.set_ylabel("velocity [mm/s]")
    ax.set_title("Commanded vs actual cart velocity with events")
    ax.grid(True, alpha=0.3)
    add_event_lines(ax, events)
    ax.legend(loc="best")
    save_fig(plots / "cmd_vs_actual_velocity_with_events.png")


def plot_signal_with_events(
    t: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    events: dict[str, object],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(t, values, linewidth=1.5, label=ylabel)
    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=0.9, alpha=0.55)
    ax.set_xlabel("move time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    add_event_lines(ax, events, include_residual=True, include_guard=True)
    ax.legend(loc="best")
    save_fig(out_path)


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
        t_arr = column_series(rows, "move_time_sec")
        swing_mm = column_series(rows, "swing_mm")
        swing_deg, swing_deg_source = swing_angle_series(rows, run)
        events = command_event_data(rows, run)
        stop_time = float(events["residual_start"])
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
        residual_mm_metrics = residual_metrics_from_series(
            t_arr, swing_mm, stop_time, prefix="residual", unit="mm")
        residual_deg_metrics = residual_metrics_from_series(
            t_arr, swing_deg, stop_time, prefix="residual_angle", unit="deg")
        summary.update(
            {
                "travel_actual_mm": travel,
                "travel_error_mm": travel - run.target_mm if math.isfinite(travel) else math.nan,
                "command_stop_time_s": stop_time,
                "run_duration_s": last_finite(rows, "move_time_sec"),
                "residual_angle_source": swing_deg_source,
                "id_first_valid_time_s": first_finite_time(rows, "T_sec"),
                "id_first_valid_T_s": finite_from_row(first_id_row, "T_sec"),
                "id_period_s": run.id_period_s,
                "id_method": run.id_method,
                "id_lowpass_hz": run.id_lowpass_hz,
                "is2_selection_window_s": run.is2_selection_window_s,
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
        summary.update(residual_mm_metrics)
        summary.update(residual_deg_metrics)
        summary.update(
            {
                # Compatibility aliases: immediate, no-guard residual in mm.
                "residual_samples": int(summary.get("residual_guard_0p0_samples", 0.0)),
                "residual_p2p_mm": summary.get("residual_guard_0p0_p2p_mm", math.nan),
                "residual_max_abs_mm": summary.get("residual_guard_0p0_max_abs_mm", math.nan),
                "residual_rms_mm": summary.get("residual_guard_0p0_rms_mm", math.nan),
                "residual_rms_demeaned_mm": summary.get(
                    "residual_guard_0p0_rms_demeaned_mm", math.nan),
                "residual_angle_samples": int(summary.get("residual_angle_guard_0p0_samples", 0.0)),
                "residual_angle_p2p_deg": summary.get("residual_angle_guard_0p0_p2p_deg", math.nan),
                "residual_angle_max_abs_deg": summary.get(
                    "residual_angle_guard_0p0_max_abs_deg", math.nan),
                "residual_angle_rms_deg": summary.get("residual_angle_guard_0p0_rms_deg", math.nan),
                "residual_angle_rms_demeaned_deg": summary.get(
                    "residual_angle_guard_0p0_rms_demeaned_deg", math.nan),
            }
        )
        write_residual_report(run_dir, summary)
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
    t = column_series(rows, "move_time_sec")
    travel = column_series(rows, "traveled_mm")
    swing = column_series(rows, "swing_mm")
    swing_angle, swing_angle_source = swing_angle_series(rows, run)
    filtered_swing = column_series(rows, "id_filtered_swing_mm")
    events = command_event_data(rows, run)
    cmd = np.asarray(events["cmd"], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(np.asarray(events["t"], dtype=float), cmd, linewidth=2)
    axes[0].set_ylabel("cmd vel [mm/s]")
    axes[1].plot(t, travel, linewidth=2)
    axes[1].axhline(run.target_mm, color="black", linestyle="--", alpha=0.4)
    axes[1].set_ylabel("travel [mm]")
    axes[2].plot(t, swing, linewidth=1.4, alpha=0.65, label="raw swing")
    if np.isfinite(filtered_swing).any():
        axes[2].plot(t, filtered_swing, linewidth=2.0, color="tab:orange", label="ID low-pass swing")
        axes[2].legend(loc="best")
    axes[2].set_ylabel("swing [mm]")
    axes[3].plot(t, swing_angle, linewidth=1.4, color="tab:green", label=swing_angle_source)
    axes[3].set_ylabel("swing angle [deg]")
    axes[3].set_xlabel("move time [s]")
    axes[0].set_title(run.run_id)
    for ax in axes:
        add_event_lines(ax, events, include_residual=False, annotate_switches=False)
        ax.grid(True, alpha=0.3)
    save_fig(plots / "run_stack.png")

    residual_start = float(events.get("residual_start", math.nan))
    if not math.isfinite(residual_start):
        residual_start = float(np.nanmax(t)) if np.isfinite(t).any() else 0.0
    rt = t - residual_start
    mask = np.isfinite(rt) & (rt >= 0.0)

    # Magnetic encoders (pose_e / pose_e_rel) carry a fixed mechanical zero
    # offset that isn't meaningful swing — demean so residual drift is easy to
    # read. Camera-sourced swing (/payload/state, /payload/pose) has no such
    # offset, so leave it as measured.
    is_encoder_source = "pose_e" in run.payload_topic

    plt.figure(figsize=(10, 4))
    if is_encoder_source:
        swing_offset_mm = float(np.nanmean(swing[mask])) if mask.any() else 0.0
        swing_demeaned = swing[mask] - swing_offset_mm
        plt.plot(rt[mask], swing[mask], linewidth=1.2, alpha=0.35, color="tab:blue", label="raw")
        plt.plot(rt[mask], swing_demeaned, linewidth=2, color="tab:blue",
                  label=f"demeaned (offset {swing_offset_mm:+.2f} mm)")
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        plt.legend(loc="best", fontsize=8)
    else:
        plt.plot(rt[mask], swing[mask], linewidth=2, color="tab:blue")
    for guard in (0.2, 0.5, 1.0):
        plt.axvline(guard, color="tab:red", linestyle="--", alpha=0.25 if guard != 0.5 else 0.55)
    plt.xlabel("time after command zero [s]")
    plt.ylabel("residual swing [mm]")
    plt.title("Residual swing [mm]")
    plt.grid(True, alpha=0.3)
    save_fig(plots / "residual_swing.png")

    plt.figure(figsize=(10, 4))
    if is_encoder_source:
        swing_angle_offset_deg = float(np.nanmean(swing_angle[mask])) if mask.any() else 0.0
        swing_angle_demeaned = swing_angle[mask] - swing_angle_offset_deg
        plt.plot(rt[mask], swing_angle[mask], linewidth=1.2, alpha=0.35, color="tab:green", label="raw")
        plt.plot(rt[mask], swing_angle_demeaned, linewidth=2, color="tab:green",
                  label=f"demeaned (offset {swing_angle_offset_deg:+.3f} deg)")
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        plt.legend(loc="best", fontsize=8)
    else:
        plt.plot(rt[mask], swing_angle[mask], linewidth=2, color="tab:green")
    for guard in (0.2, 0.5, 1.0):
        plt.axvline(guard, color="tab:red", linestyle="--", alpha=0.25 if guard != 0.5 else 0.55)
    plt.xlabel("time after command zero [s]")
    plt.ylabel("residual swing angle [deg]")
    plt.title("Residual swing angle")
    plt.grid(True, alpha=0.3)
    save_fig(plots / "residual_swing_angle.png")

    plot_velocity_with_events(rows, run, plots, events)
    plot_signal_with_events(
        t,
        swing,
        title="Payload swing [mm] with ID lock, switches, tf, and residual guard",
        ylabel="swing [mm]",
        out_path=plots / "swing_mm_with_events.png",
        events=events,
    )
    plot_signal_with_events(
        t,
        swing_angle,
        title="Payload swing angle with ID lock, switches, tf, and residual guard",
        ylabel="swing angle [deg]",
        out_path=plots / "swing_angle_with_events.png",
        events=events,
    )

    if run.method != "pulse" or run.id_method != "integral":
        candidates = [] if run.id_method == "paper2-step" else id_candidate_points(rows)
        tau_s = run.tau_s if math.isfinite(run.tau_s) and run.tau_s > 0.0 else math.nan
        chosen_id_t = last_finite(rows, "schedule_id_time_s")
        lock_t = last_finite(rows, "schedule_locked_at_s")

        def mark_time_events(ax, *, legend: bool = False):
            if math.isfinite(tau_s):
                ax.axvspan(0.0, tau_s, color="tab:green", alpha=0.07)
                ax.axvline(
                    tau_s,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                    label="tau / lock target" if legend else None,
                )
            if math.isfinite(chosen_id_t):
                ax.axvline(
                    chosen_id_t,
                    color="tab:orange",
                    linestyle="-",
                    linewidth=1.4,
                    alpha=0.9,
                    label="chosen ID" if legend else None,
                )
            if math.isfinite(lock_t) and (not math.isfinite(tau_s) or abs(lock_t - tau_s) > 0.05):
                ax.axvline(
                    lock_t,
                    color="tab:purple",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                    label="actual lock" if legend else None,
                )

        if candidates:
            cand_t = [float(r.get("id_candidate_time_s", math.nan)) for r in candidates]
            cand_T = [float(r.get("id_candidate_T_sec", math.nan)) for r in candidates]
            cand_cond = [float(r.get("id_candidate_cond_b", math.nan)) for r in candidates]
            cand_valid = [bool(int(r.get("id_candidate_valid", 0) or 0)) for r in candidates]
            valid_t = [ti for ti, ok in zip(cand_t, cand_valid) if ok]
            valid_T = [val for val, ok in zip(cand_T, cand_valid) if ok]
            valid_cond = [val for val, ok in zip(cand_cond, cand_valid) if ok]
            reject_t = [ti for ti, ok in zip(cand_t, cand_valid) if not ok]
            reject_T = [val for val, ok in zip(cand_T, cand_valid) if not ok]
            reject_cond = [val for val, ok in zip(cand_cond, cand_valid) if not ok]
            fig = plt.figure(figsize=(11, 10), constrained_layout=True)
            grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.95])
            axes = [
                fig.add_subplot(grid[0, 0]),
                fig.add_subplot(grid[1, 0]),
                fig.add_subplot(grid[2, 0]),
            ]
            axes[1].sharex(axes[0])
            axes[0].scatter(
                reject_t, reject_T, s=14, color="tab:red", alpha=0.22,
                edgecolors="none", label="rejected")
            axes[0].scatter(
                valid_t, valid_T, s=18, color="tab:blue", alpha=0.82,
                edgecolors="none", label="valid")
            axes[1].scatter(
                reject_t, reject_cond, s=14, color="tab:red", alpha=0.22,
                edgecolors="none")
            axes[1].scatter(
                valid_t, valid_cond, s=18, color="tab:blue", alpha=0.82,
                edgecolors="none")
            axes[1].set_xlabel("move time [s]")
            finite_pairs = [
                (val, cond)
                for val, cond in zip(cand_T, cand_cond)
                if math.isfinite(val) and math.isfinite(cond)
            ]
            finite_T = [val for val, _ in finite_pairs]
            finite_cond_for_T = [cond for _, cond in finite_pairs]
            valid_pairs = [
                (val, cond)
                for val, cond, ok in zip(cand_T, cand_cond, cand_valid)
                if ok and math.isfinite(val) and math.isfinite(cond)
            ]
            valid_pair_T = [val for val, _ in valid_pairs]
            valid_pair_cond = [cond for _, cond in valid_pairs]
            axes[2].scatter(
                finite_T, finite_cond_for_T, s=14, color="tab:gray",
                alpha=0.22, edgecolors="none", label="all computed")
            axes[2].scatter(
                valid_pair_T, valid_pair_cond, s=18, color="tab:purple",
                alpha=0.78, edgecolors="none", label="valid candidates")
            axes[2].set_xlabel("T estimate [s]")
            axes[2].set_ylabel("condB")
            axes[2].set_yscale("log")
            mark_time_events(axes[0], legend=True)
            mark_time_events(axes[1])
            plt.setp(axes[0].get_xticklabels(), visible=False)
        else:
            T_vals = [float(r.get("T_sec", math.nan)) for r in rows]
            cond_vals = [float(r.get("cond_b", math.nan)) for r in rows]
            two_mode = False
            fig, axes = plt.subplots(3 if two_mode else 2, 1, figsize=(10, 9 if two_mode else 7), sharex=True)
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
            axes[0].plot(t, T_vals, linewidth=1.8)
            axes[1].plot(t, cond_vals, linewidth=1.8)
            axes[1].set_xlabel("move time [s]")
            mark_time_events(axes[0], legend=True)
            mark_time_events(axes[1])
            if two_mode:
                T2_vals = [float(r.get("two_mode_T2_sec", math.nan)) for r in rows]
                ratio_vals = [float(r.get("two_mode_amp2_amp1", math.nan)) for r in rows]
                axes[0].plot(t, T2_vals, linewidth=1.3, color="tab:orange", alpha=0.8, label="T2 estimate")
                axes[0].legend(loc="best", frameon=True)
                axes[2].plot(t, ratio_vals, linewidth=1.6, color="tab:purple")
                axes[2].set_ylabel("amp2 / amp1")
                axes[2].set_xlabel("move time [s]")
                mark_time_events(axes[2])
        axes[0].set_ylabel("T estimate [s]")
        axes[1].set_ylabel("condB")
        axes[1].set_yscale("log")
        if candidates:
            axes[0].legend(loc="upper right", frameon=True)
            axes[2].legend(loc="best", frameon=True)
        for ax in axes:
            ax.grid(True, which="major", alpha=0.28)
            ax.grid(True, which="minor", alpha=0.12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_title("Online ID candidates: T estimate and conditioning")
        save_fig(plots / "id_trace.png")


def save_fig(path: Path):
    fig = plt.gcf()
    if not fig.get_constrained_layout():
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
        id_method=args.id_method,
        id_lowpass_hz=args.id_lowpass_hz,
        is2_selection_window_s=args.is2_selection_window_s,
    )
    write_metadata(args, run, run_dir)
    print(f"[single-test] Run folder: {run_dir}")
    print("[single-test] Command:")
    print(shell_quote(command))
    if args.execute:
        if not args.no_prompt:
            input("Confirm rope length, clear workspace, let the payload settle, then press Enter...")
        if not args.no_reset_origin:
            reset_encoder_origin()
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
