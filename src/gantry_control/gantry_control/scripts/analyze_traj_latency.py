#!/usr/bin/env python3
"""Decompose TRAJ STREAM latency from an instrumented experiment CSV.

The controller publishes /gantry/traj_latency with source, receive, apply,
motor-write, and encoder timestamps.  adaptive_paper_tdf_player records the
latest diagnostic record in its CSV.  This script deduplicates those records
and reports transport, controller-queue, write, and feedback-response delays.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


REQUIRED_COLUMNS = (
    'lat_encoder_read_stamp_s',
    'lat_stream_seq_applied',
    'lat_source_stamp_s',
    'lat_controller_rx_stamp_s',
    'lat_apply_begin_stamp_s',
    'lat_apply_done_stamp_s',
    'lat_applied_vx_mm_s',
    'lat_cart_x_mm',
    'lat_motor_vx_mm_s',
    'lat_position_vx_mm_s',
)


def finite(row: dict[str, str], key: str) -> float | None:
    text = row.get(key, '')
    if text in ('', 'nan', 'NaN'):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def first_crossing(
    samples: list[dict[str, float]],
    *,
    time_key: str,
    signal_key: str,
    old_value: float,
    new_value: float,
    fraction: float,
    after_s: float,
) -> float | None:
    delta = new_value - old_value
    if abs(delta) < 1.0e-12:
        return None
    previous = None
    for sample in samples:
        t = sample[time_key]
        if t < after_s:
            previous = sample
            continue
        progress = (sample[signal_key] - old_value) / delta
        if progress >= fraction:
            if previous is None:
                return t
            p0 = (previous[signal_key] - old_value) / delta
            t0 = previous[time_key]
            if p0 < fraction and progress > p0 and t > t0:
                alpha = (fraction - p0) / (progress - p0)
                # A pre-write sample and the first post-write sample can
                # interpolate to a slightly negative response delay.  That is
                # only a 100 Hz resolution artifact, not non-causal motion.
                return max(after_s, t0 + alpha * (t - t0))
            return t
        previous = sample
    return None


def milliseconds(value: float | None) -> str:
    return 'n/a' if value is None else f'{1000.0 * value:.3f}'


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Analyze instrumented command-to-motion latency CSV')
    parser.add_argument('csv_path')
    parser.add_argument(
        '--axis', choices=('x',), default='x',
        help='Current diagnostic implementation reports the tested x axis.')
    parser.add_argument(
        '--min-step-mm-s', type=float, default=50.0,
        help='Ignore command changes smaller than this value.')
    args = parser.parse_args()

    path = Path(args.csv_path).expanduser()
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    missing = [name for name in REQUIRED_COLUMNS if name not in fields]
    if missing:
        print('CSV does not contain the new latency instrumentation:')
        for name in missing:
            print(f'  missing {name}')
        print('Restart the rebuilt Mission Planner/controller and record a new run.')
        return 2

    # Deduplicate the controller diagnostics. The player and controller both
    # run near 100 Hz, so one diagnostic record may appear in adjacent CSV rows.
    by_encoder_stamp: dict[float, dict[str, float]] = {}
    for row in rows:
        stamp = finite(row, 'lat_encoder_read_stamp_s')
        x = finite(row, 'lat_cart_x_mm')
        motor_v = finite(row, 'lat_motor_vx_mm_s')
        filtered_v = finite(row, 'lat_position_vx_mm_s')
        if stamp is None or stamp <= 0.0 or x is None or motor_v is None or filtered_v is None:
            continue
        by_encoder_stamp[stamp] = {
            't': stamp,
            'x': x,
            'motor_v': motor_v,
            'filtered_v': filtered_v,
        }
    samples = sorted(by_encoder_stamp.values(), key=lambda sample: sample['t'])
    for index, sample in enumerate(samples):
        if index == 0:
            sample['raw_position_v'] = sample['filtered_v']
            continue
        previous = samples[index - 1]
        dt = sample['t'] - previous['t']
        sample['raw_position_v'] = (
            (sample['x'] - previous['x']) / dt if dt > 1.0e-9 else previous['raw_position_v']
        )

    # Extract the first CSV observation of every controller-applied sequence.
    by_sequence: dict[int, dict[str, float]] = {}
    for row in rows:
        seq_value = finite(row, 'lat_stream_seq_applied')
        if seq_value is None or seq_value <= 0.0:
            continue
        seq = int(round(seq_value))
        source = finite(row, 'lat_source_stamp_s')
        rx = finite(row, 'lat_controller_rx_stamp_s')
        begin = finite(row, 'lat_apply_begin_stamp_s')
        done = finite(row, 'lat_apply_done_stamp_s')
        applied = finite(row, 'lat_applied_vx_mm_s')
        if None in (source, rx, begin, done, applied):
            continue
        by_sequence.setdefault(seq, {
            'seq': float(seq),
            'source': source,
            'rx': rx,
            'begin': begin,
            'done': done,
            'applied': applied,
        })
    events = sorted(by_sequence.values(), key=lambda event: event['done'])

    previous_command = 0.0
    analyzed = 0
    print(f'Latency analysis: {path}')
    print(f'Unique encoder samples: {len(samples)}')
    print('All latency values below are milliseconds. Feedback crossings are')
    print('interpolated between 100 Hz samples and therefore have about 10 ms raw resolution.\n')

    for event in events:
        new_command = event['applied']
        delta = new_command - previous_command
        if abs(delta) < args.min_step_mm_s:
            previous_command = new_command
            continue
        analyzed += 1
        label = 'rise/change' if abs(new_command) > abs(previous_command) else 'stop/reduction'
        print(
            f"Event {analyzed}: seq={int(event['seq'])} {label} "
            f"{previous_command:.3f} -> {new_command:.3f} mm/s")
        controller_timed = abs(event['rx'] - event['source']) < 5.0e-5
        if controller_timed:
            print('  command path                    : controller-timed buffered knot')
            print(
                '  scheduled boundary -> apply     : '
                f"{milliseconds(event['begin'] - event['source'])}")
        else:
            print(f"  publisher -> controller receive : {milliseconds(event['rx'] - event['source'])}")
            print(f"  controller receive -> apply     : {milliseconds(event['begin'] - event['rx'])}")
        print(f"  motor API write duration         : {milliseconds(event['done'] - event['begin'])}")
        total_label = (
            'scheduled boundary -> write done'
            if controller_timed
            else 'publisher -> motor write done'
        )
        print(f"  {total_label:32s}: {milliseconds(event['done'] - event['source'])}")

        for signal_key, label_text in (
            ('motor_v', 'Teknic VelMeasured'),
            ('raw_position_v', 'raw encoder-position derivative'),
            ('filtered_v', 'filtered position velocity'),
        ):
            crossings = []
            for fraction in (0.1, 0.5, 0.9):
                crossing = first_crossing(
                    samples,
                    time_key='t',
                    signal_key=signal_key,
                    old_value=previous_command,
                    new_value=new_command,
                    fraction=fraction,
                    # Feedback cannot causally respond before the motor write
                    # completes.  Start the crossing search at that boundary;
                    # interpolation against an earlier 100 Hz sample otherwise
                    # produces a small negative latency artifact.
                    after_s=event['done'],
                )
                crossings.append(
                    None if crossing is None else crossing - event['done'])
            print(
                f"  write done -> {label_text:31s} "
                f"10/50/90%: {milliseconds(crossings[0])} / "
                f"{milliseconds(crossings[1])} / {milliseconds(crossings[2])}")
        print()
        previous_command = new_command

    if analyzed == 0:
        print(
            f'No applied command changes exceeded {args.min_step_mm_s:.1f} mm/s. '
            'Check that the rebuilt controller was restarted before the run.')
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
