#!/usr/bin/env python3
"""Run encoder_reader.py (same BOARD pins). See crane_ws/ops/encoder_reader.py."""
import runpy
import sys
from pathlib import Path

for candidate in (
    Path(__file__).resolve().parent / 'encoder_reader.py',
    Path.home() / 'crane_ws' / 'ops' / 'encoder_reader.py',
):
    if candidate.is_file():
        runpy.run_path(str(candidate), run_name='__main__')
        raise SystemExit(0)

print('encoder_reader.py not found next to this script or in ~/crane_ws/ops/', file=sys.stderr)
sys.exit(1)
