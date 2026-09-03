#!/usr/bin/env python3
"""Compatibility entry point for the PULSE/nonrobust-ZV joystick jog.

The former ZVD-front/LQG-endpoint behavior has intentionally been removed.
Use Button 10 while idle to switch between direct PULSE and two-impulse ZV.
"""

from pulse_zv_jog import main


if __name__ == '__main__':
    raise SystemExit(main())
