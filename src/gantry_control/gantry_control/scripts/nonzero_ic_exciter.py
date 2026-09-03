#!/usr/bin/env python3
"""Closed-loop pendulum swing-up controller for the nonzero-IC player.

The ``adaptive_paper_tdf_player.py`` ``nonzero-ic`` profile assumes the payload
is already swinging when its free-swing observation window opens.  This module
adds the missing stage: a resonant, amplitude-regulating trolley drive that
pumps a resting (or lightly swinging) payload up to a commanded peak swing
angle and then hands off at a swing extremum so the subsequent stationary-
trolley identification window starts from a clean state.

Design
------
* The regulated quantity is the swing *envelope*, estimated per tick from the
  measured axis angle and a causally filtered angle rate:

      A_est = hypot(theta, theta_dot / omega_n)          [degrees]

  For a pure sinusoid this equals the peak angle, so it is a smooth, per-tick
  stand-in for "the amplitude the payload would ring at if released now".

* The drive is phase-locked to the swing rate and scaled by the amplitude
  error, so it pumps energy when below target, removes energy when above, and
  fades to zero at convergence with no mode switch:

      e     = A_target - A_est
      u_a   = 0                       if |e| <= tolerance          (deadband)
              clip(e / band, -1, +1)  otherwise
      d     = tanh(theta_dot / rate_ref)                           (smooth sign)
      v_cmd = drive_sign * speed * u_a * d

* ``drive_sign`` encodes how a positive trolley velocity maps to a positive
  change in the measured angle (encoder sign * axis sign * transport lag).  It
  is auto-calibrated from whether the envelope actually grows during the first
  ~1.5 half-periods of driving, and flipped at most twice.

* Hand-off happens only at a swing extremum: enough consecutive tracked peaks
  within tolerance of the target *and* a near-zero instantaneous rate.  The
  payload then holds all its energy as potential energy, so stopping the
  trolley injects a minimal transient into the identification window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExciteConfig:
    """Static configuration for :class:`NonzeroIcExciter`."""

    target_angle_deg: float
    omega_rad_s: float
    tolerance_deg: float = 1.0
    band_deg: float = 4.0
    speed_mm_s: float = 150.0
    drive_sign: float = 1.0
    auto_calibrate_sign: bool = True
    settle_cycles: float = 1.0
    peak_velocity_deg_s: float = 8.0
    rate_filter_hz: float = 6.0
    travel_budget_mm: float = 200.0
    timeout_s: float = 30.0
    abort_angle_deg: float = 30.0
    slew_mm_s2: float = 6000.0
    kick_angle_deg: float = 1.0
    bootstrap_min_angle_deg: float = 4.0

    def __post_init__(self) -> None:
        checks = {
            'target_angle_deg': self.target_angle_deg > 0.0,
            'omega_rad_s': self.omega_rad_s > 0.0,
            'tolerance_deg': self.tolerance_deg > 0.0,
            'band_deg': self.band_deg >= self.tolerance_deg,
            'speed_mm_s': self.speed_mm_s > 0.0,
            'drive_sign': self.drive_sign in (1.0, -1.0),
            'settle_cycles': self.settle_cycles > 0.0,
            'peak_velocity_deg_s': self.peak_velocity_deg_s > 0.0,
            'rate_filter_hz': self.rate_filter_hz > 0.0,
            'travel_budget_mm': self.travel_budget_mm > 0.0,
            'timeout_s': self.timeout_s > 0.0,
            'abort_angle_deg': self.abort_angle_deg > self.target_angle_deg,
            'slew_mm_s2': self.slew_mm_s2 > 0.0,
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f'invalid ExciteConfig fields: {", ".join(bad)}')
        for name in ('target_angle_deg', 'omega_rad_s', 'speed_mm_s'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')


@dataclass
class ExciteCommand:
    """One control tick's output."""

    velocity_mm_s: float
    amplitude_est_deg: float
    peak_est_deg: float
    angle_rate_deg_s: float
    phase: str
    drive_sign: float
    converged: bool = False
    abort_reason: str | None = None


@dataclass(frozen=True)
class BoundedExciteConfig:
    """Configuration for a smooth, one-sided, return-to-anchor swing-up.

    The trolley reference is a raised cosine over each pendulum period.  It
    therefore starts at the anchor, moves only in the positive axis direction,
    and returns to the anchor with zero reference velocity every cycle.
    """

    target_angle_deg: float
    omega_rad_s: float
    tolerance_deg: float = 1.5
    speed_mm_s: float = 100.0
    initial_excursion_mm: float = 15.0
    excursion_step_mm: float = 10.0
    max_excursion_mm: float = 100.0
    position_kp_s: float = 2.0
    return_speed_mm_s: float = 30.0
    return_tolerance_mm: float = 1.0
    settle_cycles: float = 1.0
    timeout_s: float = 30.0
    abort_angle_deg: float = 30.0
    slew_mm_s2: float = 600.0

    def __post_init__(self) -> None:
        checks = {
            'target_angle_deg': self.target_angle_deg > 0.0,
            'omega_rad_s': self.omega_rad_s > 0.0,
            'tolerance_deg': self.tolerance_deg > 0.0,
            'speed_mm_s': self.speed_mm_s > 0.0,
            'initial_excursion_mm': self.initial_excursion_mm > 0.0,
            'excursion_step_mm': self.excursion_step_mm > 0.0,
            'max_excursion_mm': self.max_excursion_mm >= self.initial_excursion_mm,
            'position_kp_s': self.position_kp_s > 0.0,
            'return_speed_mm_s': self.return_speed_mm_s > 0.0,
            'return_tolerance_mm': self.return_tolerance_mm > 0.0,
            'settle_cycles': self.settle_cycles > 0.0,
            'timeout_s': self.timeout_s > 0.0,
            'abort_angle_deg': self.abort_angle_deg > self.target_angle_deg,
            'slew_mm_s2': self.slew_mm_s2 > 0.0,
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f'invalid BoundedExciteConfig fields: {", ".join(bad)}')
        for name in checks:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')


class BoundedCycleExciter:
    """Smooth resonant swing-up confined to ``anchor <= q <= anchor+D``.

    For cycle excursion ``D`` and pendulum frequency ``omega``, the reference
    is

        q_ref = D/2 * (1 - cos(omega*t))
        v_ref = D*omega/2 * sin(omega*t).

    Swing amplitude is estimated from the peak-to-peak encoder angle over a
    complete cycle.  This deliberately avoids differentiating the encoder for
    feedback, which caused command chatter in the first hardware experiment.
    """

    def __init__(self, config: BoundedExciteConfig):
        self.cfg = config
        self._t0: float | None = None
        self._prev_t: float | None = None
        self._prev_angle = 0.0
        self._rate = 0.0
        self._u_prev = 0.0
        self._cycle_index = 0
        self._cycle_min = math.inf
        self._cycle_max = -math.inf
        self._cycle_amplitude = 0.0
        self._cycles_in_band = 0
        self._returning = False
        self._phase = 'init'

        # A raised-cosine excursion D needs peak velocity D*omega/2.  Never
        # silently ask the velocity loop for more than its configured cap.
        self._speed_limited_excursion_mm = min(
            config.max_excursion_mm,
            2.0 * config.speed_mm_s / config.omega_rad_s,
        )
        self._excursion_mm = min(
            config.initial_excursion_mm, self._speed_limited_excursion_mm)

    @property
    def drive_sign(self) -> float:
        # The bounded trajectory is intentionally positive-axis only.
        return 1.0

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi / self.cfg.omega_rad_s

    @property
    def excursion_mm(self) -> float:
        return self._excursion_mm

    def _slew(self, target_mm_s: float, dt: float) -> float:
        max_step = self.cfg.slew_mm_s2 * max(dt, 1.0e-3)
        return self._u_prev + max(
            -max_step, min(max_step, target_mm_s - self._u_prev))

    def _command(
        self,
        velocity_mm_s: float,
        phase: str,
        *,
        converged: bool = False,
        abort_reason: str | None = None,
    ) -> ExciteCommand:
        return ExciteCommand(
            velocity_mm_s=velocity_mm_s,
            amplitude_est_deg=self._cycle_amplitude,
            peak_est_deg=self._cycle_amplitude,
            angle_rate_deg_s=self._rate,
            phase=phase,
            drive_sign=1.0,
            converged=converged,
            abort_reason=abort_reason,
        )

    def update(
        self,
        now: float,
        angle_deg: float,
        cart_offset_mm: float = 0.0,
    ) -> ExciteCommand:
        cfg = self.cfg
        if not (math.isfinite(angle_deg) and math.isfinite(cart_offset_mm)):
            return self._command(self._u_prev, 'stale')

        if self._t0 is None:
            self._t0 = now
            self._prev_t = now
            self._prev_angle = angle_deg
            self._cycle_min = angle_deg
            self._cycle_max = angle_deg
            self._phase = 'outbound'
            return self._command(0.0, self._phase)

        dt = now - self._prev_t
        if dt <= 0.0:
            return self._command(self._u_prev, self._phase)

        # Rate is diagnostic only.  It is not allowed to change the command.
        raw_rate = (angle_deg - self._prev_angle) / dt
        alpha = 1.0 - math.exp(-2.0 * math.pi * 2.0 * dt)
        self._rate += min(max(alpha, 0.0), 1.0) * (raw_rate - self._rate)
        elapsed = now - self._t0

        abort_reason: str | None = None
        if abs(angle_deg) > cfg.abort_angle_deg:
            abort_reason = (
                f'|angle|={abs(angle_deg):.1f}deg exceeded the abort limit '
                f'{cfg.abort_angle_deg:.1f}deg')
        elif elapsed > cfg.timeout_s:
            abort_reason = (
                f'bounded swing-up did not converge within {cfg.timeout_s:.1f}s '
                f'(cycle amplitude={self._cycle_amplitude:.1f}deg, '
                f'target={cfg.target_angle_deg:.1f}deg)')
        if abort_reason is not None:
            self._u_prev = 0.0
            self._phase = 'abort'
            self._prev_t = now
            self._prev_angle = angle_deg
            return self._command(0.0, 'abort', abort_reason=abort_reason)

        cycle_index = int(math.floor(elapsed / self.period_s))
        if cycle_index > self._cycle_index and not self._returning:
            if self._cycle_max >= self._cycle_min:
                self._cycle_amplitude = 0.5 * (
                    self._cycle_max - self._cycle_min)
            if abs(self._cycle_amplitude - cfg.target_angle_deg) <= cfg.tolerance_deg:
                self._cycles_in_band += 1
            else:
                self._cycles_in_band = 0

            # Once a complete cycle reaches the target threshold, stop adding
            # energy and return to the anchor.  Requiring another in-band cycle
            # after an overshoot would pump the payload even harder.
            if self._cycle_amplitude >= cfg.target_angle_deg - cfg.tolerance_deg:
                self._returning = True
                self._phase = 'return'
            else:
                # Coarse fixed excursion steps produced a repeatable overshoot
                # in the 5 degree hardware trials (3.6 -> 5.9 degrees).  Near
                # the gate, scale the next excursion toward the lower edge of
                # the acceptable band, while retaining the configured step as
                # an upper bound.  The 0.25 mm floor guarantees progress when
                # the response is noisy or weakly nonlinear.
                target_threshold = max(
                    cfg.target_angle_deg - cfg.tolerance_deg, 1.0e-6)
                if self._cycle_amplitude > 1.0e-6:
                    predicted_excursion = (
                        self._excursion_mm
                        * target_threshold
                        / self._cycle_amplitude
                    )
                    increment_mm = min(
                        cfg.excursion_step_mm,
                        max(0.25, predicted_excursion - self._excursion_mm),
                    )
                else:
                    increment_mm = cfg.excursion_step_mm
                self._excursion_mm = min(
                    self._excursion_mm + increment_mm,
                    self._speed_limited_excursion_mm,
                )
                self._cycle_index = cycle_index
                self._cycle_min = angle_deg
                self._cycle_max = angle_deg

        self._cycle_min = min(self._cycle_min, angle_deg)
        self._cycle_max = max(self._cycle_max, angle_deg)

        if self._returning:
            if abs(cart_offset_mm) <= cfg.return_tolerance_mm:
                target_velocity = 0.0
            else:
                target_velocity = max(
                    -cfg.return_speed_mm_s,
                    min(cfg.return_speed_mm_s, -cfg.position_kp_s * cart_offset_mm),
                )
            u = self._slew(target_velocity, dt)
            # The real trolley can coast a fraction past the anchor while the
            # return command slews to zero.  The old unconditional ``u <= 0``
            # guard then trapped it just outside the return tolerance forever:
            # the position controller requested a small positive correction,
            # but the guard suppressed it until timeout.  Permit correction
            # toward the anchor from either side while preventing commands
            # that move farther away.
            if cart_offset_mm < -cfg.return_tolerance_mm and u < 0.0:
                u = 0.0
            elif cart_offset_mm > cfg.return_tolerance_mm and u > 0.0:
                u = 0.0
            if (
                abs(cart_offset_mm) <= cfg.return_tolerance_mm
                and abs(u) <= 1.0
            ):
                u = 0.0
                self._phase = 'converged'
                self._u_prev = u
                self._prev_t = now
                self._prev_angle = angle_deg
                return self._command(0.0, 'converged', converged=True)
            self._phase = 'return'
        else:
            phase = cfg.omega_rad_s * elapsed
            q_ref = 0.5 * self._excursion_mm * (1.0 - math.cos(phase))
            v_ref = 0.5 * self._excursion_mm * cfg.omega_rad_s * math.sin(phase)
            position_correction = cfg.position_kp_s * (q_ref - cart_offset_mm)
            target_velocity = max(
                -cfg.speed_mm_s,
                min(cfg.speed_mm_s, v_ref + position_correction),
            )
            u = self._slew(target_velocity, dt)
            # Hard one-sided guards.  The feedback term normally keeps the
            # cart away from these limits; these prevent accumulated tracking
            # error from commanding farther outside the envelope.
            if cart_offset_mm <= 0.0 and u < 0.0:
                u = 0.0
            if cart_offset_mm >= cfg.max_excursion_mm and u > 0.0:
                u = 0.0
            self._phase = 'outbound' if math.sin(phase) >= 0.0 else 'inbound'

        self._u_prev = u
        self._prev_t = now
        self._prev_angle = angle_deg
        return self._command(u, self._phase)


class NonzeroIcExciter:
    """Amplitude-regulating resonant swing-up controller.

    Feed :meth:`update` the wall time, the (bias-removed) axis swing angle in
    degrees, and the signed trolley offset from where excitation began.  It
    returns the trolley velocity command and the running envelope estimate.
    """

    def __init__(self, config: ExciteConfig):
        self.cfg = config
        self._t0: float | None = None
        self._prev_t: float | None = None
        self._prev_angle = 0.0
        self._rate = 0.0
        self._prev_rate = 0.0
        self._u_prev = 0.0
        self._drive_sign = float(config.drive_sign)
        self._bootstrapped = False
        self._peak_est = 0.0
        self._peaks_in_band = 0
        self._calib_ref_amp = 0.0
        self._calib_t0: float | None = None
        self._flips = 0
        self._phase = 'init'

    @property
    def drive_sign(self) -> float:
        return self._drive_sign

    @property
    def half_period_s(self) -> float:
        return math.pi / self.cfg.omega_rad_s

    def _amplitude_est(self, angle_deg: float, rate_deg_s: float) -> float:
        return math.hypot(angle_deg, rate_deg_s / self.cfg.omega_rad_s)

    def _slew(self, u_target: float, dt: float) -> float:
        max_step = self.cfg.slew_mm_s2 * max(dt, 1.0e-3)
        return self._u_prev + max(-max_step, min(max_step, u_target - self._u_prev))

    def update(
        self,
        now: float,
        angle_deg: float,
        cart_offset_mm: float = 0.0,
    ) -> ExciteCommand:
        cfg = self.cfg

        if not math.isfinite(angle_deg):
            # Hold the last command through a bad sample rather than lurching.
            return ExciteCommand(
                self._u_prev, self._peak_est, self._peak_est, self._rate,
                'stale', self._drive_sign)

        if self._t0 is None:
            self._t0 = now
            self._prev_t = now
            self._prev_angle = angle_deg
            self._phase = 'kick'
            self._u_prev = self._drive_sign * cfg.speed_mm_s
            return ExciteCommand(
                self._u_prev, abs(angle_deg), self._peak_est, 0.0,
                self._phase, self._drive_sign)

        dt = now - self._prev_t
        if dt <= 0.0:
            amp = self._amplitude_est(angle_deg, self._rate)
            return ExciteCommand(
                self._u_prev, amp, self._peak_est, self._rate,
                self._phase, self._drive_sign)

        raw_rate = (angle_deg - self._prev_angle) / dt
        alpha = 1.0 - math.exp(-2.0 * math.pi * cfg.rate_filter_hz * dt)
        alpha = min(max(alpha, 0.0), 1.0)
        self._rate += alpha * (raw_rate - self._rate)

        elapsed = now - self._t0
        amp = self._amplitude_est(angle_deg, self._rate)

        # Peak tracking: a sign change in the rate marks an extremum at the
        # previous sample.
        if self._prev_rate != 0.0 and (self._rate > 0.0) != (self._prev_rate > 0.0):
            self._peak_est = abs(self._prev_angle)
            if abs(self._peak_est - cfg.target_angle_deg) <= cfg.tolerance_deg:
                self._peaks_in_band += 1
            else:
                self._peaks_in_band = 0

        abort_reason: str | None = None
        if abs(angle_deg) > cfg.abort_angle_deg:
            abort_reason = (
                f'|angle|={abs(angle_deg):.1f}deg exceeded the abort limit '
                f'{cfg.abort_angle_deg:.1f}deg')
        elif elapsed > cfg.timeout_s:
            abort_reason = (
                f'swing-up did not converge within {cfg.timeout_s:.1f}s '
                f'(A_est={amp:.1f}deg, target={cfg.target_angle_deg:.1f}deg)')
        if abort_reason is not None:
            self._commit(now, angle_deg, 0.0, 'abort')
            return ExciteCommand(
                0.0, amp, self._peak_est, self._rate, 'abort',
                self._drive_sign, abort_reason=abort_reason)

        needed_peaks = max(1, int(math.ceil(2.0 * cfg.settle_cycles)))
        soft_tol = 1.5 * cfg.tolerance_deg
        at_extremum = (
            abs(self._rate) <= cfg.peak_velocity_deg_s
            and abs(angle_deg) >= cfg.target_angle_deg - soft_tol
            and abs(amp - cfg.target_angle_deg) <= soft_tol
        )
        if self._peaks_in_band >= needed_peaks and at_extremum:
            self._commit(now, angle_deg, 0.0, 'converged')
            return ExciteCommand(
                0.0, amp, self._peak_est, self._rate, 'converged',
                self._drive_sign, converged=True)

        if amp >= cfg.bootstrap_min_angle_deg:
            self._bootstrapped = True

        err = cfg.target_angle_deg - amp

        # Auto-calibrate the drive sign from whether the envelope actually grows
        # while we are trying to pump it up.  It is not run when starting above
        # target (where the envelope is meant to shrink) so a correct sign is
        # never flipped away.
        pumping_up = err > cfg.tolerance_deg
        if cfg.auto_calibrate_sign and self._flips < 2 and pumping_up:
            if self._calib_t0 is None:
                self._calib_t0 = now
                self._calib_ref_amp = amp
            elif now - self._calib_t0 >= 1.5 * self.half_period_s:
                if amp - self._calib_ref_amp < 0.25 * cfg.tolerance_deg:
                    self._drive_sign = -self._drive_sign
                    self._flips += 1
                self._calib_t0 = now
                self._calib_ref_amp = amp
        elif not pumping_up:
            self._calib_t0 = None
        if amp < cfg.kick_angle_deg:
            # Open-loop resonant square wave to break symmetry from rest, before
            # there is an angle signal worth phase-locking to.
            square = 1.0 if math.sin(cfg.omega_rad_s * elapsed) >= 0.0 else -1.0
            u_cmd = self._drive_sign * cfg.speed_mm_s * square
            phase = 'kick'
        else:
            if abs(err) <= cfg.tolerance_deg:
                u_a = 0.0
            elif not self._bootstrapped and err > 0.0:
                u_a = 1.0
            else:
                u_a = max(-1.0, min(1.0, err / cfg.band_deg))
            # Resonant pump: a near-bang-bang trolley velocity that switches at
            # the swing's zero crossings puts the trolley acceleration impulses
            # on the payload's velocity extrema, where they do the most work.
            # The leading minus makes drive_sign=+1 add energy on a nominal rig.
            theta_ref = max(0.10 * cfg.target_angle_deg, cfg.tolerance_deg)
            d = -math.tanh(angle_deg / theta_ref)
            u_cmd = self._drive_sign * cfg.speed_mm_s * u_a * d
            phase = 'servo' if self._bootstrapped else 'bootstrap'

        if abs(cart_offset_mm) > cfg.travel_budget_mm:
            # Let the payload coast back inside the cart envelope.
            u_cmd = 0.0
            phase = 'coast'

        u = self._slew(u_cmd, dt)
        self._commit(now, angle_deg, u, phase)
        return ExciteCommand(
            u, amp, self._peak_est, self._rate, phase, self._drive_sign)

    def _commit(self, now: float, angle_deg: float, u: float, phase: str) -> None:
        self._prev_t = now
        self._prev_angle = angle_deg
        self._prev_rate = self._rate
        self._u_prev = u
        self._phase = phase
