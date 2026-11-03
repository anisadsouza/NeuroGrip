"""Window-level signal checks shared with the browser TypeScript port.

The waveform checks are conservative heuristics for obvious clipping plateaus,
large repeated transients, and abrupt half-window gain changes. They are not a
validated artifact classifier and do not detect high baseline noise by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalQualityConfig:
    # Electrode-detachment gate, in volts. Must sit below the ~15 uV baseline
    # noise floor: a detached electrode reads near zero, whereas a genuine
    # rest window still carries baseline noise and must not be rejected.
    min_channel_rms: float = 5e-6
    min_channel_rms_by_channel: tuple[float, ...] | None = None
    max_abs_amplitude: float = 5.0
    max_consecutive_extreme_samples: int = 3
    transient_step_threshold: float = 1e-3
    min_transient_steps: int = 16
    max_half_window_rms_ratio: float = 8.0


@dataclass(frozen=True)
class SignalQualityResult:
    ok: bool
    reason: str | None = None
    failed_channel: int | None = None


def check_signal_quality(
    channels: np.ndarray,
    config: SignalQualityConfig = SignalQualityConfig(),
) -> SignalQualityResult:
    signal = np.asarray(channels, dtype=np.float32)
    if signal.ndim != 2:
        return SignalQualityResult(False, "input must be a 2D channel matrix")
    if signal.shape[0] < 1:
        return SignalQualityResult(False, "input must contain at least one channel")
    if (
        config.min_channel_rms_by_channel is not None
        and len(config.min_channel_rms_by_channel) != signal.shape[0]
    ):
        return SignalQualityResult(False, "quality calibration channel count mismatch")

    for index, values in enumerate(signal):
        if values.size < 1:
            return SignalQualityResult(False, "channel is empty", index)
        if np.any(~np.isfinite(values)):
            return SignalQualityResult(False, "channel contains non-finite values", index)

        rms = float(np.sqrt(np.mean(np.square(values))))
        rms_floor = (
            config.min_channel_rms
            if config.min_channel_rms_by_channel is None
            else config.min_channel_rms_by_channel[index]
        )
        if not np.isfinite(rms_floor) or rms_floor <= 0:
            return SignalQualityResult(False, "invalid channel RMS calibration", index)
        if rms < rms_floor:
            reason = (
                "channel RMS below noise floor"
                if config.min_channel_rms_by_channel is None
                else "channel RMS below calibrated minimum"
            )
            return SignalQualityResult(False, reason, index)

        if float(np.max(np.abs(values))) > config.max_abs_amplitude:
            return SignalQualityResult(False, "channel amplitude exceeds expected EMG range", index)

        if (
            not isinstance(config.max_consecutive_extreme_samples, int)
            or config.max_consecutive_extreme_samples < 1
            or not np.isfinite(config.transient_step_threshold)
            or config.transient_step_threshold <= 0
            or not isinstance(config.min_transient_steps, int)
            or config.min_transient_steps < 1
            or not np.isfinite(config.max_half_window_rms_ratio)
            or config.max_half_window_rms_ratio <= 1
        ):
            return SignalQualityResult(False, "invalid waveform quality configuration", index)

        if (
            values.size >= config.max_consecutive_extreme_samples
            and float(np.max(values)) != float(np.min(values))
        ):
            for extreme in (float(np.max(values)), float(np.min(values))):
                matches = values == extreme
                run_starts = np.flatnonzero(matches & ~np.r_[False, matches[:-1]])
                run_ends = np.flatnonzero(matches & ~np.r_[matches[1:], False])
                if np.any(
                    run_ends - run_starts + 1
                    >= config.max_consecutive_extreme_samples
                ):
                    return SignalQualityResult(
                        False, "repeated extreme samples suggest clipping", index
                    )

        midpoint = values.size // 2
        if midpoint > 0 and midpoint < values.size:
            first_rms = float(np.sqrt(np.mean(np.square(values[:midpoint]))))
            second_rms = float(np.sqrt(np.mean(np.square(values[midpoint:]))))
            rms_floor = max(rms_floor, np.finfo(np.float32).tiny)
            if max(first_rms, second_rms) / max(
                min(first_rms, second_rms), rms_floor
            ) > config.max_half_window_rms_ratio:
                return SignalQualityResult(
                    False, "abrupt half-window RMS change suggests gain shift", index
                )

        if values.size > 1 and np.count_nonzero(
            np.abs(np.diff(values)) > config.transient_step_threshold
        ) >= config.min_transient_steps:
            return SignalQualityResult(
                False, "repeated large sample steps suggest transient noise", index
            )

    return SignalQualityResult(True)
