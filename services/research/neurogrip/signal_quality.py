from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalQualityConfig:
    # Electrode-detachment gate, in volts. Must sit below the ~15 uV baseline
    # noise floor: a detached electrode reads near zero, whereas a genuine
    # rest window still carries baseline noise and must not be rejected.
    min_channel_rms: float = 5e-6
    max_abs_amplitude: float = 5.0


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

    for index, values in enumerate(signal):
        if np.any(~np.isfinite(values)):
            return SignalQualityResult(False, "channel contains non-finite values", index)

        rms = float(np.sqrt(np.mean(np.square(values))))
        if rms < config.min_channel_rms:
            return SignalQualityResult(False, "channel RMS below noise floor", index)

        if float(np.max(np.abs(values))) > config.max_abs_amplitude:
            return SignalQualityResult(False, "channel amplitude exceeds expected EMG range", index)

    return SignalQualityResult(True)
