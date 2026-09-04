/**
 * Signal-quality gating. A port of
 * services/research/neurogrip/signal_quality.py: same thresholds, same order of
 * checks, same reason strings, so a rejection in the browser is explicable by
 * the Python test suite.
 */

export interface SignalQualityConfig {
  minChannelRms: number;
  maxAbsAmplitude: number;
}

export const DEFAULT_SIGNAL_QUALITY_CONFIG: SignalQualityConfig = {
  // Electrode-detachment gate, in volts. Must sit below the ~15 uV baseline
  // noise floor: a detached electrode reads near zero, whereas a genuine rest
  // window still carries baseline noise and must not be rejected.
  minChannelRms: 5e-6,
  maxAbsAmplitude: 5.0,
};

export interface SignalQualityResult {
  ok: boolean;
  reason?: string;
  failedChannel?: number;
}

export function checkSignalQuality(
  channels: Float64Array[],
  config: SignalQualityConfig,
): SignalQualityResult {
  if (channels.length < 1) {
    return { ok: false, reason: 'input must contain at least one channel' };
  }

  for (let index = 0; index < channels.length; index++) {
    const values = channels[index]!;
    if (values.length < 1) {
      return { ok: false, reason: 'channel is empty', failedChannel: index };
    }

    let sumSquares = 0;
    let maxAbs = 0;
    for (let i = 0; i < values.length; i++) {
      const value = values[i]!;
      if (!Number.isFinite(value)) {
        return { ok: false, reason: 'channel contains non-finite values', failedChannel: index };
      }
      sumSquares += value * value;
      const magnitude = Math.abs(value);
      if (magnitude > maxAbs) maxAbs = magnitude;
    }

    if (Math.sqrt(sumSquares / values.length) < config.minChannelRms) {
      return { ok: false, reason: 'channel RMS below noise floor', failedChannel: index };
    }
    if (maxAbs > config.maxAbsAmplitude) {
      return {
        ok: false,
        reason: 'channel amplitude exceeds expected EMG range',
        failedChannel: index,
      };
    }
  }

  return { ok: true };
}
