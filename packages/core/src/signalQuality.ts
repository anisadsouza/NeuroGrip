/**
 * Window-level signal checks. A port of
 * services/research/neurogrip/signal_quality.py: same thresholds, same order of
 * checks, same reason strings. Waveform checks are heuristics for obvious
 * faults, not a validated artifact classifier.
 */

export interface SignalQualityConfig {
  minChannelRms: number;
  minChannelRmsByChannel?: readonly number[];
  maxAbsAmplitude: number;
  maxConsecutiveExtremeSamples: number;
  transientStepThreshold: number;
  minTransientSteps: number;
  maxHalfWindowRmsRatio: number;
}

export const DEFAULT_SIGNAL_QUALITY_CONFIG: SignalQualityConfig = {
  // Electrode-detachment gate, in volts. Must sit below the ~15 uV baseline
  // noise floor: a detached electrode reads near zero, whereas a genuine rest
  // window still carries baseline noise and must not be rejected.
  minChannelRms: 5e-6,
  maxAbsAmplitude: 5.0,
  maxConsecutiveExtremeSamples: 3,
  transientStepThreshold: 1e-3,
  minTransientSteps: 16,
  maxHalfWindowRmsRatio: 8.0,
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
  if (
    config.minChannelRmsByChannel !== undefined &&
    config.minChannelRmsByChannel.length !== channels.length
  ) {
    return { ok: false, reason: 'quality calibration channel count mismatch' };
  }

  for (let index = 0; index < channels.length; index++) {
    const values = channels[index]!;
    if (values.length < 1) {
      return { ok: false, reason: 'channel is empty', failedChannel: index };
    }

    let sumSquares = 0;
    let maxAbs = 0;
    let maxValue = Number.NEGATIVE_INFINITY;
    let minValue = Number.POSITIVE_INFINITY;
    for (let i = 0; i < values.length; i++) {
      const value = values[i]!;
      if (!Number.isFinite(value)) {
        return { ok: false, reason: 'channel contains non-finite values', failedChannel: index };
      }
      sumSquares += value * value;
      const magnitude = Math.abs(value);
      if (magnitude > maxAbs) maxAbs = magnitude;
      if (value > maxValue) maxValue = value;
      if (value < minValue) minValue = value;
    }

    const rmsFloor = config.minChannelRmsByChannel?.[index] ?? config.minChannelRms;
    if (!Number.isFinite(rmsFloor) || rmsFloor <= 0) {
      return { ok: false, reason: 'invalid channel RMS calibration', failedChannel: index };
    }
    if (Math.sqrt(sumSquares / values.length) < rmsFloor) {
      return {
        ok: false,
        reason: config.minChannelRmsByChannel
          ? 'channel RMS below calibrated minimum'
          : 'channel RMS below noise floor',
        failedChannel: index,
      };
    }
    if (maxAbs > config.maxAbsAmplitude) {
      return {
        ok: false,
        reason: 'channel amplitude exceeds expected EMG range',
        failedChannel: index,
      };
    }

    if (
      !Number.isInteger(config.maxConsecutiveExtremeSamples) ||
      config.maxConsecutiveExtremeSamples < 1 ||
      !Number.isFinite(config.transientStepThreshold) ||
      config.transientStepThreshold <= 0 ||
      !Number.isInteger(config.minTransientSteps) ||
      config.minTransientSteps < 1 ||
      !Number.isFinite(config.maxHalfWindowRmsRatio) ||
      config.maxHalfWindowRmsRatio <= 1
    ) {
      return {
        ok: false,
        reason: 'invalid waveform quality configuration',
        failedChannel: index,
      };
    }

    if (maxValue !== minValue) {
      for (const extreme of [maxValue, minValue]) {
        let extremeRun = 0;
        let longestExtremeRun = 0;
        for (let i = 0; i < values.length; i++) {
          if (values[i] === extreme) {
            extremeRun++;
            longestExtremeRun = Math.max(longestExtremeRun, extremeRun);
          } else {
            extremeRun = 0;
          }
        }
        if (longestExtremeRun >= config.maxConsecutiveExtremeSamples) {
          return {
            ok: false,
            reason: 'repeated extreme samples suggest clipping',
            failedChannel: index,
          };
        }
      }
    }

    const midpoint = Math.floor(values.length / 2);
    if (midpoint > 0 && midpoint < values.length) {
      let firstSquares = 0;
      let secondSquares = 0;
      for (let i = 0; i < midpoint; i++) {
        firstSquares += values[i]! * values[i]!;
      }
      for (let i = midpoint; i < values.length; i++) {
        secondSquares += values[i]! * values[i]!;
      }
      const firstRms = Math.sqrt(firstSquares / midpoint);
      const secondRms = Math.sqrt(secondSquares / (values.length - midpoint));
      const denominatorFloor = Math.max(rmsFloor, Number.MIN_VALUE);
      if (Math.max(firstRms, secondRms) / Math.max(Math.min(firstRms, secondRms), denominatorFloor)
        > config.maxHalfWindowRmsRatio) {
        return {
          ok: false,
          reason: 'abrupt half-window RMS change suggests gain shift',
          failedChannel: index,
        };
      }
    }

    let largeSteps = 0;
    for (let i = 1; i < values.length; i++) {
      if (Math.abs(values[i]! - values[i - 1]!) > config.transientStepThreshold) {
        largeSteps++;
      }
    }
    if (largeSteps >= config.minTransientSteps) {
      return {
        ok: false,
        reason: 'repeated large sample steps suggest transient noise',
        failedChannel: index,
      };
    }
  }

  return { ok: true };
}
