/**
 * Handcrafted sEMG feature extraction.
 *
 * A port of services/research/neurogrip/features.py. Pinned against it by
 * test/conformance.test.ts using golden vectors generated from Python.
 */

import {
  autocorrelate,
  bandMeans,
  levinsonDurbin,
  meanFrequency,
  medianFrequency,
  periodogram,
} from './dsp.js';
import { AR_ORDER, PER_CHANNEL_FEATURES, PSD_BINS, featureNames } from './spec.js';

export interface FeatureConfig {
  samplingRateHz: number;
  zeroCrossingThreshold: number;
  slopeSignThreshold: number;
}

export const DEFAULT_FEATURE_CONFIG: FeatureConfig = {
  samplingRateHz: 2000,
  zeroCrossingThreshold: 1e-5,
  slopeSignThreshold: 1e-5,
};

function assertWellFormed(channels: Float64Array[]): void {
  if (channels.length < 1) throw new RangeError('at least one channel is required');
  const width = channels[0]!.length;
  if (width < 3) throw new RangeError('each channel needs at least three samples');
  for (const channel of channels) {
    if (channel.length !== width) {
      throw new RangeError('all channels must have the same sample count');
    }
  }
}

/** Write one channel's 17 features into `out` starting at `offset`. */
function writeChannelFeatures(
  values: Float64Array,
  config: FeatureConfig,
  out: Float32Array,
  offset: number,
): void {
  const n = values.length;

  let sumSquares = 0;
  let sumAbs = 0;
  for (let i = 0; i < n; i++) {
    sumSquares += values[i]! * values[i]!;
    sumAbs += Math.abs(values[i]!);
  }

  let waveformLength = 0;
  let zeroCrossings = 0;
  for (let i = 0; i < n - 1; i++) {
    const delta = values[i + 1]! - values[i]!;
    waveformLength += Math.abs(delta);
    // Strict product-negative test: Math.sign(0) is 0, so comparing signs
    // would count a sample resting at zero as crossing both neighbours.
    if (values[i]! * values[i + 1]! < 0 && Math.abs(delta) >= config.zeroCrossingThreshold) {
      zeroCrossings++;
    }
  }

  // Sign test on the product (dimensionless); magnitude test on the differences,
  // which share units with the threshold. Comparing the threshold against the
  // product directly is dimensionally inconsistent.
  let slopeSignChanges = 0;
  for (let i = 1; i < n - 1; i++) {
    const back = values[i]! - values[i - 1]!;
    const forward = values[i]! - values[i + 1]!;
    if (
      back * forward > 0 &&
      Math.max(Math.abs(back), Math.abs(forward)) >= config.slopeSignThreshold
    ) {
      slopeSignChanges++;
    }
  }

  out[offset + 0] = Math.sqrt(sumSquares / n);
  out[offset + 1] = sumAbs / n;
  out[offset + 2] = waveformLength;
  out[offset + 3] = zeroCrossings;
  out[offset + 4] = slopeSignChanges;

  const ar = levinsonDurbin(autocorrelate(values, AR_ORDER), AR_ORDER);
  for (let i = 0; i < AR_ORDER; i++) out[offset + 5 + i] = ar[i]!;

  const { freqs, psd } = periodogram(values, config.samplingRateHz);
  const bands = bandMeans(psd, PSD_BINS);
  for (let i = 0; i < PSD_BINS; i++) out[offset + 5 + AR_ORDER + i] = bands[i]!;

  const spectralOffset = offset + 5 + AR_ORDER + PSD_BINS;
  out[spectralOffset] = medianFrequency(freqs, psd);
  out[spectralOffset + 1] = meanFrequency(freqs, psd);
}

/**
 * The hot path. Allocates one Float32Array and no Map, because this runs on
 * every 20 ms hop inside the inference worker.
 */
export function featureVector(
  channels: Float64Array[],
  config: FeatureConfig,
): Float32Array {
  assertWellFormed(channels);
  const stride = PER_CHANNEL_FEATURES.length;
  const out = new Float32Array(channels.length * stride);
  for (let c = 0; c < channels.length; c++) {
    writeChannelFeatures(channels[c]!, config, out, c * stride);
  }
  return out;
}

/** Named view over featureVector. For debugging and conformance, not the hot path. */
export function extractFeatures(
  channels: Float64Array[],
  config: FeatureConfig,
): Map<string, number> {
  const vector = featureVector(channels, config);
  const names = featureNames(channels.length);
  const out = new Map<string, number>();
  for (let i = 0; i < names.length; i++) out.set(names[i]!, vector[i]!);
  return out;
}
