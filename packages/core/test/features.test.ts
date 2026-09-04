import { describe, expect, it } from 'vitest';
import { DEFAULT_FEATURE_CONFIG, extractFeatures, featureVector } from '../src/features.js';
import { PER_CHANNEL_FEATURES, featureCount, featureNames } from '../src/spec.js';

const config = { ...DEFAULT_FEATURE_CONFIG, zeroCrossingThreshold: 0, slopeSignThreshold: 0 };

describe('spec', () => {
  it('declares the seventeen per-channel features', () => {
    expect([...PER_CHANNEL_FEATURES]).toEqual([
      'rms', 'mav', 'wl', 'zc', 'ssc',
      'ar1', 'ar2', 'ar3', 'ar4',
      'psd1', 'psd2', 'psd3', 'psd4', 'psd5', 'psd6',
      'mdf', 'mnf',
    ]);
  });

  it('orders names channel-major, not lexicographically', () => {
    const names = featureNames(12);
    expect(names.indexOf('ch2_rms')).toBeLessThan(names.indexOf('ch10_rms'));
    expect(names).not.toEqual([...names].sort());
    expect(names.length).toBe(featureCount(12));
  });

  it('rejects non-positive channel counts', () => {
    expect(() => featureNames(0)).toThrow();
    expect(() => featureCount(0)).toThrow();
  });
});

describe('extractFeatures', () => {
  it('computes the time-domain values', () => {
    const features = extractFeatures([Float64Array.from([3, -4, 3, -4])], config);
    expect(features.get('ch1_rms')).toBeCloseTo(Math.sqrt((9 + 16 + 9 + 16) / 4), 5);
    expect(features.get('ch1_mav')).toBeCloseTo(3.5, 5);
    expect(features.get('ch1_wl')).toBeCloseTo(21, 5);
  });

  it('does not count a sample resting exactly at zero as a crossing', () => {
    const features = extractFeatures([Float64Array.from([1, 0, 1])], config);
    expect(features.get('ch1_zc')).toBe(0);
  });

  it('counts a genuine sign change', () => {
    const features = extractFeatures([Float64Array.from([1, -1, 1])], config);
    expect(features.get('ch1_zc')).toBe(2);
  });

  it('is scale-consistent for slope sign changes', () => {
    const base = [0, 1, 0, 1, 0, 1, 0];
    const small = extractFeatures(
      [Float64Array.from(base.map((v) => v * 1e-3))],
      { ...DEFAULT_FEATURE_CONFIG, slopeSignThreshold: 1e-3 * 0.5 },
    );
    const large = extractFeatures(
      [Float64Array.from(base.map((v) => v * 1e3))],
      { ...DEFAULT_FEATURE_CONFIG, slopeSignThreshold: 1e3 * 0.5 },
    );
    expect(small.get('ch1_ssc')).toBe(large.get('ch1_ssc'));
    expect(small.get('ch1_ssc')!).toBeGreaterThan(0);
  });

  it('zeroes AR and spectral features for a silent channel', () => {
    const features = extractFeatures([new Float64Array(400)], DEFAULT_FEATURE_CONFIG);
    for (const name of ['ch1_ar1', 'ch1_ar2', 'ch1_ar3', 'ch1_ar4', 'ch1_mdf', 'ch1_mnf']) {
      expect(features.get(name)).toBe(0);
    }
  });

  it('rejects channels of differing length', () => {
    expect(() =>
      extractFeatures([new Float64Array(10), new Float64Array(9)], config),
    ).toThrow();
  });

  it('rejects an empty channel list', () => {
    expect(() => extractFeatures([], config)).toThrow();
  });
});

describe('featureVector', () => {
  it('is float32 and follows spec order', () => {
    const channels = [new Float64Array(400).fill(0.01), new Float64Array(400).fill(0.02)];
    const vector = featureVector(channels, DEFAULT_FEATURE_CONFIG);
    const named = extractFeatures(channels, DEFAULT_FEATURE_CONFIG);
    expect(vector).toBeInstanceOf(Float32Array);
    expect(vector.length).toBe(featureCount(2));
    expect(vector[0]).toBeCloseTo(named.get('ch1_rms')!, 6);
    expect(vector[17]).toBeCloseTo(named.get('ch2_rms')!, 6);
  });
});
