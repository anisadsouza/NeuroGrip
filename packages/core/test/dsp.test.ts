import { describe, expect, it } from 'vitest';
import {
  autocorrelate,
  bandMeans,
  hann,
  levinsonDurbin,
  meanFrequency,
  medianFrequency,
  nextPow2,
  periodogram,
} from '../src/dsp.js';

describe('nextPow2', () => {
  it('rounds up to a power of two', () => {
    expect(nextPow2(1)).toBe(1);
    expect(nextPow2(2)).toBe(2);
    expect(nextPow2(3)).toBe(4);
    expect(nextPow2(400)).toBe(512);
  });
});

describe('hann', () => {
  it('is symmetric and zero at both ends', () => {
    const w = hann(9);
    expect(w[0]).toBeCloseTo(0, 12);
    expect(w[8]).toBeCloseTo(0, 12);
    expect(w[4]).toBeCloseTo(1, 12);
    expect(w[1]).toBeCloseTo(w[7]!, 12);
  });

  it('returns [1] for a single sample', () => {
    expect(Array.from(hann(1))).toEqual([1]);
  });
});

describe('periodogram', () => {
  it('locates a pure tone', () => {
    const fs = 1000;
    const x = new Float64Array(512);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin((2 * Math.PI * 100 * i) / fs);
    const { freqs, psd } = periodogram(x, fs);
    let peak = 0;
    for (let i = 1; i < psd.length; i++) if (psd[i]! > psd[peak]!) peak = i;
    expect(freqs[peak]).toBeCloseTo(100, 0);
    expect(psd.length).toBe(512 / 2 + 1);
  });
});

describe('bandMeans', () => {
  it('uses numpy array_split sizing', () => {
    // 7 items into 3 bands -> 3, 2, 2
    const values = Float64Array.from([1, 2, 3, 4, 5, 6, 7]);
    const means = bandMeans(values, 3);
    expect(means[0]).toBeCloseTo(2, 12);
    expect(means[1]).toBeCloseTo(4.5, 12);
    expect(means[2]).toBeCloseTo(6.5, 12);
  });

  it('emits zero for bands with no samples', () => {
    const means = bandMeans(Float64Array.from([5]), 3);
    expect(Array.from(means)).toEqual([5, 0, 0]);
  });
});

describe('autocorrelate', () => {
  it('is biased and peaks at zero lag', () => {
    const x = Float64Array.from([1, 2, 3, 4]);
    const r = autocorrelate(x, 2);
    expect(r.length).toBe(3);
    expect(r[0]).toBeCloseTo((1 + 4 + 9 + 16) / 4, 12);
    expect(r[0]!).toBeGreaterThanOrEqual(r[1]!);
  });
});

describe('levinsonDurbin', () => {
  it('returns zeros for a signal with no power', () => {
    expect(Array.from(levinsonDurbin(new Float64Array(5), 4))).toEqual([0, 0, 0, 0]);
  });

  it('recovers a known AR(2) process', () => {
    // x[n] = 1.2 x[n-1] - 0.5 x[n-2] + e  ->  a1 = -1.2, a2 = 0.5
    const n = 4000;
    const x = new Float64Array(n);
    let seed = 7;
    const random = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648 - 0.5;
    };
    for (let i = 2; i < n; i++) {
      x[i] = 1.2 * x[i - 1]! - 0.5 * x[i - 2]! + random() * 0.1;
    }
    const coeffs = levinsonDurbin(autocorrelate(x, 2), 2);
    expect(coeffs[0]).toBeCloseTo(-1.2, 1);
    expect(coeffs[1]).toBeCloseTo(0.5, 1);
  });
});

describe('frequency moments', () => {
  it('are correct on a flat spectrum', () => {
    const freqs = Float64Array.from([0, 10, 20, 30, 40]);
    const psd = Float64Array.from([1, 1, 1, 1, 1]);
    expect(meanFrequency(freqs, psd)).toBeCloseTo(20, 12);
    expect(medianFrequency(freqs, psd)).toBeCloseTo(20, 12);
  });

  it('are zero when there is no power', () => {
    const freqs = Float64Array.from([0, 10, 20]);
    const psd = new Float64Array(3);
    expect(meanFrequency(freqs, psd)).toBe(0);
    expect(medianFrequency(freqs, psd)).toBe(0);
  });
});
