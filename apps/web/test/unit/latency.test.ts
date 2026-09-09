/**
 * Both of these encode a refusal, and a refusal is exactly the kind of thing
 * that gets quietly removed by someone making a number appear sooner.
 */

import { describe, expect, it } from 'vitest';
import {
  LATENCY_WINDOW,
  MINIMUM_LATENCY_SAMPLES,
  percentile95,
  trimRing,
} from '../../src/decode/latency.js';

/** `count` measurements, ascending, so the expected rank is obvious. */
function ramp(count: number): number[] {
  return Array.from({ length: count }, (_, i) => i + 1);
}

describe('percentile95', () => {
  /**
   * A P95 over five windows is a maximum wearing a percentile's name. The
   * screen shows a dash instead, which is honest; showing a number would
   * invite it to be quoted.
   */
  it('reports nothing below the minimum sample count rather than a flattering maximum', () => {
    expect(percentile95(ramp(MINIMUM_LATENCY_SAMPLES - 1))).toBeNull();
    expect(percentile95(ramp(MINIMUM_LATENCY_SAMPLES))).not.toBeNull();
  });

  it('returns a value that was actually measured rather than an interpolation', () => {
    const samples = ramp(100);
    const p95 = percentile95(samples)!;
    expect(samples.includes(p95)).toBe(true);
    expect(p95).toBe(96);
  });

  it('does not depend on the order the measurements arrived in', () => {
    const ascending = ramp(50);
    const shuffled = [...ascending].reverse();
    expect(percentile95(shuffled)).toBe(percentile95(ascending));
  });

  it('leaves the caller list untouched rather than sorting it in place', () => {
    const samples = [5, 1, 4, 2, 3, ...ramp(20)];
    const before = [...samples];
    percentile95(samples);
    expect(samples).toEqual(before);
  });
});

describe('trimRing', () => {
  /**
   * The newest, not the oldest. Keeping the first two hundred windows would
   * freeze the P95 at whatever the machine was doing while the model was still
   * warming up, and it would never move again however long the session ran.
   */
  it('keeps the most recent measurements rather than the earliest', () => {
    let ring: number[] = [];
    for (let i = 0; i < LATENCY_WINDOW + 50; i++) ring = trimRing(ring, i);

    expect(ring.length).toBe(LATENCY_WINDOW);
    expect(ring.at(-1)).toBe(LATENCY_WINDOW + 49);
    expect(ring[0]).toBe(50);
  });

  it('grows freely until it reaches the limit', () => {
    let ring: number[] = [];
    for (let i = 0; i < 10; i++) ring = trimRing(ring, i);
    expect(ring.length).toBe(10);
  });
});
