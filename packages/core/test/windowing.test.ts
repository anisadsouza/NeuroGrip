import { describe, expect, it } from 'vitest';
import {
  DEFAULT_WINDOW_CONFIG,
  MultiChannelRingBuffer,
  hopSamples,
  windowSamples,
} from '../src/windowing.js';

describe('window geometry', () => {
  it('derives sample counts from milliseconds and rate', () => {
    expect(windowSamples(DEFAULT_WINDOW_CONFIG)).toBe(400);
    expect(hopSamples(DEFAULT_WINDOW_CONFIG)).toBe(40);
  });

  it('rejects non-positive geometry', () => {
    expect(() => windowSamples({ ...DEFAULT_WINDOW_CONFIG, windowMs: 0 })).toThrow();
    expect(() => hopSamples({ ...DEFAULT_WINDOW_CONFIG, hopMs: 0 })).toThrow();
  });
});

describe('MultiChannelRingBuffer', () => {
  it('returns null until a full window is available', () => {
    const buffer = new MultiChannelRingBuffer(2, 16);
    expect(buffer.readLatestWindow(4)).toBeNull();
    buffer.push([Float64Array.from([1, 2, 3]), Float64Array.from([4, 5, 6])]);
    expect(buffer.readLatestWindow(4)).toBeNull();
    buffer.push([Float64Array.from([7]), Float64Array.from([8])]);
    expect(buffer.readLatestWindow(4)).not.toBeNull();
  });

  it('returns the most recent samples in order', () => {
    const buffer = new MultiChannelRingBuffer(1, 8);
    buffer.push([Float64Array.from([1, 2, 3, 4, 5])]);
    expect(Array.from(buffer.readLatestWindow(3)![0]!)).toEqual([3, 4, 5]);
  });

  it('wraps correctly once capacity is exceeded', () => {
    const buffer = new MultiChannelRingBuffer(1, 4);
    buffer.push([Float64Array.from([1, 2, 3, 4, 5, 6])]);
    expect(Array.from(buffer.readLatestWindow(4)![0]!)).toEqual([3, 4, 5, 6]);
  });

  it('keeps channels aligned across a wrap', () => {
    const buffer = new MultiChannelRingBuffer(2, 4);
    buffer.push([Float64Array.from([1, 2, 3, 4, 5, 6]), Float64Array.from([10, 20, 30, 40, 50, 60])]);
    const window = buffer.readLatestWindow(4)!;
    expect(Array.from(window[0]!)).toEqual([3, 4, 5, 6]);
    expect(Array.from(window[1]!)).toEqual([30, 40, 50, 60]);
  });

  it('caps available at capacity', () => {
    const buffer = new MultiChannelRingBuffer(1, 4);
    buffer.push([Float64Array.from([1, 2, 3, 4, 5, 6])]);
    expect(buffer.available).toBe(4);
  });

  it('rejects a window wider than capacity', () => {
    const buffer = new MultiChannelRingBuffer(1, 4);
    expect(() => buffer.readLatestWindow(5)).toThrow();
  });

  it('rejects a frame with the wrong channel count', () => {
    const buffer = new MultiChannelRingBuffer(2, 8);
    expect(() => buffer.push([Float64Array.from([1])])).toThrow();
  });
});

describe('MultiChannelRingBuffer: the scratch buffer hazard', () => {
  /**
   * `readLatestWindow` returns views into shared scratch, which is documented
   * on the method and is the right trade for a hot path: a fresh allocation
   * every hop at 50 Hz is garbage the collector has to chase.
   *
   * The cost is that the result has a lifetime, and a lifetime is exactly the
   * kind of contract that gets broken by an `async` keyword added elsewhere.
   * The browser worker was concurrent by accident for this reason: two decodes
   * overlapping meant the first computed its RMS and its display trace from
   * the second one's samples, with no error and no visible cause beyond a
   * garbled trace. This test states the hazard so the next reader meets it
   * here rather than in a bug report.
   */
  it('invalidates a previous result on the next read rather than returning independent copies', () => {
    const ring = new MultiChannelRingBuffer(1, 16);
    ring.push([new Float64Array([1, 1, 1, 1])]);

    const first = ring.readLatestWindow(4)!;
    expect(first[0]![0]).toBe(1);

    ring.push([new Float64Array([9, 9, 9, 9])]);
    const second = ring.readLatestWindow(4)!;

    // The same memory. A caller holding `first` across this read is now
    // looking at `second`, which is why the worker handles one message at a
    // time rather than awaiting inference with a window still in hand.
    expect(second[0]![0]).toBe(9);
    expect(first[0]![0]).toBe(9);
    expect(first[0]!.buffer).toBe(second[0]!.buffer);
  });

  it('gives a caller who copies a window a value that survives the next read', () => {
    const ring = new MultiChannelRingBuffer(1, 16);
    ring.push([new Float64Array([1, 1, 1, 1])]);

    const copied = Float64Array.from(ring.readLatestWindow(4)![0]!);
    ring.push([new Float64Array([9, 9, 9, 9])]);
    ring.readLatestWindow(4);

    expect(copied[0]).toBe(1);
  });
});
