/**
 * The three functions between the wire and the screen.
 *
 * Two of them the wearer reads directly -- the oscilloscope trace and the
 * activation ring -- and the third decides which electrode is which. None of
 * them can fail loudly: a transposed ring still draws eight plausible lanes,
 * and a decimation that loses the sign alternation still draws a line.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { channelRms, decimate, deinterleave } from '../../src/worker/frames.js';
import {
  type ReplayBundle,
  type ReplayClock,
  ReplaySource,
  loadReplayBundle,
} from '../../src/sources/replaySource.js';

/** A clock that runs exactly one frame when asked. */
class OneFrameClock implements ReplayClock {
  private time = 0;
  private queued: (() => void) | null = null;

  now(): number {
    return this.time;
  }
  schedule(callback: () => void): number {
    this.queued = callback;
    return 1;
  }
  cancel(): void {
    this.queued = null;
  }
  advance(ms: number): void {
    this.time += ms;
    const frame = this.queued;
    this.queued = null;
    frame?.();
  }
}

function committedBundleFetch(url: string): Promise<Response> {
  const name = url.endsWith('.json') ? 'emg-replay.json' : 'emg-replay.bin';
  const bytes = readFileSync(
    fileURLToPath(new URL(`../../../../apps/web/public/replay/${name}`, import.meta.url)),
  );
  return Promise.resolve(new Response(new Uint8Array(bytes)));
}

const bundle: ReplayBundle = await loadReplayBundle('/replay', committedBundleFetch);

describe('decimate', () => {
  /**
   * The exact failure the docstring names. Peak-picking on an alternating
   * waveform lands on positive peaks in consecutive buckets and the trace
   * drifts off the baseline into smooth undulation -- which still looks like a
   * signal, which is why it needs a gate rather than an eye.
   */
  it('keeps both extremes of each bucket rather than only the peak', () => {
    const width = 400;
    const alternating = new Float64Array(width);
    for (let i = 0; i < width; i++) alternating[i] = i % 2 === 0 ? 1 : -1;

    const out = decimate([alternating], 40);

    let low = Infinity;
    let high = -Infinity;
    for (const value of out) {
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
    expect(low).toBe(-1);
    expect(high).toBe(1);
  });

  /**
   * Without the alternation the polyline doubles back on itself every other
   * point and the envelope reads as a row of spikes rather than a band.
   */
  it('alternates the order bucket to bucket so the polyline zig-zags', () => {
    const width = 400;
    const alternating = new Float64Array(width);
    for (let i = 0; i < width; i++) alternating[i] = i % 2 === 0 ? 1 : -1;

    const out = decimate([alternating], 40);

    // Bucket 0 emits low then high; bucket 1 emits high then low.
    expect(out[0]).toBe(-1);
    expect(out[1]).toBe(1);
    expect(out[2]).toBe(1);
    expect(out[3]).toBe(-1);
  });

  it('writes exactly one row per channel and never past the row', () => {
    const width = 400;
    const points = 41;
    const channels = [new Float64Array(width).fill(1), new Float64Array(width).fill(2)];

    const out = decimate(channels, points);

    expect(out.length).toBe(channels.length * points);
    expect(out[points - 1]).toBeDefined();
    // The second channel's row starts where the first ends, not one short.
    expect(out[points]).toBe(2);
  });

  it('handles a window shorter than the requested points rather than reading past its end', () => {
    const out = decimate([new Float64Array([1, -1, 1, -1])], 40);
    for (const value of out) expect(Number.isFinite(value)).toBe(true);
  });
});

describe('channelRms', () => {
  it('reports the magnitude of a constant channel as that magnitude', () => {
    expect(channelRms([new Float64Array(100).fill(3)])[0]).toBeCloseTo(3, 12);
  });

  it('is blind to sign, so a full-scale oscillation reads as full scale', () => {
    const wave = new Float64Array(100);
    for (let i = 0; i < 100; i++) wave[i] = i % 2 === 0 ? 2 : -2;
    expect(channelRms([wave])[0]).toBeCloseTo(2, 12);
  });

  it('reports one figure per channel, in channel order', () => {
    const rms = channelRms([new Float64Array(10).fill(1), new Float64Array(10).fill(5)]);
    expect(rms).toEqual([1, 5]);
  });
});

describe('deinterleave', () => {
  /**
   * The gate that matters here, and the same shape as the conformance suite
   * one layer up: two implementations of one layout, pinned against each
   * other. The replay source writes `chunk[c * wanted + i]`; if the worker
   * ever reads it as interleaved, the electrode ring transposes and the
   * decoder receives a plausible eight-channel window belonging to no arm.
   */
  it('round-trips a chunk laid out exactly as the replay source writes one', () => {
    const clock = new OneFrameClock();
    let chunk: Float32Array | null = null;
    const source = new ReplaySource(bundle, (samples) => {
      chunk = samples;
    }, clock);

    source.start();
    clock.advance(20);

    const { nChannels, samplesPerGesture } = bundle.manifest;
    const channels = deinterleave(chunk!, nChannels);
    const segment = bundle.segments[0]!;

    expect(channels.length).toBe(nChannels);
    for (let c = 0; c < nChannels; c++) {
      for (let i = 0; i < 5; i++) {
        expect(channels[c]![i], `channel ${c} sample ${i}`).toBeCloseTo(
          segment[c * samplesPerGesture + i]!,
          12,
        );
      }
    }
  });

  it('refuses a chunk that does not divide into whole channels rather than truncating', () => {
    expect(() => deinterleave(new Float32Array(10), 3)).toThrow(RangeError);
  });
});
