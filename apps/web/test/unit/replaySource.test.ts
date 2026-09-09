/**
 * The replay source paces the whole application. Every latency figure the Live
 * screen reports is measured against samples this class delivered, so a source
 * that quietly plays back in slow motion, or that dumps a stall's whole
 * backlog in one burst, produces numbers that look fine and mean nothing.
 *
 * The clock is injected rather than patched globally because this suite
 * contains no mocks anywhere: a ManualClock the test constructs and hands in is
 * an argument, not a modified global.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  type ReplayBundle,
  type ReplayClock,
  ReplaySource,
  loadReplayBundle,
} from '../../src/sources/replaySource.js';

/** A clock the test advances by hand, running frames only when asked. */
class ManualClock implements ReplayClock {
  private time = 0;
  private queued: (() => void) | null = null;
  private nextHandle = 1;

  now(): number {
    return this.time;
  }

  schedule(callback: () => void): number {
    this.queued = callback;
    return this.nextHandle++;
  }

  cancel(): void {
    this.queued = null;
  }

  get hasPendingFrame(): boolean {
    return this.queued !== null;
  }

  /** Move time forward, then run the frame that was waiting. */
  advance(ms: number): void {
    this.time += ms;
    const frame = this.queued;
    this.queued = null;
    frame?.();
  }
}

/** Read a repository file as it is committed, by path from the root. */
function readRepoFile(relativePath: string): Buffer {
  return readFileSync(fileURLToPath(new URL(`../../../../${relativePath}`, import.meta.url)));
}

/** Serve the committed replay bundle without a network or a global fetch. */
function committedBundleFetch(url: string): Promise<Response> {
  const name = url.endsWith('.json') ? 'emg-replay.json' : 'emg-replay.bin';
  const bytes = readRepoFile(`apps/web/public/replay/${name}`);
  return Promise.resolve(
    new Response(new Uint8Array(bytes), {
      status: 200,
      headers: { 'content-type': name.endsWith('.json') ? 'application/json' : 'application/octet-stream' },
    }),
  );
}

const bundle: ReplayBundle = await loadReplayBundle('/replay', committedBundleFetch);

describe('loadReplayBundle', () => {
  /**
   * The byte-length guard and the int16-to-volts scaling are only ever
   * exercised against the real bundle. Reading the committed bytes turns both
   * into a live gate on the artifact rather than on a synthetic stand-in.
   */
  it('decodes the committed bundle to the shape its manifest declares', () => {
    expect(bundle.segments.length).toBe(bundle.manifest.gestures.length);
    for (let g = 0; g < bundle.segments.length; g++) {
      expect(bundle.segments[g]!.length, bundle.manifest.gestures[g]).toBe(
        bundle.manifest.nChannels * bundle.manifest.samplesPerGesture,
      );
    }
  });

  it('converts counts to volts in the range surface EMG actually occupies', () => {
    let peak = 0;
    for (const segment of bundle.segments) {
      for (const value of segment) peak = Math.max(peak, Math.abs(value));
    }
    // Millivolts would mean the scaling was dropped; microvolts far below this
    // would mean it was applied twice.
    expect(peak).toBeGreaterThan(1e-5);
    expect(peak).toBeLessThan(5e-3);
  });

  it('refuses a bundle whose byte count disagrees with its manifest rather than decoding garbage', async () => {
    const truncating = (url: string): Promise<Response> =>
      url.endsWith('.json')
        ? committedBundleFetch(url)
        : Promise.resolve(new Response(new Uint8Array(16)));

    await expect(loadReplayBundle('/replay', truncating)).rejects.toThrow();
  });
});

describe('ReplaySource: pacing', () => {
  /**
   * The documented safety behaviour. After a long stall, replaying every
   * missed sample at once would hand the decoder one enormous burst and make
   * the latency reading meaningless; the source drops the backlog instead.
   */
  it('caps a stall at a quarter second of samples rather than replaying the whole backlog', () => {
    const clock = new ManualClock();
    const chunks: number[] = [];
    const source = new ReplaySource(bundle, (samples, nChannels) => {
      chunks.push(samples.length / nChannels);
    }, clock);

    source.start();
    clock.advance(2000);

    const { samplingRateHz } = bundle.manifest;
    expect(chunks.at(-1)).toBe(samplingRateHz / 4);
    expect(chunks.at(-1)).toBeLessThan(2 * samplingRateHz);
  });

  it('delivers the number of samples the elapsed time calls for', () => {
    const clock = new ManualClock();
    const chunks: number[] = [];
    const source = new ReplaySource(bundle, (samples, nChannels) => {
      chunks.push(samples.length / nChannels);
    }, clock);

    source.start();
    clock.advance(20);

    expect(chunks.at(-1)).toBe(Math.round((20 / 1000) * bundle.manifest.samplingRateHz));
  });

  it('lays a chunk out channel-major so the worker can de-interleave it', () => {
    const clock = new ManualClock();
    let chunk: Float32Array | null = null;
    const source = new ReplaySource(bundle, (samples) => {
      chunk = samples;
    }, clock);

    source.start();
    clock.advance(20);

    const wanted = Math.round((20 / 1000) * bundle.manifest.samplingRateHz);
    const segment = bundle.segments[0]!;
    const { samplesPerGesture, nChannels } = bundle.manifest;
    for (let c = 0; c < nChannels; c++) {
      expect(chunk![c * wanted], `channel ${c}`).toBe(segment[c * samplesPerGesture]);
    }
  });

  /**
   * A real wearer transitions from one activation into the next mid-signal.
   * Restarting at the segment boundary on every switch would hand the decoder
   * an unrealistically clean onset and flatter every commitment figure.
   */
  it('carries the cursor across a gesture change rather than restarting the segment', () => {
    const clock = new ManualClock();
    let chunk: Float32Array | null = null;
    const source = new ReplaySource(bundle, (samples) => {
      chunk = samples;
    }, clock);

    source.start();
    clock.advance(20);
    source.selectGesture(1);
    clock.advance(20);

    const wanted = Math.round((20 / 1000) * bundle.manifest.samplingRateHz);
    const segment = bundle.segments[1]!;
    // The second chunk starts where the first left off, in the new segment.
    expect(chunk![0]).toBe(segment[wanted]);
  });

  it('rejects a gesture index outside the bundle rather than reading past the end', () => {
    const source = new ReplaySource(bundle, () => {}, new ManualClock());
    expect(() => source.selectGesture(bundle.segments.length)).toThrow(RangeError);
    expect(() => source.selectGesture(-1)).toThrow(RangeError);
  });

  it('cancels the pending frame on stop rather than leaving one queued', () => {
    const clock = new ManualClock();
    const source = new ReplaySource(bundle, () => {}, clock);

    source.start();
    clock.advance(20);
    expect(clock.hasPendingFrame).toBe(true);

    source.stop();
    expect(clock.hasPendingFrame).toBe(false);
    expect(source.isRunning).toBe(false);
  });
});
