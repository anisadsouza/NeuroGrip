/**
 * Replays recorded sEMG at the sampling rate.
 *
 * The signal is simulated, not human. That is stated in the manifest and shown
 * in the interface; it must never be presented as a recording of a person.
 *
 * The source paces itself off wall-clock time rather than a fixed timer
 * interval, so a browser that throttles timers (a background tab, a busy main
 * thread) resumes with the correct number of samples rather than silently
 * playing back in slow motion and flattering the latency figures.
 */

import { assertFeatureSpecCompatible } from '@neurogrip/core';

export interface ReplayManifest {
  readonly formatVersion: number;
  readonly featureSpecVersion: number;
  readonly samplingRateHz: number;
  readonly nChannels: number;
  readonly samplesPerGesture: number;
  readonly voltsPerCount: number;
  readonly gestures: readonly string[];
  /**
   * Per-electrode weighting for each functional muscle group, computed from
   * the forearm volume-conduction model in `neurogrip.anatomy`.
   *
   * Carried in the manifest rather than reimplemented here for the same reason
   * the simulator is not ported to the browser: a second copy of the anatomy
   * would be a second thing to keep in step. Optional, so a bundle written
   * before this field existed still loads and the hand simply shows no drive.
   */
  readonly muscleGroups?: {
    readonly digit_flexor: readonly number[];
    readonly digit_extensor: readonly number[];
    readonly wrist_flexor: readonly number[];
    readonly wrist_extensor: readonly number[];
  };
  readonly source: string;
  /** Present for recorded bundles, e.g. `S12`; absent for simulation. */
  readonly subject?: string;
  readonly caveat: string;
}

/** Human-readable provenance. Never infer “recorded” from the URL or name. */
export function replaySourceLabel(manifest: ReplayManifest): string {
  return manifest.source === 'ninapro_db2'
    ? `Recorded · NinaPro DB2 · subject ${manifest.subject ?? 'unknown'}`
    : 'Simulated · NeuroGrip motor-unit model';
}

export interface ReplayBundle {
  readonly manifest: ReplayManifest;
  /** Volts, indexed [gesture][channel * samplesPerGesture + t]. */
  readonly segments: readonly Float32Array[];
}

/**
 * How the bundle is fetched. Defaulted, so no caller changes; present so a
 * test can hand in the committed bytes rather than patching a global.
 */
export type FetchLike = (url: string) => Promise<Response>;

async function fetchManifest(url: string, fetchLike: FetchLike): Promise<ReplayManifest> {
  const response = await fetchLike(url);
  if (!response.ok) throw new Error(`replay manifest not found at ${url}`);
  const body = await response.text();
  try {
    return JSON.parse(body) as ReplayManifest;
  } catch {
    // Vite and many static hosts return index.html for an unknown route. Naming
    // that condition makes a wrong base URL or missing public asset fixable.
    const kind = body.trimStart().startsWith('<') ? 'HTML' : 'invalid JSON';
    throw new Error(`replay manifest at ${url} returned ${kind}`);
  }
}

export async function loadReplayBundle(
  baseUrl: string,
  fetchLike: FetchLike = (url) => fetch(url),
): Promise<ReplayBundle> {
  const [manifest, buffer] = await Promise.all([
    fetchManifest(`${baseUrl}/emg-replay.json`, fetchLike),
    fetchLike(`${baseUrl}/emg-replay.bin`).then((r) => {
      if (!r.ok) throw new Error(`replay samples not found at ${baseUrl}`);
      return r.arrayBuffer();
    }),
  ]);

  // A bundle generated under a different feature specification would feed the
  // decoder a signal it was not fitted on. Refused here rather than in the
  // worker, so the failure names the bundle that caused it.
  assertFeatureSpecCompatible({
    source: 'emg-replay.json',
    featureSpecVersion: manifest.featureSpecVersion,
  });

  const perGesture = manifest.nChannels * manifest.samplesPerGesture;
  const expected = manifest.gestures.length * perGesture * 2;
  if (buffer.byteLength !== expected) {
    throw new Error(
      `replay bundle is ${buffer.byteLength} bytes, manifest describes ${expected}`,
    );
  }

  const counts = new Int16Array(buffer);
  const segments: Float32Array[] = [];
  for (let g = 0; g < manifest.gestures.length; g++) {
    const volts = new Float32Array(perGesture);
    const offset = g * perGesture;
    for (let i = 0; i < perGesture; i++) {
      volts[i] = counts[offset + i]! * manifest.voltsPerCount;
    }
    segments.push(volts);
  }

  return { manifest, segments };
}

export type SampleSink = (samples: Float32Array, nChannels: number) => void;

/**
 * The clock and scheduler the source paces itself against.
 *
 * Injected rather than reached for globally so that a test can advance time
 * deliberately and assert what the source does with a stall. Wall-clock pacing
 * is the whole point of this class, and a behaviour that only ever runs
 * against a real clock cannot be tested at all without waiting for one.
 */
export interface ReplayClock {
  now(): number;
  schedule(callback: () => void): number;
  cancel(handle: number): void;
}

export const WALL_CLOCK: ReplayClock = {
  now: () => performance.now(),
  schedule: (callback) => requestAnimationFrame(callback),
  cancel: (handle) => cancelAnimationFrame(handle),
};

export class ReplaySource {
  private gestureIndex = 0;
  private cursor = 0;
  private lastTick = 0;
  private frame: number | null = null;
  private running = false;

  constructor(
    private readonly bundle: ReplayBundle,
    private readonly sink: SampleSink,
    private readonly clock: ReplayClock = WALL_CLOCK,
  ) {}

  get isRunning(): boolean {
    return this.running;
  }

  get gesture(): string {
    return this.bundle.manifest.gestures[this.gestureIndex]!;
  }

  /**
   * Switch the gesture being replayed.
   *
   * The cursor is deliberately NOT reset: a real wearer transitions from one
   * activation into the next mid-signal, and restarting at the segment
   * boundary every time would hand the decoder an unrealistically clean onset.
   */
  selectGesture(index: number): void {
    if (index < 0 || index >= this.bundle.segments.length) {
      throw new RangeError(`gesture index ${index} out of range`);
    }
    this.gestureIndex = index;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.lastTick = this.clock.now();
    this.tick();
  }

  stop(): void {
    this.running = false;
    if (this.frame !== null) {
      this.clock.cancel(this.frame);
      this.frame = null;
    }
  }

  private tick = (): void => {
    if (!this.running) return;

    const now = this.clock.now();
    const elapsed = now - this.lastTick;
    this.lastTick = now;

    const { samplingRateHz, nChannels, samplesPerGesture } = this.bundle.manifest;
    // Cap the catch-up. After a long stall, replaying every missed sample at
    // once would produce one enormous burst and a meaningless latency reading;
    // dropping the backlog is the honest behaviour.
    const wanted = Math.min(
      Math.round((elapsed / 1000) * samplingRateHz),
      samplingRateHz / 4,
    );

    if (wanted > 0) {
      const segment = this.bundle.segments[this.gestureIndex]!;
      const chunk = new Float32Array(nChannels * wanted);
      for (let c = 0; c < nChannels; c++) {
        for (let i = 0; i < wanted; i++) {
          const t = (this.cursor + i) % samplesPerGesture;
          chunk[c * wanted + i] = segment[c * samplesPerGesture + t]!;
        }
      }
      this.cursor = (this.cursor + wanted) % samplesPerGesture;
      this.sink(chunk, nChannels);
    }

    this.frame = this.clock.schedule(this.tick);
  };
}
