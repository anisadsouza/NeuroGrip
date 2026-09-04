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

export interface ReplayManifest {
  readonly formatVersion: number;
  readonly featureSpecVersion: number;
  readonly samplingRateHz: number;
  readonly nChannels: number;
  readonly samplesPerGesture: number;
  readonly voltsPerCount: number;
  readonly gestures: readonly string[];
  readonly source: string;
  readonly caveat: string;
}

export interface ReplayBundle {
  readonly manifest: ReplayManifest;
  /** Volts, indexed [gesture][channel * samplesPerGesture + t]. */
  readonly segments: readonly Float32Array[];
}

export async function loadReplayBundle(baseUrl: string): Promise<ReplayBundle> {
  const [manifest, buffer] = await Promise.all([
    fetch(`${baseUrl}/emg-replay.json`).then((r) => {
      if (!r.ok) throw new Error(`replay manifest not found at ${baseUrl}`);
      return r.json() as Promise<ReplayManifest>;
    }),
    fetch(`${baseUrl}/emg-replay.bin`).then((r) => {
      if (!r.ok) throw new Error(`replay samples not found at ${baseUrl}`);
      return r.arrayBuffer();
    }),
  ]);

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

export class ReplaySource {
  private gestureIndex = 0;
  private cursor = 0;
  private lastTick = 0;
  private frame: number | null = null;
  private running = false;

  constructor(
    private readonly bundle: ReplayBundle,
    private readonly sink: SampleSink,
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
    this.lastTick = performance.now();
    this.tick();
  }

  stop(): void {
    this.running = false;
    if (this.frame !== null) {
      cancelAnimationFrame(this.frame);
      this.frame = null;
    }
  }

  private tick = (): void => {
    if (!this.running) return;

    const now = performance.now();
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

    this.frame = requestAnimationFrame(this.tick);
  };
}
