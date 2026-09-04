/**
 * Streaming front end: window geometry and a preallocated ring buffer.
 *
 * Mirrors services/research/neurogrip/windowing.py for geometry. The 20 ms hop
 * (rather than the report's 100 ms) is what makes progressive actuation
 * possible: at a 10 Hz decision rate there is nothing to accumulate between
 * decisions.
 */

export interface WindowConfig {
  windowMs: number;
  hopMs: number;
  samplingRateHz: number;
}

export const DEFAULT_WINDOW_CONFIG: WindowConfig = {
  windowMs: 200,
  hopMs: 20,
  samplingRateHz: 2000,
};

export function windowSamples(config: WindowConfig): number {
  if (config.windowMs <= 0) throw new RangeError('windowMs must be positive');
  return Math.trunc((config.windowMs * config.samplingRateHz) / 1000);
}

export function hopSamples(config: WindowConfig): number {
  if (config.hopMs <= 0) throw new RangeError('hopMs must be positive');
  return Math.trunc((config.hopMs * config.samplingRateHz) / 1000);
}

/**
 * Fixed-capacity multi-channel ring buffer.
 *
 * Preallocates and never grows. Allocating inside the 50 Hz inference loop
 * invites garbage-collection pauses, which would land directly in the P95
 * latency figure this project treats as a safety property.
 */
export class MultiChannelRingBuffer {
  private readonly buffers: Float64Array[];
  private readonly scratch: Float64Array[];
  private writeIndex = 0;
  private filled = 0;

  constructor(
    public readonly nChannels: number,
    public readonly capacitySamples: number,
  ) {
    if (nChannels < 1) throw new RangeError('nChannels must be at least 1');
    if (capacitySamples < 1) throw new RangeError('capacitySamples must be at least 1');
    this.buffers = Array.from({ length: nChannels }, () => new Float64Array(capacitySamples));
    this.scratch = Array.from({ length: nChannels }, () => new Float64Array(capacitySamples));
  }

  get available(): number {
    return this.filled;
  }

  push(frame: Float64Array[]): void {
    if (frame.length !== this.nChannels) {
      throw new RangeError(`expected ${this.nChannels} channels, received ${frame.length}`);
    }
    const count = frame[0]!.length;
    for (const channel of frame) {
      if (channel.length !== count) throw new RangeError('channels must be equal length');
    }

    for (let i = 0; i < count; i++) {
      const slot = (this.writeIndex + i) % this.capacitySamples;
      for (let c = 0; c < this.nChannels; c++) {
        this.buffers[c]![slot] = frame[c]![i]!;
      }
    }
    this.writeIndex = (this.writeIndex + count) % this.capacitySamples;
    this.filled = Math.min(this.capacitySamples, this.filled + count);
  }

  /**
   * The most recent `width` samples per channel, oldest first, or null if not
   * enough data has arrived. The returned arrays are reused scratch views and
   * are invalidated by the next call - copy them if you need to retain them.
   */
  readLatestWindow(width: number): Float64Array[] | null {
    if (width < 1) throw new RangeError('width must be positive');
    if (width > this.capacitySamples) {
      throw new RangeError('width exceeds buffer capacity');
    }
    if (this.filled < width) return null;

    const start = (this.writeIndex - width + this.capacitySamples) % this.capacitySamples;
    const out: Float64Array[] = [];
    for (let c = 0; c < this.nChannels; c++) {
      const view = this.scratch[c]!.subarray(0, width);
      for (let i = 0; i < width; i++) {
        view[i] = this.buffers[c]![(start + i) % this.capacitySamples]!;
      }
      out.push(view);
    }
    return out;
  }
}
