/**
 * Pure frame arithmetic, kept out of the worker module.
 *
 * These three functions are the worker's only non-trivial logic that does not
 * involve ONNX, and while they lived beside `import * as ort from
 * 'onnxruntime-web'` they were untestable at any price: importing them meant
 * instantiating a WASM runtime. Here they are ordinary functions.
 */

/**
 * Decimate a window for drawing, preserving the envelope.
 *
 * Min/max decimation, not peak-picking. Taking the largest-magnitude sample in
 * each bucket loses the sign alternation that makes a waveform look like a
 * waveform: consecutive buckets can both land on positive peaks and the trace
 * drifts off the baseline into smooth undulation. Emitting the minimum and the
 * maximum of each bucket in order keeps both extremes, so the drawn envelope
 * matches what an oscilloscope would show.
 */
export function decimate(channels: Float64Array[], points: number): Float32Array {
  const out = new Float32Array(channels.length * points);
  const width = channels[0]!.length;
  const buckets = Math.max(1, Math.floor(points / 2));
  const stride = width / buckets;

  for (let c = 0; c < channels.length; c++) {
    const source = channels[c]!;
    const base = c * points;
    for (let b = 0; b < buckets; b++) {
      const start = Math.floor(b * stride);
      const end = Math.max(start + 1, Math.min(width, Math.floor((b + 1) * stride)));

      let low = source[start]!;
      let high = source[start]!;
      for (let i = start + 1; i < end; i++) {
        const value = source[i]!;
        if (value < low) low = value;
        if (value > high) high = value;
      }

      // Alternate the order bucket to bucket so the polyline zig-zags through
      // the envelope rather than doubling back on itself every other point.
      const first = b % 2 === 0 ? low : high;
      const second = b % 2 === 0 ? high : low;
      out[base + b * 2] = first;
      if (b * 2 + 1 < points) out[base + b * 2 + 1] = second;
    }
  }
  return out;
}


/**
 * Root-mean-square per channel, in the units the samples arrived in.
 *
 * This drives the activation ring and the hand's colour, so it is a
 * measurement the wearer reads, not an internal quantity.
 */
export function channelRms(window: readonly Float64Array[]): number[] {
  const out: number[] = [];
  for (const channel of window) {
    let sum = 0;
    for (let i = 0; i < channel.length; i++) sum += channel[i]! * channel[i]!;
    out.push(Math.sqrt(sum / channel.length));
  }
  return out;
}

/**
 * Split a channel-major chunk into one array per channel.
 *
 * The layout is channel-major blocks -- every sample of channel 0, then every
 * sample of channel 1 -- and not interleaved. Reading it as interleaved would
 * not error; it would silently transpose the electrode ring, so the decoder
 * would receive a plausible eight-channel window belonging to no arm. The
 * round-trip against the replay source's own layout is what pins this.
 */
export function deinterleave(samples: Float32Array, nChannels: number): Float64Array[] {
  const count = samples.length / nChannels;
  if (!Number.isInteger(count)) {
    throw new RangeError(`${samples.length} samples do not divide into ${nChannels} channels`);
  }
  const channels: Float64Array[] = [];
  for (let c = 0; c < nChannels; c++) {
    const view = new Float64Array(count);
    for (let i = 0; i < count; i++) view[i] = samples[c * count + i]!;
    channels.push(view);
  }
  return channels;
}
