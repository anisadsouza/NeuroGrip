/**
 * Numeric primitives for the feature pipeline.
 *
 * A port of services/research/neurogrip/dsp.py, algorithm for algorithm.
 * The two implementations are pinned against each other by
 * test/conformance.test.ts. Do not "improve" anything here without changing
 * the Python side and regenerating the golden vectors.
 */

export function nextPow2(n: number): number {
  if (n < 1) throw new RangeError('n must be positive');
  let power = 1;
  while (power < n) power *= 2;
  return power;
}

export function hann(n: number): Float64Array {
  if (n < 1) throw new RangeError('n must be positive');
  const w = new Float64Array(n);
  if (n === 1) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  }
  return w;
}

interface Twiddle {
  cos: Float64Array;
  sin: Float64Array;
}

const twiddleCache = new Map<number, Twiddle>();

function twiddles(n: number): Twiddle {
  const cached = twiddleCache.get(n);
  if (cached) return cached;
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  for (let k = 0; k < n / 2; k++) {
    const angle = (-2 * Math.PI * k) / n;
    cos[k] = Math.cos(angle);
    sin[k] = Math.sin(angle);
  }
  const table = { cos, sin };
  twiddleCache.set(n, table);
  return table;
}

/**
 * In-place iterative radix-2 Cooley-Tukey FFT.
 *
 * Twiddles come from a precomputed exact table rather than being accumulated
 * incrementally across butterflies: incremental accumulation drifts, and
 * conformance with numpy matters more than the microseconds saved.
 */
function fftInPlace(real: Float64Array, imag: Float64Array): void {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tempReal = real[i]!;
      real[i] = real[j]!;
      real[j] = tempReal;
      const tempImag = imag[i]!;
      imag[i] = imag[j]!;
      imag[j] = tempImag;
    }
  }

  const table = twiddles(n);
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const stride = n / len;
    for (let start = 0; start < n; start += len) {
      for (let k = 0; k < half; k++) {
        const wr = table.cos[k * stride]!;
        const wi = table.sin[k * stride]!;
        const a = start + k;
        const b = a + half;
        const vr = real[b]! * wr - imag[b]! * wi;
        const vi = real[b]! * wi + imag[b]! * wr;
        real[b] = real[a]! - vr;
        imag[b] = imag[a]! - vi;
        real[a] = real[a]! + vr;
        imag[a] = imag[a]! + vi;
      }
    }
  }
}

export function periodogram(
  x: Float64Array,
  fs: number,
): { freqs: Float64Array; psd: Float64Array } {
  const n = x.length;
  if (n < 1) throw new RangeError('signal must not be empty');

  const window = hann(n);
  const nfft = nextPow2(n);
  const real = new Float64Array(nfft);
  const imag = new Float64Array(nfft);

  let windowEnergy = 0;
  for (let i = 0; i < n; i++) {
    real[i] = x[i]! * window[i]!;
    windowEnergy += window[i]! * window[i]!;
  }

  fftInPlace(real, imag);

  const bins = nfft / 2 + 1;
  const psd = new Float64Array(bins);
  const scale = fs * windowEnergy;
  if (scale > 0) {
    for (let k = 0; k < bins; k++) {
      psd[k] = (real[k]! * real[k]! + imag[k]! * imag[k]!) / scale;
    }
    // Fold negative-frequency power into positive bins, excluding DC and Nyquist.
    for (let k = 1; k < bins - 1; k++) psd[k] = psd[k]! * 2;
  }

  const freqs = new Float64Array(bins);
  for (let k = 0; k < bins; k++) freqs[k] = (k * fs) / nfft;

  return { freqs, psd };
}

/** Mean of each of nBands contiguous groups, matching numpy array_split sizing. */
export function bandMeans(values: Float64Array, nBands: number): Float64Array {
  if (nBands < 1) throw new RangeError('nBands must be positive');
  const out = new Float64Array(nBands);
  const base = Math.floor(values.length / nBands);
  const remainder = values.length % nBands;

  let start = 0;
  for (let band = 0; band < nBands; band++) {
    const size = base + (band < remainder ? 1 : 0);
    if (size === 0) {
      out[band] = 0;
      continue;
    }
    let sum = 0;
    for (let i = 0; i < size; i++) sum += values[start + i]!;
    out[band] = sum / size;
    start += size;
  }
  return out;
}

/** Biased autocorrelation r[k] = sum(x[n] * x[n+k]) / N for k = 0..maxLag. */
export function autocorrelate(x: Float64Array, maxLag: number): Float64Array {
  if (maxLag < 0) throw new RangeError('maxLag must not be negative');
  const n = x.length;
  const out = new Float64Array(maxLag + 1);
  const limit = Math.min(maxLag, n - 1);
  for (let lag = 0; lag <= limit; lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) sum += x[i]! * x[i + lag]!;
    out[lag] = sum / n;
  }
  return out;
}

/**
 * AR coefficients a[1..order] for x[n] = -sum(a[i] * x[n-i]) + e[n].
 * Returns zeros for a signal with no power, which is correct for a silent channel.
 */
export function levinsonDurbin(r: Float64Array, order: number): Float64Array {
  if (order < 1) throw new RangeError('order must be positive');
  const zeros = new Float64Array(order);
  if (r.length < order + 1 || r[0]! <= 0) return zeros;

  const a = new Float64Array(order + 1);
  a[0] = 1;
  let error = r[0]!;

  for (let m = 1; m <= order; m++) {
    let acc = r[m]!;
    for (let i = 1; i < m; i++) acc += a[i]! * r[m - i]!;
    const reflection = -acc / error;

    // Snapshot a[1..m-1] before updating: the update reads it reversed.
    const previous = a.slice(1, m);
    a[m] = reflection;
    for (let i = 1; i < m; i++) {
      a[i] = previous[i - 1]! + reflection * previous[m - 1 - i]!;
    }

    error *= 1 - reflection * reflection;
    if (error <= 0) return zeros;
  }

  return a.slice(1);
}

export function medianFrequency(freqs: Float64Array, psd: Float64Array): number {
  let total = 0;
  for (let i = 0; i < psd.length; i++) total += psd[i]!;
  if (total <= 0) return 0;

  const target = 0.5 * total;
  let cumulative = 0;
  for (let i = 0; i < psd.length; i++) {
    cumulative += psd[i]!;
    if (cumulative >= target) return freqs[i]!;
  }
  return freqs[freqs.length - 1]!;
}

export function meanFrequency(freqs: Float64Array, psd: Float64Array): number {
  let total = 0;
  let weighted = 0;
  for (let i = 0; i < psd.length; i++) {
    total += psd[i]!;
    weighted += freqs[i]! * psd[i]!;
  }
  return total <= 0 ? 0 : weighted / total;
}
