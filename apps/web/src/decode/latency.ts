/**
 * How the Live screen summarises what it measured.
 *
 * Both functions encode a refusal. The percentile refuses to report a figure
 * from too few samples, because a P95 over five windows is a maximum wearing a
 * percentile's name. The ring refuses to grow without bound, because a session
 * left running would otherwise accumulate samples until the tab died.
 */

/** Below this many samples, no percentile is reported at all. */
export const MINIMUM_LATENCY_SAMPLES = 20;

/** How many recent measurements the screen keeps. */
export const LATENCY_WINDOW = 200;

/**
 * The 95th percentile by nearest rank, or null if too few samples have arrived.
 *
 * Nearest rank, deliberately: `sorted[floor(n * 0.95)]` returns a value that
 * was actually measured. Note this is NOT the estimator `experiments.py` uses
 * for the Python-side figure -- `np.percentile` interpolates between the two
 * neighbouring samples -- so the two numbers are close but not the same
 * statistic, and should not be compared to the last decimal.
 */
export function percentile95(
  samples: readonly number[],
  minimumSamples: number = MINIMUM_LATENCY_SAMPLES,
): number | null {
  if (samples.length < minimumSamples) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length * 0.95)]!;
}

/**
 * Append a measurement, keeping the newest `limit`.
 *
 * The newest, not the oldest. A P95 computed over the first two hundred
 * windows of a session would freeze at whatever the machine was doing while
 * the model was still warming up.
 */
export function trimRing(
  previous: readonly number[],
  next: number,
  limit: number = LATENCY_WINDOW,
): number[] {
  const appended = [...previous, next];
  return appended.length > limit ? appended.slice(appended.length - limit) : appended;
}
