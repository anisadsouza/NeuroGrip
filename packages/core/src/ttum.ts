/**
 * Time to Useful Motion.
 *
 * Accuracy says whether the decoder was right. It says nothing about the thing
 * a wearer actually experiences, which is how long they push before the hand
 * does anything. That is what progressive actuation is for, and it needs its
 * own number or the mechanism's whole claim rests on a description.
 *
 * TTUM is the number of hops from a trial's first decodable window to the
 * first hop at which the leading gesture reaches `motionOnset` of its
 * risk-weighted boundary -- the first moment of visible, still reversible
 * motion. At a 20 ms hop every value is an exact multiple of 20 ms, so the
 * distribution is exact rather than binned.
 *
 * Two figures, not one. **Any motion** is when the hand first moves at all.
 * **Correct motion** is when it first moves toward the gesture the wearer
 * intended; motion toward the wrong grip is not useful motion, and on a real
 * prosthesis it is the failure that matters.
 *
 * Trials that never reach the onset are **censored**: counted, never assigned
 * a finite value, and never dropped. There is no mean in the summary for the
 * same reason -- under censoring a mean is either wrong or an unstated
 * imputation, and percentiles alongside a censored count are the honest form.
 *
 * What this measures does NOT include the 200 ms of window fill that precedes
 * the first decision. That latency is additive: a wearer experiences roughly
 * TTUM + 200 ms.
 */

import {
  DEFAULT_EVIDENCE_CONFIG,
  EvidenceAccumulator,
  type EvidenceConfig,
} from './evidence.js';

export interface TtumTrial {
  /** Posteriors per hop, in time order. */
  readonly posteriors: readonly (readonly number[])[];
  /** The gesture the wearer was performing. */
  readonly trueClass: number;
}

export interface TtumOutcome {
  /** Hops until the hand first moved, or null if it never did. */
  readonly hopsToMotion: number | null;
  /** Hops until it first moved toward the intended gesture, or null. */
  readonly hopsToCorrectMotion: number | null;
  /** Hops until commitment latched, or null. */
  readonly hopsToLatch: number | null;
  readonly latchedClass: number | null;
  readonly timedOut: boolean;
  readonly hops: number;
}

export interface TtumDistribution {
  readonly nMoved: number;
  readonly nCensored: number;
  readonly p50Ms: number | null;
  readonly p90Ms: number | null;
  readonly p95Ms: number | null;
  readonly maxMs: number | null;
  /** Milliseconds to count, so the exactness of the hop grid stays visible. */
  readonly histogramMs: Record<string, number>;
}

export interface TtumSummary {
  readonly nTrials: number;
  readonly hopMs: number;
  readonly anyMotion: TtumDistribution;
  readonly correctMotion: TtumDistribution;
  readonly latch: {
    readonly nLatched: number;
    readonly nCensored: number;
    readonly nTimedOut: number;
    readonly accuracyAtLatch: number | null;
    readonly p50Ms: number | null;
    readonly p95Ms: number | null;
  };
  readonly perGesture: Record<
    string,
    { nTrials: number; nMoved: number; p50Ms: number | null; risk: number }
  >;
}

/**
 * Replay one trial through the accumulator and report when things happened.
 *
 * A fresh accumulator per trial: a trial is one attempt, and evidence carried
 * across from the previous one would make the second gesture in a sequence
 * look faster than the first.
 */
export function runTrial(
  trial: TtumTrial,
  risks: readonly number[],
  overrides: Partial<EvidenceConfig> = {},
): TtumOutcome {
  const config = { ...DEFAULT_EVIDENCE_CONFIG, ...overrides };
  const accumulator = new EvidenceAccumulator(risks, config);

  let hopsToCorrectMotion: number | null = null;
  let hopsToLatch: number | null = null;
  let latchedClass: number | null = null;
  let timedOut = false;
  let hops = 0;

  for (const posteriors of trial.posteriors) {
    hops++;
    const decision = accumulator.update(posteriors);

    if (hopsToCorrectMotion === null && decision.moving && decision.leader === trial.trueClass) {
      hopsToCorrectMotion = hops;
    }
    if (decision.latched) {
      hopsToLatch = hops;
      latchedClass = decision.latchedClass;
      timedOut = decision.timedOut;
      break;
    }
  }

  return {
    hopsToMotion: accumulator.hopsToMotion,
    hopsToCorrectMotion,
    hopsToLatch,
    latchedClass,
    timedOut,
    hops,
  };
}

/** Nearest rank over a sorted sample, or null if the sample is empty. */
function percentile(sorted: readonly number[], fraction: number): number | null {
  if (sorted.length === 0) return null;
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))]!;
}

/**
 * Summarise one set of hop counts, where null means the trial never got there.
 *
 * `nMoved + nCensored` always equals the number of trials, and a test asserts
 * it: quietly dropping the trials that never moved is the single easiest way
 * to publish a TTUM figure that is better than the system.
 */
function distribution(hops: readonly (number | null)[], hopMs: number): TtumDistribution {
  const finite = hops.filter((value): value is number => value !== null).sort((a, b) => a - b);
  const milliseconds = finite.map((value) => value * hopMs);

  const histogramMs: Record<string, number> = {};
  for (const ms of milliseconds) {
    histogramMs[String(ms)] = (histogramMs[String(ms)] ?? 0) + 1;
  }

  return {
    nMoved: finite.length,
    nCensored: hops.length - finite.length,
    p50Ms: percentile(milliseconds, 0.5),
    p90Ms: percentile(milliseconds, 0.9),
    p95Ms: percentile(milliseconds, 0.95),
    maxMs: milliseconds.length > 0 ? milliseconds[milliseconds.length - 1]! : null,
    histogramMs,
  };
}

export function summariseTtum(
  outcomes: readonly TtumOutcome[],
  trueClasses: readonly number[],
  gestures: readonly string[],
  risks: readonly number[],
  hopMs = 20,
): TtumSummary {
  if (outcomes.length !== trueClasses.length) {
    throw new RangeError(
      `${outcomes.length} outcomes against ${trueClasses.length} labels`,
    );
  }

  // Indexed rather than filtered-then-matched: two trials can produce
  // identical outcome objects, and `indexOf` would score the wrong one.
  const latchMs: number[] = [];
  let nLatched = 0;
  let correctLatches = 0;
  for (let index = 0; index < outcomes.length; index++) {
    const outcome = outcomes[index]!;
    if (outcome.hopsToLatch === null) continue;
    nLatched++;
    latchMs.push(outcome.hopsToLatch * hopMs);
    if (outcome.latchedClass === trueClasses[index]) correctLatches++;
  }
  latchMs.sort((a, b) => a - b);

  const perGesture: TtumSummary['perGesture'] = {};
  for (let index = 0; index < gestures.length; index++) {
    const name = gestures[index]!;
    const mine = outcomes.filter((_, i) => trueClasses[i] === index);
    const ms = mine
      .map((outcome) => outcome.hopsToMotion)
      .filter((value): value is number => value !== null)
      .map((value) => value * hopMs)
      .sort((a, b) => a - b);
    perGesture[name] = {
      nTrials: mine.length,
      nMoved: ms.length,
      p50Ms: percentile(ms, 0.5),
      risk: risks[index] ?? 0,
    };
  }

  return {
    nTrials: outcomes.length,
    hopMs,
    anyMotion: distribution(
      outcomes.map((outcome) => outcome.hopsToMotion),
      hopMs,
    ),
    correctMotion: distribution(
      outcomes.map((outcome) => outcome.hopsToCorrectMotion),
      hopMs,
    ),
    latch: {
      nLatched,
      nCensored: outcomes.length - nLatched,
      nTimedOut: outcomes.filter((outcome) => outcome.timedOut).length,
      accuracyAtLatch: nLatched > 0 ? correctLatches / nLatched : null,
      p50Ms: percentile(latchMs, 0.5),
      p95Ms: percentile(latchMs, 0.95),
    },
    perGesture,
  };
}
