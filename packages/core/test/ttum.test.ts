/**
 * The metric that makes progressive actuation checkable.
 *
 * Every claim this project makes about the evidence accumulator is a claim
 * about time: motion starts early, high-cost gestures wait longer, nothing
 * hangs. Accuracy cannot see any of it. If TTUM is computed wrongly -- and the
 * two easy wrongs are dropping the trials that never moved and losing the risk
 * weighting -- the number still looks reasonable and flatters the mechanism it
 * was built to test.
 */

import { describe, expect, it } from 'vitest';
import { DEFAULT_EVIDENCE_CONFIG } from '../src/evidence.js';
import { type TtumTrial, runTrial, summariseTtum } from '../src/ttum.js';

/** Posteriors concentrated on `winner` at the given confidence. */
function peaked(nClasses: number, winner: number, confidence: number): number[] {
  const rest = (1 - confidence) / (nClasses - 1);
  return Array.from({ length: nClasses }, (_, i) => (i === winner ? confidence : rest));
}

/** A trial of `hops` identical windows, all favouring `winner`. */
function steadyTrial(
  nClasses: number,
  winner: number,
  confidence: number,
  hops: number,
): TtumTrial {
  return {
    posteriors: Array.from({ length: hops }, () => peaked(nClasses, winner, confidence)),
    trueClass: winner,
  };
}

describe('runTrial', () => {
  it('reports the hop motion began on rather than the hop evidence started', () => {
    const outcome = runTrial(steadyTrial(3, 1, 0.95, 41), [0, 0.1, 0.1]);

    expect(outcome.hopsToMotion).not.toBeNull();
    // Motion is a fifth of the way to the boundary, so it cannot be the first
    // hop: a metric that reported 1 would mean the onset had been lost.
    expect(outcome.hopsToMotion!).toBeGreaterThan(1);
    expect(outcome.hopsToMotion!).toBeLessThan(outcome.hopsToLatch!);
  });

  /**
   * The mechanism claim, stated as an inequality. Identical evidence, two
   * costs: the expensive gesture must take longer to reach the same fraction
   * of its own boundary, because its boundary is further away. This is the one
   * assertion here that fails if the risk weighting is ever dropped.
   */
  it('starts a costly gesture moving later than a cheap one on identical evidence', () => {
    const cheap = runTrial(steadyTrial(2, 1, 0.95, 60), [0, 0.1]);
    const costly = runTrial(steadyTrial(2, 1, 0.95, 60), [0, 1.0]);

    expect(cheap.hopsToMotion).not.toBeNull();
    expect(costly.hopsToMotion).not.toBeNull();
    expect(costly.hopsToMotion!).toBeGreaterThan(cheap.hopsToMotion!);
    expect(costly.hopsToLatch!).toBeGreaterThan(cheap.hopsToLatch!);
  });

  /**
   * Ambiguous evidence produces no motion, and that must be recorded as
   * "never" rather than as the trial's length. Reporting the length would put
   * a finite, plausible-looking number into the distribution for a trial in
   * which the hand did nothing at all.
   */
  it('reports a trial that never separates as censored rather than as its length', () => {
    const flat: TtumTrial = {
      posteriors: Array.from({ length: 41 }, () => [0.5, 0.5]),
      trueClass: 1,
    };
    const outcome = runTrial(flat, [0, 1.0]);

    expect(outcome.hopsToMotion).toBeNull();
    expect(outcome.hopsToLatch).toBeNull();
    expect(outcome.hops).toBe(41);
  });

  it('separates motion toward the wrong gesture from motion toward the right one', () => {
    // Confident, and confidently wrong: the hand moves, but not usefully.
    const misled: TtumTrial = {
      posteriors: Array.from({ length: 41 }, () => peaked(3, 2, 0.95)),
      trueClass: 1,
    };
    const outcome = runTrial(misled, [0, 0.1, 0.1]);

    expect(outcome.hopsToMotion).not.toBeNull();
    expect(outcome.hopsToCorrectMotion).toBeNull();
  });

  it('stops at the latch rather than accumulating past a committed decision', () => {
    const outcome = runTrial(steadyTrial(2, 1, 0.99, 200), [0, 0.1]);
    expect(outcome.hops).toBe(outcome.hopsToLatch);
  });

  it('reports the safe fallback as a timeout rather than as a decision', () => {
    // Long enough to pass maxHops with evidence that never separates.
    const flat: TtumTrial = {
      posteriors: Array.from({ length: DEFAULT_EVIDENCE_CONFIG.maxHops + 5 }, () => [0.5, 0.5]),
      trueClass: 1,
    };
    const outcome = runTrial(flat, [0, 1.0]);

    expect(outcome.timedOut).toBe(true);
    expect(outcome.latchedClass).toBe(DEFAULT_EVIDENCE_CONFIG.safeClassIndex);
  });
});

describe('summariseTtum', () => {
  const gestures = ['rest', 'open_hand', 'fist'];
  const risks = [0, 0.1, 1.0];

  /**
   * The arithmetic that catches a flattered figure. Trials that never moved
   * are the slow ones by definition, so dropping them lowers every percentile
   * -- and nothing else in the summary would look wrong.
   */
  it('accounts for every trial as either moved or censored', () => {
    const outcomes = [
      runTrial(steadyTrial(3, 1, 0.95, 41), risks),
      runTrial(steadyTrial(3, 2, 0.95, 41), risks),
      runTrial({ posteriors: Array.from({ length: 41 }, () => [0.34, 0.33, 0.33]), trueClass: 1 }, risks),
    ];
    const summary = summariseTtum(outcomes, [1, 2, 1], gestures, risks);

    expect(summary.anyMotion.nMoved + summary.anyMotion.nCensored).toBe(summary.nTrials);
    expect(summary.correctMotion.nMoved + summary.correctMotion.nCensored).toBe(summary.nTrials);
    expect(summary.latch.nLatched + summary.latch.nCensored).toBe(summary.nTrials);
  });

  /**
   * The hop grid is the unit of measurement. A value between two hops would
   * mean an interpolation had crept in, and the distribution would be claiming
   * a resolution the system does not have.
   */
  it('reports times on the hop grid rather than between hops', () => {
    const outcomes = [runTrial(steadyTrial(3, 1, 0.95, 41), risks)];
    const summary = summariseTtum(outcomes, [1], gestures, risks);

    for (const value of [summary.anyMotion.p50Ms, summary.anyMotion.maxMs]) {
      expect(value).not.toBeNull();
      expect(value! % summary.hopMs, `${value} ms is not a whole hop`).toBe(0);
    }
    for (const key of Object.keys(summary.anyMotion.histogramMs)) {
      expect(Number(key) % summary.hopMs, key).toBe(0);
    }
  });

  it('reports nothing rather than zero when no trial ever moved', () => {
    const flat = runTrial(
      { posteriors: Array.from({ length: 41 }, () => [0.34, 0.33, 0.33]), trueClass: 1 },
      risks,
    );
    const summary = summariseTtum([flat], [1], gestures, risks);

    expect(summary.anyMotion.nMoved).toBe(0);
    expect(summary.anyMotion.p50Ms).toBeNull();
    expect(summary.anyMotion.maxMs).toBeNull();
  });

  /**
   * The per-gesture table is where a reader sees the risk mechanism as a
   * trend. It has to carry the cost beside the time, or the trend is a
   * coincidence the reader has to take on trust.
   */
  it('carries each gesture commit cost beside its time', () => {
    const outcomes = [
      runTrial(steadyTrial(3, 1, 0.95, 41), risks),
      runTrial(steadyTrial(3, 2, 0.95, 41), risks),
    ];
    const summary = summariseTtum(outcomes, [1, 2], gestures, risks);

    expect(summary.perGesture.open_hand!.risk).toBe(0.1);
    expect(summary.perGesture.fist!.risk).toBe(1.0);
    expect(summary.perGesture.fist!.p50Ms!).toBeGreaterThan(
      summary.perGesture.open_hand!.p50Ms!,
    );
  });

  it('scores latch accuracy against the trial label rather than against the leader', () => {
    const right = runTrial(steadyTrial(3, 1, 0.95, 41), risks);
    const wrong = runTrial(
      { posteriors: Array.from({ length: 41 }, () => peaked(3, 2, 0.95)), trueClass: 1 },
      risks,
    );
    const summary = summariseTtum([right, wrong], [1, 1], gestures, risks);

    expect(summary.latch.nLatched).toBe(2);
    expect(summary.latch.accuracyAtLatch).toBeCloseTo(0.5, 12);
  });

  it('refuses a label list that does not match the outcomes rather than aligning them silently', () => {
    const outcome = runTrial(steadyTrial(3, 1, 0.95, 41), risks);
    expect(() => summariseTtum([outcome], [1, 2], gestures, risks)).toThrow(RangeError);
  });
});
