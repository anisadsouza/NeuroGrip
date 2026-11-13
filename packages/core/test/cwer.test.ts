/**
 * Cost-weighted error rate.
 *
 * Accuracy counts a fist latched in place of an open hand and an open hand
 * latched in place of a fist as the same mistake. They are not: one closes a
 * power grip on whatever is in front of the wearer. CWER charges each wrong
 * commitment the commit cost of the gesture it wrongly committed to, so the
 * risk weighting in the accumulator is scored by the quantity it was built to
 * reduce. The two easy ways to flatter it are charging the safe fallback as an
 * error and dropping the trials that never committed; both are tested here.
 */

import { describe, expect, it } from 'vitest';
import { DEFAULT_EVIDENCE_CONFIG } from '../src/evidence.js';
import { costWeightedErrorRate } from '../src/cwer.js';
import { type TtumOutcome, type TtumTrial, runTrial } from '../src/ttum.js';

const gestures = ['rest', 'open_hand', 'fist'];
const costs = [0, 0.1, 1.0];

function peaked(nClasses: number, winner: number, confidence: number): number[] {
  const rest = (1 - confidence) / (nClasses - 1);
  return Array.from({ length: nClasses }, (_, i) => (i === winner ? confidence : rest));
}

function trial(winner: number, trueClass: number, hops = 41, confidence = 0.95): TtumTrial {
  return {
    posteriors: Array.from({ length: hops }, () => peaked(3, winner, confidence)),
    trueClass,
  };
}

function run(t: TtumTrial): TtumOutcome {
  return runTrial(t, costs);
}

describe('costWeightedErrorRate', () => {
  it('charges a wrong commitment the cost of the gesture it wrongly committed to', () => {
    // Intended open hand, latched fist: the expensive mistake.
    const fistInError = costWeightedErrorRate([run(trial(2, 1))], [1], gestures, costs);
    // Intended fist, latched open hand: the cheap one.
    const openInError = costWeightedErrorRate([run(trial(1, 2))], [2], gestures, costs);

    expect(fistInError.errorRate).toBe(1);
    expect(openInError.errorRate).toBe(1);
    expect(fistInError.cwer).toBeCloseTo(1.0, 12);
    expect(openInError.cwer).toBeCloseTo(0.1, 12);
  });

  it('charges nothing for a correct commitment', () => {
    const summary = costWeightedErrorRate(
      [run(trial(1, 1)), run(trial(2, 2))],
      [1, 2],
      gestures,
      costs,
    );
    expect(summary.nWrongLatches).toBe(0);
    expect(summary.cwer).toBe(0);
  });

  /**
   * The urgency fallback is a refusal to guess. An unintended rest is not a
   * physical event, so charging it would penalise the safety behaviour the
   * mechanism exists to provide. It is still counted, separately, so a run
   * that "wins" CWER by never committing is visible as exactly that.
   */
  it('counts the safe fallback separately and does not charge it as a wrong commitment', () => {
    const flat: TtumTrial = {
      posteriors: Array.from({ length: DEFAULT_EVIDENCE_CONFIG.maxHops + 5 }, () => [
        0.34, 0.33, 0.33,
      ]),
      trueClass: 2,
    };
    const summary = costWeightedErrorRate([run(flat)], [2], gestures, costs);

    expect(summary.nSafeFallbacks).toBe(1);
    expect(summary.nWrongLatches).toBe(0);
    expect(summary.cwer).toBe(0);
  });

  it('divides by every trial, including the ones that never committed', () => {
    const neverCommits: TtumTrial = {
      posteriors: Array.from({ length: 41 }, () => [0.34, 0.33, 0.33]),
      trueClass: 1,
    };
    const summary = costWeightedErrorRate(
      [run(trial(2, 1)), run(neverCommits)],
      [1, 1],
      gestures,
      costs,
    );

    expect(summary.nTrials).toBe(2);
    expect(summary.nUncommitted).toBe(1);
    expect(
      summary.nCorrectLatches + summary.nWrongLatches + summary.nSafeFallbacks + summary.nUncommitted,
    ).toBe(summary.nTrials);
    expect(summary.cwer).toBeCloseTo(0.5, 12);
  });

  it('itemises each confusion by intended and committed gesture', () => {
    const summary = costWeightedErrorRate(
      [run(trial(2, 1)), run(trial(2, 1)), run(trial(1, 2))],
      [1, 1, 2],
      gestures,
      costs,
    );
    const pairs = Object.fromEntries(
      summary.confusions.map((item) => [`${item.intended}->${item.committed}`, item]),
    );

    expect(pairs['open_hand->fist']).toMatchObject({ count: 2, unitCost: 1.0 });
    expect(pairs['fist->open_hand']).toMatchObject({ count: 1, unitCost: 0.1 });
    // Ordered by total cost, so the costliest confusion is read first.
    expect(summary.confusions[0]!.committed).toBe('fist');
  });

  it('refuses mismatched labels or costs rather than aligning them silently', () => {
    const outcome = run(trial(1, 1));
    expect(() => costWeightedErrorRate([outcome], [1, 2], gestures, costs)).toThrow(RangeError);
    expect(() => costWeightedErrorRate([outcome], [1], gestures, [0, 1])).toThrow(RangeError);
  });

  it('reports nothing rather than zero for an empty run', () => {
    const summary = costWeightedErrorRate([], [], gestures, costs);
    expect(summary.cwer).toBeNull();
    expect(summary.errorRate).toBeNull();
  });
});
