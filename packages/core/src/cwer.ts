/**
 * Cost-Weighted Error Rate.
 *
 * Accuracy treats every wrong commitment alike. The evidence accumulator does
 * not: it makes a power grip clear a higher bar than an opening hand because
 * the two mistakes are not equally recoverable. CWER scores the system in that
 * same currency. Each wrong commitment is charged the commit cost of the
 * gesture the hand wrongly committed to -- the same `COMMIT_COST` table the
 * accumulator's boundaries come from -- and the total is divided by every
 * trial.
 *
 *     CWER = (1 / N) * sum over wrong latches of cost[committed class]
 *
 * With costs in [0, 1], CWER is in [0, 1] and never exceeds the plain error
 * rate. The gap between the two is what risk weighting buys.
 *
 * Three outcomes are not charged, and each is counted so none can hide:
 *
 * - a **correct** latch;
 * - the **safe fallback** -- urgency ran out and the hand returned to rest. An
 *   unintended rest is a refusal to guess, not a physical event;
 * - an **uncommitted** trial that ended before any latch.
 *
 * A system that drove CWER to zero by never committing would show it as a
 * fallback and uncommitted count equal to N, which is why those counts are
 * part of the result rather than a footnote.
 */

import type { TtumOutcome } from './ttum.js';

export interface CwerConfusion {
  readonly intended: string;
  readonly committed: string;
  readonly count: number;
  /** Commit cost of the wrongly committed gesture. */
  readonly unitCost: number;
  readonly totalCost: number;
}

export interface CwerSummary {
  readonly nTrials: number;
  readonly nCorrectLatches: number;
  readonly nWrongLatches: number;
  readonly nSafeFallbacks: number;
  readonly nUncommitted: number;
  /** Wrong latches over all trials, or null for an empty run. */
  readonly errorRate: number | null;
  /** Cost-weighted wrong latches over all trials, or null for an empty run. */
  readonly cwer: number | null;
  /** Wrong commitments by pair, costliest total first. */
  readonly confusions: readonly CwerConfusion[];
}

export function costWeightedErrorRate(
  outcomes: readonly TtumOutcome[],
  trueClasses: readonly number[],
  gestures: readonly string[],
  costs: readonly number[],
): CwerSummary {
  if (outcomes.length !== trueClasses.length) {
    throw new RangeError(`${outcomes.length} outcomes against ${trueClasses.length} labels`);
  }
  if (costs.length !== gestures.length) {
    throw new RangeError(`${costs.length} costs against ${gestures.length} gestures`);
  }

  let nCorrectLatches = 0;
  let nWrongLatches = 0;
  let nSafeFallbacks = 0;
  let nUncommitted = 0;
  let charged = 0;
  const pairs = new Map<string, { intended: number; committed: number; count: number }>();

  for (let index = 0; index < outcomes.length; index++) {
    const outcome = outcomes[index]!;
    const intended = trueClasses[index]!;
    if (outcome.latchedClass === null) {
      nUncommitted++;
      continue;
    }
    if (outcome.timedOut) {
      nSafeFallbacks++;
      continue;
    }
    if (outcome.latchedClass === intended) {
      nCorrectLatches++;
      continue;
    }
    nWrongLatches++;
    charged += costs[outcome.latchedClass] ?? 0;
    const key = `${intended}:${outcome.latchedClass}`;
    const entry = pairs.get(key) ?? { intended, committed: outcome.latchedClass, count: 0 };
    entry.count++;
    pairs.set(key, entry);
  }

  const confusions = [...pairs.values()]
    .map(({ intended, committed, count }) => {
      const unitCost = costs[committed] ?? 0;
      return {
        intended: gestures[intended] ?? String(intended),
        committed: gestures[committed] ?? String(committed),
        count,
        unitCost,
        totalCost: unitCost * count,
      };
    })
    .sort((a, b) => b.totalCost - a.totalCost || b.count - a.count);

  const n = outcomes.length;
  return {
    nTrials: n,
    nCorrectLatches,
    nWrongLatches,
    nSafeFallbacks,
    nUncommitted,
    errorRate: n > 0 ? nWrongLatches / n : null,
    cwer: n > 0 ? charged / n : null,
    confusions,
  };
}
