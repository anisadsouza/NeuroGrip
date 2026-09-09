/**
 * The risk vector is the only thing standing between a decoder's guess and a
 * power grip closing. Its contents are a safety judgement recorded elsewhere;
 * what this file gates is that the judgement survives the trip into the
 * accumulator intact and in the right order.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  COMMIT_COST,
  UNKNOWN_COMMIT_COST,
  labelsFor,
  risksFor,
} from '../../src/decode/commitCost.js';

const decoder = JSON.parse(
  readFileSync(fileURLToPath(new URL('../../../../artifacts/decoder.json', import.meta.url)), 'utf-8'),
) as { gestures: string[] };

describe('risksFor', () => {
  /**
   * The accumulator indexes this vector by class index. A vector built in any
   * other order silently hands the fist's boundary to whichever gesture sits
   * at that position, and the only symptom is a grip committing early.
   */
  it('follows the order it is given rather than the table ordering', () => {
    const forward = risksFor(['fist', 'open_hand']);
    const reversed = risksFor(['open_hand', 'fist']);

    expect(forward).toEqual([1.0, 0.1]);
    expect(reversed).toEqual([0.1, 1.0]);
  });

  /**
   * A gesture the table has never heard of is not known to be safe. Costing it
   * at zero would let a class introduced by a retrained model commit on the
   * first hop with no boundary at all.
   */
  it('costs an unknown gesture at the middling default rather than at nothing', () => {
    expect(risksFor(['a_gesture_nobody_has_named'])).toEqual([UNKNOWN_COMMIT_COST]);
    expect(UNKNOWN_COMMIT_COST).toBeGreaterThan(0);
  });

  it('gives every gesture the committed decoder emits an explicit cost', () => {
    for (const name of decoder.gestures) {
      expect(COMMIT_COST[name], name).toBeDefined();
    }
  });

  /**
   * The ordering claim is worth nothing if the class list itself drifts. This
   * pins the vector to the model actually shipped.
   */
  it('produces one risk per class the decoder emits', () => {
    expect(risksFor(decoder.gestures).length).toBe(decoder.gestures.length);
  });

  it('holds the closing grips above the opening hand rather than treating them alike', () => {
    expect(COMMIT_COST.fist!).toBeGreaterThan(COMMIT_COST.open_hand!);
    expect(COMMIT_COST.spherical_grip!).toBeGreaterThan(COMMIT_COST.point!);
    expect(COMMIT_COST.rest).toBe(0);
  });
});

describe('labelsFor', () => {
  it('gives every gesture the decoder emits wearer-facing copy rather than an identifier', () => {
    for (const label of labelsFor(decoder.gestures)) {
      expect(label, label).not.toMatch(/_/);
    }
  });

  it('falls back to the raw name rather than rendering undefined', () => {
    expect(labelsFor(['unnamed_gesture'])).toEqual(['unnamed_gesture']);
  });
});
