import { describe, expect, it } from 'vitest';
import { decisionOutputFor } from '../../src/decode/decisionOutput.js';

const gestures = ['rest', 'fist', 'pinch'] as const;

describe('decisionOutputFor', () => {
  it('keeps the latched hand pose on the committed class if the leader changes', () => {
    expect(
      decisionOutputFor({
        gestures,
        leader: 2,
        latched: true,
        latchedClass: 1,
        timedOut: false,
        commitment: 1,
      }),
    ).toEqual({ index: 1, gesture: 'fist', handCommitment: 1 });
  });

  it('shows the leading class while evidence is still reversible', () => {
    expect(
      decisionOutputFor({
        gestures,
        leader: 2,
        latched: false,
        latchedClass: null,
        timedOut: false,
        commitment: 0.6,
      }),
    ).toEqual({ index: 2, gesture: 'pinch', handCommitment: 0.6 });
  });

  it('returns the hand to rest after the evidence times out', () => {
    expect(
      decisionOutputFor({
        gestures,
        leader: 2,
        latched: false,
        latchedClass: null,
        timedOut: true,
        commitment: 1,
      }),
    ).toEqual({ index: 0, gesture: 'rest', handCommitment: 0 });
  });

  it('does not report travel when the committed output is rest', () => {
    expect(
      decisionOutputFor({
        gestures,
        leader: 2,
        latched: true,
        latchedClass: 0,
        timedOut: false,
        commitment: 1,
      }),
    ).toEqual({ index: 0, gesture: 'rest', handCommitment: 0 });
  });
});
