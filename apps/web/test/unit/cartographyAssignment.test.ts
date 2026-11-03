import { describe, expect, it } from 'vitest';
import { COMMIT_COST } from '../../src/decode/commitCost.js';
import { assignFunctionsByRisk } from '../../src/cartography/assignment.js';

describe('assignFunctionsByRisk', () => {
  it('uses the Hungarian assignment to reserve high-risk functions for reliable prototypes', () => {
    const assignments = assignFunctionsByRisk(
      ['reliable', 'uncertain'],
      [0.05, 0.8],
      ['fist', 'open_hand'],
    );

    expect(assignments.map(({ function: assigned }) => assigned)).toEqual([
      'fist',
      'open_hand',
    ]);
    expect(assignments[0]!.expectedCommitCost).toBeCloseTo(0.05);
    expect(assignments[1]!.expectedCommitCost).toBeCloseTo(0.08);
  });

  it('matches each prototype once and supports more functions than prototypes', () => {
    const assignments = assignFunctionsByRisk(
      ['p1', 'p2'],
      [0.1, 0.3],
      ['fist', 'open_hand', 'pinch'],
    );

    expect(new Set(assignments.map(({ function: assigned }) => assigned)).size).toBe(2);
    expect(assignments.reduce((sum, item) => sum + item.expectedCommitCost, 0))
      .toBeLessThanOrEqual(
        0.1 * COMMIT_COST['fist']! + 0.3 * COMMIT_COST['open_hand']!,
      );
  });

  it('rejects invalid risks and impossible assignment shapes', () => {
    expect(() => assignFunctionsByRisk(['p'], [1.1])).toThrow(RangeError);
    expect(() => assignFunctionsByRisk(['p1', 'p2'], [0.2])).toThrow(RangeError);
    expect(() => assignFunctionsByRisk(['p1', 'p2'], [0.2, 0.3], ['fist']))
      .toThrow(RangeError);
  });
});
