import { describe, expect, it } from 'vitest';
import {
  capsulePath,
  polylinePath,
  smoothClosedPath,
  smoothOpenPath,
  type Point,
} from '../src/handPaths.js';

/** Every coordinate in a path, in order. */
function numbers(d: string): number[] {
  return (d.match(/-?\d+(?:\.\d+)?/g) ?? []).map(Number);
}

const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);

describe('capsulePath', () => {
  const p1: Point = { x: 0, y: 0 };
  const p2: Point = { x: 40, y: 0 };

  it('starts and ends on the circles it spans', () => {
    const d = capsulePath(p1, 10, p2, 7);
    const [x0, y0, x1, y1] = numbers(d) as [number, number, number, number];
    // The move-to lies on the proximal circle, the first line-to on the distal.
    expect(distance({ x: x0, y: y0 }, p1)).toBeCloseTo(10, 1);
    expect(distance({ x: x1, y: y1 }, p2)).toBeCloseTo(7, 1);
  });

  it('closes the outline', () => {
    expect(capsulePath(p1, 10, p2, 7).endsWith('Z')).toBe(true);
  });

  it('gives the wider joint the larger cap', () => {
    // Nearly equal radii over a long bone: each cap is close to a half turn,
    // the proximal one just over and the distal one just under.
    const d = capsulePath(p1, 10, p2, 9);
    const flags = [...d.matchAll(/A[^A]*?0 (\d) 0/g)].map((m) => Number(m[1]));
    expect(flags).toEqual([0, 1]);
  });

  it('falls back to a single circle when one joint swallows the other', () => {
    const d = capsulePath(p1, 20, { x: 2, y: 0 }, 4);
    expect(numbers(d).every(Number.isFinite)).toBe(true);
    expect(d).toContain('A20 20');
  });

  it('survives coincident joints without producing NaN', () => {
    const d = capsulePath(p1, 8, { x: 0, y: 0 }, 8);
    expect(d).not.toContain('NaN');
    expect(numbers(d).every(Number.isFinite)).toBe(true);
  });

  it('produces only finite coordinates for realistic phalanges', () => {
    for (const r2 of [2, 5, 8, 9.9]) {
      for (const angle of [0, 1, 2, 3, 4, 5, 6]) {
        const end = { x: 45 * Math.cos(angle), y: 45 * Math.sin(angle) };
        const d = capsulePath(p1, 10, end, r2);
        expect(numbers(d).every(Number.isFinite), `r2=${r2} angle=${angle}`).toBe(true);
      }
    }
  });
});

describe('smoothClosedPath', () => {
  const square: Point[] = [
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 10, y: 10 },
    { x: 0, y: 10 },
  ];

  it('emits one cubic per point and closes', () => {
    const d = smoothClosedPath(square);
    expect(d.startsWith('M0 0')).toBe(true);
    expect(d.endsWith('Z')).toBe(true);
    expect(d.split('C').length - 1).toBe(square.length);
  });

  it('passes through every point it was given', () => {
    const d = smoothClosedPath(square);
    for (const point of square.slice(1)) {
      expect(d).toContain(`${point.x} ${point.y}`);
    }
  });

  it('handles degenerate input without throwing', () => {
    expect(smoothClosedPath([])).toBe('');
    expect(numbers(smoothClosedPath([{ x: 1, y: 2 }])).every(Number.isFinite)).toBe(true);
    expect(numbers(smoothClosedPath(square)).every(Number.isFinite)).toBe(true);
  });
});

describe('smoothOpenPath', () => {
  const arc: Point[] = [
    { x: 0, y: 0 },
    { x: 5, y: 8 },
    { x: 14, y: 10 },
    { x: 22, y: 6 },
  ];

  it('emits one cubic per gap and does not close', () => {
    const d = smoothOpenPath(arc);
    expect(d.split('C').length - 1).toBe(arc.length - 1);
    expect(d.endsWith('Z')).toBe(false);
  });

  it('produces only finite coordinates', () => {
    expect(numbers(smoothOpenPath(arc)).every(Number.isFinite)).toBe(true);
  });
});

describe('polylinePath', () => {
  const corners: Point[] = [
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 12, y: 20 },
    { x: -2, y: 20 },
  ];

  it('emits straight segments and nothing else', () => {
    const d = polylinePath(corners);
    expect(d).toBe('M0 0L10 0L12 20L-2 20');
    expect(d).not.toContain('C');
  });

  it('closes only when asked', () => {
    expect(polylinePath(corners, true).endsWith('Z')).toBe(true);
    expect(polylinePath(corners).endsWith('Z')).toBe(false);
  });

  /** The reason this exists: smoothing a straight-sided shaft narrows it. */
  it('keeps a straight-sided shape at its stated width', () => {
    const straight = polylinePath(corners, true);
    expect(straight).toContain('10 0');
    expect(smoothClosedPath(corners)).not.toContain('L');
  });

  it('handles an empty list', () => {
    expect(polylinePath([])).toBe('');
  });
});
