/**
 * Path geometry for drawing a solved hand.
 *
 * This lives beside the kinematics rather than in the web application because
 * it is trigonometry, not markup: the tangent construction below is the kind
 * of thing that is subtly wrong in a way you notice only as a slightly
 * malformed knuckle, and it deserves the same tests as everything else here.
 * SVG path data is just a serialisation of the result.
 *
 * Two constructions:
 *
 *   `capsulePath` -- the outline of the convex hull of two circles. Each
 *   phalanx is drawn as one of these, tapering from a wider proximal joint to
 *   a narrower distal one, which is what turns a stick figure into a limb with
 *   volume. Drawn distal-over-proximal with an opaque fill, the seam where two
 *   capsules meet reads as the joint.
 *
 *   `smoothClosedPath` / `smoothOpenPath` -- a Catmull-Rom spline through a
 *   set of points, emitted as cubic Beziers. The palm is a soft shape with
 *   knuckle scallops and a thenar bulge; a polygon through the same points
 *   reads as a machine part.
 */

// The 2-D point type is declared once, beside the kinematics that produce it.
export type { Point } from './posture.js';
import type { Point } from './posture.js';

const round = (value: number) => (Math.round(value * 100) / 100).toString();

/**
 * A full circle as path data.
 *
 * Exported because the renderer draws joint bearings, axle pins and contact
 * pads as circles, and packing several into one path lets a whole digit take a
 * single fill rather than compounding one overlay per part.
 */
export function circlePath(centre: Point, r: number): string {
  const { x, y } = centre;
  return (
    `M${round(x - r)} ${round(y)}` +
    `A${round(r)} ${round(r)} 0 1 0 ${round(x + r)} ${round(y)}` +
    `A${round(r)} ${round(r)} 0 1 0 ${round(x - r)} ${round(y)}Z`
  );
}

/**
 * The outline of the convex hull of two circles.
 *
 * The two straight edges are the external tangents, meeting the circles at
 * angle `alpha = acos((r1 - r2) / L)` either side of the centre line. Each end
 * is then capped by the arc on its own far side, so the distal cap spans
 * `2 * alpha` and the proximal one the remainder.
 */
export function capsulePath(p1: Point, r1: number, p2: Point, r2: number): string {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const length = Math.hypot(dx, dy);

  // One circle contains the other: the hull is just the larger of them.
  if (!(length > 1e-9) || length <= Math.abs(r1 - r2)) {
    return r1 >= r2 ? circlePath(p1, r1) : circlePath(p2, r2);
  }

  const theta = Math.atan2(dy, dx);
  const alpha = Math.acos(Math.max(-1, Math.min(1, (r1 - r2) / length)));

  const at = (centre: Point, r: number, angle: number): Point => ({
    x: centre.x + r * Math.cos(angle),
    y: centre.y + r * Math.sin(angle),
  });

  const a1 = at(p1, r1, theta + alpha);
  const a2 = at(p2, r2, theta + alpha);
  const b1 = at(p1, r1, theta - alpha);
  const b2 = at(p2, r2, theta - alpha);

  // Both arcs run in the direction of decreasing angle, which is
  // anticlockwise on screen, so the whole outline winds consistently.
  const distalLarge = 2 * alpha > Math.PI ? 1 : 0;
  const proximalLarge = 2 * Math.PI - 2 * alpha > Math.PI ? 1 : 0;

  return (
    `M${round(a1.x)} ${round(a1.y)}` +
    `L${round(a2.x)} ${round(a2.y)}` +
    `A${round(r2)} ${round(r2)} 0 ${distalLarge} 0 ${round(b2.x)} ${round(b2.y)}` +
    `L${round(b1.x)} ${round(b1.y)}` +
    `A${round(r1)} ${round(r1)} 0 ${proximalLarge} 0 ${round(a1.x)} ${round(a1.y)}Z`
  );
}

/**
 * Straight segments through every point.
 *
 * For anything whose real edges are straight -- the forearm shaft, the ghost's
 * joint chains -- smoothing is not neutral: a Catmull-Rom through four corners
 * bows the sides inward and narrows the shape by a visible margin.
 */
export function polylinePath(points: readonly Point[], close = false): string {
  if (points.length === 0) return '';
  const d =
    `M${round(points[0]!.x)} ${round(points[0]!.y)}` +
    points.slice(1).map((p) => `L${round(p.x)} ${round(p.y)}`).join('');
  return close ? `${d}Z` : d;
}

/** Catmull-Rom control points for the segment from `p1` to `p2`. */
function segment(p0: Point, p1: Point, p2: Point, p3: Point): string {
  const c1 = { x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6 };
  const c2 = { x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6 };
  return (
    `C${round(c1.x)} ${round(c1.y)} ${round(c2.x)} ${round(c2.y)} ` +
    `${round(p2.x)} ${round(p2.y)}`
  );
}

/** A closed Catmull-Rom spline through every point, as cubic Beziers. */
export function smoothClosedPath(points: readonly Point[]): string {
  const n = points.length;
  if (n === 0) return '';
  if (n < 3) {
    return (
      `M${round(points[0]!.x)} ${round(points[0]!.y)}` +
      points.slice(1).map((p) => `L${round(p.x)} ${round(p.y)}`).join('') +
      'Z'
    );
  }

  const at = (i: number) => points[((i % n) + n) % n]!;
  let d = `M${round(points[0]!.x)} ${round(points[0]!.y)}`;
  for (let i = 0; i < n; i++) {
    d += segment(at(i - 1), at(i), at(i + 1), at(i + 2));
  }
  return `${d}Z`;
}

/** An open Catmull-Rom spline, with the end tangents clamped. */
export function smoothOpenPath(points: readonly Point[]): string {
  const n = points.length;
  if (n === 0) return '';
  if (n < 3) {
    return (
      `M${round(points[0]!.x)} ${round(points[0]!.y)}` +
      points.slice(1).map((p) => `L${round(p.x)} ${round(p.y)}`).join('')
    );
  }

  const at = (i: number) => points[Math.max(0, Math.min(n - 1, i))]!;
  let d = `M${round(points[0]!.x)} ${round(points[0]!.y)}`;
  for (let i = 0; i < n - 1; i++) {
    d += segment(at(i - 1), at(i), at(i + 1), at(i + 2));
  }
  return d;
}
