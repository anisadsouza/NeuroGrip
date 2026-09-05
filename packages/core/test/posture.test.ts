import { describe, expect, it } from 'vitest';
import { DEFAULT_EVIDENCE_CONFIG } from '../src/evidence.js';
import {
  GESTURE_POSTURES,
  HAND_VIEWBOX,
  HINGE_RADIUS,
  REST_POSTURE,
  actuate,
  actuatorExcursion,
  blendPosture,
  digitDrive,
  digitExcursion,
  jointTravel,
  postureFor,
  projectHand,
  solveHand,
  type HandPosture,
  type Vec3,
} from '../src/posture.js';

/** The simulator's vocabulary. If this list moves, the hand must move with it. */
const VOCABULARY = [
  'rest',
  'fist',
  'open_hand',
  'pinch',
  'point',
  'wrist_flexion',
  'wrist_extension',
  'thumb_up',
  'two_finger',
  'spherical_grip',
] as const;

const norm = (v: Vec3) => Math.hypot(v.x, v.y, v.z);
const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });

/** Every drawn point with the half-width it is drawn at. */
function drawnExtent(
  skeleton: ReturnType<typeof solveHand>,
  projected: ReturnType<typeof projectHand>,
) {
  const items: { point: { x: number; y: number }; radius: number }[] = [];
  projected.digits.forEach((chain, digit) => {
    chain.forEach((point, joint) => {
      items.push({ point, radius: skeleton.radii[digit]![joint]! });
    });
  });
  for (const point of projected.hinge) {
    items.push({ point, radius: HINGE_RADIUS });
  }
  for (const point of [
    ...projected.palm,
    ...projected.forearm,
    ...projected.seams.flat(),
    ...projected.tendons.flat(),
  ]) {
    items.push({ point, radius: 0 });
  }
  return items;
}

/** Fingertip of each digit; the thumb is digit 0. */
function tips(posture: HandPosture): Vec3[] {
  return solveHand(posture).digits.map((chain) => chain[chain.length - 1]!);
}

describe('actuatorExcursion', () => {
  it('holds the hand still until the motion onset is reached', () => {
    expect(actuatorExcursion(0)).toBe(0);
    expect(actuatorExcursion(0.1, 0.2)).toBe(0);
    expect(actuatorExcursion(0.2, 0.2)).toBe(0);
  });

  it('reaches full excursion exactly at commitment', () => {
    expect(actuatorExcursion(1)).toBe(1);
  });

  it('rises monotonically between onset and commitment', () => {
    let previous = -1;
    for (let i = 0; i <= 100; i++) {
      const value = actuatorExcursion(i / 100);
      expect(value).toBeGreaterThanOrEqual(previous);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
      previous = value;
    }
  });

  it('clamps commitment outside the unit interval', () => {
    expect(actuatorExcursion(-3)).toBe(0);
    expect(actuatorExcursion(4)).toBe(1);
  });

  /**
   * The commitment bar draws its onset mark from the evidence config. If the
   * hand used a different one the bar would promise motion at a point where
   * the hand had not started, which is exactly the kind of silent divergence
   * this repository takes seriously elsewhere.
   */
  it('defaults to the same onset the commitment bar draws', () => {
    const onset = DEFAULT_EVIDENCE_CONFIG.motionOnset;
    expect(actuatorExcursion(onset)).toBe(0);
    expect(actuatorExcursion(onset + 1e-6)).toBeGreaterThan(0);
  });
});

describe('the gesture postures', () => {
  it('covers the whole decoder vocabulary', () => {
    for (const name of VOCABULARY) {
      expect(postureFor(name), name).toBeDefined();
    }
  });

  it('reports an unknown gesture rather than silently posing rest', () => {
    expect(postureFor('handstand')).toBeUndefined();
  });

  it('keeps every joint value inside its declared range', () => {
    for (const [name, posture] of Object.entries(GESTURE_POSTURES)) {
      for (const flexion of posture.digits) {
        expect(flexion, name).toBeGreaterThanOrEqual(0);
        expect(flexion, name).toBeLessThanOrEqual(1);
      }
      expect(posture.opposition, name).toBeGreaterThanOrEqual(0);
      expect(posture.opposition, name).toBeLessThanOrEqual(1);
      expect(posture.spread, name).toBeGreaterThanOrEqual(0);
      expect(posture.spread, name).toBeLessThanOrEqual(1);
      expect(posture.wrist, name).toBeGreaterThanOrEqual(-1);
      expect(posture.wrist, name).toBeLessThanOrEqual(1);
    }
  });

  /**
   * The same argument as the simulator's gain-degeneracy test, one layer up:
   * two gestures the decoder can tell apart but the hand renders identically
   * would make the display a worse instrument than the decoder behind it.
   */
  it('renders no two gestures to the same hand', () => {
    const names = Object.keys(GESTURE_POSTURES);
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        const a = tips(GESTURE_POSTURES[names[i]!]!);
        const b = tips(GESTURE_POSTURES[names[j]!]!);
        const separation = Math.max(...a.map((tip, k) => norm(sub(tip, b[k]!))));
        expect(separation, names[i] + ' vs ' + names[j]).toBeGreaterThan(8);
      }
    }
  });
});

describe('blendPosture', () => {
  it('returns the endpoints exactly', () => {
    const fist = GESTURE_POSTURES.fist!;
    expect(blendPosture(REST_POSTURE, fist, 0)).toEqual(REST_POSTURE);
    expect(blendPosture(REST_POSTURE, fist, 1)).toEqual(fist);
  });

  it('clamps the blend fraction', () => {
    const fist = GESTURE_POSTURES.fist!;
    expect(blendPosture(REST_POSTURE, fist, -1)).toEqual(REST_POSTURE);
    expect(blendPosture(REST_POSTURE, fist, 9)).toEqual(fist);
  });

  it('moves the fingertips continuously between the endpoints', () => {
    const fist = GESTURE_POSTURES.fist!;
    const steps = 60;
    let previous = tips(REST_POSTURE)[2]!;
    let total = 0;

    for (let i = 1; i <= steps; i++) {
      const current = tips(blendPosture(REST_POSTURE, fist, i / steps))[2]!;
      const step = norm(sub(current, previous));
      // Continuous: every frame is a small move, and none is a jump.
      expect(step, 'step ' + i).toBeGreaterThan(0);
      expect(step, 'step ' + i).toBeLessThan(15);
      total += step;
      previous = current;
    }

    expect(total).toBeGreaterThan(50);
    expect(previous).toEqual(tips(fist)[2]!);
  });
});

describe('solveHand', () => {
  it('produces only finite coordinates', () => {
    for (const posture of Object.values(GESTURE_POSTURES)) {
      const skeleton = solveHand(posture);
      for (const chain of skeleton.digits) {
        for (const point of chain) {
          expect(Number.isFinite(point.x + point.y + point.z)).toBe(true);
        }
      }
    }
  });

  /** Phalanges are bone. Flexion may not stretch them. */
  it('preserves every segment length under any flexion', () => {
    const reference = solveHand(GESTURE_POSTURES.open_hand!);
    for (const posture of Object.values(GESTURE_POSTURES)) {
      const skeleton = solveHand(posture);
      skeleton.digits.forEach((chain, digit) => {
        for (let i = 1; i < chain.length; i++) {
          const length = norm(sub(chain[i]!, chain[i - 1]!));
          const expected = norm(
            sub(reference.digits[digit]![i]!, reference.digits[digit]![i - 1]!),
          );
          expect(length).toBeCloseTo(expected, 6);
        }
      });
    }
  });

  it('curls the fingertips toward the wrist when the hand closes', () => {
    const open = tips(GESTURE_POSTURES.open_hand!);
    const closed = tips(GESTURE_POSTURES.fist!);
    for (let digit = 1; digit < 5; digit++) {
      expect(norm(closed[digit]!), 'digit ' + digit).toBeLessThan(norm(open[digit]!));
    }
  });

  it('splays the fingers wider open than closed', () => {
    const span = (posture: HandPosture) => {
      const xs = tips(posture)
        .slice(1)
        .map((tip) => tip.x);
      return Math.max(...xs) - Math.min(...xs);
    };
    expect(span(GESTURE_POSTURES.open_hand!)).toBeGreaterThan(span(GESTURE_POSTURES.fist!));
  });

  it('leaves the pointing finger extended past the folded ones', () => {
    const [, index, middle, ring, little] = tips(GESTURE_POSTURES.point!);
    for (const folded of [middle!, ring!, little!]) {
      expect(index!.y).toBeGreaterThan(folded.y);
    }
  });

  it('raises the thumb clear of the folded fingers for a thumbs-up', () => {
    const [thumb, ...fingers] = tips(GESTURE_POSTURES.thumb_up!);
    for (const finger of fingers) {
      expect(thumb!.y).toBeGreaterThan(finger.y);
    }
  });

  it('carries the fingertips palmar when the wrist flexes and dorsal when it extends', () => {
    const flexed = GESTURE_POSTURES.wrist_flexion!;
    const extended = GESTURE_POSTURES.wrist_extension!;
    expect(tips(flexed)[2]!.z).toBeGreaterThan(tips({ ...flexed, wrist: 0 })[2]!.z);
    expect(tips(extended)[2]!.z).toBeLessThan(tips({ ...extended, wrist: 0 })[2]!.z);
  });

  /**
   * The forearm is the reference the wrist angle is read against. If it moved
   * with the hand there would be no angle to see, only a drawing that slid
   * around the panel.
   */
  it('holds the forearm still whatever the hand does', () => {
    const reference = solveHand(REST_POSTURE).forearm;
    for (const posture of Object.values(GESTURE_POSTURES)) {
      expect(solveHand(posture).forearm).toEqual(reference);
    }
    // Its distal edge is the wrist, so it meets the origin the hand turns
    // about rather than floating below the palm.
    const distalEdge = [...reference].sort((a, b) => b.y - a.y).slice(0, 2);
    for (const point of distalEdge) {
      expect(Math.abs(point.y)).toBeLessThan(10);
    }
  });
});

describe('projectHand', () => {
  /**
   * The stated requirement is that nothing overflows on any device. A fixed
   * view box means the hand must never leave it at any point in any gesture's
   * travel. An auto-fitting box would rescale the hand every frame, which
   * would read as the hand growing rather than moving.
   */
  it('keeps every gesture inside the view box across its whole travel', () => {
    const right = HAND_VIEWBOX.x + HAND_VIEWBOX.width;
    const bottom = HAND_VIEWBOX.y + HAND_VIEWBOX.height;

    for (const [name, target] of Object.entries(GESTURE_POSTURES)) {
      for (let step = 0; step <= 20; step++) {
        const posture = blendPosture(REST_POSTURE, target, step / 20);
        const skeleton = solveHand(posture);
        const projected = projectHand(skeleton);

        // Joints are drawn as discs, so the drawn extent is the centre plus
        // the half-width. Testing centres alone would pass a hand whose
        // knuckles hang over the edge.
        for (const { point, radius } of drawnExtent(skeleton, projected)) {
          expect(point.x - radius, name + ' x').toBeGreaterThanOrEqual(HAND_VIEWBOX.x);
          expect(point.x + radius, name + ' x').toBeLessThanOrEqual(right);
          expect(point.y - radius, name + ' y').toBeGreaterThanOrEqual(HAND_VIEWBOX.y);
          expect(point.y + radius, name + ' y').toBeLessThanOrEqual(bottom);
        }
      }
    }
  });

  /**
   * The other half of the containment test. On its own, containment can be
   * satisfied by enlarging the box until nothing touches it -- the same move
   * as widening a tolerance until a failure passes. This pins the box to the
   * drawing it is supposed to bound.
   */
  it('sizes the view box to the drawing rather than padding it out', () => {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (const target of Object.values(GESTURE_POSTURES)) {
      for (let step = 0; step <= 20; step++) {
        const skeleton = solveHand(blendPosture(REST_POSTURE, target, step / 20));
        for (const { point, radius } of drawnExtent(skeleton, projectHand(skeleton))) {
          minX = Math.min(minX, point.x - radius);
          maxX = Math.max(maxX, point.x + radius);
          minY = Math.min(minY, point.y - radius);
          maxY = Math.max(maxY, point.y + radius);
        }
      }
    }

    // Enough room for the stroke and a hairline of air, and no more.
    const slack = 12;
    expect(minX - HAND_VIEWBOX.x).toBeLessThan(slack);
    expect(HAND_VIEWBOX.x + HAND_VIEWBOX.width - maxX).toBeLessThan(slack);
    expect(minY - HAND_VIEWBOX.y).toBeLessThan(slack);
    expect(HAND_VIEWBOX.y + HAND_VIEWBOX.height - maxY).toBeLessThan(slack);
  });

  /**
   * The depth ordering is what makes a closing fist read as a solid hand
   * rather than a tangle of crossing outlines, so it has to be the right way
   * round: a finger that has curled toward the viewer must sort in front of
   * one that has not, and in front of the palm it is closing over.
   */
  it('puts a curled finger in front of an extended one and in front of the palm', () => {
    const closed = projectHand(solveHand(GESTURE_POSTURES.fist!));
    const open = projectHand(solveHand(GESTURE_POSTURES.open_hand!));

    for (let digit = 1; digit < 5; digit++) {
      expect(closed.digitDepth[digit]!, 'digit ' + digit).toBeGreaterThan(
        open.digitDepth[digit]!,
      );
      expect(closed.digitDepth[digit]!, 'digit ' + digit).toBeGreaterThan(closed.palmDepth);
    }
  });

  it('brings the thumb forward as it opposes', () => {
    const abducted = projectHand(solveHand(GESTURE_POSTURES.thumb_up!));
    const opposed = projectHand(solveHand(GESTURE_POSTURES.pinch!));
    expect(opposed.digitDepth[0]!).toBeGreaterThan(abducted.digitDepth[0]!);
  });

  it('projects a distal point above a proximal one in screen space', () => {
    const projected = projectHand(solveHand(GESTURE_POSTURES.open_hand!));
    const middle = projected.digits[2]!;
    expect(middle[middle.length - 1]!.y).toBeLessThan(middle[0]!.y);
  });
});

describe('the underactuated transmission', () => {
  /**
   * The mechanism this models: one tendon, the knuckle takes the load first,
   * the joints beyond it follow as the tendon takes up. If all three moved
   * together there would be no mechanism, only an interpolation.
   */
  it('moves the knuckle before the joints beyond it', () => {
    const early = 0.2;
    expect(jointTravel(0, early)).toBeGreaterThan(jointTravel(1, early));
    expect(jointTravel(1, early)).toBeGreaterThanOrEqual(jointTravel(2, early));
    expect(jointTravel(2, early)).toBe(0);
  });

  it('brings every joint to full travel by full command', () => {
    for (let joint = 0; joint < 3; joint++) {
      expect(jointTravel(joint, 1), 'joint ' + joint).toBeCloseTo(1, 9);
      expect(jointTravel(joint, 0), 'joint ' + joint).toBeCloseTo(0, 9);
    }
  });

  it('advances every joint monotonically and stays in range', () => {
    for (let joint = 0; joint < 3; joint++) {
      let previous = -1;
      for (let i = 0; i <= 100; i++) {
        const travel = jointTravel(joint, i / 100);
        expect(travel, 'joint ' + joint).toBeGreaterThanOrEqual(previous);
        expect(travel).toBeGreaterThanOrEqual(0);
        expect(travel).toBeLessThanOrEqual(1);
        previous = travel;
      }
    }
  });

  it('clamps a command outside the unit interval', () => {
    expect(jointTravel(0, -1)).toBe(0);
    expect(jointTravel(2, 5)).toBe(1);
  });

  /**
   * The coupling has to survive the trip through the kinematics, not just hold
   * in `jointTravel`. Measured as the turn at each joint of the drawn hand.
   */
  it('turns the knuckle further than the joints beyond it while closing', () => {
    const turn = (a: Vec3, b: Vec3, c: Vec3) => {
      const u = sub(b, a);
      const v = sub(c, b);
      const cosine = (u.x * v.x + u.y * v.y + u.z * v.z) / (norm(u) * norm(v) || 1);
      return Math.acos(Math.max(-1, Math.min(1, cosine)));
    };

    for (let step = 1; step <= 6; step++) {
      const chain = solveHand(
        actuate(REST_POSTURE, GESTURE_POSTURES.fist!, step / 10),
      ).digits[2]!;
      const knuckle = turn(
        { x: chain[0]!.x, y: chain[0]!.y - 1, z: chain[0]!.z },
        chain[0]!,
        chain[1]!,
      );
      const middle = turn(chain[0]!, chain[1]!, chain[2]!);
      const fingertip = turn(chain[1]!, chain[2]!, chain[3]!);
      expect(knuckle, 'step ' + step).toBeGreaterThan(middle - 1e-9);
      expect(middle, 'step ' + step).toBeGreaterThan(fingertip - 1e-9);
    }
  });
});

describe('the closing order', () => {
  it('starts the little finger before the index', () => {
    expect(digitExcursion(4, 0.15)).toBeGreaterThan(digitExcursion(1, 0.15));
  });

  it('lands every digit exactly at full travel', () => {
    for (let digit = 0; digit < 5; digit++) {
      expect(digitExcursion(digit, 1), 'digit ' + digit).toBeCloseTo(1, 9);
      expect(digitExcursion(digit, 0), 'digit ' + digit).toBe(0);
    }
  });

  it('keeps the endpoints of an actuated move exact', () => {
    const fist = GESTURE_POSTURES.fist!;
    expect(actuate(REST_POSTURE, fist, 0)).toEqual(REST_POSTURE);
    expect(actuate(REST_POSTURE, fist, 1)).toEqual(fist);
  });

  it('produces a hand mid-move that no single blend fraction produces', () => {
    const fist = GESTURE_POSTURES.fist!;
    const sequenced = actuate(REST_POSTURE, fist, 0.35);
    const spread =
      Math.max(...sequenced.digits.slice(1)) - Math.min(...sequenced.digits.slice(1));
    // A uniform blend gives all four fingers all-but-identical flexion; the
    // closing order has to open a real gap between them.
    const uniform = blendPosture(REST_POSTURE, fist, 0.35);
    const uniformSpread =
      Math.max(...uniform.digits.slice(1)) - Math.min(...uniform.digits.slice(1));
    expect(spread).toBeGreaterThan(uniformSpread + 0.05);
  });

  it('stays inside the view box with the closing order applied', () => {
    const right = HAND_VIEWBOX.x + HAND_VIEWBOX.width;
    const bottom = HAND_VIEWBOX.y + HAND_VIEWBOX.height;
    for (const [name, target] of Object.entries(GESTURE_POSTURES)) {
      for (let step = 0; step <= 20; step++) {
        const skeleton = solveHand(actuate(REST_POSTURE, target, step / 20));
        for (const { point, radius } of drawnExtent(skeleton, projectHand(skeleton))) {
          expect(point.x - radius, name).toBeGreaterThanOrEqual(HAND_VIEWBOX.x);
          expect(point.x + radius, name).toBeLessThanOrEqual(right);
          expect(point.y - radius, name).toBeGreaterThanOrEqual(HAND_VIEWBOX.y);
          expect(point.y + radius, name).toBeLessThanOrEqual(bottom);
        }
      }
    }
  });
});

describe('digitDrive', () => {
  it('is positive for digits that are closing and negative for those opening', () => {
    const closing = digitDrive(GESTURE_POSTURES.fist!);
    for (const value of closing) expect(value).toBeGreaterThan(0);

    const opening = digitDrive(GESTURE_POSTURES.open_hand!);
    for (const value of opening) expect(value).toBeLessThan(0);
  });

  /** `point` is the case that makes per-digit drive worth showing at all. */
  it('separates the extending finger from the folding ones in a point', () => {
    const [, index, middle] = digitDrive(GESTURE_POSTURES.point!);
    expect(index!).toBeLessThan(0);
    expect(middle!).toBeGreaterThan(0);
  });

  it('is zero everywhere at rest', () => {
    for (const value of digitDrive(REST_POSTURE)) expect(value).toBe(0);
  });
});
