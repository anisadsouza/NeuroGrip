/**
 * The actuator command, and the hand kinematics that render it.
 *
 * The evidence accumulator produces a commitment fraction. This module turns
 * that fraction into a hand posture, and the posture into coordinates. It is
 * the second half of progressive actuation: without it, "the hand begins
 * moving reversibly as evidence accrues" is a claim in a document rather than
 * something a wearer can watch happen.
 *
 * Three things live here, all pure and all testable:
 *
 *   1. `actuatorExcursion` -- commitment to travel. Zero below the motion
 *      onset, one at commitment, linear between. The onset defaults to the
 *      same constant the commitment bar draws its mark from, so the bar cannot
 *      promise motion the hand has not begun.
 *
 *   2. `GESTURE_POSTURES` -- a target joint configuration per gesture, blended
 *      from rest by the excursion.
 *
 *   3. `solveHand` / `projectHand` -- forward kinematics in three dimensions,
 *      then a fixed oblique projection.
 *
 * On the third: it would be simpler to draw the hand flat and fake the curl.
 * It would also be wrong. Finger flexion and wrist flexion are both sagittal
 * motions, invisible in a true palmar view, so a flat drawing has to either
 * hide the dominant motion in this vocabulary or misrepresent it as sideways
 * deviation. Solving in 3-D and projecting obliquely costs about sixty lines
 * of trigonometry and shows both the curl and the spread honestly.
 *
 * Frame: origin at the wrist, +x toward the little finger, +y toward the
 * fingertips, +z out of the palm. Angles in radians internally, declared in
 * degrees. Lengths in millimetres, roughly anatomical, so the proportions are
 * a real hand rather than a pleasing one.
 */

import { DEFAULT_EVIDENCE_CONFIG } from './evidence.js';

export interface Vec3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

export interface Point {
  readonly x: number;
  readonly y: number;
}

/**
 * A hand posture.
 *
 * Five flexion values plus two whole-hand degrees of freedom. This is far
 * fewer than a hand has, and deliberately so: it is the set a myoelectric
 * prosthesis can actually command, which is what the display should show.
 */
export interface HandPosture {
  /** Flexion per digit in [0, 1], thumb first, then index to little. */
  readonly digits: readonly [number, number, number, number, number];
  /** Thumb rotation across the palm: 0 abducted, 1 fully opposed. */
  readonly opposition: number;
  /** Finger splay: 0 adducted, 1 splayed. */
  readonly spread: number;
  /** Wrist angle: -1 fully extended, 0 neutral, +1 fully flexed. */
  readonly wrist: number;
}

const DEG = Math.PI / 180;

// ---- Anatomy ------------------------------------------------------------
// Segment lengths and joint ranges. These are the model; everything else is
// consequence.

/**
 * Metacarpophalangeal joints, radial to ulnar; the thumb's carpometacarpal
 * joint comes first, low on the radial edge.
 *
 * Proportioned to a measured adult hand rather than to what looks tidy. Palm
 * breadth across the knuckles is close to palm length, and an earlier version
 * that halved the breadth drew something that read as a stick with fingers.
 */
const MCP: readonly Vec3[] = [
  { x: -30, y: 22, z: 6 },
  { x: -28, y: 92, z: 0 },
  { x: -9.5, y: 97, z: 0 },
  { x: 9.5, y: 94, z: 0 },
  { x: 27, y: 85, z: 0 },
];

/** Phalanx lengths, proximal to distal. The thumb has one fewer bone but the
 *  metacarpal moves with it, so all five digits are three-segment chains. */
const PHALANGES: readonly (readonly number[])[] = [
  [38, 28, 22], // thumb
  [45, 25, 20], // index
  [50, 30, 22], // middle
  [46, 27, 21], // ring
  [36, 21, 18], // little
];

/**
 * Link half-widths at each joint, base to tip.
 *
 * These set the digit's silhouette. An earlier version drew the joints as
 * bosses wider than the links between them, on the theory that a linkage has
 * visible bearings -- it read as a string of beads and lost the shape of a
 * hand entirely. The joints are now inscribed instead: the silhouette is the
 * link, and the pivot shows as a ring drawn inside it at `JOINT_RING`.
 */
const DIGIT_RADII: readonly (readonly number[])[] = [
  [10, 8.8, 8, 6.8], // thumb
  [8.4, 7.6, 6.8, 5.9], // index
  [8.8, 8, 7.2, 6.3], // middle
  [8.4, 7.6, 6.8, 5.9], // ring
  [7.6, 6.8, 5.9, 5.2], // little
];

/** Shaft width between joints, as a fraction of the joint half-width. Just
 *  under one, so each joint reads as a slight swelling rather than a boss. */
export const LINK_WAIST = 0.88;

/** Pivot ring radius, inscribed in the link. */
export const JOINT_RING = 0.66;

/** Axle pin radius, as a fraction of the link half-width. */
export const AXLE_RADIUS = 0.22;

/** Contact pad radius at a fingertip, as a fraction of the tip half-width. */
export const PAD_RADIUS = 0.52;

/** How far the tendon stands off the joint centres, in bearing radii. */
const TENDON_OFFSET = 0.72;

/** Half-width of the wrist hinge bosses. These are real bosses -- the hinge
 *  is the one joint whose hardware is outside the hand's silhouette. */
export const HINGE_RADIUS = 7.5;

/** Joint travel at full flexion, in degrees. */
const FLEXION_RANGE: readonly (readonly number[])[] = [
  [40, 55, 78], // thumb: carpometacarpal, metacarpophalangeal, interphalangeal
  [88, 100, 62],
  [88, 100, 62],
  [88, 100, 62],
  [88, 100, 62],
];

/**
 * When each joint moves, as a fraction of the digit's commanded flexion.
 *
 * A tendon-driven prosthetic finger is underactuated: one motor pulls one
 * tendon, the metacarpophalangeal joint takes the load first, and the two
 * joints beyond it follow through passive coupling as the tendon continues to
 * take up. So the joints do not move together -- the knuckle leads, the middle
 * joint follows, the fingertip curls last. Interpolating all three in lockstep
 * is the tell of an animated drawing rather than a mechanism.
 *
 * Each entry is the [start, end] of that joint's travel within the digit's
 * command range. They overlap, because a real transmission is compliant rather
 * than sequenced by a controller.
 */
const JOINT_COUPLING: readonly (readonly [number, number])[] = [
  [0, 0.62], // metacarpophalangeal: driven, leads
  [0.14, 0.88], // proximal interphalangeal: coupled, follows
  [0.34, 1], // distal interphalangeal: coupled, last to take up
];

/**
 * Closing order across the digits, as a fraction of the digit's travel that is
 * spent waiting.
 *
 * A hand closing on an object does not close as a slab: the little finger
 * arrives first and the index last, which is what lets a grasp conform to
 * something before it is gripped. Prosthetic hands are programmed the same
 * way. The offsets are small -- a fifth of the travel end to end -- because
 * this is a legibility cue, not a stagger anyone should have to wait through.
 */
const DIGIT_LEAD: readonly number[] = [0.2, 0.18, 0.12, 0.06, 0];

/** Smoothstep, so a joint enters and leaves its travel without a corner. */
function ease(t: number): number {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}

/**
 * How far one joint has travelled, given the digit's commanded flexion.
 *
 * Returns a fraction of that joint's own range, so `solveHand` multiplies it
 * by the anatomical travel and nothing else has to know about the coupling.
 */
export function jointTravel(joint: number, digitFlexion: number): number {
  const window = JOINT_COUPLING[joint];
  if (!window) return clamp01(digitFlexion);
  const [start, end] = window;
  if (end <= start) return clamp01(digitFlexion) >= end ? 1 : 0;
  return ease((clamp01(digitFlexion) - start) / (end - start));
}

/**
 * Excursion for one digit, given the hand's overall travel.
 *
 * Applies the closing order: a digit that leads starts sooner and reaches its
 * posture sooner, and every digit still arrives exactly at full excursion.
 */
export function digitExcursion(digit: number, excursion: number): number {
  const lead = DIGIT_LEAD[digit] ?? 0;
  if (lead <= 0) return clamp01(excursion);
  return clamp01((clamp01(excursion) - lead) / (1 - lead));
}

/** Finger abduction at full spread, in degrees; positive is toward the little
 *  finger. The middle finger barely moves, which is why it is the axis the
 *  others splay around. */
const ABDUCTION = [-15, -4, 7, 18];

/** The thumb swings from radially abducted toward the palm as it opposes. */
const THUMB_ABDUCTION_OPEN = -34;
const THUMB_ABDUCTION_OPPOSED = -20;
const THUMB_OPPOSITION_RANGE = 62;

/**
 * Wrist travel at |wrist| = 1, in degrees.
 *
 * Less than a wrist can do. The camera looks nearly along the axis the wrist
 * turns about, so past about thirty-five degrees of flexion the chassis
 * foreshortens into an edge-on sliver -- an honest orthographic projection of
 * a pose nobody could interpret. The display understates the angle so that it
 * can show it at all, which is the trade a side view would avoid at the cost
 * of losing every finger.
 */
const WRIST_RANGE = 34;

/**
 * The palm chassis, anticlockwise from the radial wrist corner.
 *
 * A machined plate, not a palm: straight runs and chamfered corners, drawn
 * without smoothing. The digit bearings mount along its distal edge and stand
 * proud of it, so the knuckle line is a row of bosses rather than a scalloped
 * outline. The radial side is wider because it houses the thumb actuator,
 * which is where the bulk sits on a real hand too.
 */
const PALM: readonly Vec3[] = [
  { x: -28, y: 4, z: 0 },
  { x: -33, y: 18, z: 0 }, // chamfer
  { x: -35, y: 46, z: 0 }, // thumb actuator housing
  { x: -33, y: 74, z: 0 },
  { x: -30, y: 96, z: 0 },
  { x: -25, y: 101, z: 0 }, // chamfer
  { x: -9.5, y: 104, z: 0 },
  { x: 9.5, y: 101, z: 0 },
  { x: 24, y: 95, z: 0 },
  { x: 31, y: 84, z: 0 }, // chamfer
  { x: 32, y: 40, z: 0 },
  { x: 29, y: 12, z: 0 }, // chamfer
  { x: 28, y: 4, z: 0 },
];

/**
 * Panel seams on the chassis.
 *
 * Three straight lines: the knuckle rail linking the four finger mounts, the
 * split between the finger block and the thumb housing, and the transverse
 * seam where the plate meets the wrist assembly. They are the mechanical
 * equivalent of the palmar creases they replaced -- a small amount of interior
 * structure that stops a filled outline reading as a paddle.
 */
const SEAMS: readonly (readonly Vec3[])[] = [
  [
    { x: -28, y: 92, z: 0 },
    { x: -9.5, y: 97, z: 0 },
    { x: 9.5, y: 94, z: 0 },
    { x: 27, y: 85, z: 0 },
  ],
  [
    { x: -24, y: 8, z: 0 },
    { x: -22, y: 44, z: 0 },
    { x: -26, y: 78, z: 0 },
  ],
  [
    { x: -27, y: 26, z: 0 },
    { x: 29, y: 26, z: 0 },
  ],
];

/**
 * The thumb actuator housing, as a closed panel on the radial side.
 *
 * The thumb is the one digit whose drive cannot run straight up the chassis,
 * so on a real hand it gets its own motor block. Drawing it is what makes the
 * plate read as an assembly rather than a paddle.
 */
const HOUSING: readonly Vec3[] = [
  { x: -31, y: 22, z: 0 },
  { x: -18, y: 26, z: 0 },
  { x: -16, y: 56, z: 0 },
  { x: -25, y: 70, z: 0 },
  { x: -32, y: 62, z: 0 },
];

/**
 * The wrist hinge axle, drawn as a boss at each end.
 *
 * The hinge axis runs medio-laterally, so from this view its two ends are
 * circles either side of the wrist. They belong to the forearm and never
 * rotate: the hand turns about them, which is what makes the joint legible.
 */
const WRIST_HINGE: readonly Vec3[] = [
  { x: -27, y: 1, z: 0 },
  { x: 27, y: 1, z: 0 },
];

/**
 * The forearm shaft: a closed, straight-sided shape drawn behind the palm.
 *
 * It is drawn because the wrist angle is meaningless without something to
 * measure it against, and it is deliberately rigid: the hand turns relative to
 * the forearm, so the gap that opens between this wrist edge and the palm's is
 * the joint. Its top edge tucks a little above the wrist so the palm covers
 * the seam when the wrist is neutral, and its bottom edge lands exactly on the
 * frame, so the arm reads as cropped by the panel rather than as a hand
 * standing on a plinth.
 */
const FOREARM: readonly Vec3[] = [
  { x: -27, y: 6, z: 0 },
  { x: 27, y: 6, z: 0 },
  { x: 29, y: -66, z: 0 },
  { x: -29, y: -66, z: 0 },
];

// ---- Vector helpers -----------------------------------------------------

function rotateX(v: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return { x: v.x, y: v.y * c - v.z * s, z: v.y * s + v.z * c };
}

function rotateY(v: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return { x: v.x * c + v.z * s, y: v.y, z: -v.x * s + v.z * c };
}

function rotateZ(v: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return { x: v.x * c - v.y * s, y: v.x * s + v.y * c, z: v.z };
}

/** Rodrigues rotation of `v` about a unit `axis`. */
function rotateAbout(v: Vec3, axis: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const dot = axis.x * v.x + axis.y * v.y + axis.z * v.z;
  const cross = {
    x: axis.y * v.z - axis.z * v.y,
    y: axis.z * v.x - axis.x * v.z,
    z: axis.x * v.y - axis.y * v.x,
  };
  return {
    x: v.x * c + cross.x * s + axis.x * dot * (1 - c),
    y: v.y * c + cross.y * s + axis.y * dot * (1 - c),
    z: v.z * c + cross.z * s + axis.z * dot * (1 - c),
  };
}

const clamp01 = (value: number) => (value < 0 ? 0 : value > 1 ? 1 : value);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// ---- Postures -----------------------------------------------------------

/**
 * The posture the hand returns to.
 *
 * Not a flat hand: a relaxed hand rests in slight flexion, increasing toward
 * the little finger. Drawing rest as a flat plane would make every gesture
 * look like it starts from a salute.
 */
export const REST_POSTURE: HandPosture = {
  digits: [0.34, 0.28, 0.3, 0.32, 0.34],
  opposition: 0.35,
  spread: 0.18,
  wrist: 0,
};

/**
 * Target postures, keyed by the decoder's class names.
 *
 * These describe the prosthesis command, not a photograph of a hand. Each has
 * to be distinguishable from all the others at a glance, which is asserted
 * rather than assumed: see the degeneracy test in `test/posture.test.ts`.
 */
export const GESTURE_POSTURES: Readonly<Record<string, HandPosture>> = {
  rest: REST_POSTURE,
  fist: {
    digits: [0.7, 1, 1, 1, 1],
    opposition: 0.85,
    spread: 0,
    wrist: 0,
  },
  open_hand: {
    digits: [0.02, 0.02, 0.02, 0.02, 0.02],
    opposition: 0.05,
    spread: 1,
    wrist: 0,
  },
  pinch: {
    digits: [0.55, 0.62, 0.34, 0.36, 0.38],
    opposition: 0.95,
    spread: 0.05,
    wrist: 0,
  },
  point: {
    digits: [0.72, 0.03, 1, 1, 1],
    opposition: 0.8,
    spread: 0.1,
    wrist: 0,
  },
  wrist_flexion: {
    digits: [0.3, 0.28, 0.3, 0.32, 0.34],
    opposition: 0.35,
    spread: 0.2,
    wrist: 1,
  },
  wrist_extension: {
    digits: [0.3, 0.28, 0.3, 0.32, 0.34],
    opposition: 0.35,
    spread: 0.2,
    wrist: -1,
  },
  thumb_up: {
    digits: [0, 1, 1, 1, 1],
    opposition: 0,
    spread: 0,
    wrist: 0,
  },
  two_finger: {
    digits: [0.8, 0.05, 0.05, 1, 1],
    opposition: 0.85,
    // Full splay: at anything less the two extended fingers project onto each
    // other and the gesture reads as one finger.
    spread: 1,
    wrist: 0,
  },
  spherical_grip: {
    digits: [0.45, 0.5, 0.52, 0.55, 0.58],
    opposition: 0.55,
    spread: 0.55,
    wrist: 0,
  },
};

/**
 * The posture for a gesture, or undefined if there is none.
 *
 * Undefined rather than a rest fallback on purpose. A gesture the decoder can
 * emit but the hand cannot draw is a gap the caller should surface, not paper
 * over by showing a resting hand as though nothing were wrong.
 */
export function postureFor(gesture: string): HandPosture | undefined {
  return Object.prototype.hasOwnProperty.call(GESTURE_POSTURES, gesture)
    ? GESTURE_POSTURES[gesture]
    : undefined;
}

/**
 * Commitment fraction to actuator travel.
 *
 * Below the motion onset the hand has not moved at all -- that is what makes
 * the onset mark on the commitment bar mean something. Above it, travel is
 * proportional to remaining evidence, reaching full excursion exactly when the
 * gesture latches.
 */
export function actuatorExcursion(
  commitment: number,
  motionOnset: number = DEFAULT_EVIDENCE_CONFIG.motionOnset,
): number {
  const c = clamp01(commitment);
  if (c <= motionOnset) return 0;
  if (motionOnset >= 1) return c >= 1 ? 1 : 0;
  return clamp01((c - motionOnset) / (1 - motionOnset));
}

/** Linear blend between two postures. `t` is clamped to [0, 1]. */
export function blendPosture(from: HandPosture, to: HandPosture, t: number): HandPosture {
  const f = clamp01(t);
  if (f === 0) return from;
  if (f === 1) return to;
  return {
    digits: from.digits.map((value, i) => lerp(value, to.digits[i]!, f)) as unknown as [
      number,
      number,
      number,
      number,
      number,
    ],
    opposition: lerp(from.opposition, to.opposition, f),
    spread: lerp(from.spread, to.spread, f),
    wrist: lerp(from.wrist, to.wrist, f),
  };
}

/**
 * The posture at a given point in the actuator's travel.
 *
 * `blendPosture` moves every digit together. This applies the closing order on
 * top, so intermediate frames show a hand conforming to something rather than
 * a slab shutting. Both endpoints are still exact: at zero travel it is the
 * rest posture, at full travel it is the target, and nothing arrives late.
 */
export function actuate(
  from: HandPosture,
  to: HandPosture,
  excursion: number,
): HandPosture {
  const t = clamp01(excursion);
  if (t === 0) return from;
  if (t === 1) return to;

  const digits = from.digits.map((value, digit) =>
    lerp(value, to.digits[digit]!, digitExcursion(digit, t)),
  ) as unknown as [number, number, number, number, number];

  return {
    digits,
    // Opposition is a thumb degree of freedom, so it keeps the thumb's timing.
    opposition: lerp(from.opposition, to.opposition, digitExcursion(0, t)),
    spread: lerp(from.spread, to.spread, t),
    wrist: lerp(from.wrist, to.wrist, t),
  };
}

/**
 * Signed drive per digit: positive is closing, negative is opening.
 *
 * Crossed with the muscle-group activation read off the electrodes, this is
 * what says which tendon is under load. The geometry shows what the decoder
 * decided; this is the axis along which the raw signal can be shown on the
 * same object.
 */
export function digitDrive(
  posture: HandPosture,
  reference: HandPosture = REST_POSTURE,
): readonly number[] {
  return posture.digits.map((value, digit) => value - reference.digits[digit]!);
}

// ---- Kinematics ---------------------------------------------------------

export interface HandSkeleton {
  /** One chain per digit, base joint first and fingertip last. Thumb is 0. */
  readonly digits: readonly (readonly Vec3[])[];
  /** Bearing half-width at each joint, matching `digits` point for point. */
  readonly radii: readonly (readonly number[])[];
  /** Drive tendon per digit, routed over the palmar side of each bearing. */
  readonly tendons: readonly (readonly Vec3[])[];
  /** Closed chassis outline. */
  readonly palm: readonly Vec3[];
  /** Panel seams, drawn as open lines over the chassis. */
  readonly seams: readonly (readonly Vec3[])[];
  /** The thumb actuator housing, a closed panel on the chassis. */
  readonly housing: readonly Vec3[];
  /** The two ends of the wrist hinge axle. Fixed to the forearm. */
  readonly hinge: readonly Vec3[];
  /** Forearm outline, proximal to wrist and back. Fixed: it never rotates. */
  readonly forearm: readonly Vec3[];
}

const UP: Vec3 = { x: 0, y: 1, z: 0 };
const ULNAR: Vec3 = { x: 1, y: 0, z: 0 };

const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });

function normalise(v: Vec3): Vec3 {
  const length = Math.hypot(v.x, v.y, v.z);
  return length > 0 ? { x: v.x / length, y: v.y / length, z: v.z / length } : UP;
}

/** A digit's solved geometry: joint centres, and the tendon routed over them. */
interface SolvedDigit {
  readonly points: Vec3[];
  readonly tendon: Vec3[];
}

/**
 * One digit as a chain of rigid segments.
 *
 * `frame` orients the digit before flexion: it rotates both the direction the
 * digit points and the axis it hinges about, so a splayed finger still curls
 * in its own plane rather than in the hand's.
 */
function solveDigit(
  base: Vec3,
  lengths: readonly number[],
  flexions: readonly number[],
  radii: readonly number[],
  frame: (v: Vec3) => Vec3,
): SolvedDigit {
  const direction = frame(UP);
  const axis = frame(ULNAR);
  const points: Vec3[] = [base];
  const segments: Vec3[] = [];

  let cumulative = 0;
  let current = base;
  for (let i = 0; i < lengths.length; i++) {
    cumulative += flexions[i]!;
    const d = rotateAbout(direction, axis, cumulative);
    segments.push(d);
    current = {
      x: current.x + d.x * lengths[i]!,
      y: current.y + d.y * lengths[i]!,
      z: current.z + d.z * lengths[i]!,
    };
    points.push(current);
  }

  // The tendon runs over the palmar side of each bearing. At a joint it takes
  // the bisector of the two segments meeting there, which is where a real
  // tendon sits as it wraps a pulley -- and it is why the visible run of
  // tendon across a joint shortens as that joint closes.
  const tendon = points.map((point, i) => {
    const before = segments[i - 1];
    const after = segments[i];
    const along = before && after ? normalise(add(before, after)) : (after ?? before)!;
    const palmar = rotateAbout(along, axis, Math.PI / 2);
    const offset = radii[i]! * TENDON_OFFSET;
    return {
      x: point.x + palmar.x * offset,
      y: point.y + palmar.y * offset,
      z: point.z + palmar.z * offset,
    };
  });

  return { points, tendon };
}

/** Forward kinematics for a whole hand, in the wrist frame. */
export function solveHand(posture: HandPosture): HandSkeleton {
  const wristAngle = posture.wrist * WRIST_RANGE * DEG;
  const atWrist = (v: Vec3) => rotateX(v, wristAngle);

  const digits: SolvedDigit[] = [];

  // The thumb: radial abduction in the palm plane, then opposition about the
  // long axis of the hand, which is what carries it across the palm.
  const thumbAbduction =
    lerp(THUMB_ABDUCTION_OPEN, THUMB_ABDUCTION_OPPOSED, clamp01(posture.opposition)) * DEG;
  const oppositionAngle = clamp01(posture.opposition) * THUMB_OPPOSITION_RANGE * DEG;
  digits.push(
    solveDigit(
      MCP[0]!,
      PHALANGES[0]!,
      FLEXION_RANGE[0]!.map(
        (range, joint) => range * jointTravel(joint, posture.digits[0]) * DEG,
      ),
      DIGIT_RADII[0]!,
      (v) => rotateY(rotateZ(v, -thumbAbduction), oppositionAngle),
    ),
  );

  for (let finger = 0; finger < 4; finger++) {
    const abduction = ABDUCTION[finger]! * clamp01(posture.spread) * DEG;
    const flexion = posture.digits[finger + 1]!;
    digits.push(
      solveDigit(
        MCP[finger + 1]!,
        PHALANGES[finger + 1]!,
        FLEXION_RANGE[finger + 1]!.map(
          (range, joint) => range * jointTravel(joint, flexion) * DEG,
        ),
        DIGIT_RADII[finger + 1]!,
        (v) => rotateZ(v, -abduction),
      ),
    );
  }

  return {
    digits: digits.map(({ points }) => points.map(atWrist)),
    radii: DIGIT_RADII,
    tendons: digits.map(({ tendon }) => tendon.map(atWrist)),
    palm: PALM.map(atWrist),
    seams: SEAMS.map((line) => line.map(atWrist)),
    housing: HOUSING.map(atWrist),
    // The hinge belongs to the forearm, so it does not turn with the hand.
    hinge: WRIST_HINGE,
    forearm: FOREARM,
  };
}

// ---- Projection ---------------------------------------------------------

export interface HandView {
  /** Rotation about the hand's long axis, in degrees. Reveals the palm. */
  readonly yaw: number;
  /** Rotation about the medio-lateral axis, in degrees. Adds elevation. */
  readonly pitch: number;
}

/**
 * The camera.
 *
 * Yawed off the palmar normal so that flexion -- which is straight toward the
 * viewer in a true palmar view, and therefore invisible -- becomes a sideways
 * sweep toward the thumb, which is where a closing hand actually goes.
 *
 * Pitch is zero deliberately. Pitch rotates about the same axis the wrist
 * turns about, so any non-zero value foreshortens flexion and extension by
 * different amounts and makes one of the two look broken while the other
 * looks fine. At zero the display treats them alike.
 */
export const DEFAULT_HAND_VIEW: HandView = { yaw: -34, pitch: 0 };

/**
 * Fixed drawing bounds, in projected units.
 *
 * Fixed, not fitted. A box recomputed per frame would rescale the hand as it
 * moved, so a closing fist would appear to grow rather than close. The
 * containment test sweeps every gesture across its whole travel against these
 * numbers -- including each joint's drawn half-width, not just its centre --
 * which is the guarantee that the drawing never overflows.
 */
export const HAND_VIEWBOX = { x: -116, y: -212, width: 179, height: 278 } as const;

/**
 * A projected point, carrying the camera-space depth it came from.
 *
 * The depth is what lets the renderer sort parts back to front. Without it a
 * closing fist draws as a tangle of crossing outlines; with it, the digits
 * that have curled toward the viewer cover the palm, and the hand reads as a
 * solid object.
 */
export interface ProjectedPoint extends Point {
  /** Larger is nearer the viewer. */
  readonly depth: number;
}

export interface ProjectedHand {
  readonly digits: readonly (readonly ProjectedPoint[])[];
  /** Mean depth per digit, for back-to-front ordering. */
  readonly digitDepth: readonly number[];
  readonly tendons: readonly (readonly ProjectedPoint[])[];
  readonly palm: readonly ProjectedPoint[];
  readonly palmDepth: number;
  readonly seams: readonly (readonly ProjectedPoint[])[];
  readonly housing: readonly ProjectedPoint[];
  readonly hinge: readonly ProjectedPoint[];
  readonly forearm: readonly ProjectedPoint[];
}

function project(v: Vec3, view: HandView): ProjectedPoint {
  const rotated = rotateX(rotateY(v, view.yaw * DEG), view.pitch * DEG);
  // Screen y grows downward; the hand's y grows toward the fingertips.
  return { x: rotated.x, y: -rotated.y, depth: rotated.z };
}

const meanDepth = (points: readonly ProjectedPoint[]) =>
  points.reduce((total, p) => total + p.depth, 0) / Math.max(1, points.length);

/** Orthographic projection of a solved hand into drawing coordinates. */
export function projectHand(
  skeleton: HandSkeleton,
  view: HandView = DEFAULT_HAND_VIEW,
): ProjectedHand {
  const to2d = (v: Vec3) => project(v, view);
  const digits = skeleton.digits.map((chain) => chain.map(to2d));
  const palm = skeleton.palm.map(to2d);
  return {
    digits,
    // Weighted toward the tip, which is the end that actually swings forward
    // and the end whose overlap the eye reads first.
    digitDepth: digits.map(
      (chain) => (meanDepth(chain) + chain[chain.length - 1]!.depth) / 2,
    ),
    tendons: skeleton.tendons.map((line) => line.map(to2d)),
    palm,
    palmDepth: meanDepth(palm),
    seams: skeleton.seams.map((line) => line.map(to2d)),
    housing: skeleton.housing.map(to2d),
    hinge: skeleton.hinge.map(to2d),
    forearm: skeleton.forearm.map(to2d),
  };
}
