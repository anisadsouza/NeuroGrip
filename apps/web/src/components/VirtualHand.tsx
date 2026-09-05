/**
 * The hand.
 *
 * The commitment bar shows evidence accruing against a boundary. This shows
 * what that evidence is doing to the prosthesis, which is the only part the
 * wearer would feel. Together they are the whole of progressive actuation: an
 * abstract quantity on the left, a physical consequence on the right, driven
 * by the same number.
 *
 * ---
 *
 * **It is drawn as a machine, because it is one.** This is a prosthesis, not a
 * hand: bearings at every joint, narrow link shafts between them, a machined
 * chassis, a wrist hinge with a visible axle, and one drive tendon per digit
 * routed over the palmar side of each bearing. Drawing flesh would be a
 * picture of the thing the device replaces rather than the device itself.
 *
 * **Two independent truths are shown on one object.** The geometry is what the
 * decoder decided -- posture, and how far through its travel it is. The colour
 * is what the electrodes are reading right now, before any decoding: carmine
 * on a digit means the muscle group that drives that digit in that direction
 * is under load. They can disagree, and when they do that is worth seeing.
 *
 * Carmine is the palette's one chromatic hue and it means muscle activation
 * and nothing else, so this is the one place the hand is allowed colour at
 * all. The commit green is a state, not a measurement, and appears only once
 * the gesture is held.
 *
 * **Every frame is a mechanism, not an interpolation.** The joints do not move
 * together: the knuckle takes the load first and the joints beyond it follow
 * through passive coupling, and the little finger leads the index into a
 * grasp. Both come from `posture.ts`, both are tested, and both are why an
 * intermediate frame looks like a hand closing on something rather than a
 * shape being tweened.
 *
 * Parts are painted back to front by projected depth, so a digit that has
 * curled toward the viewer covers the chassis it is closing over.
 */

import { useMemo } from 'react';
import {
  AXLE_RADIUS,
  HAND_VIEWBOX,
  HINGE_RADIUS,
  JOINT_RING,
  LINK_WAIST,
  PAD_RADIUS,
  REST_POSTURE,
  actuate,
  actuatorExcursion,
  capsulePath,
  circlePath,
  digitDrive,
  polylinePath,
  postureFor,
  projectHand,
  solveHand,
  type HandPosture,
  type ProjectedPoint,
} from '@neurogrip/core';

/** Muscle-group drive read off the electrode ring, each in [0, 1]. */
export interface MuscleDrive {
  readonly digitFlexor: number;
  readonly digitExtensor: number;
  readonly wristFlexor: number;
  readonly wristExtensor: number;
}

export const NO_DRIVE: MuscleDrive = {
  digitFlexor: 0,
  digitExtensor: 0,
  wristFlexor: 0,
  wristExtensor: 0,
};

export interface VirtualHandProps {
  /** The gesture the decoder currently leads with. */
  readonly gesture: string;
  /** Plain-language name, for the caption and the screen-reader label. */
  readonly label: string;
  /** Progress toward commitment, in [0, 1]. */
  readonly commitment: number;
  readonly latched: boolean;
  readonly reversible: boolean;
  readonly timedOut: boolean;
  /** Live muscle activation, for the drive tint. */
  readonly drive?: MuscleDrive;
}

/**
 * Joint travel at which a digit counts as fully driven.
 *
 * A digit barely off its resting angle is not under meaningful load, so
 * without this the tint would light the whole hand at the first flicker of
 * signal regardless of what the hand was doing.
 */
const FULL_DRIVE_TRAVEL = 0.45;

/** Wrist angle, as a fraction of full travel, that counts as fully driven. */
const FULL_DRIVE_WRIST = 0.5;

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

interface DigitDrawing {
  readonly key: string;
  readonly depth: number;
  /** The digit silhouette: link shafts capped by the joints they run between. */
  readonly links: string;
  /** Pivot rings, inscribed in the silhouette rather than proud of it. */
  readonly pivots: string;
  /** Axle pins at each joint centre. */
  readonly axles: string;
  /** The contact pad at the fingertip. */
  readonly pad: string;
  readonly tendon: string;
  /** How hard this digit is being driven, in [0, 1]. */
  readonly load: number;
}

interface Drawing {
  readonly digits: readonly DigitDrawing[];
  readonly palmDepth: number;
  readonly palm: string;
  readonly seams: readonly string[];
  readonly housing: string;
  readonly hinge: readonly ProjectedPoint[];
  readonly forearm: string;
  readonly skeleton: readonly string[];
}

function draw(posture: HandPosture, drive: MuscleDrive): Drawing {
  const skeleton = solveHand(posture);
  const projected = projectHand(skeleton);
  const commanded = digitDrive(posture);

  const digits = projected.digits.map((chain, digit): DigitDrawing => {
    const radii = skeleton.radii[digit]!;
    const last = chain.length - 1;

    // Signed: closing pulls on the flexor group, opening on the extensor.
    const travel = commanded[digit] ?? 0;
    const group = travel >= 0 ? drive.digitFlexor : drive.digitExtensor;
    const load = clamp01(Math.abs(travel) / FULL_DRIVE_TRAVEL) * clamp01(group);

    return {
      key: `digit-${digit}`,
      depth: projected.digitDepth[digit]!,
      // Shaft plus joint, unioned into one silhouette. The shaft is just
      // narrower than the joints it runs between, so each joint reads as a
      // slight swelling rather than a bead threaded on a wire.
      links:
        chain
          .slice(0, -1)
          .map((joint, i) =>
            capsulePath(
              joint,
              radii[i]! * LINK_WAIST,
              chain[i + 1]!,
              radii[i + 1]! * LINK_WAIST,
            ),
          )
          .join('') + chain.map((joint, i) => circlePath(joint, radii[i]!)).join(''),
      // Every joint but the last: a fingertip is the end of the chain, not a
      // pivot, and drawing hardware there stacked a ring, an axle and a
      // contact pad into a bullseye.
      pivots: chain
        .slice(0, -1)
        .map((joint, i) => circlePath(joint, radii[i]! * JOINT_RING))
        .join(''),
      axles: chain
        .slice(0, -1)
        .map((joint, i) => circlePath(joint, radii[i]! * AXLE_RADIUS))
        .join(''),
      pad: circlePath(projected.tendons[digit]![last]!, radii[last]! * PAD_RADIUS),
      tendon: polylinePath(projected.tendons[digit]!),
      load,
    };
  });

  return {
    digits,
    palmDepth: projected.palmDepth,
    palm: polylinePath(projected.palm, true),
    seams: projected.seams.map((line) => polylinePath(line)),
    housing: polylinePath(projected.housing, true),
    hinge: projected.hinge,
    forearm: polylinePath(projected.forearm, true),
    // The retract ghost: joint centres and chassis outline only.
    skeleton: [
      polylinePath(projected.palm, true),
      ...projected.digits.map((chain) => polylinePath(chain)),
    ],
  };
}

const REST_DRAWING = draw(REST_POSTURE, NO_DRIVE);

export function VirtualHand({
  gesture,
  label,
  commitment,
  latched,
  reversible,
  timedOut,
  drive = NO_DRIVE,
}: VirtualHandProps) {
  const target = postureFor(gesture);
  const excursion = target ? actuatorExcursion(commitment) : 0;

  const posture = useMemo(
    () => (target ? actuate(REST_POSTURE, target, excursion) : REST_POSTURE),
    [target, excursion],
  );
  const drawing = useMemo(() => draw(posture, drive), [posture, drive]);

  // Shown only while the motion can still be taken back. Once the gesture is
  // held, the hand is not going back to rest on its own and drawing the ghost
  // would be a promise the system is no longer making.
  const showGhost = reversible && excursion > 0;

  const wristLoad =
    clamp01(Math.abs(posture.wrist) / FULL_DRIVE_WRIST) *
    clamp01(posture.wrist >= 0 ? drive.wristFlexor : drive.wristExtensor);

  const state = timedOut
    ? 'at rest, no clear intent'
    : latched
      ? `holding ${label}`
      : excursion > 0
        ? `moving toward ${label}, still reversible`
        : 'at rest';

  // Back to front, with the chassis taking its turn among the digits so a
  // curled digit covers it and an extended one does not.
  const ordered = useMemo(
    () =>
      [
        ...drawing.digits.map((digit) => ({ depth: digit.depth, digit })),
        { depth: drawing.palmDepth, digit: null as DigitDrawing | null },
      ].sort((a, b) => a.depth - b.depth),
    [drawing],
  );

  const activeDrive = Math.max(
    drive.digitFlexor,
    drive.digitExtensor,
    drive.wristFlexor,
    drive.wristExtensor,
  );

  return (
    <figure className="hand" data-latched={latched || undefined}>
      <figcaption className="hand-caption">Hand</figcaption>

      <svg
        className="hand-figure"
        viewBox={`${HAND_VIEWBOX.x} ${HAND_VIEWBOX.y} ${HAND_VIEWBOX.width} ${HAND_VIEWBOX.height}`}
        role="img"
        aria-label={`Prosthetic hand, ${state}. ${Math.round(excursion * 100)} percent of full travel, muscle drive ${Math.round(activeDrive * 100)} percent.`}
      >
        <path className="hand-forearm" d={drawing.forearm} />

        {/* The wrist hinge axle. Fixed to the forearm; the hand turns on it. */}
        {drawing.hinge.map((point, i) => (
          <g key={i}>
            <circle className="hand-link" cx={point.x} cy={point.y} r={HINGE_RADIUS} />
            <circle
              className="hand-drive"
              cx={point.x}
              cy={point.y}
              r={HINGE_RADIUS}
              style={{ opacity: wristLoad }}
            />
            <circle
              className="hand-axle"
              cx={point.x}
              cy={point.y}
              r={HINGE_RADIUS * 0.32}
            />
          </g>
        ))}

        {showGhost ? (
          <g className="hand-ghost" aria-hidden="true">
            {REST_DRAWING.skeleton.map((d, i) => (
              <path key={i} d={d} />
            ))}
          </g>
        ) : null}

        {ordered.map(({ digit }) =>
          digit === null ? (
            <g key="chassis">
              <path className="hand-chassis" d={drawing.palm} />
              <path className="hand-housing" d={drawing.housing} />
              {drawing.seams.map((d, i) => (
                <path key={i} className="hand-seam" d={d} />
              ))}
            </g>
          ) : (
            <g key={digit.key}>
              <path className="hand-link" d={digit.links} />
              {/* The signal layer: one overlay per digit, so the whole digit
                  takes a single wash rather than each part compounding. */}
              <path
                className="hand-drive"
                d={digit.links}
                style={{ opacity: digit.load }}
              />
              <path className="hand-pivot" d={digit.pivots} />
              <path className="hand-axle" d={digit.axles} />
              <path
                className="hand-tendon"
                d={digit.tendon}
                style={{ opacity: 0.3 + 0.7 * digit.load }}
                strokeWidth={1 + 1.4 * digit.load}
              />
              <path className="hand-pad" d={digit.pad} />
            </g>
          ),
        )}
      </svg>

      <p className="hand-state" data-latched={latched || undefined}>
        {timedOut
          ? 'No clear intent. Held at rest.'
          : latched
            ? 'Held. Press Release to open.'
            : excursion > 0
              ? 'Moving. Returns to rest if the evidence turns.'
              : 'Waiting for evidence.'}
      </p>

      <p className="hand-legend">
        Shape is the decoded intent.{' '}
        <span className="hand-legend-key" aria-hidden="true" /> is live muscle drive
        on that joint, straight off the electrodes.
      </p>

      {target ? null : (
        <p className="notice notice-stop hand-missing" role="status">
          No hand posture is defined for <code>{gesture}</code>.
        </p>
      )}
    </figure>
  );
}
