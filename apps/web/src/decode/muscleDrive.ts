/**
 * Muscle-group drive, straight off the electrodes.
 *
 * This deliberately does not pass through the decoder. The hand's shape is
 * what the decoder concluded; its colour is what the muscles are doing, and
 * the two are worth being able to disagree -- a wearer straining against a
 * gesture the decoder has not recognised should see the strain.
 *
 * The weighting comes from the replay manifest, which computes it from the
 * same forearm anatomy the simulator mixes through, rather than being
 * reimplemented here where it would be a second thing to keep in step.
 */

import { type MuscleDrive, NO_DRIVE } from '../components/VirtualHand.js';

export interface MuscleGroupWeights {
  readonly digit_flexor: readonly number[];
  readonly digit_extensor: readonly number[];
  readonly wrist_flexor: readonly number[];
  readonly wrist_extensor: readonly number[];
}

/**
 * Channel RMS that reads as full drive for a group, in volts.
 *
 * A group's drive is a weighted mean over the ring. The weights concentrate on
 * the electrodes nearest that muscle, so the mean tracks the loudest channel
 * fairly closely rather than being dragged down by the far side of the
 * forearm -- measured, the group drive peaks at 656 uV against a per-channel
 * peak of 953. Seven hundred puts the loudest gesture in the vocabulary at
 * about 0.94 and pins nothing.
 *
 * Two earlier values were guessed at from the simulator's stated RMS range and
 * both saturated on a power grip, which made a strong contraction and a
 * fatigued one draw the same colour. This one comes from the signal.
 */
export const FULL_DRIVE_VOLTS = 7e-4;

/**
 * Project per-channel RMS onto the four functional groups.
 *
 * Absent weights or absent RMS yield no drive rather than a guess: a hand
 * drawn with invented colour is worse than a hand drawn without any.
 */
export function projectDrive(
  groups: MuscleGroupWeights | undefined,
  channelRms: readonly number[] | undefined,
  fullDriveVolts: number = FULL_DRIVE_VOLTS,
): MuscleDrive {
  if (!groups || !channelRms) return NO_DRIVE;

  const project = (weights: readonly number[]): number => {
    let total = 0;
    // A channel the RMS array does not reach counts as silent rather than as
    // NaN: a short array means a dropped message, not a hot electrode.
    for (let i = 0; i < weights.length; i++) total += weights[i]! * (channelRms[i] ?? 0);
    return Math.min(1, Math.max(0, total / fullDriveVolts));
  };

  return {
    digitFlexor: project(groups.digit_flexor),
    digitExtensor: project(groups.digit_extensor),
    wristFlexor: project(groups.wrist_flexor),
    wristExtensor: project(groups.wrist_extensor),
  };
}
