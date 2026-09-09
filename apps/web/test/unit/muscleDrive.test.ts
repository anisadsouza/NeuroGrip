/**
 * The hand's colour is a measurement, not a label -- it is the one place in
 * the interface where carmine means muscle rather than emphasis. A projection
 * that saturates, or that produces NaN from a short message, would paint a
 * number that is not true of the wearer's arm.
 */

import { describe, expect, it } from 'vitest';
import { NO_DRIVE } from '../../src/components/VirtualHand.js';
import {
  FULL_DRIVE_VOLTS,
  type MuscleGroupWeights,
  projectDrive,
} from '../../src/decode/muscleDrive.js';

/** Eight electrodes, all weight on the first, so the arithmetic is checkable. */
const FIRST_ELECTRODE: MuscleGroupWeights = {
  digit_flexor: [1, 0, 0, 0, 0, 0, 0, 0],
  digit_extensor: [0, 1, 0, 0, 0, 0, 0, 0],
  wrist_flexor: [0, 0, 1, 0, 0, 0, 0, 0],
  wrist_extensor: [0, 0, 0, 1, 0, 0, 0, 0],
};

describe('projectDrive', () => {
  it('scales a channel against the full-drive level rather than against itself', () => {
    const rms = [FULL_DRIVE_VOLTS / 2, 0, 0, 0, 0, 0, 0, 0];
    expect(projectDrive(FIRST_ELECTRODE, rms).digitFlexor).toBeCloseTo(0.5, 6);
  });

  /**
   * A contraction beyond the full-drive level must read as full, not as more
   * than full: the hand's fill is a fraction and a value above one would paint
   * outside the digit.
   */
  it('saturates at full drive rather than exceeding it', () => {
    const rms = [FULL_DRIVE_VOLTS * 10, 0, 0, 0, 0, 0, 0, 0];
    expect(projectDrive(FIRST_ELECTRODE, rms).digitFlexor).toBe(1);
  });

  it('floors at zero rather than passing a negative through', () => {
    const rms = [-FULL_DRIVE_VOLTS, 0, 0, 0, 0, 0, 0, 0];
    expect(projectDrive(FIRST_ELECTRODE, rms).digitFlexor).toBe(0);
  });

  /**
   * A short RMS array means a dropped or truncated message, not a hot
   * electrode. Treating the missing channels as silent keeps the number
   * finite; the alternative is NaN reaching a fill percentage.
   */
  it('treats channels the message did not reach as silent rather than as NaN', () => {
    const drive = projectDrive(FIRST_ELECTRODE, [FULL_DRIVE_VOLTS]);
    for (const [group, value] of Object.entries(drive)) {
      expect(Number.isFinite(value), group).toBe(true);
    }
    expect(drive.digitExtensor).toBe(0);
  });

  /**
   * A hand drawn with invented colour is worse than a hand drawn without any:
   * a bundle written before muscleGroups existed must show no drive, not a
   * plausible-looking guess.
   */
  it('shows no drive at all when the manifest carries no weights', () => {
    expect(projectDrive(undefined, [FULL_DRIVE_VOLTS])).toEqual(NO_DRIVE);
    expect(projectDrive(FIRST_ELECTRODE, undefined)).toEqual(NO_DRIVE);
  });
});
