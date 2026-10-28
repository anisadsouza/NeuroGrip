import { describe, expect, it } from 'vitest';
import { virtualCommandFor } from '../../src/decode/virtualCommand.js';

describe('virtualCommandFor', () => {
  it.each([
    ['rest', 'hold_rest'],
    ['fist', 'close_fist'],
    ['open_hand', 'open_hand'],
    ['pinch', 'close_thumb_index'],
    ['point', 'extend_index'],
    ['wrist_flexion', 'flex_wrist'],
    ['wrist_extension', 'extend_wrist'],
    ['thumb_up', 'raise_thumb'],
    ['two_finger', 'extend_index_middle'],
    ['spherical_grip', 'close_spherical_grip'],
  ])('maps %s to the virtual action %s', (gesture, command) => {
    expect(virtualCommandFor(gesture)).toBe(command);
  });

  it('does not invent a command for an unknown class', () => {
    expect(virtualCommandFor('unknown')).toBeNull();
  });
});
