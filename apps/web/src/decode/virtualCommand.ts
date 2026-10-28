const VIRTUAL_COMMANDS: Readonly<Record<string, string>> = {
  rest: 'hold_rest',
  fist: 'close_fist',
  open_hand: 'open_hand',
  pinch: 'close_thumb_index',
  point: 'extend_index',
  wrist_flexion: 'flex_wrist',
  wrist_extension: 'extend_wrist',
  thumb_up: 'raise_thumb',
  two_finger: 'extend_index_middle',
  spherical_grip: 'close_spherical_grip',
};

export function virtualCommandFor(gesture: string): string | null {
  return VIRTUAL_COMMANDS[gesture] ?? null;
}
