/**
 * What each gesture costs to get wrong.
 *
 * This is a safety judgement, not a tuning parameter. The question for each
 * gesture is how hard the mistake is to undo: a power grip closing on
 * something fragile, or on a person, is the expensive error. Opening a hand
 * drops what you were holding, which is recoverable. Rest costs nothing, which
 * is why it is also the fallback when nothing else commits.
 *
 * It lives in its own module because three things need it and they must not
 * derive it separately: the accumulator running in the worker, the commitment
 * bar drawing the boundaries, and the TTUM run that reports how long motion
 * takes to begin. A risk vector written twice is a risk vector that will one
 * day disagree with itself, and the symptom would be a metric that flatters a
 * boundary the wearer never had.
 */

/**
 * The cost of an unintended commitment, keyed by the model's class name.
 *
 * A gesture absent from this table takes a middling cost rather than a free
 * one: an unknown gesture is not known to be safe.
 */
export const COMMIT_COST: Readonly<Record<string, number>> = {
  rest: 0,
  fist: 1.0,
  spherical_grip: 1.0,
  pinch: 0.6,
  two_finger: 0.6,
  wrist_flexion: 0.3,
  wrist_extension: 0.3,
  point: 0.2,
  thumb_up: 0.2,
  open_hand: 0.1,
};

/** The cost assumed for a gesture this table has never heard of. */
export const UNKNOWN_COMMIT_COST = 0.5;

/** Plain-language names. The model's identifiers are not wearer-facing copy. */
export const GESTURE_LABEL: Readonly<Record<string, string>> = {
  rest: 'Rest',
  fist: 'Close fist',
  open_hand: 'Open hand',
  pinch: 'Pinch',
  point: 'Point',
  wrist_flexion: 'Bend wrist in',
  wrist_extension: 'Bend wrist back',
  thumb_up: 'Thumb up',
  two_finger: 'Two-finger grip',
  spherical_grip: 'Hold a ball',
};

/**
 * Risks in the order the model emits its classes.
 *
 * Order is the whole point: the accumulator indexes this by class index, so a
 * vector built in any other order silently assigns the fist's boundary to
 * whatever gesture happens to sit at position one.
 */
export function risksFor(gestures: readonly string[]): number[] {
  return gestures.map((name) => COMMIT_COST[name] ?? UNKNOWN_COMMIT_COST);
}

/** Wearer-facing labels in the same order, falling back to the raw name. */
export function labelsFor(gestures: readonly string[]): string[] {
  return gestures.map((name) => GESTURE_LABEL[name] ?? name);
}
