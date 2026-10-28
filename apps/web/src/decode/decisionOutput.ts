export interface DecisionOutputInput {
  readonly gestures: readonly string[];
  readonly leader: number;
  readonly latched: boolean;
  readonly latchedClass: number | null;
  readonly timedOut: boolean;
  readonly commitment: number;
}

export interface DecisionOutput {
  readonly gesture: string;
  readonly index: number;
  readonly handCommitment: number;
}

export function decisionOutputFor(input: DecisionOutputInput): DecisionOutput {
  const index = input.timedOut
    ? input.gestures.indexOf('rest')
    : input.latched && input.latchedClass !== null
      ? input.latchedClass
      : input.leader;
  const gesture = input.gestures[index] ?? '';
  return {
    index,
    gesture,
    handCommitment: gesture === 'rest' ? 0 : input.commitment,
  };
}
