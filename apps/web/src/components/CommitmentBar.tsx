/**
 * The commitment bar.
 *
 * This is the one element in the interface that earns visual weight, because
 * it shows something no other myoelectric interface shows: a decision being
 * formed, with a cost attached to it.
 *
 * What it has to make legible, in one glance:
 *
 *   - how far the current gesture has progressed toward commitment
 *   - where that gesture's boundary sits, which depends on how costly it is
 *   - that the motion is still reversible, or that it no longer is
 *   - which other gestures are competing, and how far behind they are
 *
 * A progress bar shows the first of those. The rest are the point. So the bar
 * carries a boundary post whose position IS the risk weighting, competing
 * gestures as recessed ticks on the same scale, and a state that changes at the
 * moment reversibility is lost.
 *
 * The scale is shared by every gesture, which is what makes the asymmetry
 * visible: a costly grip's post stands further right than a cheap one's, on the
 * same ruler.
 */

import { useEffect, useRef } from 'react';

export interface CommitmentBarProps {
  readonly gestures: readonly string[];
  readonly leader: number;
  readonly commitment: number;
  readonly latched: boolean;
  readonly reversible: boolean;
  readonly timedOut: boolean;
  /** Accumulated evidence per gesture, in log-evidence units. */
  readonly evidence: readonly number[];
  /** The leader's boundary, after risk weighting and urgency collapse. */
  readonly effectiveThreshold: number;
  /** Commit cost per gesture, so every boundary can be drawn on one scale. */
  readonly risks: readonly number[];
  readonly baseThreshold: number;
  readonly riskWeight: number;
  readonly motionOnset: number;
}

export function CommitmentBar({
  gestures,
  leader,
  commitment,
  latched,
  reversible,
  timedOut,
  evidence,
  effectiveThreshold,
  risks,
  baseThreshold,
  riskWeight,
  motionOnset,
}: CommitmentBarProps) {
  const liveRef = useRef<HTMLDivElement>(null);
  const announced = useRef<string>('');

  // The widest boundary any gesture could demand sets the scale, so a costly
  // gesture's post genuinely sits further along than a cheap one's.
  const maxThreshold = Math.max(
    baseThreshold + riskWeight * Math.max(...risks, 0),
    effectiveThreshold,
  );
  const toPercent = (value: number) => Math.min(100, (value / maxThreshold) * 100);

  const leaderName = gestures[leader] ?? 'unknown';
  const boundaryPercent = toPercent(effectiveThreshold);
  const fillPercent = toPercent(evidence[leader] ?? 0);
  const onsetPercent = boundaryPercent * motionOnset;

  const state = timedOut
    ? 'timed out'
    : latched
      ? 'held'
      : reversible
        ? 'moving, reversible'
        : 'listening';

  // Announce only committed and abandoned states. Narrating every hop would
  // make a screen reader unusable at 50 Hz.
  useEffect(() => {
    const message = latched
      ? timedOut
        ? 'No clear intent. Holding at rest.'
        : `${leaderName} held.`
      : '';
    if (message && message !== announced.current && liveRef.current) {
      announced.current = message;
      liveRef.current.textContent = message;
    }
    if (!latched) announced.current = '';
  }, [latched, timedOut, leaderName]);

  return (
    <section className="commitment" aria-labelledby="commitment-heading">
      <header className="commitment-head">
        <h2 id="commitment-heading">Commitment</h2>
        <p className="commitment-state" data-state={latched ? 'latched' : 'open'}>
          {state}
        </p>
      </header>

      <div
        className="commitment-track ng-graticule"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(commitment * 100)}
        aria-valuetext={`${leaderName}, ${Math.round(commitment * 100)} percent committed, ${state}`}
      >
        <div
          className="commitment-fill"
          data-latched={latched || undefined}
          style={{ width: `${fillPercent}%` }}
        />

        {/* Where motion begins. Before this the hand has not moved at all. */}
        <div
          className="commitment-onset"
          style={{ left: `${onsetPercent}%` }}
          aria-hidden="true"
        />

        {/* Every gesture's boundary on one scale. The leader's is emphasised;
            the others are there so the asymmetry is visible rather than
            asserted. */}
        {gestures.map((name, index) => (
          <div
            key={name}
            className="commitment-post"
            data-leader={index === leader || undefined}
            style={{ left: `${toPercent(baseThreshold + riskWeight * (risks[index] ?? 0))}%` }}
            aria-hidden="true"
          />
        ))}

        {/* Competing gestures, on the same ruler as the leader. */}
        {evidence.map((value, index) =>
          index === leader || value <= 0 ? null : (
            <div
              key={gestures[index] ?? index}
              className="commitment-rival"
              style={{ left: `${toPercent(value)}%` }}
              aria-hidden="true"
            />
          ),
        )}
      </div>

      <footer className="commitment-foot">
        <span className="commitment-leader">{leaderName}</span>
        <span className="ng-num commitment-percent">
          {(commitment * 100).toFixed(0)}
          <span className="commitment-unit"> %</span>
        </span>
      </footer>

      <p className="commitment-legend">
        The hand starts moving at the first mark and can still be taken back.
        It holds only at this gesture&rsquo;s post
        {risks[leader] !== undefined && risks[leader]! > 0
          ? ', which sits further out because this grip is harder to undo.'
          : '.'}
      </p>

      <div ref={liveRef} role="status" aria-live="polite" className="visually-hidden" />
    </section>
  );
}
