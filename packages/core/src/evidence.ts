/**
 * Risk-weighted evidence accumulation and progressive actuation.
 *
 * Pillar 2 of the NeuroGrip design. Instead of thresholding a single window's
 * posterior, this accumulates log-evidence across successive 20 ms hops and
 * drives the hand *progressively*: motion begins while the decision is still
 * reversible and latches only at full commitment.
 *
 * Two properties distinguish it from a confidence threshold:
 *
 *   1. Each gesture carries its own decision boundary, raised in proportion to
 *      the cost of falsely committing to it. Closing a power grip near a face
 *      must clear a higher bar than opening a hand, because the two mistakes
 *      are not equally recoverable. A single global confidence threshold cannot
 *      express that.
 *
 *   2. The actuator command is continuous, not binary. Useful motion starts at
 *      a fraction of the boundary and retracts if the evidence turns, so the
 *      wearer sees intent being formed rather than waiting through a silent
 *      deliberation window and then getting a discrete jump.
 *
 * The accumulator is a leaky competing race: each class accrues its log
 * likelihood ratio against its strongest rival, decays toward zero, and is
 * floored at zero so a class that fell behind can recover promptly.
 *
 * Boundaries collapse with elapsed time (an urgency signal, as in
 * sequential-sampling models of decision making) so the system cannot hang
 * indefinitely. If nothing commits before `maxHops`, it falls back to the safe
 * class -- doing nothing -- rather than forcing a guess at a grip.
 */

export interface EvidenceConfig {
  /** Base decision boundary, in log-evidence units. */
  baseThreshold: number;
  /** Boundary added per unit of commit cost: theta_k = base + weight * risk_k. */
  riskWeight: number;
  /** Commitment fraction at which the hand starts to move, still reversibly. */
  motionOnset: number;
  /** Per-hop decay of accumulated evidence, in [0, 1). */
  leak: number;
  /** Hops of deliberation before boundaries begin to collapse. */
  urgencyOnsetHops: number;
  /** Fraction of the boundary remaining at maximum urgency, in (0, 1]. */
  urgencyFloor: number;
  /** Hops after which the safe fallback is taken if nothing has committed. */
  maxHops: number;
  /** Class treated as "do nothing" -- the state it is always safe to fall back to. */
  safeClassIndex: number;
  /** Probability floor, so a zero posterior cannot produce -Infinity. */
  epsilon: number;
}

export const DEFAULT_EVIDENCE_CONFIG: EvidenceConfig = {
  // Sized so a confident decoder takes roughly ten hops (200 ms) to commit,
  // with motion beginning after about two (40 ms). That gap is the mechanism:
  // ~160 ms of visible, retractable motion before the grip latches. An earlier
  // value of 6 latched in two hops, which left nothing to retract and made
  // "progressive" actuation indistinguishable from a confidence threshold.
  baseThreshold: 24,
  // Risk 1.0 raises the bar by half, risk 2.0 doubles it.
  riskWeight: 12,
  motionOnset: 0.2,
  leak: 0.05,
  // At a 20 ms hop these are 500 ms of deliberation and a 1.5 s hard stop.
  urgencyOnsetHops: 25,
  urgencyFloor: 0.35,
  maxHops: 75,
  safeClassIndex: 0,
  epsilon: 1e-12,
};

export interface Decision {
  /** Class currently holding the most evidence. */
  leader: number;
  /** Progress toward commitment for the leader, in [0, 1]. */
  commitment: number;
  /** The hand is moving. True from `motionOnset` upward. */
  moving: boolean;
  /** Moving, but the motion can still be taken back. */
  reversible: boolean;
  /** Commitment reached; the gesture is held until reset. */
  latched: boolean;
  /** The committed class, or null while still deliberating. */
  latchedClass: number | null;
  /** Deliberation ran past `maxHops` and the safe fallback was taken. */
  timedOut: boolean;
  /** Hops since construction or the last reset. */
  hops: number;
  /** The leader's boundary after risk weighting and urgency collapse. */
  effectiveThreshold: number;
  /** Accumulated evidence per class. A copy; safe to retain. */
  evidence: readonly number[];
}

export class EvidenceAccumulator {
  private readonly config: EvidenceConfig;
  private readonly risks: Float64Array;
  private readonly evidence: Float64Array;
  private hopCount = 0;
  private latchedIndex: number | null = null;
  private timedOutFlag = false;
  private motionHop: number | null = null;

  constructor(risks: readonly number[], overrides: Partial<EvidenceConfig> = {}) {
    if (risks.length < 2) {
      throw new RangeError('evidence accumulation needs at least two classes');
    }
    for (const risk of risks) {
      if (!(risk >= 0) || !Number.isFinite(risk)) {
        throw new RangeError(`commit cost must be finite and non-negative, got ${risk}`);
      }
    }

    this.config = { ...DEFAULT_EVIDENCE_CONFIG, ...overrides };
    if (this.config.leak < 0 || this.config.leak >= 1) {
      throw new RangeError('leak must be in [0, 1)');
    }
    if (this.config.urgencyFloor <= 0 || this.config.urgencyFloor > 1) {
      throw new RangeError('urgencyFloor must be in (0, 1]');
    }
    if (
      this.config.safeClassIndex < 0 ||
      this.config.safeClassIndex >= risks.length
    ) {
      throw new RangeError('safeClassIndex is out of range');
    }

    this.risks = Float64Array.from(risks);
    this.evidence = new Float64Array(risks.length);
  }

  get nClasses(): number {
    return this.evidence.length;
  }

  /** Hop at which motion first began, or null if it has not. The TTUM metric. */
  get hopsToMotion(): number | null {
    return this.motionHop;
  }

  evidenceFor(index: number): number {
    if (index < 0 || index >= this.evidence.length) {
      throw new RangeError(`class index ${index} out of range`);
    }
    return this.evidence[index]!;
  }

  reset(): void {
    this.evidence.fill(0);
    this.hopCount = 0;
    this.latchedIndex = null;
    this.timedOutFlag = false;
    this.motionHop = null;
  }

  /** Boundary for a class after risk weighting and the current urgency collapse. */
  private thresholdFor(index: number): number {
    const base = this.config.baseThreshold + this.config.riskWeight * this.risks[index]!;
    const { urgencyOnsetHops, maxHops, urgencyFloor } = this.config;

    if (this.hopCount <= urgencyOnsetHops || maxHops <= urgencyOnsetHops) return base;

    const progress = Math.min(
      1,
      (this.hopCount - urgencyOnsetHops) / (maxHops - urgencyOnsetHops),
    );
    return base * (1 - progress * (1 - urgencyFloor));
  }

  /**
   * Advance one hop.
   *
   * `posteriors` need not be normalised; they are renormalised internally so a
   * caller passing unnormalised scores gets the same answer.
   */
  update(posteriors: ArrayLike<number>): Decision {
    if (posteriors.length !== this.evidence.length) {
      throw new RangeError(
        `expected ${this.evidence.length} posteriors, received ${posteriors.length}`,
      );
    }

    this.hopCount += 1;

    let total = 0;
    for (let i = 0; i < posteriors.length; i++) {
      const value = posteriors[i]!;
      if (!Number.isFinite(value) || value < 0) {
        throw new RangeError('posteriors must be finite and non-negative');
      }
      total += value;
    }
    if (total <= 0) throw new RangeError('posteriors must not sum to zero');

    if (this.latchedIndex === null) {
      this.accumulate(posteriors, total);
    }

    const leader = this.argmaxEvidence();
    const threshold = this.thresholdFor(leader);
    const commitment =
      this.latchedIndex !== null
        ? 1
        : Math.min(1, Math.max(0, this.evidence[leader]! / threshold));

    if (this.latchedIndex === null && commitment >= 1) {
      this.latchedIndex = leader;
    }

    // The hard stop. Falling back to the safe class is a deliberate refusal to
    // guess: an unintended grip is a physical event, an unintended rest is not.
    if (this.latchedIndex === null && this.hopCount >= this.config.maxHops) {
      this.latchedIndex = this.config.safeClassIndex;
      this.timedOutFlag = true;
    }

    const latched = this.latchedIndex !== null;
    const finalCommitment = latched ? 1 : commitment;
    const moving = finalCommitment >= this.config.motionOnset;
    if (moving && this.motionHop === null) this.motionHop = this.hopCount;

    return {
      leader: latched ? this.latchedIndex! : leader,
      commitment: finalCommitment,
      moving,
      reversible: moving && !latched,
      latched,
      latchedClass: this.latchedIndex,
      timedOut: this.timedOutFlag,
      hops: this.hopCount,
      effectiveThreshold: threshold,
      evidence: Array.from(this.evidence),
    };
  }

  /**
   * Leaky competing accumulation.
   *
   * Each class accrues the log ratio of its own probability to that of its
   * strongest rival, so evidence measures separation from the best alternative
   * rather than absolute probability. Floored at zero: a class that fell behind
   * should be able to recover on the next few hops rather than having to climb
   * out of an arbitrarily deep hole, which is what keeps switching responsive.
   */
  private accumulate(posteriors: ArrayLike<number>, total: number): void {
    const { epsilon, leak } = this.config;
    const n = this.evidence.length;

    const normalised = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      normalised[i] = Math.max(epsilon, posteriors[i]! / total);
    }

    // Largest and second largest, so each class can be scored against its
    // strongest rival in one pass rather than n passes.
    let best = 0;
    let second = 1;
    if (normalised[1]! > normalised[0]!) {
      best = 1;
      second = 0;
    }
    for (let i = 2; i < n; i++) {
      if (normalised[i]! > normalised[best]!) {
        second = best;
        best = i;
      } else if (normalised[i]! > normalised[second]!) {
        second = i;
      }
    }

    for (let i = 0; i < n; i++) {
      const rival = i === best ? normalised[second]! : normalised[best]!;
      const logRatio = Math.log(normalised[i]!) - Math.log(rival);
      this.evidence[i] = Math.max(0, this.evidence[i]! * (1 - leak) + logRatio);
    }
  }

  private argmaxEvidence(): number {
    let best = 0;
    for (let i = 1; i < this.evidence.length; i++) {
      if (this.evidence[i]! > this.evidence[best]!) best = i;
    }
    return best;
  }
}
