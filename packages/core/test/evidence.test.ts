import { describe, expect, it } from 'vitest';
import { DEFAULT_EVIDENCE_CONFIG, EvidenceAccumulator } from '../src/evidence.js';

const N = 4;

/** Posteriors concentrated on `winner`, the rest sharing the remainder. */
function peaked(winner: number, mass = 0.9, n = N): number[] {
  const rest = (1 - mass) / (n - 1);
  return Array.from({ length: n }, (_, i) => (i === winner ? mass : rest));
}

const uniform = (n = N) => Array.from({ length: n }, () => 1 / n);

/** Feed the same posteriors until latched; return hops taken, or null. */
function hopsToLatch(acc: EvidenceAccumulator, posteriors: number[], limit = 500) {
  for (let hop = 1; hop <= limit; hop++) {
    const decision = acc.update(posteriors);
    if (decision.latched) return hop;
  }
  return null;
}

describe('EvidenceAccumulator: starting state', () => {
  it('starts with no evidence, no motion and nothing latched', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    const decision = acc.update(uniform());
    expect(decision.commitment).toBeCloseTo(0, 6);
    expect(decision.moving).toBe(false);
    expect(decision.latched).toBe(false);
    expect(decision.latchedClass).toBeNull();
    expect(decision.hops).toBe(1);
  });

  it('rejects a posterior vector of the wrong length', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    expect(() => acc.update([0.5, 0.5])).toThrow();
  });

  it('requires at least two classes', () => {
    expect(() => new EvidenceAccumulator([0])).toThrow();
  });
});

describe('EvidenceAccumulator: accumulation', () => {
  it('raises commitment monotonically under consistent evidence', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    let previous = -1;
    for (let hop = 0; hop < 8; hop++) {
      const { commitment, latched } = acc.update(peaked(2));
      if (latched) break;
      expect(commitment).toBeGreaterThanOrEqual(previous);
      previous = commitment;
    }
    expect(previous).toBeGreaterThan(0);
  });

  it('identifies the correct leader', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    acc.update(peaked(3));
    expect(acc.update(peaked(3)).leader).toBe(3);
  });

  it('latches once commitment reaches one', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    const hops = hopsToLatch(acc, peaked(1));
    expect(hops).not.toBeNull();
    const after = acc.update(peaked(1));
    expect(after.latched).toBe(true);
    expect(after.latchedClass).toBe(1);
    expect(after.commitment).toBe(1);
  });

  it('keeps commitment near zero on ambiguous input', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    for (let hop = 0; hop < 20; hop++) {
      const { commitment, latched } = acc.update(uniform());
      expect(latched).toBe(false);
      expect(commitment).toBeLessThan(0.05);
    }
  });
});

describe('EvidenceAccumulator: risk-weighted asymmetry', () => {
  it('makes a high-cost gesture require more evidence than a cheap one', () => {
    // This is the mechanism's central claim: identical evidence, different
    // commitment cost, therefore different time to commit.
    const cheap = new EvidenceAccumulator([0, 0, 0, 0]);
    const costly = new EvidenceAccumulator([0, 1, 0, 0]);

    const cheapHops = hopsToLatch(cheap, peaked(1));
    const costlyHops = hopsToLatch(costly, peaked(1));

    expect(cheapHops).not.toBeNull();
    expect(costlyHops).not.toBeNull();
    expect(costlyHops!).toBeGreaterThan(cheapHops!);
  });

  it('raises the effective threshold in proportion to risk', () => {
    const cheap = new EvidenceAccumulator([0, 0, 0, 0]);
    const costly = new EvidenceAccumulator([0, 2, 0, 0]);
    const a = cheap.update(peaked(1));
    const b = costly.update(peaked(1));
    expect(b.effectiveThreshold).toBeGreaterThan(a.effectiveThreshold);
    expect(b.effectiveThreshold).toBeCloseTo(
      DEFAULT_EVIDENCE_CONFIG.baseThreshold + 2 * DEFAULT_EVIDENCE_CONFIG.riskWeight,
      6,
    );
  });

  it('does not let risk affect a gesture that is not leading', () => {
    const acc = new EvidenceAccumulator([0, 5, 0, 0]);
    const decision = acc.update(peaked(2));
    expect(decision.leader).toBe(2);
    expect(decision.effectiveThreshold).toBeCloseTo(
      DEFAULT_EVIDENCE_CONFIG.baseThreshold,
      6,
    );
  });

  it('rejects negative risk', () => {
    expect(() => new EvidenceAccumulator([0, -1, 0, 0])).toThrow();
  });
});

describe('EvidenceAccumulator: reversibility', () => {
  it('begins motion before it commits', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    let sawReversibleMotion = false;
    for (let hop = 0; hop < 200; hop++) {
      const decision = acc.update(peaked(2));
      if (decision.moving && !decision.latched) sawReversibleMotion = true;
      if (decision.latched) break;
    }
    expect(sawReversibleMotion).toBe(true);
  });

  it('retracts when the evidence flips before commitment', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    for (let hop = 0; hop < 4; hop++) acc.update(peaked(2));
    const peakEvidence = acc.evidenceFor(2);
    expect(peakEvidence).toBeGreaterThan(0);
    expect(acc.update(peaked(2)).latched).toBe(false);

    for (let hop = 0; hop < 4; hop++) acc.update(peaked(0));
    // The abandoned gesture gives its evidence back rather than holding it.
    expect(acc.evidenceFor(2)).toBeLessThan(peakEvidence);
  });

  it('opens a usable window between first motion and commitment', () => {
    // The mechanism's whole claim: motion starts well before the grip latches,
    // leaving time to change your mind. A two-hop commit would have none.
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    let motionHop: number | null = null;
    let latchHop: number | null = null;
    for (let hop = 1; hop <= 200; hop++) {
      const decision = acc.update(peaked(1));
      if (decision.moving && motionHop === null) motionHop = hop;
      if (decision.latched) {
        latchHop = hop;
        break;
      }
    }
    expect(motionHop).not.toBeNull();
    expect(latchHop).not.toBeNull();
    expect(latchHop! - motionHop!).toBeGreaterThanOrEqual(5);
  });

  it('never reports negative evidence', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    for (let hop = 0; hop < 30; hop++) acc.update(peaked(0));
    for (let index = 0; index < N; index++) {
      expect(acc.evidenceFor(index)).toBeGreaterThanOrEqual(0);
    }
  });

  it('stays latched once committed, even against contrary evidence', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    hopsToLatch(acc, peaked(1));
    for (let hop = 0; hop < 20; hop++) {
      const decision = acc.update(peaked(3));
      expect(decision.latched).toBe(true);
      expect(decision.latchedClass).toBe(1);
    }
  });

  it('clears everything on reset', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    hopsToLatch(acc, peaked(1));
    acc.reset();
    const decision = acc.update(uniform());
    expect(decision.latched).toBe(false);
    expect(decision.latchedClass).toBeNull();
    expect(decision.hops).toBe(1);
    expect(decision.commitment).toBeCloseTo(0, 6);
  });
});

describe('EvidenceAccumulator: urgency and the safe fallback', () => {
  it('collapses the threshold as hops accumulate', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0], {
      urgencyOnsetHops: 2,
      maxHops: 40,
    });
    const early = acc.update(uniform()).effectiveThreshold;
    for (let hop = 0; hop < 20; hop++) acc.update(uniform());
    const late = acc.update(uniform()).effectiveThreshold;
    expect(late).toBeLessThan(early);
  });

  it('never collapses the threshold below the configured floor', () => {
    const config = { urgencyOnsetHops: 1, maxHops: 10, urgencyFloor: 0.4 };
    const acc = new EvidenceAccumulator([0, 0, 0, 0], config);
    let lowest = Infinity;
    for (let hop = 0; hop < 10; hop++) {
      lowest = Math.min(lowest, acc.update(uniform()).effectiveThreshold);
    }
    expect(lowest).toBeGreaterThanOrEqual(
      DEFAULT_EVIDENCE_CONFIG.baseThreshold * 0.4 - 1e-9,
    );
  });

  it('falls back to the safe class rather than forcing a guess', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0], {
      maxHops: 12,
      safeClassIndex: 0,
    });
    let decision = acc.update(uniform());
    for (let hop = 0; hop < 20 && !decision.timedOut; hop++) {
      decision = acc.update(uniform());
    }
    expect(decision.timedOut).toBe(true);
    expect(decision.latchedClass).toBe(0);
    expect(decision.latched).toBe(true);
  });

  it('does not time out when evidence commits in time', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0], { maxHops: 500 });
    hopsToLatch(acc, peaked(1));
    expect(acc.update(peaked(1)).timedOut).toBe(false);
  });
});

describe('EvidenceAccumulator: numerical safety and metrics', () => {
  it('survives zero probabilities without producing NaN or Infinity', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    for (let hop = 0; hop < 10; hop++) {
      const decision = acc.update([1, 0, 0, 0]);
      expect(Number.isFinite(decision.commitment)).toBe(true);
      expect(Number.isFinite(decision.effectiveThreshold)).toBe(true);
      for (const value of decision.evidence) expect(Number.isFinite(value)).toBe(true);
    }
  });

  it('normalises posteriors that do not sum to one', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    const scaled = new EvidenceAccumulator([0, 0, 0, 0]);
    for (let hop = 0; hop < 5; hop++) {
      acc.update(peaked(1));
      scaled.update(peaked(1).map((p) => p * 7));
    }
    expect(scaled.evidenceFor(1)).toBeCloseTo(acc.evidenceFor(1), 8);
  });

  it('reports hops-to-motion, the time-to-useful-motion metric', () => {
    const acc = new EvidenceAccumulator([0, 0, 0, 0]);
    expect(acc.hopsToMotion).toBeNull();
    let latchHop = 0;
    for (let hop = 1; hop <= 200; hop++) {
      const decision = acc.update(peaked(1));
      if (decision.latched) {
        latchHop = hop;
        break;
      }
    }
    expect(acc.hopsToMotion).not.toBeNull();
    expect(acc.hopsToMotion!).toBeLessThan(latchHop);
  });

  it('is deterministic', () => {
    const a = new EvidenceAccumulator([0, 1, 0, 0]);
    const b = new EvidenceAccumulator([0, 1, 0, 0]);
    for (let hop = 0; hop < 15; hop++) {
      expect(a.update(peaked(1)).commitment).toBe(b.update(peaked(1)).commitment);
    }
  });
});
