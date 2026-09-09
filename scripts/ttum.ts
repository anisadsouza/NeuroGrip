/**
 * Replay the corpus through the real evidence accumulator and report TTUM.
 *
 * Python writes what the decoder believed, window by window, out of fold. This
 * reads those posteriors and feeds them through `EvidenceAccumulator` -- the
 * same class the browser worker runs, not a copy of it. That is the whole
 * reason this step is TypeScript: a Python reimplementation would be a second
 * accumulator owing a second conformance gate, which is exactly the trap
 * CLAUDE.md keeps the simulator out of the browser to avoid.
 *
 *     python -m neurogrip.experiments compare --out artifacts/model_comparison.json \
 *         --posteriors artifacts/posteriors
 *     npm run ttum
 *
 * Run with vite-node, as the design reference sheet is.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_EVIDENCE_CONFIG } from '../packages/core/src/evidence.js';
import { type TtumOutcome, runTrial, summariseTtum } from '../packages/core/src/ttum.js';
import { risksFor } from '../apps/web/src/decode/commitCost.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

interface PosteriorManifest {
  formatVersion: number;
  featureSpecVersion: number;
  model: string;
  outOfFold: boolean;
  gestures: string[];
  nWindows: number;
  nClasses: number;
  windowMs: number;
  hopMs: number;
  trials: { subject: number; label: number; rep: number; start: number; length: number }[];
  source: string;
  caveat: string;
}

function main(): void {
  const stem = join(ROOT, 'artifacts', 'posteriors');
  let manifest: PosteriorManifest;
  try {
    manifest = JSON.parse(readFileSync(`${stem}.json`, 'utf-8')) as PosteriorManifest;
  } catch {
    console.error(
      'artifacts/posteriors.json not found. Generate it first:\n' +
        '  python -m neurogrip.experiments compare --out artifacts/model_comparison.json \\\n' +
        '      --posteriors artifacts/posteriors',
    );
    process.exit(1);
    return;
  }

  const bytes = readFileSync(`${stem}.bin`);
  const flat = new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const expected = manifest.nWindows * manifest.nClasses;
  if (flat.length !== expected) {
    throw new Error(`posteriors.bin holds ${flat.length} values, manifest describes ${expected}`);
  }

  // One risk vector, shared with the browser by import rather than by being
  // written down twice.
  const risks = risksFor(manifest.gestures);

  const outcomes: TtumOutcome[] = [];
  const labels: number[] = [];
  for (const trial of manifest.trials) {
    const posteriors: number[][] = [];
    for (let hop = 0; hop < trial.length; hop++) {
      const offset = (trial.start + hop) * manifest.nClasses;
      posteriors.push(Array.from(flat.subarray(offset, offset + manifest.nClasses)));
    }
    outcomes.push(runTrial({ posteriors, trueClass: trial.label }, risks));
    labels.push(trial.label);
  }

  const summary = summariseTtum(outcomes, labels, manifest.gestures, risks, manifest.hopMs);
  const hopsPerTrial = manifest.trials[0]?.length ?? 0;

  const payload = {
    formatVersion: 1,
    model: manifest.model,
    source: manifest.source,
    outOfFold: manifest.outOfFold,
    windowMs: manifest.windowMs,
    hopsPerTrial,
    evidenceConfig: DEFAULT_EVIDENCE_CONFIG,
    risks: Object.fromEntries(manifest.gestures.map((name, i) => [name, risks[i]!])),
    ...summary,
    definition:
      'Time to Useful Motion: hops from a trial first decodable window to the ' +
      'first hop at which the leading gesture reaches motionOnset of its ' +
      'risk-weighted boundary, reported in milliseconds at the stated hop. ' +
      '"Correct motion" additionally requires the leader to be the trial true ' +
      'class. Trials that never reach the onset are censored: counted, never ' +
      'assigned a finite value, never dropped. No mean is reported, because ' +
      'under censoring a mean is either wrong or an unstated imputation. ' +
      'The 200 ms of window fill before the first decision is excluded and is ' +
      'additive: a wearer experiences roughly TTUM plus 200 ms.',
    reachability:
      `Each trial is ${hopsPerTrial} hops, and the urgency timeout fires at ` +
      `${DEFAULT_EVIDENCE_CONFIG.maxHops}, so within a trial the timeout is ` +
      'structurally unreachable and censoring is the only failure mode. ' +
      'Lengthening the repetition past about 1.5 s would activate the timeout ' +
      'path and change what this metric measures.',
    caveat: manifest.caveat,
  };

  const out = join(ROOT, 'artifacts', 'ttum.json');
  writeFileSync(out, `${JSON.stringify(payload, null, 2)}\n`, 'utf-8');

  console.log(`TTUM over ${summary.nTrials} out-of-fold trials (${manifest.model})\n`);
  console.log(
    `  any motion      p50 ${summary.anyMotion.p50Ms} ms   ` +
      `p95 ${summary.anyMotion.p95Ms} ms   censored ${summary.anyMotion.nCensored}`,
  );
  console.log(
    `  correct motion  p50 ${summary.correctMotion.p50Ms} ms   ` +
      `p95 ${summary.correctMotion.p95Ms} ms   censored ${summary.correctMotion.nCensored}`,
  );
  console.log(
    `  latch           p50 ${summary.latch.p50Ms} ms   ` +
      `accuracy ${summary.latch.accuracyAtLatch?.toFixed(4) ?? '-'}   ` +
      `timed out ${summary.latch.nTimedOut}`,
  );
  console.log(`\nwritten to ${out}`);
}

main();
