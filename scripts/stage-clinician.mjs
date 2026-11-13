import { copyFile, mkdir, readFile, stat } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const source = resolve(process.argv[2] ?? 'artifacts/electrode_shift_robustness.json');
const target = resolve(process.argv[3] ?? 'apps/web/public/clinician/robustness.json');

const report = JSON.parse(await readFile(source, 'utf8'));
if (
  !Array.isArray(report.shifts) ||
  report.shifts.length === 0 ||
  !report.corpus ||
  !report.acceptance
) {
  throw new Error(`Robustness report has an unexpected shape: ${source}`);
}

for (const point of report.shifts) {
  if (
    !Number.isFinite(point.shift_rad) ||
    !Number.isFinite(point.mean_accuracy) ||
    !Number.isFinite(point.degradation_pp) ||
    !Number.isFinite(point.ci95_low) ||
    !Number.isFinite(point.ci95_high)
  ) {
    throw new Error(`Robustness report contains an invalid shift result: ${source}`);
  }
}

await mkdir(dirname(target), { recursive: true });
await copyFile(source, target);
const { size } = await stat(target);
if (size === 0) throw new Error(`Staged robustness report is empty: ${target}`);
console.log(`staged simulated robustness report to ${target}`);

// Decision timing: TTUM and CWER from the simulator's out-of-fold posteriors.
const ttumSource = resolve(process.argv[4] ?? 'artifacts/ttum.json');
const ttumTarget = resolve(process.argv[5] ?? 'apps/web/public/clinician/ttum.json');
const ttum = JSON.parse(await readFile(ttumSource, 'utf8'));
if (
  ttum.source !== 'simulator' ||
  !ttum.anyMotion ||
  !ttum.latch ||
  !ttum.cwer?.riskWeighted ||
  !ttum.cwer?.uniformBoundary ||
  ttum.anyMotion.nMoved + ttum.anyMotion.nCensored !== ttum.nTrials
) {
  throw new Error(`TTUM report has an unexpected shape or lacks CWER: ${ttumSource}`);
}
await copyFile(ttumSource, ttumTarget);
console.log(`staged simulated TTUM and CWER report to ${ttumTarget}`);
