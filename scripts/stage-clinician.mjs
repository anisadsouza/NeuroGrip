import { copyFile, mkdir, stat } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const source = resolve(process.argv[2] ?? 'artifacts/electrode_shift_robustness.json');
const target = resolve(process.argv[3] ?? 'apps/web/public/clinician/robustness.json');

const report = JSON.parse(await (await import('node:fs/promises')).readFile(source, 'utf8'));
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
