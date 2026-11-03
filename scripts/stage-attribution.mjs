import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const source = resolve(process.argv[2] ?? 'artifacts/attribution_demo.json');
const target = resolve(process.argv[3] ?? 'apps/web/public/attribution/demo.json');
const report = JSON.parse(await readFile(source, 'utf8'));

const expectedCauses = new Set([
  'physiological',
  'behavioral',
  'model_capacity',
  'drift',
]);
if (
  report.source !== 'synthetic_fault_injections' ||
  !Array.isArray(report.cases) ||
  report.cases.length !== expectedCauses.size ||
  report.cases.some(
    (item) =>
      !expectedCauses.has(item.injected_cause) ||
      !expectedCauses.has(item.cause) ||
      !Array.isArray(item.channel_attribution) ||
      !item.scores ||
      Math.abs(Object.values(item.scores).reduce((sum, score) => sum + score, 0) - 1) > 0.002,
  )
) {
  throw new Error(`Attribution demo report has an unexpected shape: ${source}`);
}

await mkdir(dirname(target), { recursive: true });
await writeFile(target, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(`staged synthetic attribution demo to ${target}`);
