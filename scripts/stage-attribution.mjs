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

// The end-to-end sessions and the coaching localisation check are staged
// beside the controlled injections. Both are simulator-only.
const sessionSource = resolve(process.argv[4] ?? 'artifacts/router_session.json');
const sessionTarget = resolve(process.argv[5] ?? 'apps/web/public/attribution/session.json');
const session = JSON.parse(await readFile(sessionSource, 'utf8'));
if (
  session.source !== 'simulated_sessions' ||
  !Array.isArray(session.sessions) ||
  session.sessions.length !== expectedCauses.size ||
  !session.per_fault ||
  session.sessions.some(
    (item) =>
      !expectedCauses.has(item.injected_cause) ||
      !Number.isFinite(item.n_error_windows) ||
      Object.values(item.routed).reduce((sum, count) => sum + count, 0) !== item.n_error_windows,
  )
) {
  throw new Error(`Router session report has an unexpected shape: ${sessionSource}`);
}
await writeFile(sessionTarget, `${JSON.stringify(session, null, 2)}\n`, 'utf8');
console.log(`staged simulated router sessions to ${sessionTarget}`);

const coachingSource = resolve(process.argv[6] ?? 'artifacts/coaching_localisation.json');
const coachingTarget = resolve(process.argv[7] ?? 'apps/web/public/attribution/coaching.json');
const coaching = JSON.parse(await readFile(coachingSource, 'utf8'));
if (
  coaching.source !== 'simulator' ||
  !Number.isFinite(coaching.recovery_rate) ||
  coaching.n_recovered + coaching.n_wrong_group_or_direction + coaching.n_declined !==
    coaching.n_error_windows
) {
  throw new Error(`Coaching localisation report has an unexpected shape: ${coachingSource}`);
}
// The browser shows the pooled result and per-group table; per-window
// examples stay in the artifact.
const { cases: _cases, ...coachingSummary } = coaching;
await writeFile(coachingTarget, `${JSON.stringify(coachingSummary, null, 2)}\n`, 'utf8');
console.log(`staged simulated coaching localisation to ${coachingTarget}`);
