import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const source = resolve(process.argv[2] ?? 'artifacts/cartography_gate.json');
const target = resolve(process.argv[3] ?? 'apps/web/public/cartography/demo.json');

const report = JSON.parse(await readFile(source, 'utf8'));
if (!report.criterion || !report.cohorts?.intact || !report.cohorts?.finger_atrophy) {
  throw new Error(`Cartography gate report has an unexpected shape: ${source}`);
}

const demoSubject = report.cohorts.intact.subjects?.[0];
if (
  !demoSubject?.map_points?.length ||
  !demoSubject?.prototype_comparisons?.length
) {
  throw new Error(`Cartography report has no map or per-prototype metrics: ${source}`);
}
if (
  !demoSubject.teach_back?.targets?.length ||
  demoSubject.teach_back.targets.some(
    (target) => target.trajectory_map.length !== target.attempts_made,
  ) ||
  !report.cohorts.intact.teach_back ||
  !report.cohorts.finger_atrophy.teach_back
) {
  throw new Error(`Cartography report has no teach-back results: ${source}`);
}

const staged = {
  criterion: report.criterion,
  cohorts: Object.fromEntries(
    Object.entries(report.cohorts).map(([name, cohort]) => [
      name,
      {
        wins: cohort.wins,
        n_subjects: cohort.n_subjects,
        weighted_selection: cohort.weighted_selection,
        teach_back: cohort.teach_back,
        subjects: [name === 'intact' ? demoSubject : cohort.subjects[0]],
      },
    ]),
  ),
  teach_back_definition: report.teach_back_definition,
  caveat: report.caveat,
};

await mkdir(dirname(target), { recursive: true });
await writeFile(target, `${JSON.stringify(staged, null, 2)}\n`, 'utf8');
console.log(`staged simulated Cartography demo to ${target}`);
