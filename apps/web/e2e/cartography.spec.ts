import { expect, test } from '@playwright/test';

const teachBack = {
  n_targets: 5,
  n_learned: 4,
  n_censored: 1,
  median_attempts_to_criterion: 13,
  median_recall_at_exact_target: 1,
  median_recall_after_teach_back: 1,
  n_passing_gate: 3,
  n_passing_gate_and_retained: 3,
};

const demoReport = {
  teach_back_definition: 'Simulated reward-driven learner.',
  criterion: {
    learnable_recall: 0.8,
    description: 'Simulated per-prototype gate.',
  },
  cohorts: {
    intact: {
      wins: 2,
      n_subjects: 8,
      weighted_selection: {
        separability_ratio: 0.9722,
        excitation_change: -0.0227,
        fatigue_change: -0.0133,
      },
      teach_back: teachBack,
      subjects: [{
        subject: 'carto00',
        teach_back: {
          config: { max_attempts: 60, window: 5, hits: 3 },
          targets: [{
            prototype: 'proto_1',
            attempts_to_criterion: 7,
            attempts_made: 7,
            distances: [0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.35],
            on_target: [false, false, true, false, true, true, true],
            trajectory_map: [[0.5, 0.5], [0.48, 0.5], [0.45, 0.5], [0.44, 0.5], [0.42, 0.5], [0.41, 0.5], [0.4, 0.5]],
            target_map: [0.4, 0.5],
            recall_at_exact_target: 0.9,
            recall_after_teach_back: 0.85,
            passes_gate: true,
          }],
        },
        map_points: [[0.2, 0.3], [0.7, 0.8]],
        pose_effort: [0.2, 0.6],
        prototypes_map: [[0.4, 0.5]],
        prototype_comparisons: [{
          prototype: 'proto_1',
          matched_catalogue_function: 'point',
          recall: 0.9,
          recall_threshold: 0.8,
          nearest_rival: 'rest',
          separation: 1.2,
          catalogue_nearest_rival: 'rest',
          catalogue_separation: 1.1,
          learnable: true,
          beats_catalogue_separation: true,
          passes: true,
        }],
      }],
    },
    finger_atrophy: {
      wins: 5,
      n_subjects: 8,
      weighted_selection: {
        separability_ratio: 0.9077,
        excitation_change: -0.0461,
        fatigue_change: -0.0265,
      },
      teach_back: { ...teachBack, n_learned: 3 },
      subjects: [],
    },
  },
  caveat: 'Simulated users only. Not a human result.',
};

test('Cartography presents the simulated map, per-prototype gate, and assignment', async ({
  page,
}) => {
  await page.route('**/cartography/demo.json', (route) =>
    route.fulfill({ json: demoReport }),
  );
  await page.goto('/#cartography');
  const tabs = page.getByRole('tablist', { name: 'Neuromotor Cartography pages' });

  await expect(page.getByRole('heading', { name: 'Neuromotor Cartography' })).toBeVisible();
  await expect(page.locator('main').getByRole('note')).toContainText('Simulated people');
  await expect(page.getByRole('img', { name: /Simulated reachable activation map/ })).toBeVisible();
  await expect(page.locator('.cartography-readout')).toContainText('1/1');

  await tabs.getByRole('tab', { name: 'Teach-back' }).click();
  const teach = page.getByRole('region', { name: 'Teach-back' });
  await expect(teach.getByText('Learned at attempt 7')).toBeVisible();
  await expect(teach).toContainText('85.0%');
  const slider = teach.getByRole('slider');
  await expect(slider).toHaveValue('7');
  await slider.focus();
  await page.keyboard.press('Home');
  await expect(teach.getByText('Attempt 1 of 7')).toBeVisible();
  await expect(teach.getByRole('img', { name: /Attempts 1 to 1 toward target 1/ })).toBeVisible();
  await teach.getByRole('textbox', { name: 'Name this target' }).fill('Flick');
  await expect(teach.getByRole('button', { name: /Flick/ })).toBeVisible();

  await tabs.getByRole('tab', { name: 'Targets' }).click();
  const prototypeTable = page.getByRole('table', {
    name: 'Per-prototype held-out repetition gate and illustrative function assignment',
  });
  await expect(prototypeTable).toContainText('90.0% (≥80%)');
  await expect(prototypeTable).toContainText('Yes');

  await tabs.getByRole('tab', { name: 'Cohorts' }).click();
  const cohorts = page.locator('.cartography-cohorts table');
  await expect(cohorts).toContainText('2/8');
  await expect(cohorts).toContainText('5/8');
  await expect(cohorts).toContainText('4/5 · median 13 attempts');
  await expect(cohorts).toContainText('Weak finger muscles');

  await page.setViewportSize({ width: 360, height: 900 });
  const dimensions = await page.evaluate(() => ({
    pageWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.pageWidth).toBeLessThanOrEqual(dimensions.viewportWidth + 1);
});
