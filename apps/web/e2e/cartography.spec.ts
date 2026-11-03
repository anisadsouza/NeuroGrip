import { expect, test } from '@playwright/test';

const demoReport = {
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
      subjects: [{
        subject: 'carto00',
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
  await page.goto('/');
  await page.getByRole('button', { name: 'Cartography' }).click();

  await expect(page.getByRole('heading', { name: 'Neuromotor Cartography' })).toBeVisible();
  await expect(page.locator('.cartography-screen > .notice[role="note"]'))
    .toContainText('synthetic subjects');
  await expect(page.getByRole('img', { name: /Simulated reachable activation map/ })).toBeVisible();
  const prototypeTable = page.getByRole('table', {
    name: 'Per-prototype held-out repetition gate and illustrative function assignment',
  });
  await expect(prototypeTable).toContainText('90.0% (≥80%)');
  await expect(prototypeTable).toContainText('Pass');
  await expect(page.locator('.cartography-cohorts table')).toContainText('2/8');
  await expect(page.locator('.cartography-cohorts table')).toContainText('5/8');

  await page.setViewportSize({ width: 360, height: 900 });
  const dimensions = await page.evaluate(() => ({
    pageWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.pageWidth).toBeLessThanOrEqual(dimensions.viewportWidth + 1);
});
