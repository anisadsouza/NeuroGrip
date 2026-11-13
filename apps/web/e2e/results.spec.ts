import { existsSync, readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const HELDOUT_REPLAY = join(ROOT, 'apps', 'web', 'public', 'replay', 'ninapro');
const HELDOUT_MODEL = join(ROOT, 'apps', 'web', 'public', 'models', 'ninapro');
const HAS_HELDOUT = existsSync(join(HELDOUT_REPLAY, 'emg-replay.json')) &&
  existsSync(join(HELDOUT_MODEL, 'decoder.onnx'));
const HAS_HELDOUT_RESULT = existsSync(join(ROOT, 'apps', 'web', 'public', 'results', 'ninapro_holdout.json'));

test('Accuracy opens on real data when present, else on the model comparison', async ({ page }) => {
  await page.goto('/#results');
  await expect(page.getByRole('heading', { name: 'Accuracy', exact: true })).toBeVisible();

  if (HAS_HELDOUT_RESULT) {
    const tabs = page.getByRole('tablist', { name: 'Accuracy pages' });
    await expect(page.getByRole('heading', { name: /S\d.* → S\d+/ })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Most common wrong predictions' })).toBeVisible();
    const evaluation = JSON.parse(
      readFileSync(join(ROOT, 'apps', 'web', 'public', 'results', 'ninapro_holdout.json'), 'utf8'),
    ) as { accuracy: number; gesture_order: string[] };
    await expect(page.locator('section[aria-labelledby="heldout-heading"]'))
      .toContainText(`${(evaluation.accuracy * 100).toFixed(1)}%`);
    await tabs.getByRole('tab', { name: 'Mistakes' }).click();
    await expect(page.locator('.confusion-matrix tbody tr')).toHaveCount(evaluation.gesture_order.length);
    await tabs.getByRole('tab', { name: 'Model comparison' }).click();
  } else {
    // Without the licensed recordings there is one page and no empty card:
    // the comparison's method note says where recorded results would appear.
    await expect(page.getByRole('tablist')).toHaveCount(0);
  }

  const simulated = page.getByRole('region', { name: 'Simulated sEMG model comparison' });
  await expect(simulated).toContainText('RBF SVM');
  await expect(simulated.locator('tbody tr')).toHaveCount(3);
  await simulated.getByRole('button', { name: 'Method and limits' }).click();
  await expect(page.getByRole('dialog', { name: 'Simulated sEMG model comparison' }))
    .toContainText('simulated sEMG');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);
});

test('Results keeps the held-out confusion matrix inside a scrollable panel on mobile', async ({
  page,
}) => {
  test.skip(!HAS_HELDOUT_RESULT, 'No held-out evaluation staged.');
  await page.setViewportSize({ width: 360, height: 900 });
  await page.goto('/#results');
  await page.getByRole('tab', { name: 'Mistakes' }).click();

  const dimensions = await page.evaluate(() => {
    const matrix = document.querySelector('.matrix-scroll');
    return {
      pageWidth: document.documentElement.scrollWidth,
      viewportWidth: document.documentElement.clientWidth,
      matrixWidth: matrix?.clientWidth ?? 0,
      matrixContentWidth: matrix?.scrollWidth ?? 0,
    };
  });
  expect(dimensions.pageWidth).toBeLessThanOrEqual(dimensions.viewportWidth + 1);
  expect(dimensions.matrixContentWidth).toBeGreaterThan(dimensions.matrixWidth);
});

test('Live loads the recorded model only when held-out assets are staged', async ({ page }) => {
  test.skip(!HAS_HELDOUT, 'No held-out NinaPro model/replay staged.');
  const failures: string[] = [];
  page.on('pageerror', (error) => failures.push(error.message));

  await page.goto('/#live');
  await page.getByLabel('Choose the signal').selectOption('ninapro');
  await expect(page.getByText(/Recorded · NinaPro DB2/)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole('button', { name: 'Start decoding' })).toBeEnabled();
  await page.getByRole('button', { name: 'Start decoding' }).click();
  await expect(
    page.locator('.live-decoder dl > div').filter({ hasText: 'Posterior' }).locator('dd'),
  ).toHaveText(/\d+\.\d+%/, { timeout: 15_000 });
  expect(failures, failures.join('\n')).toEqual([]);
});

test('Live offers the recorded person only when it is installed, and never strands the user', async ({
  page,
}) => {
  test.skip(HAS_HELDOUT, 'Recorded assets are staged here; the missing-asset path is not reachable.');
  await page.goto('/#live');
  const option = page.getByRole('option', { name: /Recorded person/ });
  await expect(option).toHaveText(/not in this build/);
  await expect(option).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Start decoding' })).toBeEnabled({ timeout: 30_000 });
});
