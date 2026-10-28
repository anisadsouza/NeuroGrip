import { existsSync, readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const HELDOUT_REPLAY = join(ROOT, 'apps', 'web', 'public', 'replay', 'ninapro');
const HELDOUT_MODEL = join(ROOT, 'apps', 'web', 'public', 'models', 'ninapro');
const HAS_HELDOUT = existsSync(join(HELDOUT_REPLAY, 'emg-replay.json')) &&
  existsSync(join(HELDOUT_MODEL, 'decoder.onnx'));

test('Results keeps simulator and recorded evaluations in separate sections', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Results' }).click();

  await expect(page.getByRole('heading', { name: 'Simulated sEMG' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Recorded NinaPro DB2' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Subject-held-out NinaPro evaluation' }))
    .toBeVisible();

  if (HAS_HELDOUT) {
    const evaluation = JSON.parse(
      readFileSync(join(ROOT, 'apps', 'web', 'dist', 'results', 'ninapro_holdout.json'), 'utf8'),
    ) as {
      accuracy: number;
      confusion_matrix: number[][];
      expected_calibration_error: number;
      gesture_order: string[];
      n_test: number;
      timeline: unknown[];
    };
    const latency = JSON.parse(
      readFileSync(
        join(ROOT, 'apps', 'web', 'dist', 'results', 'browser_latency_ninapro.json'),
        'utf8',
      ),
    ) as {
      n: number;
      p95Ms: number;
      rejectedWindows: number;
    };
    const heldoutPanel = page.locator('section[aria-labelledby="heldout-heading"]');
    const browserLatencyPanel = page.locator(
      'section[aria-labelledby="held-out-ninapro-browser-decode-latency-heading"]',
    );

    await expect(page.getByRole('heading', { name: 'Performed vs decoded timeline' }))
      .toBeVisible();
    await expect(page.getByText('Held-out confusion matrix', { exact: false })).toBeVisible();
    await expect(page.getByText('CPU ONNX P95')).toBeVisible();
    await expect(page.getByText(/Finger abduction \(proxy/).first()).toBeVisible();
    await expect(heldoutPanel).toContainText(`${(evaluation.accuracy * 100).toFixed(1)}%`);
    await expect(heldoutPanel).toContainText(
      evaluation.expected_calibration_error.toFixed(3),
    );
    await expect(heldoutPanel).toContainText(`${evaluation.n_test.toLocaleString()} windows`);
    await expect(heldoutPanel.locator('.confusion-matrix tbody tr'))
      .toHaveCount(evaluation.gesture_order.length);
    await expect(heldoutPanel.locator('.performed-timeline li')).toHaveCount(
      evaluation.timeline.length,
    );
    for (const [rowIndex, expectedRow] of evaluation.confusion_matrix.entries()) {
      const cells = heldoutPanel.locator('.confusion-matrix tbody tr').nth(rowIndex).locator('td');
      await expect(cells).toHaveCount(expectedRow.length);
      for (const [cellIndex, value] of expectedRow.entries()) {
        await expect(cells.nth(cellIndex)).toHaveText(String(value));
      }
    }
    await expect(browserLatencyPanel).toContainText(`${latency.p95Ms.toFixed(3)} ms`);
    await expect(browserLatencyPanel).toContainText(
      `${latency.n} decoded · ${latency.rejectedWindows} rejected`,
    );
  } else {
    await expect(page.getByText(/No held-out export staged/)).toBeVisible();
  }
});

test('Results keeps the held-out confusion matrix inside a scrollable panel on mobile', async ({
  page,
}) => {
  test.skip(!HAS_HELDOUT, 'No held-out evaluation staged.');
  await page.setViewportSize({ width: 360, height: 900 });
  await page.goto('/');
  await page.getByRole('button', { name: 'Results' }).click();
  await expect(page.getByRole('heading', { name: 'Performed vs decoded timeline' })).toBeVisible();

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

  await page.goto('/');
  await page.getByLabel('Replay source').selectOption('ninapro');
  await expect(page.getByText(/Recorded · NinaPro DB2/)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole('button', { name: 'Start decoding' })).toBeEnabled();
  await expect(page.getByText(/per-channel rest calibration/i)).toBeVisible();
  await page.getByRole('button', { name: 'Start decoding' }).click();
  await expect(
    page.getByText('Latency, this window').locator('..').locator('dd'),
  ).toHaveText(/\d+\.\d+ ms/, { timeout: 15_000 });
  expect(failures, failures.join('\n')).toEqual([]);
});
