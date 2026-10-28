import { existsSync, readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const HELDOUT_REPLAY = join(ROOT, 'apps', 'web', 'public', 'replay', 'ninapro');
const HELDOUT_MODEL = join(ROOT, 'apps', 'web', 'public', 'models', 'ninapro');
const HAS_HELDOUT = existsSync(join(HELDOUT_REPLAY, 'emg-replay.json')) &&
  existsSync(join(HELDOUT_MODEL, 'decoder.onnx'));

test('Results prioritizes held-out NinaPro metrics and keeps diagnostics optional', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Results' }).click();

  await expect(page.getByRole('heading', { name: 'Recorded-data results' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'S1 → S10' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Most common wrong predictions' })).toBeVisible();

  if (HAS_HELDOUT) {
    const evaluation = JSON.parse(
      readFileSync(join(ROOT, 'apps', 'web', 'dist', 'results', 'ninapro_holdout.json'), 'utf8'),
    ) as {
      accuracy: number;
      confusion_matrix: number[][];
      expected_calibration_error: number;
      gesture_order: string[];
      dataset_movements: Record<string, string>;
      n_test: number;
      timeline: unknown[];
      latency: {
        p95_ms: number;
      };
      quality_gate: {
        accepted: number;
        rejected: number;
        acceptance_rate: number;
        top1_accuracy_on_accepted: number | null;
        rejected_by_performed_gesture: Record<string, number>;
      };
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
      stablePredictionWindows: number;
      correctStablePredictions: number;
      stablePredictionAccuracy: number | null;
      zeroSignalRejected: boolean;
    };
    const heldoutPanel = page.locator('section[aria-labelledby="heldout-heading"]');
    await expect(page.getByText('Model comparison and simulator results')).toBeVisible();
    await expect(page.getByText('View confusion matrix')).toBeVisible();
    await expect(page.getByText('View performed-versus-decoded timeline')).toBeVisible();
    await expect(heldoutPanel.locator('details[open]')).toHaveCount(0);
    await expect(heldoutPanel.locator('.results-metrics > div')).toHaveCount(5);
    await expect(heldoutPanel.locator('.confusion-matrix')).not.toBeVisible();
    await expect(heldoutPanel).toContainText(`${(evaluation.accuracy * 100).toFixed(1)}%`);
    await expect(heldoutPanel).toContainText(
      evaluation.expected_calibration_error.toFixed(3),
    );
    await expect(heldoutPanel).toContainText(
      `${(evaluation.quality_gate.acceptance_rate * 100).toFixed(1)}%`,
    );
    await expect(heldoutPanel).toContainText(
      `${evaluation.quality_gate.accepted.toLocaleString()} accepted · ` +
        `${evaluation.quality_gate.rejected.toLocaleString()} rejected`,
    );
    await expect(heldoutPanel).toContainText(`${evaluation.n_test.toLocaleString()} windows`);
    await page.getByText('View confusion matrix').click();
    await expect(heldoutPanel.locator('.confusion-matrix tbody tr'))
      .toHaveCount(evaluation.gesture_order.length);
    await expect(page.getByText('Held-out confusion matrix', { exact: false })).toBeVisible();
    for (const [rowIndex, expectedRow] of evaluation.confusion_matrix.entries()) {
      const cells = heldoutPanel.locator('.confusion-matrix tbody tr').nth(rowIndex).locator('td');
      await expect(cells).toHaveCount(expectedRow.length);
      for (const [cellIndex, value] of expectedRow.entries()) {
        await expect(cells.nth(cellIndex)).toHaveText(String(value));
      }
    }
    await page.getByText('View performed-versus-decoded timeline').click();
    await expect(heldoutPanel.locator('.performed-timeline li')).toHaveCount(
      evaluation.timeline.length,
    );
    await page.getByText('Evaluation and quality-gate notes').click();
    await expect(heldoutPanel).toContainText(
      `CPU ONNX inference P95: ${evaluation.latency.p95_ms.toFixed(2)} ms`,
    );
    const rejectedLabelSummary = Object.entries(evaluation.quality_gate.rejected_by_performed_gesture)
      .filter(([, count]) => count > 0)
      .map(([gesture, count]) => `${evaluation.dataset_movements[gesture] ?? gesture}: ${count}`)
      .join(', ');
    await expect(heldoutPanel).toContainText(`Rejected performed labels: ${rejectedLabelSummary}.`);
    await page.getByText('Model comparison and simulator results').click();
    await expect(page.getByRole('heading', { name: 'Simulated sEMG model comparison' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Recorded NinaPro DB2 model comparison' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'NinaPro browser diagnostics' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Simulator browser diagnostics' })).toBeVisible();
    const browserLatencyPanel = page.getByRole('region', { name: 'NinaPro browser diagnostics' });
    await expect(browserLatencyPanel).toContainText(`${latency.p95Ms.toFixed(3)} ms`);
    if (typeof latency.zeroSignalRejected === 'boolean') {
      await expect(browserLatencyPanel).toContainText(
        `Zero-signal control${latency.zeroSignalRejected ? 'rejected' : 'not rejected'}`,
      );
    }
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
  await page.getByText('View confusion matrix').click();

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
  await expect(page.getByText('Model probability')).toBeVisible();
  await expect(
    page.getByText('Model probability').locator('..').locator('dd'),
  ).toHaveText(/\d+\.\d+%/);
  await expect(
    page.getByText('Virtual actuator').locator('..').locator('dd'),
  ).not.toHaveText('—');
  await expect(page.getByText('Virtual output only. No physical actuator or motor is connected.'))
    .toBeVisible();
  expect(failures, failures.join('\n')).toEqual([]);
});
