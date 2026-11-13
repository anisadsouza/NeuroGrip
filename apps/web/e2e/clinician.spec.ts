import AxeBuilder from '@axe-core/playwright';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const RESULTS = join(ROOT, 'apps', 'web', 'public', 'results');
const HAS_RECORDED_LATENCY = existsSync(join(RESULTS, 'browser_latency_ninapro.json'));
const HAS_HELDOUT = existsSync(join(RESULTS, 'ninapro_holdout.json'));

test('Step 6 pages through timing, robustness, latency, records and limits', async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto('/');
  await page.getByRole('navigation', { name: 'Guided path' })
    .getByRole('button', { name: 'Step 6: Speed, safety and limits' }).click();
  await expect(page.getByRole('heading', { name: 'Speed, safety and limits', exact: true }))
    .toBeVisible();
  const tabs = page.getByRole('tablist', { name: 'Speed, safety and limits pages' });

  const audit = async () => {
    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze();
    expect(results.violations.map(({ id }) => id), JSON.stringify(results.violations, null, 2))
      .toEqual([]);
  };

  await expect(page.getByRole('heading', { name: 'Decision timing and cost-weighted error' }))
    .toBeVisible();
  const cwer = page.getByRole('region', { name: 'Cost-weighted error rate' });
  await expect(cwer).toContainText('Cost-weighted error rate');
  const weighted = cwer.locator('dl > div')
    .filter({ has: page.getByText('CWER, risk-weighted boundaries', { exact: true }) });
  await expect(weighted.locator('dd')).toHaveText(/^\d+\.\d{2}%$/);
  await expect(page.getByRole('region', { name: 'Time to useful motion' }))
    .toContainText('lower bound');
  await audit();

  await tabs.getByRole('tab', { name: 'Robustness' }).click();
  await expect(page.getByRole('img', { name: /Simulated accuracy by electrode shift/ }))
    .toBeVisible();
  await expect(page.getByText('Bound not met')).toBeVisible();
  await expect(page.getByRole('table', {
    name: 'Per-shift simulated LOSO accuracy and interval',
  })).toContainText('0.4 rad');
  await audit();

  await tabs.getByRole('tab', { name: 'Latency' }).click();
  await expect(page.getByRole('heading', { name: 'Browser latency' })).toBeVisible();
  const sources = HAS_RECORDED_LATENCY
    ? ['Recorded NinaPro replay · S10', 'Simulated replay']
    : ['Simulated replay'];
  for (const name of sources) {
    const region = page.getByRole('region', { name });
    const metric = region.locator('dl > div').first();
    await expect(metric.locator('dd')).toHaveText(/^\d+\.\d+ ms$/);
  }
  await audit();

  // With no held-out record staged there is nothing to show, so the page is
  // left out rather than offered and then found empty (design rule 4).
  const records = tabs.getByRole('tab', { name: 'Records' });
  if (HAS_HELDOUT) {
    await records.click();
    await expect(page.getByRole('region', { name: 'Held-out study record' }))
      .toContainText('NinaPro DB2');
    await expect(page.getByRole('region', { name: 'Held-out study record' }))
      .toContainText('S10');
    await audit();
  } else {
    await expect(records).toHaveCount(0);
  }

  // The path promises limits; the last page states them, and the last step
  // offers no Next button because there is nowhere further to go.
  await tabs.getByRole('tab', { name: 'Limits' }).click();
  await expect(page.getByRole('heading', { name: 'Not yet shown' })).toBeVisible();
  await expect(page.locator('.screen-body')).toContainText('Not a medical device');
  await expect(page.getByRole('navigation', { name: 'Step through the path' })
    .getByRole('button', { name: /^Next/ })).toHaveCount(0);
  await audit();

  await page.setViewportSize({ width: 360, height: 900 });
  const dimensions = await page.evaluate(() => ({
    pageWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.pageWidth).toBeLessThanOrEqual(dimensions.viewportWidth + 1);
  expect(pageErrors).toEqual([]);
});
