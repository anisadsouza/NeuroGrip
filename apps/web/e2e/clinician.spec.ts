import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('Research review exposes Evaluate, Robustness, Latency, and Records reports', async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto('/');
  await page.getByRole('button', { name: 'Research review' }).click();
  await expect(page.getByRole('heading', { name: 'Research review' })).toBeVisible();
  await expect(page.locator('.clinician-header .notice')).toContainText('No patient records');

  const audit = async () => {
    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze();
    expect(results.violations.map(({ id }) => id), JSON.stringify(results.violations, null, 2))
      .toEqual([]);
  };

  await page.getByRole('button', { name: 'Evaluate' }).click();
  await expect(page.getByRole('heading', { name: 'Recorded-data results' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'S1 → S10' })).toBeVisible();
  await audit();

  await page.getByRole('button', { name: 'Robustness' }).click();
  await expect(page.getByRole('img', { name: /Simulated accuracy by electrode shift/ }))
    .toBeVisible();
  await expect(page.getByText('Bound not met')).toBeVisible();
  await expect(page.getByRole('table', {
    name: 'Per-shift simulated LOSO accuracy and interval',
  })).toContainText('0.4 rad');
  await audit();

  await page.getByRole('button', { name: 'Latency' }).click();
  await expect(page.getByRole('heading', { name: 'Browser latency' })).toBeVisible();
  for (const name of ['Recorded NinaPro replay · S10', 'Simulated replay']) {
    const region = page.getByRole('region', { name });
    const metric = region.locator('dl > div')
      .filter({ has: page.getByText('Worker P95', { exact: true }) });
    await expect(metric.locator('dd')).toHaveText(/^\d+\.\d+ ms$/);
  }
  await audit();

  await page.getByRole('button', { name: 'Records' }).click();
  await expect(page.getByRole('region', { name: 'Held-out study record' }))
    .toContainText('NinaPro DB2');
  await expect(page.getByRole('region', { name: 'Held-out study record' }))
    .toContainText('S10');
  await audit();

  await page.setViewportSize({ width: 360, height: 900 });
  const dimensions = await page.evaluate(() => ({
    pageWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.pageWidth).toBeLessThanOrEqual(dimensions.viewportWidth + 1);
  expect(pageErrors).toEqual([]);
});
