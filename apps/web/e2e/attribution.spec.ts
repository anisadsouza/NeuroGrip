import { expect, test } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test('Attribution routes injected causes and presents bounded channel coaching', async ({
  page,
}) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Attribution' }).click();

  await expect(page.getByRole('heading', { name: 'Error attribution and coaching' }))
    .toBeVisible();
  await expect(page.locator('.attribution-screen > .notice[role="note"]'))
    .toContainText('Synthetic fault injections only');
  await expect(page.locator('.attribution-case-picker button')).toHaveCount(4);
  await expect(page.locator('.attribution-screen > .notice[role="note"]'))
    .toContainText('4/4');

  await page.getByRole('button', { name: /behavioral deviation/i }).click();
  await expect(page.getByRole('heading', { name: 'Behavioral deviation' })).toBeVisible();
  await expect(page.locator('.attribution-route-check')).toHaveText('Expected route');
  await expect(page.getByRole('img', {
    name: 'Forearm electrode ring showing channels with the largest feature deviations',
  })).toBeVisible();
  await expect(page.locator('.attribution-coaching'))
    .toContainText('does not identify a specific muscle or finger');
  await expect(page.getByRole('meter')).toHaveCount(4);
  const accessibility = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  expect(accessibility.violations.map(({ id }) => id)).toEqual([]);

  await page.setViewportSize({ width: 360, height: 900 });
  const dimensions = await page.evaluate(() => ({
    pageWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.pageWidth).toBeLessThanOrEqual(dimensions.viewportWidth + 1);
});
