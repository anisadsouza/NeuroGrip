import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

async function audit(page: Page) {
  const accessibility = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  expect(accessibility.violations.map(({ id }) => id)).toEqual([]);
}

test('Attribution routes designed mistakes and presents bounded sensor feedback', async ({ page }) => {
  await page.goto('/#attribution');
  await expect(page.getByRole('heading', { name: 'Error attribution and coaching' })).toBeVisible();
  await expect(page.locator('main').getByRole('note')).toContainText('Synthetic fault injections only');
  await expect(page.locator('main').getByRole('note')).toContainText('4/4');

  const designed = page.getByRole('group', { name: 'Injected error scenario' });
  await expect(designed.getByRole('button')).toHaveCount(4);
  await designed.getByRole('button', { name: /Technique: done differently/ }).click();
  await expect(page.getByRole('heading', { name: 'Technique: done differently' })).toBeVisible();
  await expect(page.locator('.attribution-route-check')).toHaveText('Expected route');
  await expect(page.getByRole('img', {
    name: 'Forearm electrode ring showing channels with the largest feature deviations',
  })).toBeVisible();
  await expect(page.locator('.attribution-detail .attribution-coaching'))
    .toContainText('does not identify a specific muscle or finger');
  await expect(page.locator('.attribution-detail').getByRole('meter')).toHaveCount(4);
  await audit(page);

  await page.setViewportSize({ width: 360, height: 900 });
  const dimensions = await page.evaluate(() => ({
    pageWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.pageWidth).toBeLessThanOrEqual(dimensions.viewportWidth + 1);
});

test('Attribution runs simulated sessions end to end and checks the coaching', async ({ page }) => {
  await page.goto('/#attribution');
  const tabs = page.getByRole('tablist', { name: 'Error attribution and coaching pages' });

  await tabs.getByRole('tab', { name: 'Sessions' }).click();
  const sessions = page.getByRole('region', { name: 'Simulated sessions, end to end' });
  await expect(sessions.locator('.cartography-readout')).toContainText(/\d+\/20/);
  await expect(sessions.getByRole('table', { name: 'Routing by injected fault, across all seeds' })
    .locator('tbody tr')).toHaveCount(4);
  await audit(page);

  await tabs.getByRole('tab', { name: 'One session' }).click();
  const picker = page.getByRole('group', { name: /Simulated session, seed/ });
  await expect(picker.getByRole('button')).toHaveCount(4);
  await picker.getByRole('button', { name: /Technique: done differently/ }).click();
  await expect(picker.getByRole('button', { name: /Technique: done differently/ }))
    .toHaveAttribute('aria-pressed', 'true');
  const detail = page.getByRole('region', { name: 'One simulated session' });
  await expect(detail.getByRole('list', { name: 'Estimated muscle-group deviation' })
    .getByRole('listitem')).toHaveCount(4);
  await expect(detail.locator('.attribution-coaching')).toContainText('attempt was read as');
  await expect(detail).toContainText('does not identify a single finger');
  await expect(detail.getByRole('meter')).toHaveCount(5);
  await audit(page);

  await tabs.getByRole('tab', { name: 'Coaching check' }).click();
  const coaching = page.getByRole('region', { name: 'Does the coaching name the right muscles?' });
  await expect(coaching.locator('.cartography-readout')).toContainText(/\d+\.\d%/);
  await expect(coaching.locator('.cartography-readout')).toContainText('chance 12.5%');
  await expect(coaching.locator('tbody tr')).toHaveCount(4);
  await audit(page);
});
