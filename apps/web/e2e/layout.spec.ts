/**
 * The frame's promises, checked in a real browser.
 *
 * 1. On a desktop window every page of every screen fits: nothing below the
 *    fold, nothing that needs the page to scroll. Content that cannot fit is
 *    split into pages instead, and this test is what holds that line.
 * 2. Nothing in the interface asks the reader to leave it: no terminal
 *    commands, no "run this script" instructions.
 * 3. The plain-language layer works: every screen is reachable, every term
 *    can be explained in place, and the glossary pages through its groups.
 */

import { expect, test } from '@playwright/test';

const SCREENS = ['overview', 'live', 'results', 'cartography', 'attribution', 'review', 'glossary'];

for (const [width, height] of [
  [1440, 900],
  [1280, 800],
] as const) {
  test(`every page fits a ${width} x ${height} window without scrolling`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    const overflowing: string[] = [];
    for (const screen of SCREENS) {
      await page.goto(`/#${screen}`);
      await page.waitForSelector('main h2');
      await page.waitForLoadState('networkidle');
      const tabs = page.getByRole('tab');
      const count = Math.max(1, await tabs.count());
      for (let index = 0; index < count; index++) {
        if (index > 0) await tabs.nth(index).click();
        const overflow = await page.evaluate(() => {
          const body = document.querySelector('.screen-body') as HTMLElement | null;
          return {
            page: document.documentElement.scrollHeight - window.innerHeight,
            stage: body ? body.scrollHeight - body.clientHeight : 0,
          };
        });
        if (overflow.page > 1 || overflow.stage > 1) {
          overflowing.push(`${screen} page ${index + 1}: page +${overflow.page}px, stage +${overflow.stage}px`);
        }
      }
    }
    expect(overflowing, overflowing.join('\n')).toEqual([]);

    // The idea step must show its whole argument at once: the call to action,
    // the film and all four mechanisms, none of them clipped by the stage. A
    // stage that clips its own content passes the scroll check and still
    // hides it.
    await page.goto('/#overview');
    const clipped = await page.evaluate(() => {
      const stage = document.querySelector('.stage')!.getBoundingClientRect();
      const parts = ['.overview-intro .button-primary', '.hero-frame', '.mechanisms li:last-child'];
      return parts.filter((selector) => {
        const box = document.querySelector(selector)!.getBoundingClientRect();
        return box.bottom > stage.bottom || box.right > stage.right || box.height === 0;
      });
    });
    expect(clipped, `clipped on the idea step: ${clipped.join(', ')}`).toEqual([]);
  });
}

test('no screen tells the reader to run a command', async ({ page }) => {
  const offenders: string[] = [];
  for (const screen of SCREENS) {
    await page.goto(`/#${screen}`);
    await page.waitForSelector('main h2');
    await page.waitForLoadState('networkidle');
    const tabs = page.getByRole('tab');
    const count = Math.max(1, await tabs.count());
    for (let index = 0; index < count; index++) {
      if (index > 0) await tabs.nth(index).click();
      const text = await page.locator('main').innerText();
      for (const pattern of [/npm run/i, /python -m/i, /\bnpx\b/i, /build:assets/i]) {
        if (pattern.test(text)) offenders.push(`${screen} page ${index + 1}: ${pattern}`);
      }
    }
  }
  expect(offenders, offenders.join('\n')).toEqual([]);
});

test('the guided path walks every step in order and the URL remembers it', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Reads the forearm. Moves only when sure.' }))
    .toBeVisible();
  const path = page.getByRole('navigation', { name: 'Guided path' });
  const stepper = page.getByRole('navigation', { name: 'Step through the path' });
  const order = ['live', 'results', 'cartography', 'attribution', 'review'];
  for (const [index, id] of order.entries()) {
    await stepper.getByRole('button', { name: /^Next/ }).click();
    await expect(page).toHaveURL(new RegExp(`#${id}$`));
    await expect(path.getByRole('button', { name: new RegExp(`^Step ${index + 2}:`) }))
      .toHaveAttribute('aria-current', 'step');
  }
  await expect(stepper.getByRole('button', { name: /^Next/ })).toHaveCount(0);

  await page.getByRole('button', { name: 'Words explained' }).click();
  await expect(page).toHaveURL(/#glossary$/);
  await page.goBack();
  await expect(page).toHaveURL(/#review$/);
});

test('the page control is a keyboard tab list', async ({ page }) => {
  await page.goto('/#glossary');
  const tabs = page.getByRole('tab');
  await expect(tabs).toHaveCount(5);
  await tabs.first().focus();
  await page.keyboard.press('ArrowRight');
  await expect(tabs.nth(1)).toHaveAttribute('aria-selected', 'true');
  await expect(tabs.nth(1)).toBeFocused();
  await page.keyboard.press('End');
  await expect(tabs.nth(4)).toHaveAttribute('aria-selected', 'true');
  await page.getByRole('button', { name: 'Previous page' }).click();
  await expect(tabs.nth(3)).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('.pager-count')).toHaveText('4 / 5');
});

test('a technical term explains itself in place', async ({ page }) => {
  await page.goto('/#live');
  const term = page.getByRole('button', { name: 'Latency' });
  await term.focus();
  const tip = page.getByRole('tooltip').filter({ hasText: 'How long one decision takes' });
  await expect(tip).toBeVisible();
});
