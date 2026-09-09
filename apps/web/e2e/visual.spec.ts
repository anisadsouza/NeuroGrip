/**
 * Layout, at the sizes the design system commits to and in both themes.
 *
 * The assertions are structural rather than pixel comparisons. A golden image
 * fails on a font-rendering change and a real regression identically, and the
 * failure carries no diagnosis either way; what is worth gating is the small
 * set of things that would actually make the instrument unusable. Screenshots
 * are still captured and attached, for a person to look at.
 *
 * The safety banner is in here because ER-4 makes it a requirement, not a
 * decoration. A 360 px layout that pushes it off the top of the page satisfies
 * every component test and breaks the requirement, and a screenshot diff of a
 * page nobody scrolled would not catch it either.
 */

import { expect, test, type Page } from '@playwright/test';

const WIDTHS = [360, 768, 1280, 1920];
const THEMES = ['light', 'dark'] as const;

/** Set the stored theme before the app's first paint, as its own toggle does. */
async function withTheme(page: Page, theme: (typeof THEMES)[number]): Promise<void> {
  await page.addInitScript(
    ([key, value]) => window.localStorage.setItem(key as string, value as string),
    ['neurogrip.theme', theme],
  );
}

for (const theme of THEMES) {
  for (const width of WIDTHS) {
    test(`${theme} theme at ${width}px keeps the instrument readable`, async ({ page }) => {
      await withTheme(page, theme);
      await page.setViewportSize({ width, height: 900 });
      await page.goto('/');
      await page.waitForSelector('main');

      // The body never scrolls sideways. A wide table or trace must scroll
      // inside its own container instead.
      const overflow = await page.evaluate(() => ({
        scrollWidth: document.documentElement.scrollWidth,
        clientWidth: document.documentElement.clientWidth,
      }));
      expect(
        overflow.scrollWidth,
        `page scrolls horizontally: ${overflow.scrollWidth} > ${overflow.clientWidth}`,
      ).toBeLessThanOrEqual(overflow.clientWidth + 1);

      // ER-4: the disclaimer is permanent, so it must be on the page at every
      // size rather than only at the one the design was drawn at.
      const banner = page.getByText(/Research prototype/i).first();
      await expect(banner).toBeVisible();

      // Nothing wider than the viewport. This is what an unscrollable body
      // hides: a panel can overflow its parent and be clipped rather than
      // scrolled, and the wearer simply never sees the right-hand edge.
      const tooWide = await page.evaluate((viewport) => {
        const offenders: string[] = [];
        for (const element of document.querySelectorAll('body *')) {
          const box = element.getBoundingClientRect();
          if (box.width > viewport + 1) {
            offenders.push(`${element.tagName.toLowerCase()}.${element.className} ${box.width}px`);
          }
        }
        return offenders;
      }, width);
      expect(tooWide, tooWide.join('\n')).toEqual([]);

      test.info().attach(`${theme}-${width}.png`, {
        body: await page.screenshot({ fullPage: true }),
        contentType: 'image/png',
      });
    });
  }
}

/**
 * The palette is declared twice -- once under the media query, once under the
 * attribute -- because CSS cannot share them. contrast.test.ts asserts the two
 * declarations are identical as text; this asserts they resolve identically in
 * a browser, which is the thing that was actually promised.
 */
test('the stored dark theme and the system dark preference resolve to the same palette', async ({
  browser,
}) => {
  const stored = await browser.newContext();
  const storedPage = await stored.newPage();
  await withTheme(storedPage, 'dark');
  await storedPage.goto('/');
  const fromAttribute = await storedPage.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--ng-paper').trim(),
  );

  const system = await browser.newContext({ colorScheme: 'dark' });
  const systemPage = await system.newPage();
  await systemPage.goto('/');
  const fromMediaQuery = await systemPage.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--ng-paper').trim(),
  );

  expect(fromAttribute).not.toBe('');
  expect(fromAttribute).toBe(fromMediaQuery);

  await stored.close();
  await system.close();
});
