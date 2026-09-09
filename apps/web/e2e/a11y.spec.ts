/**
 * axe-core over the real page.
 *
 * The design system's contrast test parses the token file and computes ratios
 * from it, which proves the palette is legible in principle. This proves the
 * page built out of that palette is legible in fact -- that the tokens reached
 * the elements, that no rule overrode them, and that the ARIA the components
 * declare is well formed once composed.
 *
 * The intended users of this application are people with disabilities. An
 * accessibility claim in the walkthrough that no test enforces is a claim
 * about intent, not about the software.
 */

import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

const THEMES = ['light', 'dark'] as const;
// Desktop and the narrowest supported width: a reflowed layout is a different
// page as far as contrast and focus order are concerned.
const WIDTHS = [1280, 360];

async function withTheme(page: Page, theme: (typeof THEMES)[number]): Promise<void> {
  await page.addInitScript(
    ([key, value]) => window.localStorage.setItem(key as string, value as string),
    ['neurogrip.theme', theme],
  );
}

for (const theme of THEMES) {
  for (const width of WIDTHS) {
    test(`${theme} theme at ${width}px has no WCAG A or AA violations`, async ({ page }) => {
      await withTheme(page, theme);
      await page.setViewportSize({ width, height: 900 });
      await page.goto('/');
      await page.waitForSelector('main');

      const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
        .analyze();

      const described = results.violations.map(
        (violation) =>
          `${violation.id} (${violation.impact}): ${violation.help}\n` +
          violation.nodes.map((node) => `    ${node.target.join(' ')}`).join('\n'),
      );

      expect(described, described.join('\n\n')).toEqual([]);
    });
  }
}

/**
 * The radiogroup contract, checked independently of the unit test that drives
 * it. axe's aria-required-children and aria-required-attr rules see the
 * composed page, so a group whose roles were correct in isolation and broken
 * once nested still fails here.
 */
test('the gesture picker is a well-formed radio group in the composed page', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('[role="radiogroup"]');

  const results = await new AxeBuilder({ page }).include('[role="radiogroup"]').analyze();
  expect(results.violations.map((v) => v.id)).toEqual([]);

  const radios = page.locator('[role="radiogroup"] [role="radio"]');
  await expect(radios.first()).toBeVisible();

  // Exactly one tabbable radio. Ten would mean the roving tabindex is gone and
  // a keyboard user tabs through the whole vocabulary to reach the transport.
  const tabbable = await page.locator('[role="radiogroup"] [role="radio"][tabindex="0"]').count();
  expect(tabbable, 'the group must be a single tab stop').toBe(1);
});
