/**
 * The keyboard path through the application, walked.
 *
 * The unit test proves GestureChoices honours the radiogroup contract in
 * isolation. This proves the contract survives composition -- that the group
 * really is one stop in the page's tab order, and that the focus ring the
 * design system declares actually paints. contrast.test.ts can only assert the
 * :focus-visible rule exists in the token file; whether it applies to a real
 * focused button is a question only a browser answers.
 */

import { expect, test } from '@playwright/test';

/** What has focus, described the way a person would recognise it. */
async function focused(page: import('@playwright/test').Page): Promise<string> {
  return page.evaluate(() => {
    const element = document.activeElement;
    if (!element || element === document.body) return 'body';
    const role = element.getAttribute('role');
    const label = element.getAttribute('aria-label') ?? element.textContent?.trim() ?? '';
    return `${element.tagName.toLowerCase()}${role ? `[${role}]` : ''}: ${label.slice(0, 40)}`;
  });
}

test('the gesture group is one stop in the page tab order rather than ten', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('[role="radiogroup"] [role="radio"]');

  const radioCount = await page.locator('[role="radiogroup"] [role="radio"]').count();
  expect(radioCount).toBeGreaterThan(1);

  // Tab until focus first lands inside the group, then once more, and assert
  // it has left. Ten radios each holding a tab stop would fail on the second.
  let stopsToEnter = 0;
  for (let i = 0; i < 40; i++) {
    await page.keyboard.press('Tab');
    stopsToEnter++;
    const inside = await page.evaluate(
      () => document.activeElement?.closest('[role="radiogroup"]') !== null,
    );
    if (inside) break;
  }
  expect(stopsToEnter, 'never reached the gesture group by tabbing').toBeLessThan(40);

  await page.keyboard.press('Tab');
  const stillInside = await page.evaluate(
    () => document.activeElement?.closest('[role="radiogroup"]') !== null,
  );
  expect(stillInside, `one tab left the group at ${await focused(page)}`).toBe(false);
});

test('the arrow keys move and select within the group rather than scrolling the page', async ({
  page,
}) => {
  await page.goto('/');
  await page.waitForSelector('[role="radiogroup"] [role="radio"]');

  await page.locator('[role="radiogroup"] [role="radio"][tabindex="0"]').focus();
  const before = await page.locator('[role="radio"][aria-checked="true"]').textContent();

  await page.keyboard.press('ArrowDown');

  const after = await page.locator('[role="radio"][aria-checked="true"]').textContent();
  expect(after).not.toBe(before);
  expect(await page.evaluate(() => document.activeElement?.getAttribute('aria-checked'))).toBe(
    'true',
  );
  expect(await page.evaluate(() => window.scrollY)).toBe(0);
});

/**
 * A focus ring declared in a token file and never applied is the most common
 * way a keyboard-accessible interface turns out not to be: every control is
 * reachable, and the user cannot see where they are.
 */
test('a focused control paints a visible focus ring rather than only declaring one', async ({
  page,
}) => {
  await page.goto('/');
  await page.waitForSelector('button');

  await page.keyboard.press('Tab');

  const outline = await page.evaluate(() => {
    const element = document.activeElement as HTMLElement | null;
    if (!element) return null;
    const style = getComputedStyle(element);
    return {
      width: style.outlineWidth,
      style: style.outlineStyle,
      colour: style.outlineColor,
    };
  });

  expect(outline).not.toBeNull();
  expect(outline!.style).not.toBe('none');
  expect(parseFloat(outline!.width)).toBeGreaterThan(0);
});
