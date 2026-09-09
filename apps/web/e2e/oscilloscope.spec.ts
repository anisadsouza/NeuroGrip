/**
 * The one place the canvas actually draws.
 *
 * Oscilloscope.tsx returns early when its container has no layout, which is
 * unconditionally true under jsdom -- so a component test of this file would
 * render, assert, pass, and exercise none of the drawing code. That is worse
 * than no test, because it reads like coverage. The jsdom test covers the
 * caption and the stated scale; the pixels are checked here.
 *
 * What is asserted is that the trace responds to the signal. A canvas that
 * draws its graticule and nothing else looks plausible in a screenshot and
 * means the wearer cannot see their own electrodes.
 */

import { expect, test } from '@playwright/test';

test('the trace draws and changes as the signal arrives', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('canvas');

  // The backing store is sized from layout, so a zero here means the early
  // return fired and nothing was ever drawn.
  const size = await page.evaluate(() => {
    const canvas = document.querySelector('canvas') as HTMLCanvasElement;
    return { width: canvas.width, height: canvas.height };
  });
  expect(size.width).toBeGreaterThan(0);
  expect(size.height).toBeGreaterThan(0);

  const start = page.getByRole('button', { name: /start decoding/i });
  await expect(start).toBeEnabled({ timeout: 60_000 });
  await start.click();

  /** A cheap digest of the canvas, enough to tell two frames apart. */
  const digest = () =>
    page.evaluate(() => {
      const canvas = document.querySelector('canvas') as HTMLCanvasElement;
      const context = canvas.getContext('2d')!;
      const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
      let sum = 0;
      for (let i = 0; i < data.length; i += 97) sum += data[i]!;
      return sum;
    });

  await page.waitForTimeout(400);
  const first = await digest();
  await page.waitForTimeout(400);
  const second = await digest();

  expect(first, 'the canvas is blank').toBeGreaterThan(0);
  expect(second, 'the trace never changed, so nothing is being drawn').not.toBe(first);
});
