/**
 * The Phase 1 gate, finally measured rather than observed.
 *
 * The repo's rule is that every number in the docs traces to an artifact
 * written by a runnable command. The browser P95 was the exception: it was
 * read off the Live screen by hand and copied into CLAUDE.md. This spec runs
 * the real worker over a thousand windows and writes what it measured.
 *
 * The threading assertion matters as much as the latency one. Cross-origin
 * isolation is a header away from vanishing, and when it does the app still
 * works -- roughly twice as slow, silently. Asserting it here means the
 * regression fails a build rather than being noticed months later in a
 * latency figure nobody was watching.
 */

import { expect, test } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

/** The project's stated budget, from the report's NFR-2. */
const BUDGET_MS = 10;

interface BenchResult {
  n: number;
  threaded: boolean;
  backend: string;
  meanMs: number;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  maxMs: number;
  featureP95Ms: number;
  inferenceP95Ms: number;
  hopMs: number;
  note: string;
}

test('the decode path meets its latency budget in the browser it ships in', async ({ page }) => {
  const failures: string[] = [];
  page.on('pageerror', (error) => failures.push(error.message));

  await page.goto('/bench.html');

  await page.waitForFunction(
    () =>
      (window as unknown as { __benchResult?: unknown; __benchError?: string }).__benchResult !==
        undefined ||
      (window as unknown as { __benchError?: string }).__benchError !== undefined,
    undefined,
    { timeout: 120_000 },
  );

  const benchError = await page.evaluate(
    () => (window as unknown as { __benchError?: string }).__benchError,
  );
  expect(benchError, `bench failed: ${benchError ?? ''}`).toBeUndefined();

  const result = (await page.evaluate(
    () => (window as unknown as { __benchResult: BenchResult }).__benchResult,
  )) as BenchResult;

  expect(failures, failures.join('\n')).toEqual([]);
  expect(result.n).toBe(1000);

  // Not an optimisation: without threads the deployed latency is a different
  // number from the one the docs publish, and nothing else would say so.
  expect(
    result.threaded,
    'cross-origin isolation is absent, so inference ran single-threaded',
  ).toBe(true);

  expect(
    result.p95Ms,
    `P95 ${result.p95Ms.toFixed(2)} ms against a ${BUDGET_MS} ms budget`,
  ).toBeLessThanOrEqual(BUDGET_MS);

  const artifactsDir = join(ROOT, 'artifacts');
  mkdirSync(artifactsDir, { recursive: true });
  writeFileSync(
    join(artifactsDir, 'browser_latency.json'),
    `${JSON.stringify(
      {
        measuredAt: new Date().toISOString().slice(0, 10),
        budgetMs: BUDGET_MS,
        meetsBudget: result.p95Ms <= BUDGET_MS,
        ...result,
      },
      null,
      2,
    )}\n`,
  );

  test.info().attach('browser_latency.json', {
    body: JSON.stringify(result, null, 2),
    contentType: 'application/json',
  });
});

test('the bench is a harness rather than a page the app links to', async ({ page }) => {
  await page.goto('/');
  const links = await page.locator('a[href*="bench"]').count();
  expect(links, 'the bench must not be reachable from the application').toBe(0);
});
