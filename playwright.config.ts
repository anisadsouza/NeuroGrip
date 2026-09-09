/**
 * The browser tier.
 *
 * Everything here needs a real browser to mean anything: WASM threads exist
 * only under cross-origin isolation, the oscilloscope's canvas path is skipped
 * entirely where an element has no layout, and a horizontal overflow is a fact
 * about a rendered page rather than about a component tree.
 *
 * Chromium only. Cross-origin isolation, threaded WASM and stable layout
 * metrics are what is being measured, and adding two more engines would triple
 * the CI time to re-measure the same three things.
 */

import { defineConfig, devices } from '@playwright/test';

const PORT = 4173;

export default defineConfig({
  testDir: './apps/web/e2e',
  // A P95 is a statement about a machine under a known load. Letting other
  // specs compete for cores while it runs would measure the runner, not the
  // decoder.
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : [['list']],
  timeout: 120_000,
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'retain-on-failure',
    ...devices['Desktop Chrome'],
  },
  webServer: {
    // Staging the model first: apps/web/public/models is gitignored, so a
    // fresh clone has no decoder for the build to copy in.
    command:
      'node scripts/stage-model.mjs && npm run build --workspace @neurogrip/web && ' +
      `npm run preview --workspace @neurogrip/web -- --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    stdout: 'pipe',
  },
});
