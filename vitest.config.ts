import { defineConfig } from 'vitest/config';

/**
 * One `npm test` across the workspace.
 *
 * The directory entries resolve to each package's own vitest.config.ts, so the
 * pure-numeric suites keep running in the node environment they were written
 * for, and only apps/web pays for a DOM.
 */
export default defineConfig({
  test: {
    projects: [
      'packages/core',
      'packages/design',
      'apps/web/vitest.unit.config.ts',
      'apps/web/vitest.dom.config.ts',
    ],
  },
});
