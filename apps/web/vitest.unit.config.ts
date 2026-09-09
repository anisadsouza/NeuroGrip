import { defineConfig } from 'vitest/config';
import { NEUROGRIP_ALIASES } from './aliases.js';

/**
 * The pure decode logic — risk vectors, muscle projection, frame decimation.
 * None of it has any business loading a DOM, so it runs in node beside the
 * rest of the numeric suites.
 */
export default defineConfig({
  resolve: { alias: NEUROGRIP_ALIASES },
  test: {
    name: 'web-unit',
    include: ['test/unit/**/*.test.ts'],
    environment: 'node',
  },
});
