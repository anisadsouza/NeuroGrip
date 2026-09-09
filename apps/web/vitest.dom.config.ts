import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';
import { NEUROGRIP_ALIASES } from './aliases.js';

/**
 * Components rendered to assert the ARIA contracts a screen-reader user
 * depends on. These need real focus and tabIndex semantics, which is why the
 * environment is jsdom and the key events come from user-event rather than
 * synthesised directly.
 */
export default defineConfig({
  plugins: [react()],
  resolve: { alias: NEUROGRIP_ALIASES },
  test: {
    name: 'web-dom',
    include: ['test/dom/**/*.test.tsx'],
    environment: 'jsdom',
  },
});
