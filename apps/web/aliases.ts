import { fileURLToPath } from 'node:url';

/**
 * Where the workspace packages actually live on disk.
 *
 * Array form, and order matters: the subpath entry must be tried before the
 * bare package entry, or '@neurogrip/design/tokens.css' resolves to the package
 * index and Vite reports the file as missing.
 *
 * This lives in its own module because both the app build and the test runner
 * need it. Two independently written copies of one mapping is the same defect
 * as two copies of a risk vector: they agree until the day they quietly do not.
 */
export const NEUROGRIP_ALIASES = [
  {
    find: '@neurogrip/design/tokens.css',
    replacement: fileURLToPath(new URL('../../packages/design/src/tokens.css', import.meta.url)),
  },
  {
    find: '@neurogrip/design',
    replacement: fileURLToPath(new URL('../../packages/design/src/index.ts', import.meta.url)),
  },
  {
    find: '@neurogrip/core',
    replacement: fileURLToPath(new URL('../../packages/core/src/index.ts', import.meta.url)),
  },
];
