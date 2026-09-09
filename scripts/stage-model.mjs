/**
 * Copy the trained decoder from artifacts/ into the app's public directory.
 *
 * This exists as its own script because two callers need it and only one of
 * them can retrain. `build-assets.mjs` runs it after the Python export; CI and
 * the browser bench run it alone, because `apps/web/public/models/` is
 * gitignored and therefore absent on a fresh clone, while `artifacts/` is
 * tracked. Without this step a freshly cloned checkout builds an app that
 * 404s on its own decoder.
 */

import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const artifacts = join(root, 'artifacts');
const models = join(root, 'apps', 'web', 'public', 'models');

export function stageModel() {
  mkdirSync(models, { recursive: true });
  const staged = [];
  for (const name of ['decoder.onnx', 'decoder.json']) {
    const from = join(artifacts, name);
    if (!existsSync(from) || statSync(from).size === 0) {
      throw new Error(
        `artifacts/${name} is missing or empty. Run: npm run build:assets`,
      );
    }
    const to = join(models, name);
    copyFileSync(from, to);
    staged.push(to);
  }
  return staged;
}

// Run directly, rather than imported by build-assets.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  try {
    for (const path of stageModel()) {
      const kb = (statSync(path).size / 1024).toFixed(0);
      console.log(`  ${path.replace(root, '.').padEnd(48)} ${kb} KB`);
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}
