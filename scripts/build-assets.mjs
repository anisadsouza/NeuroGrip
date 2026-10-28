/**
 * Regenerates every asset the web application consumes.
 *
 * This exists as a script rather than a chain of shell commands in
 * package.json for one reason: `python` on PATH is usually not the interpreter
 * that has this project installed. An npm script inherits the shell's PATH, not
 * an activated virtualenv, so `python -m neurogrip.replay` fails with
 * ModuleNotFoundError on a machine where everything is set up correctly. That
 * is a confusing failure to hand someone whose only mistake was following the
 * README.
 *
 *   node scripts/build-assets.mjs
 */

import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { stageModel } from './stage-model.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/**
 * Find an interpreter that can actually import `neurogrip`.
 *
 * The project's own virtualenv is tried first, then whatever is on PATH, so a
 * contributor using conda, uv or a system install is not forced into ours.
 */
function findPython() {
  const candidates = [
    join(root, '.venv', 'Scripts', 'python.exe'), // Windows
    join(root, '.venv', 'bin', 'python'), // POSIX
    'python',
    'python3',
  ];

  for (const candidate of candidates) {
    if (candidate.includes('.venv') && !existsSync(candidate)) continue;
    // This shell's Node is sometimes launched under Rosetta while the project
    // virtualenv has arm64 scientific wheels. Force the arm64 slice when it is
    // available; otherwise fall back to the native interpreter unchanged.
    const commands = process.platform === 'darwin' && process.arch === 'x64'
      ? [['/usr/bin/arch', ['-arm64', candidate]], [candidate, []]]
      : [[candidate, []]];
    for (const [command, prefix] of commands) {
      const probe = spawnSync(command, [...prefix, '-c', 'import neurogrip'], {
        cwd: root,
        stdio: 'ignore',
      });
      if (probe.status === 0) return { command, prefix, display: [...prefix, command === candidate ? '' : candidate].filter(Boolean).join(' ') };
    }
  }

  console.error(
    'No Python interpreter on this machine can import `neurogrip`.\n' +
      'Install it first:\n\n' +
      '    pip install -e "./services/research[dev,ml]"\n',
  );
  process.exit(1);
}

function run(python, args, description) {
  process.stdout.write(`${description}\n`);
  const result = spawnSync(python.command, [...python.prefix, ...args], { cwd: root, stdio: 'inherit' });
  if (result.status !== 0) {
    console.error(`\nFailed: ${description}`);
    process.exit(result.status ?? 1);
  }
}

const python = findPython();
console.log(`using ${python.display}\n`);

run(
  python,
  ['-m', 'neurogrip.replay', '--out', 'apps/web/public/replay'],
  'Generating the replay bundle…',
);
run(
  python,
  ['-m', 'neurogrip.experiments', 'export', '--out', 'artifacts', '--model', 'rbf_svm'],
  'Training and exporting the decoder…',
);

const models = join(root, 'apps', 'web', 'public', 'models');
stageModel();

// Results are optional: recorded DB2 artifacts do not exist until a licensed
// dataset run has occurred. Copy whatever experiments have produced, never
// fabricate a recorded result for the screen.
const results = join(root, 'apps', 'web', 'public', 'results');
mkdirSync(results, { recursive: true });
for (const name of [
  'model_comparison.json',
  'model_comparison_ninapro.json',
  'browser_latency.json',
  'browser_latency_ninapro.json',
]) {
  const source = join(root, 'artifacts', name);
  if (existsSync(source)) copyFileSync(source, join(results, name));
}

// A subject-held-out NinaPro export is optional and never replaces the
// simulator-trained decoder. If any part exists, require and stage the full
// model/evaluation/replay set so the UI cannot mix partial artifacts.
const heldoutFiles = [
  [join(root, 'artifacts', 'ninapro_holdout', 'model', 'decoder.onnx'),
    join(models, 'ninapro', 'decoder.onnx')],
  [join(root, 'artifacts', 'ninapro_holdout', 'model', 'decoder.json'),
    join(models, 'ninapro', 'decoder.json')],
  [join(root, 'artifacts', 'ninapro_holdout', 'evaluation.json'),
    join(results, 'ninapro_holdout.json')],
  [join(root, 'apps', 'web', 'public', 'replay', 'ninapro', 'emg-replay.bin'),
    join(root, 'apps', 'web', 'public', 'replay', 'ninapro', 'emg-replay.bin')],
  [join(root, 'apps', 'web', 'public', 'replay', 'ninapro', 'emg-replay.json'),
    join(root, 'apps', 'web', 'public', 'replay', 'ninapro', 'emg-replay.json')],
];
const hasHeldoutAssets = heldoutFiles.some(([source]) => existsSync(source));
if (hasHeldoutAssets) {
  for (const [source, destination] of heldoutFiles) {
    if (!existsSync(source) || statSync(source).size === 0) {
      console.error(`Incomplete NinaPro held-out assets: ${source} is missing or empty.`);
      process.exit(1);
    }
    if (source !== destination) {
      mkdirSync(dirname(destination), { recursive: true });
      copyFileSync(source, destination);
    }
  }
}

// Verify rather than assume: an empty or missing file here produces a runtime
// error in the browser that looks nothing like its cause.
const required = [
  join(models, 'decoder.onnx'),
  join(models, 'decoder.json'),
  join(root, 'apps', 'web', 'public', 'replay', 'emg-replay.bin'),
  join(root, 'apps', 'web', 'public', 'replay', 'emg-replay.json'),
  ...(hasHeldoutAssets
    ? heldoutFiles.map(([, destination]) => destination)
    : []),
];

console.log('');
for (const path of required) {
  if (!existsSync(path) || statSync(path).size === 0) {
    console.error(`Missing or empty after build: ${path}`);
    process.exit(1);
  }
  const kb = (statSync(path).size / 1024).toFixed(0);
  console.log(`  ${path.replace(root, '.').padEnd(48)} ${kb} KB`);
}
console.log('\nAssets ready. Start the app with: npm run dev:web');
