/**
 * Generates a design-system reference sheet: the palette, the type scale, the
 * graticule, and every icon on its construction grid.
 *
 * This exists to be looked at. A token file reads as plausible no matter what
 * the values are; the only way to know whether the carmine is right against
 * the ground, or whether the icons share optical weight, is to render them
 * side by side and inspect.
 *
 *   npx vite-node packages/design/scripts/reference.ts
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ICON_NAMES, ICONS, renderIconSvg } from '../src/icons.js';

const here = dirname(fileURLToPath(import.meta.url));
const tokens = readFileSync(resolve(here, '../src/tokens.css'), 'utf-8');

const SWATCHES: ReadonlyArray<[string, string]> = [
  ['--ng-paper', 'ground'],
  ['--ng-paper-sunk', 'measurement well'],
  ['--ng-paper-raised', 'raised surface'],
  ['--ng-ink', 'text'],
  ['--ng-ink-muted', 'secondary text'],
  ['--ng-rule', 'hairline'],
  ['--ng-muscle', 'muscle activation'],
  ['--ng-commit', 'latched'],
  ['--ng-caution', 'drift, fatigue'],
  ['--ng-stop', 'rejected'],
];

const swatches = SWATCHES.map(
  ([token, role]) => `
  <div class="swatch">
    <div class="chip${token.includes('rule') ? ' chip-rule' : ''}" style="${
      token.includes('rule')
        ? `border-top-color: var(${token})`
        : `background: var(${token})`
    }"></div>
    <div class="swatch-name ng-num">${token.replace('--ng-', '')}</div>
    <div class="swatch-role">${role}</div>
  </div>`,
).join('');

const icons = ICON_NAMES.map(
  (name) => `
  <figure class="icon">
    <div class="icon-cell">${renderIconSvg(name, 24)}</div>
    <figcaption>
      <span class="icon-name">${name}</span>
      <span class="icon-note">${ICONS[name].note}</span>
    </figcaption>
  </figure>`,
).join('');

/** A biphasic MUAP burst, so the graticule is shown carrying real data. */
function trace(width: number, height: number): string {
  const points: string[] = [];
  let seed = 12345;
  const random = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648 - 0.5;
  };
  for (let x = 0; x <= width; x += 2) {
    const envelope = Math.exp(-(((x - width * 0.45) / (width * 0.16)) ** 2));
    const spike = random() * envelope * height * 0.42;
    const baseline = random() * height * 0.03;
    points.push(`${x},${(height / 2 + spike + baseline).toFixed(1)}`);
  }
  return `<polyline points="${points.join(' ')}" fill="none" stroke="var(--ng-muscle)" stroke-width="1.25"/>`;
}

const html = `<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuroGrip design reference</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
${tokens}

.sheet { padding: var(--ng-space-8); max-width: 1100px; margin: 0 auto; }
h1 { font-size: var(--ng-text-xl); font-weight: 700; margin: 0 0 var(--ng-space-2); letter-spacing: -0.01em; }
.lede { color: var(--ng-ink-muted); max-width: var(--ng-measure); margin: 0 0 var(--ng-space-8); }
h2 { font-size: var(--ng-text-lg); font-weight: 700; margin: var(--ng-space-12) 0 var(--ng-space-1); }
.section-note { color: var(--ng-ink-muted); font-size: var(--ng-text-sm); max-width: var(--ng-measure); margin: 0 0 var(--ng-space-4); }
hr { border: 0; border-top: var(--ng-hairline) solid var(--ng-rule); margin: var(--ng-space-4) 0; }

.swatches { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: var(--ng-space-4); }
.chip { height: 48px; border: var(--ng-hairline) solid var(--ng-rule-strong); }
/* A hairline colour shown as a 48px block misrepresents it. Show it as a line. */
.chip-rule { border: 0; border-top-width: var(--ng-hairline); border-top-style: solid; height: 48px; background: transparent; }
.swatch-name { font-size: var(--ng-text-xs); margin-top: var(--ng-space-2); }
.swatch-role { font-size: var(--ng-text-xs); color: var(--ng-ink-muted); }

.type-row { display: flex; align-items: baseline; gap: var(--ng-space-4); padding: var(--ng-space-2) 0; border-bottom: var(--ng-hairline) solid var(--ng-rule); }
.type-label { font-size: var(--ng-text-xs); color: var(--ng-ink-faint); width: 90px; flex: none; }

.bed { height: 132px; position: relative; border: var(--ng-hairline) solid var(--ng-rule-strong); }
.bed svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.cal { position: absolute; right: var(--ng-space-3); bottom: var(--ng-space-2); font-size: var(--ng-text-xs); color: var(--ng-ink-muted); }

.icons { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 0; border-top: var(--ng-hairline) solid var(--ng-rule); }
.icon { display: flex; gap: var(--ng-space-3); margin: 0; padding: var(--ng-space-3); border-bottom: var(--ng-hairline) solid var(--ng-rule); border-right: var(--ng-hairline) solid var(--ng-rule); align-items: flex-start; }
.icon-cell { width: 24px; height: 24px; flex: none; color: var(--ng-ink); }
.icon-name { display: block; font-size: var(--ng-text-sm); font-weight: 700; }
.icon-note { display: block; font-size: var(--ng-text-xs); color: var(--ng-ink-muted); line-height: 1.4; }

.readout { display: flex; gap: var(--ng-space-8); flex-wrap: wrap; }
.metric-value { font-size: var(--ng-text-readout); line-height: 1; }
.metric-label { font-size: var(--ng-text-xs); color: var(--ng-ink-muted); margin-top: var(--ng-space-1); }
.metric-unit { font-size: var(--ng-text-sm); color: var(--ng-ink-muted); }
</style>
</head>
<body>
<div class="sheet">
  <h1>NeuroGrip design reference</h1>
  <p class="lede">The instrument this interface imitates is a chart recorder, not a dashboard. Two rules follow: the graticule appears only behind measured data, and carmine encodes muscle activation and nothing else.</p>

  <h2>Palette</h2>
  <p class="section-note">Achromatic, with one meaning-bearing hue. Carmine follows anatomical illustration convention for muscle; it is never used to emphasise a heading or a control.</p>
  <div class="swatches">${swatches}</div>

  <h2>Type</h2>
  <p class="section-note">Atkinson Hyperlegible for interface text, because it was designed for maximum character distinction at low vision and the intended users of this system are people with disabilities. IBM Plex Mono for numerals only, where tabular alignment is a functional requirement.</p>
  <div class="type-row"><span class="type-label">xl / 24</span><span style="font-size: var(--ng-text-xl); font-weight: 700">Live decoding</span></div>
  <div class="type-row"><span class="type-label">lg / 18</span><span style="font-size: var(--ng-text-lg); font-weight: 700">Commitment</span></div>
  <div class="type-row"><span class="type-label">base / 15</span><span>The hand begins moving as evidence accrues, and latches only at full commitment.</span></div>
  <div class="type-row"><span class="type-label">sm / 13</span><span style="font-size: var(--ng-text-sm)">Held-out subject 4 of 8</span></div>
  <div class="type-row"><span class="type-label">xs / 11</span><span style="font-size: var(--ng-text-xs)">microvolts, RMS per channel</span></div>
  <div class="type-row"><span class="type-label">numerals</span><span class="ng-num" style="font-size: var(--ng-text-lg)">0123456789 · 95.63 % · 1.91 ms · 158–354 µV</span></div>

  <h2>Readout</h2>
  <p class="section-note">One large number per panel, in tabular figures so the digits do not jitter as the value changes.</p>
  <div class="readout">
    <div><div class="metric-value ng-num">1.91<span class="metric-unit"> ms</span></div><div class="metric-label">end-to-end latency, P95</div></div>
    <div><div class="metric-value ng-num">95.63<span class="metric-unit"> %</span></div><div class="metric-label">leave-one-subject-out accuracy</div></div>
    <div><div class="metric-value ng-num" style="color: var(--ng-muscle)">312<span class="metric-unit"> µV</span></div><div class="metric-label">channel 4 activation</div></div>
  </div>

  <h2>Graticule</h2>
  <p class="section-note">The grid means measurement. It appears here because there is a real trace with a real scale; it never appears as chrome.</p>
  <div class="bed ng-graticule">
    <svg viewBox="0 0 720 132" preserveAspectRatio="none">${trace(720, 132)}</svg>
    <div class="cal ng-num">200 µV · 40 ms/div</div>
  </div>

  <h2>Icons</h2>
  <p class="section-note">Constructed on a 24-pixel grid at 1.5 stroke with butt caps and miter joins. The vocabulary comes from the subject — action potential, electrode ring, forearm cross-section — not from a generic UI kit.</p>
  <div class="icons">${icons}</div>
</div>
</body>
</html>
`;

// docs/, not artifacts/: this sheet is design documentation meant to be
// read and reviewed, not a model build output.
const out = resolve(here, '../../../docs/design/reference.html');
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, html, 'utf-8');
console.log(`wrote ${out} (${ICON_NAMES.length} icons, ${SWATCHES.length} swatches)`);
