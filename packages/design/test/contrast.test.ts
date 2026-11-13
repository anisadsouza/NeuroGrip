/**
 * Contrast is a requirement, not an aspiration.
 *
 * These tests parse the real token file and compute WCAG contrast ratios for
 * every foreground/background pair the interface actually uses, in both
 * themes. A palette change that breaks legibility fails the build rather than
 * shipping and being discovered by someone who cannot read it.
 *
 * The intended users include people with disabilities, so this is the one
 * place in the design system where a number is non-negotiable.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const css = readFileSync(
  fileURLToPath(new URL('../src/tokens.css', import.meta.url)),
  'utf-8',
);

/** Pull `--name: #hex;` declarations out of a `:root`-like block. */
function parseBlock(selector: string): Record<string, string> {
  const start = css.indexOf(selector);
  if (start === -1) throw new Error(`selector not found: ${selector}`);
  const open = css.indexOf('{', start);
  const close = css.indexOf('}', open);
  const body = css.slice(open + 1, close);

  const tokens: Record<string, string> = {};
  for (const match of body.matchAll(/(--ng-[\w-]+)\s*:\s*(#[0-9a-fA-F]{3,8})\s*;/g)) {
    tokens[match[1]!] = match[2]!;
  }
  return tokens;
}

const light = parseBlock(':root {');
const dark = parseBlock(":root[data-theme='dark']");

function channel(value: number): number {
  const c = value / 255;
  return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function luminance(hex: string): number {
  const clean = hex.replace('#', '');
  const full =
    clean.length === 3
      ? clean.split('').map((c) => c + c).join('')
      : clean.slice(0, 6);
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi! + 0.05) / (lo! + 0.05);
}

/** Foreground, background, minimum ratio, why this pair exists. */
const TEXT_PAIRS: ReadonlyArray<[string, string, number, string]> = [
  ['--ng-ink', '--ng-paper', 4.5, 'body text on the ground'],
  ['--ng-ink', '--ng-paper-raised', 4.5, 'body text on a raised surface'],
  ['--ng-ink', '--ng-paper-sunk', 4.5, 'labels inside a measurement well'],
  ['--ng-ink-muted', '--ng-paper', 4.5, 'secondary text'],
  ['--ng-ink-muted', '--ng-paper-sunk', 4.5, 'axis labels in a plot bed'],
  ['--ng-muscle', '--ng-paper', 4.5, 'activation readout text'],
  ['--ng-commit', '--ng-paper', 4.5, 'the latched-state label'],
  ['--ng-caution', '--ng-paper', 4.5, 'drift and fatigue warnings'],
  ['--ng-stop', '--ng-paper', 4.5, 'rejected-window message'],
  ['--ng-accent', '--ng-paper', 4.5, 'accent links and the active step label'],
  ['--ng-accent', '--ng-paper-raised', 4.5, 'accent text on a panel'],
  ['--ng-accent', '--ng-accent-wash', 4.5, 'the current step on its tinted ground'],
  ['--ng-on-accent', '--ng-accent', 4.5, 'label on the primary button'],
  ['--ng-ink', '--ng-accent-wash', 4.5, 'text in a selected row'],
];


/** Non-text marks only need 3:1 under WCAG 1.4.11. */
const GRAPHIC_PAIRS: ReadonlyArray<[string, string, number, string]> = [
  ['--ng-muscle', '--ng-paper-sunk', 3, 'the signal trace against its bed'],
  ['--ng-commit', '--ng-paper-sunk', 3, 'the commitment bar fill'],
  ['--ng-ink', '--ng-paper-sunk', 3, 'plotted lines'],
  ['--ng-signal', '--ng-paper-sunk', 3, 'the signal trace against its bed'],
  ['--ng-signal', '--ng-paper', 3, 'signal marks and recorded-data badges'],
  ['--ng-signal', '--ng-signal-wash', 3, 'a signal icon on its tile'],
  ['--ng-evidence', '--ng-paper', 3, 'the evidence fill edge'],
  ['--ng-evidence', '--ng-evidence-wash', 3, 'an evidence icon on its tile'],
  ['--ng-grip', '--ng-paper', 3, 'committed-grip marks'],
  ['--ng-grip', '--ng-grip-wash', 3, 'a grip icon on its tile'],
  ['--ng-muscle', '--ng-muscle-wash', 3, 'a muscle icon on its tile'],
  // Listed here and NOT in TEXT_PAIRS, which is the whole point of the token:
  // --ng-ink-faint clears 3:1 for a mark and does not clear 4.5:1 for prose.
  // It is for hairlines, ghost outlines and the rest posture behind a moving
  // hand. Setting body copy in it produces text a low-vision reader cannot
  // read, which is how it was found -- axe caught it on the hand legend, where
  // the token file alone could not see the misuse.
  // Against paper-raised, which is the ground the hand figure is actually
  // drawn on. It does NOT clear 3:1 on paper-sunk, so a faint mark must not be
  // moved into a plot bed without being restated in a darker ink.
  ['--ng-ink-faint', '--ng-paper-raised', 3, 'the rest posture behind a moving hand'],
  ['--ng-accent', '--ng-paper-sunk', 3, 'evidence fill edge in the commitment bar'],
  // Deliberately NOT listed: --ng-rule / --ng-rule-strong against the ground.
  // WCAG 1.4.11 covers graphics needed to *understand content* and the visual
  // boundaries of *interactive controls*. A structural separator between two
  // regions is neither -- the content reads correctly without it, exactly as
  // table borders do. Forcing a hairline separator to 3:1 would make every
  // rule in the interface heavier than the data it separates, which inverts
  // the hierarchy this design depends on. Any rule that becomes the sole
  // boundary of a control must be added here.
];

for (const [themeName, tokens] of [
  ['light', light],
  ['dark', dark],
] as const) {
  describe(`${themeName} theme contrast`, () => {
    it('defines every token the interface references', () => {
      const required = new Set(
        [...TEXT_PAIRS, ...GRAPHIC_PAIRS].flatMap(([fg, bg]) => [fg, bg]),
      );
      for (const name of required) {
        expect(tokens[name], `${name} missing from ${themeName}`).toBeDefined();
      }
    });

    for (const [fg, bg, minimum, why] of TEXT_PAIRS) {
      it(`${why} meets ${minimum}:1`, () => {
        const ratio = contrast(tokens[fg]!, tokens[bg]!);
        expect(
          ratio,
          `${fg} on ${bg} is ${ratio.toFixed(2)}:1, needs ${minimum}:1`,
        ).toBeGreaterThanOrEqual(minimum);
      });
    }

    for (const [fg, bg, minimum, why] of GRAPHIC_PAIRS) {
      it(`${why} meets ${minimum}:1`, () => {
        const ratio = contrast(tokens[fg]!, tokens[bg]!);
        expect(
          ratio,
          `${fg} on ${bg} is ${ratio.toFixed(2)}:1, needs ${minimum}:1`,
        ).toBeGreaterThanOrEqual(minimum);
      });
    }
  });
}

describe('palette discipline', () => {
  it('has no gradients', () => {
    // linear-gradient is permitted in exactly one place: drawing the graticule
    // as hairlines, which is a grid, not a colour ramp.
    const gradients = [...css.matchAll(/gradient\(/g)].length;
    const graticuleLines = [...css.matchAll(/var\(--ng-graticule-\w+\) var\(--ng-hairline\)/g)]
      .length;
    expect(gradients).toBe(graticuleLines);
  });

  it('has no glows or soft shadows', () => {
    expect(css).not.toMatch(/box-shadow:\s*(?!none)/);
    expect(css).not.toMatch(/text-shadow:\s*(?!none)/);
    expect(css).not.toMatch(/filter:\s*.*blur/);
  });

  it('defines a radius scale that increases with surface size', () => {
    // Corners are rounded on request. What still has to hold is that the scale
    // is ordered: a nested surface must never carry a larger radius than the
    // surface containing it, or the nesting reads as a mistake.
    // Parsed by line scan rather than a constructed regex: the escaping needed
    // to build one inside a template literal is a reliable source of silent
    // no-match bugs.
    const radius = (name: string): number => {
      for (const line of css.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith(`${name}:`)) continue;
        const value = Number.parseInt(trimmed.slice(name.length + 1).trim(), 10);
        if (Number.isFinite(value)) return value;
      }
      throw new Error(`${name} is not defined`);
    };

    const small = radius('--ng-radius-sm');
    const base = radius('--ng-radius');
    const large = radius('--ng-radius-lg');
    expect(small).toBeGreaterThan(0);
    expect(base).toBeGreaterThan(small);
    expect(large).toBeGreaterThan(base);
    // Nothing so round it stops reading as an instrument panel.
    expect(large).toBeLessThanOrEqual(12);
  });

  it('respects reduced motion', () => {
    expect(css).toMatch(/@media \(prefers-reduced-motion: reduce\)/);
  });

  it('defines the dark theme by explicit declaration, not inversion', () => {
    expect(css).toMatch(/@media \(prefers-color-scheme: dark\)/);
    expect(css).toMatch(/:root\[data-theme='dark'\]/);
    expect(css).toMatch(/:root:not\(\[data-theme='light'\]\)/);
    // Dark is its own palette: its ground must not be a numeric inversion.
    expect(dark['--ng-paper']).not.toBe('#0c0b0d');
  });

  it('keeps the two dark declarations identical', () => {
    // The dark palette is declared twice: once under prefers-color-scheme for
    // viewers who never touched the toggle, once under [data-theme] for those
    // who did. CSS cannot share them without a preprocessor, so they are
    // duplicated -- and duplication drifts. If these diverge, the toggle and
    // the system preference produce different colours for the same theme.
    const media = parseBlock(":root:not([data-theme='light'])");
    for (const [name, value] of Object.entries(dark)) {
      expect(media[name], `${name} differs between the two dark blocks`).toBe(value);
    }
    expect(Object.keys(media).sort()).toEqual(Object.keys(dark).sort());
  });

  it('carries exactly the declared hue families', () => {
    // Premise changed 2026-11-08 (evening) at the owner's request: the chrome
    // has ONE accent. The four story hues survive only as data colours inside
    // figures, and the state colours keep their meaning. A hue outside these
    // lists means someone has started decorating.
    const family = (name: string) => name.replace(/--ng-|on-|-weak|-wash|-strong/g, '');
    const hues = new Set(
      Object.keys(light)
        .filter((name) => /accent|muscle|signal|evidence|grip|commit|caution|stop/.test(name))
        .map(family),
    );
    expect([...hues].sort()).toEqual(
      ['accent', 'caution', 'commit', 'evidence', 'grip', 'muscle', 'signal', 'stop'],
    );
  });

  it('keeps data colours out of the chrome', () => {
    // shell.css is the frame and every screen's layout. Data hues belong to
    // figures (app.css and the SVG/canvas components), never to navigation,
    // headings, buttons or panels.
    const shell = readFileSync(
      fileURLToPath(new URL('../../../apps/web/src/shell.css', import.meta.url)),
      'utf-8',
    );
    const leaks = [...shell.matchAll(/var\(--ng-(muscle|signal|evidence|grip)[\w-]*\)/g)].map(
      (match) => match[0],
    );
    expect(leaks, `data hue used in chrome: ${leaks.join(', ')}`).toEqual([]);
  });
});
