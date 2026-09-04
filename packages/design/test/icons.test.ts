import { describe, expect, it } from 'vitest';
import {
  ICON_CANVAS,
  ICON_NAMES,
  ICON_SAFE_MAX,
  ICON_SAFE_MIN,
  ICON_STROKE,
  ICONS,
  iconCoordinates,
  renderIconSvg,
} from '../src/icons.js';

describe('icon set', () => {
  it('is not empty and every name resolves', () => {
    expect(ICON_NAMES.length).toBeGreaterThan(10);
    for (const name of ICON_NAMES) expect(ICONS[name]).toBeDefined();
  });

  it('documents why each mark looks the way it does', () => {
    for (const name of ICON_NAMES) {
      expect(ICONS[name].note.length, `${name} has no note`).toBeGreaterThan(20);
    }
  });

  it('gives every icon at least one element', () => {
    for (const name of ICON_NAMES) {
      expect(ICONS[name].elements.length, `${name} is empty`).toBeGreaterThan(0);
    }
  });
});

describe('construction grid', () => {
  it('keeps every coordinate inside the canvas', () => {
    for (const name of ICON_NAMES) {
      for (const value of iconCoordinates(name)) {
        expect(value, `${name} has coordinate ${value} outside 0..${ICON_CANVAS}`)
          .toBeGreaterThanOrEqual(0);
        expect(value, `${name} has coordinate ${value} outside 0..${ICON_CANVAS}`)
          .toBeLessThanOrEqual(ICON_CANVAS);
      }
    }
  });

  it('keeps circles within the safe area', () => {
    // Paths carry arc flags and curve control points that are legitimately
    // allowed near the edge, so only circles -- whose extents are unambiguous
    // -- are held to the optical margin.
    for (const name of ICON_NAMES) {
      for (const element of ICONS[name].elements) {
        if (element.kind !== 'circle') continue;
        expect(element.cx - element.r, `${name} circle overflows left`)
          .toBeGreaterThanOrEqual(ICON_SAFE_MIN);
        expect(element.cx + element.r, `${name} circle overflows right`)
          .toBeLessThanOrEqual(ICON_SAFE_MAX);
        expect(element.cy - element.r, `${name} circle overflows top`)
          .toBeGreaterThanOrEqual(ICON_SAFE_MIN);
        expect(element.cy + element.r, `${name} circle overflows bottom`)
          .toBeLessThanOrEqual(ICON_SAFE_MAX);
      }
    }
  });

  it('uses fill only where filled-versus-hollow is the meaning', () => {
    const filled = ICON_NAMES.filter((name) =>
      ICONS[name].elements.some((element) => element.filled),
    );
    // Each entry earns its fill by making filled-versus-hollow the distinction
    // the mark exists to draw:
    //   record      live versus idle
    //   electrode   seated versus open contact
    //   theme       the light half versus the dark half
    //   commitment  the filled portion of the bar versus the empty remainder
    //   manifold    the peak of the reachability contour
    //   alert       the full stop under the bar, which must not read as a ring
    // Adding a name here without a reason on this list is how a line icon set
    // drifts into a mixed one.
    expect(filled.sort()).toEqual(
      ['alert', 'commitment', 'electrode', 'manifold', 'record', 'theme'].sort(),
    );
  });
});

describe('rendered SVG', () => {
  it('is well formed and carries the construction stroke', () => {
    const svg = renderIconSvg('waveform');
    expect(svg.startsWith('<svg')).toBe(true);
    expect(svg.endsWith('</svg>')).toBe(true);
    expect(svg).toContain(`viewBox="0 0 ${ICON_CANVAS} ${ICON_CANVAS}"`);
    expect(svg).toContain(`stroke-width="${ICON_STROKE}"`);
  });

  it('uses machined caps and joins, never rounded', () => {
    for (const name of ICON_NAMES) {
      const svg = renderIconSvg(name);
      expect(svg).toContain('stroke-linecap="butt"');
      expect(svg).toContain('stroke-linejoin="miter"');
      expect(svg).not.toContain('round');
    }
  });

  it('inherits colour rather than hard-coding it', () => {
    for (const name of ICON_NAMES) {
      const svg = renderIconSvg(name);
      expect(svg).not.toMatch(/#[0-9a-fA-F]{3,8}/);
      expect(svg).toContain('currentColor');
    }
  });

  it('is hidden from assistive technology unless given a label', () => {
    expect(renderIconSvg('play')).toContain('aria-hidden="true"');
    const labelled = renderIconSvg('play', 24, 'Start the stream');
    expect(labelled).toContain('role="img"');
    expect(labelled).toContain('aria-label="Start the stream"');
    expect(labelled).not.toContain('aria-hidden');
  });

  it('scales without changing the drawing', () => {
    const small = renderIconSvg('forearm', 16);
    const large = renderIconSvg('forearm', 48);
    expect(small).toContain('width="16"');
    expect(large).toContain('width="48"');
    // The geometry is identical; only the box changes.
    expect(small.replace(/width="16" height="16"/, 'X')).toBe(
      large.replace(/width="48" height="48"/, 'X'),
    );
  });
});

describe('subject vocabulary', () => {
  it('carries the marks this domain actually needs', () => {
    for (const required of [
      'electrode',
      'forearm',
      'waveform',
      'spectrum',
      'commitment',
      'manifold',
      'drift',
    ]) {
      expect(ICON_NAMES).toContain(required);
    }
  });

  it('avoids the generic UI-kit vocabulary', () => {
    for (const generic of ['gear', 'cog', 'bell', 'hamburger', 'menu', 'star', 'heart']) {
      expect(ICON_NAMES).not.toContain(generic);
    }
  });

  it('places eight electrodes on the forearm ring, matching the hardware', () => {
    const ticks = ICONS.forearm.elements.filter((e) => e.kind === 'path');
    expect(ticks).toHaveLength(8);
  });
});
