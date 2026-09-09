/**
 * NeuroGrip icon set.
 *
 * Every icon is constructed on the same grid rather than drawn freehand, so
 * they share optical weight and can be checked mechanically.
 *
 * CONSTRUCTION GRID
 *   Canvas        24 x 24
 *   Safe area     2 .. 22 on both axes (2px optical margin on every side)
 *   Key stops     2, 6, 12, 18, 22 -- quarters of the safe area
 *   Ring radius   7.5 for anything electrode-shaped, 8 for anything whole-limb
 *   Stroke        1.5, butt caps, miter joins
 *
 * Square caps and miter joins are not a style choice: this is an instrument,
 * and its marks should look machined rather than handwritten. Rounded caps
 * would soften every icon toward the generic UI-kit look the design avoids.
 *
 * Fill is used in exactly three places, each time because filled/unfilled is
 * the semantic difference the icon exists to show: a live recording dot, a
 * seated electrode, and the light/dark half-disc.
 *
 * The vocabulary is drawn from the subject rather than from a generic UI kit:
 * the biphasic motor-unit action potential, the electrode ring, the forearm
 * cross-section, the power spectrum, and the commitment bar with its boundary
 * mark. There is no gear, no bell, no hamburger.
 */

export type IconElement =
  | { readonly kind: 'path'; readonly d: string; readonly filled?: boolean }
  | {
      readonly kind: 'circle';
      readonly cx: number;
      readonly cy: number;
      readonly r: number;
      readonly filled?: boolean;
    };

export interface IconDefinition {
  /** One line on why this mark looks the way it does. */
  readonly note: string;
  readonly elements: readonly IconElement[];
}

export const ICON_CANVAS = 24;
export const ICON_SAFE_MIN = 2;
export const ICON_SAFE_MAX = 22;
export const ICON_STROKE = 1.5;

const path = (d: string, filled = false): IconElement => ({ kind: 'path', d, filled });
const circle = (cx: number, cy: number, r: number, filled = false): IconElement => ({
  kind: 'circle',
  cx,
  cy,
  r,
  filled,
});

/** Eight electrode ticks around the forearm ring, at the sensor positions. */
function electrodeTicks(inner: number, outer: number): IconElement[] {
  const ticks: IconElement[] = [];
  for (let i = 0; i < 8; i++) {
    const angle = (i * Math.PI) / 4;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const round = (v: number) => Math.round(v * 100) / 100;
    ticks.push(
      path(
        `M${round(12 + inner * cos)} ${round(12 + inner * sin)}` +
          `L${round(12 + outer * cos)} ${round(12 + outer * sin)}`,
      ),
    );
  }
  return ticks;
}

export const ICONS = {
  /* ---- Subject vocabulary --------------------------------------------- */

  electrode: {
    note: 'A seated electrode: contact ring with the conductive centre filled.',
    elements: [circle(12, 12, 7.5), circle(12, 12, 2.5, true)],
  },

  forearm: {
    note: 'Forearm cross-section with the eight-electrode ring, viewed distally.',
    elements: [circle(12, 12, 7.5), ...electrodeTicks(7.5, 9.8)],
  },

  waveform: {
    note: 'A biphasic motor-unit action potential -- the signature trace of this field.',
    elements: [path('M2 13 H7 L9 6 L12 20 L14 11 L16 13 H22')],
  },

  spectrum: {
    note: 'Power spectral density: band powers over a baseline.',
    elements: [
      path('M2 20 H22'),
      path('M5 20 V10'),
      path('M9 20 V6'),
      path('M12 20 V12'),
      path('M16 20 V15'),
      path('M19 20 V17'),
    ],
  },

  commitment: {
    note: 'The commitment bar: a track filled part-way, with the risk boundary standing beyond the fill.',
    elements: [
      path('M3 9 H21 V15 H3 Z'),
      path('M3 9 H12 V15 H3 Z', true),
      path('M17 5.5 V18.5'),
    ],
  },

  manifold: {
    note: 'The cartograph: nested reachability contours around an off-centre peak.',
    elements: [
      path('M12 4.5 C17.5 4.5 20.5 8 20 12.5 C19.5 17 15.5 19.5 11 19.5 C6.5 19.5 3.5 16 3.5 11.5 C3.5 7 7 4.5 12 4.5 Z'),
      path('M11 8.5 C14 8.5 15.5 10 15.5 12 C15.5 14 13.5 15.5 11 15.5 C8.5 15.5 7.5 14 7.5 12 C7.5 10 8.5 8.5 11 8.5 Z'),
      circle(11.4, 12, 1.1, true),
    ],
  },

  drift: {
    note: 'A trace departing from its baseline: the drift and fatigue indicator.',
    elements: [path('M2 16 H22'), path('M2 16 L9 15 L15 10.5 L22 4')],
  },

  /* ---- Transport ------------------------------------------------------- */

  play: {
    note: 'Start the stream. A hard-edged triangle; rounding it would soften the whole set.',
    elements: [path('M8 4 L19 12 L8 20 Z')],
  },

  pause: {
    note: 'Hold the stream. Two bars at the same weight as every other stroke.',
    elements: [path('M9 5 V19'), path('M15 5 V19')],
  },

  stop: {
    note: 'End the session. A hollow square, so stopping reads as less final than recording.',
    elements: [path('M6 6 H18 V18 H6 Z')],
  },

  record: {
    note: 'Capturing. Filled, because filled is what distinguishes live from idle.',
    elements: [circle(12, 12, 6, true)],
  },

  reset: {
    note: 'Return to the starting state: a three-quarter turn back to the origin.',
    elements: [path('M12 4 A8 8 0 1 0 20 12'), path('M9 1 L12 4 L9 7')],
  },

  /* ---- State and control ----------------------------------------------- */

  alert: {
    note: 'Something needs attention. Triangle, because triangles mean caution.',
    elements: [path('M12 3 L22 20 H2 Z'), path('M12 9 V14'), circle(12, 17, 0.9, true)],
  },

  check: {
    note: 'Passed, within tolerance, accepted.',
    elements: [path('M4 13 L9.5 18.5 L20 6')],
  },

  close: {
    note: 'Dismiss. A true diagonal cross rather than a rotated plus, so it cannot be misread as add.',
    elements: [path('M5 5 L19 19'), path('M19 5 L5 19')],
  },

  chevronDown: {
    note: 'Disclose the detail beneath.',
    elements: [path('M5 9 L12 16 L19 9')],
  },

  theme: {
    note: 'Light and dark, as one disc half-filled.',
    elements: [circle(12, 12, 7.5), path('M12 4.5 A7.5 7.5 0 0 1 12 19.5 Z', true)],
  },

  sliders: {
    note: 'Adjust parameters. Sliders, not a gear -- these are continuous values.',
    elements: [
      path('M2 7 H22'),
      path('M2 12 H22'),
      path('M2 17 H22'),
      path('M8 4.5 V9.5'),
      path('M16 9.5 V14.5'),
      path('M6 14.5 V19.5'),
    ],
  },
} as const satisfies Record<string, IconDefinition>;

export type IconName = keyof typeof ICONS;

export const ICON_NAMES = Object.keys(ICONS) as IconName[];

/**
 * Render an icon as a standalone SVG string.
 *
 * `currentColor` throughout, so an icon takes the colour of the text it sits
 * beside and never needs a colour prop.
 */
export function renderIconSvg(name: IconName, size = 24, title?: string): string {
  const definition = ICONS[name];
  const body = definition.elements
    .map((element) => {
      const fill = element.filled ? 'currentColor' : 'none';
      const stroke = element.filled ? 'none' : 'currentColor';
      if (element.kind === 'circle') {
        return `<circle cx="${element.cx}" cy="${element.cy}" r="${element.r}" fill="${fill}" stroke="${stroke}"/>`;
      }
      return `<path d="${element.d}" fill="${fill}" stroke="${stroke}"/>`;
    })
    .join('');

  const label = title
    ? ` role="img" aria-label="${title}"`
    : ' aria-hidden="true" focusable="false"';

  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" ` +
    `viewBox="0 0 ${ICON_CANVAS} ${ICON_CANVAS}" stroke-width="${ICON_STROKE}" ` +
    `stroke-linecap="butt" stroke-linejoin="miter"${label}>${body}</svg>`
  );
}

/** Every numeric coordinate in an icon, for mechanical bounds checking. */
export function iconCoordinates(name: IconName): number[] {
  const values: number[] = [];
  for (const element of ICONS[name].elements) {
    if (element.kind === 'circle') {
      values.push(
        element.cx - element.r,
        element.cx + element.r,
        element.cy - element.r,
        element.cy + element.r,
      );
      continue;
    }
    // Path commands: pull every number, ignoring the letters between them.
    // Arc flags are single digits and land inside the safe area anyway.
    for (const match of element.d.matchAll(/-?\d+(?:\.\d+)?/g)) {
      values.push(Number.parseFloat(match[0]));
    }
  }
  return values;
}
