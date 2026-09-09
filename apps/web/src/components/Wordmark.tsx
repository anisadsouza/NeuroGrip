/**
 * The NeuroGrip wordmark: a signal plotted through the name.
 *
 * Two ideas in one object.
 *
 * The letters are **constructed, not typeset** -- drawn from four primitives
 * on a grid of cap height 100 and stem 13: the rectangle, the half ellipse,
 * the ring and the diagonal cut. Every coordinate below falls out of those two
 * numbers. It is the same discipline the icon set is built on, so the mark and
 * the icons read as one family, and no display face is loaded only to say the
 * name once.
 *
 * Through them runs a **surface-EMG trace**: quiet at the left, a burst of
 * motor-unit activity through the middle, settling to a held line under GRiP.
 * Signal in, grip out, read left to right. The letters are cut away around it
 * rather than sitting behind it, so the trace is plotted *through* the word --
 * one object, not a logo with a graph stuck beside it.
 *
 * The weight is heavy and the tracking tight, so the word reads as one mass
 * with the trace cut through it. Counters are correspondingly small -- at this
 * stem a generous counter makes a letter read as a ring rather than as a solid
 * form -- and the mark is set small in the frame, where its density does the
 * work that size would otherwise have to.
 *
 * The trace **fades at both ends** rather than stopping. A line that ends in a
 * cap looks cropped, as though the mark were a detail of something larger; one
 * that dissolves reads as a signal passing through, which is what it is. The
 * same fade is applied to the channel cut out of the letters, so where the
 * trace has faded the letters are whole again.
 *
 * The lowercase **i** is the one irregularity: its dot is the only
 * free-standing disc in the mark, so the eye lands there, and it matches how
 * the product already writes itself -- NeuroGrip, not NEUROGRIP.
 *
 * Monochrome, always. Carmine means muscle activation in this interface and
 * nothing else, so the mark is not allowed to borrow it.
 */

import { useId } from 'react';

/** Cap height. Every measure below is a fraction of this. */
const CAP = 100;

/** Stem width. Heavy: this is the weight that makes the mark read as one mass. */
const STEM = 28;

/** Space between letters. Tight, so the word closes up into that mass. */
const TRACK = 10;

/** Where the trace reaches full strength, as a fraction of the mark's width. */
const FADE = 0.16;

/** The plotted trace. A hairline against the letters, as in a figure. */
const TRACE_STROKE = 3;

/**
 * The channel cut through the letters.
 *
 * Barely wider than the trace itself. A wide channel does not read as a signal
 * passing behind the word -- it erases whatever it crosses, and the first
 * casualty is the E, whose waist bar it removes outright. A narrow one nicks
 * the letters and leaves the eye to close them again.
 */
const TRACE_CHANNEL = TRACE_STROKE + 3.5;

/**
 * How far the trace departs from the midline at its loudest.
 *
 * An earlier draft ran to a third of the cap and the burst ate the O and the G
 * outright -- a mark you cannot read is not a mark. The signal modulates the
 * word; it does not replace it.
 */
const TRACE_GAIN = 15;

/** An axis-aligned rectangle, as path data. */
const rect = (x: number, y: number, w: number, h: number): string =>
  `M${x},${y}h${w}v${h}h${-w}z`;

/** A closed polygon through the given points. */
const poly = (points: readonly [number, number][]): string =>
  `M${points.map(([x, y]) => `${x},${y}`).join('L')}z`;

/**
 * A bowl: the half ellipse that closes R and P.
 *
 * Elliptical rather than circular so reach and height are independent -- a
 * semicircular bowl wide enough to balance the O would have to stand taller
 * than the cap. Drawn as a single subpath, out along the outer curve and back
 * along the counter, so it fills correctly under the non-zero rule that the
 * overlapping bars elsewhere require.
 */
function bowl(top: number, height: number, reach: number): string {
  const rx = reach - STEM;
  const ry = height / 2;
  const cy = top + ry;
  return (
    `M${STEM},${top}` +
    `A${rx},${ry} 0 0 1 ${STEM},${top + height}` +
    `V${cy + (ry - STEM)}` +
    `A${rx - STEM},${ry - STEM} 0 0 0 ${STEM},${cy - (ry - STEM)}` +
    'z'
  );
}

/** A ring, as two concentric circles. Needs the even-odd rule to stay open. */
function ring(cx: number, cy: number, outer: number): string {
  const inner = outer - STEM;
  return (
    `M${cx - outer},${cy}a${outer},${outer} 0 1 0 ${outer * 2},0a${outer},${outer} 0 1 0 ${-outer * 2},0z` +
    `M${cx - inner},${cy}a${inner},${inner} 0 1 0 ${inner * 2},0a${inner},${inner} 0 1 0 ${-inner * 2},0z`
  );
}

// Tall enough that the counter survives the stem: at this weight a shorter
// bowl closes to a slit and the letter reads as a solid block.
const BOWL_HEIGHT = 76;
const BOWL_REACH = 70;

interface Glyph {
  readonly width: number;
  /** Filled non-zero: bars are allowed to overlap the stem. */
  readonly path: string;
  /** Filled even-odd, for the letters built as a true ring. */
  readonly openPath?: string;
  /**
   * Cut away from this letter before anything is drawn.
   *
   * Only the G needs it: a G is a ring with its upper right removed, and
   * subtraction is the one operation a fill rule cannot express -- an
   * overlapping subpath would invert the ring rather than open it.
   */
  readonly notch?: string;
}

/** The diagonal's vertical run across the N, at the stem's own thickness. */
const N_WIDTH = 92;
const N_RISE = (STEM * CAP) / (N_WIDTH - 2 * STEM);

const GLYPHS: Record<string, Glyph> = {
  // Two stems joined by a diagonal cut of the same thickness.
  N: {
    width: N_WIDTH,
    path: poly([
      [0, 0],
      [STEM, 0],
      [N_WIDTH - STEM, CAP - N_RISE],
      [N_WIDTH - STEM, 0],
      [N_WIDTH, 0],
      [N_WIDTH, CAP],
      [N_WIDTH - STEM, CAP],
      [STEM, N_RISE],
      [STEM, CAP],
      [0, CAP],
    ]),
  },

  // Three bars off one stem. The waist bar stops short of the others, which is
  // what keeps the letter from reading as a closed box.
  E: {
    width: 70,
    path:
      rect(0, 0, STEM, CAP) +
      rect(0, 0, 70, STEM) +
      // The waist sits a little above centre, as it does in a book face: at
      // true centre the letter looks bottom-heavy.
      rect(0, (CAP - STEM) / 2 - 5, 56, STEM) +
      rect(0, CAP - STEM, 70, STEM),
  },

  // Two stems closed by a half disc, the counter struck from the same centre.
  U: (() => {
    const width = 78;
    const outer = width / 2;
    const inner = outer - STEM;
    const shoulder = CAP - outer;
    return {
      width,
      path:
        `M0,0V${shoulder}A${outer},${outer} 0 0 0 ${width},${shoulder}V0` +
        `H${width - STEM}V${shoulder}A${inner},${inner} 0 0 1 ${STEM},${shoulder}V0z`,
    };
  })(),

  // A true ring. With the trace passing through it, this is the letter that
  // most looks like an electrode.
  O: { width: CAP, path: '', openPath: ring(CAP / 2, CAP / 2, CAP / 2) },

  // A ring with its upper right cut away and a bar run back to its centre.
  G: {
    width: CAP,
    path: rect(CAP / 2, (CAP - STEM) / 2, CAP / 2, STEM),
    openPath: ring(CAP / 2, CAP / 2, CAP / 2),
    notch: rect(CAP / 2, 0, CAP / 2, (CAP - STEM) / 2 - 5),
  },

  R: {
    width: 84,
    path:
      rect(0, 0, STEM, CAP) +
      bowl(0, BOWL_HEIGHT, BOWL_REACH) +
      // The leg leaves the stem at the bowl's closing edge and lands on the
      // baseline at the bowl's full reach, so R and P share a footprint.
      poly([
        [STEM, BOWL_HEIGHT - STEM - 2],
        [STEM + STEM, BOWL_HEIGHT - STEM - 2],
        [84, CAP],
        [84 - STEM - 4, CAP],
      ]),
  },

  P: { width: BOWL_REACH, path: rect(0, 0, STEM, CAP) + bowl(0, BOWL_HEIGHT, BOWL_REACH) },

  // Lowercase, and the only free-standing disc in the mark.
  i: (() => {
    const dot = STEM * 0.72;
    return {
      width: STEM,
      path:
        rect(0, 46, STEM, CAP - 46) +
        `M${STEM / 2 - dot},${dot}a${dot},${dot} 0 1 0 ${dot * 2},0a${dot},${dot} 0 1 0 ${-dot * 2},0z`,
    };
  })(),
};

const WORD = [...'NEUROGRiP'];

const LAID_OUT = WORD.reduce<{ glyphs: { glyph: Glyph; x: number; key: string }[]; x: number }>(
  (state, letter, index) => {
    const glyph = GLYPHS[letter]!;
    state.glyphs.push({ glyph, x: state.x, key: `${letter}-${index}` });
    state.x += glyph.width + TRACK;
    return state;
  },
  { glyphs: [], x: 0 },
);

const WIDTH = LAID_OUT.x - TRACK;

/**
 * The trace, as offsets from the midline in fractions of the gain.
 *
 * Written out rather than generated: this is a drawing, and a drawing wants to
 * be the same every time it is rendered. The shape is one gesture -- baseline
 * noise, recruitment, a peak, then the settle into a held contraction -- the
 * same sequence the commitment bar shows one panel over.
 */
const TRACE: readonly number[] = [
  0, 0.05, -0.07, 0.06, -0.04, 0.2, -0.36, 0.55, -0.72, 0.88, -1, 0.8, -0.94,
  0.64, -0.76, 0.92, -0.57, 0.46, -0.62, 0.36, -0.42, 0.27, -0.31, 0.19, -0.21,
  0.13, -0.15, 0.09, -0.1, 0.06, -0.05, 0.04, -0.03, 0.02, 0,
];

/** The trace as a polyline across the full width of the mark. */
const TRACE_PATH = TRACE.map((offset, index) => {
  const x = (index / (TRACE.length - 1)) * WIDTH;
  return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${(CAP / 2 + offset * TRACE_GAIN).toFixed(1)}`;
}).join('');

export interface WordmarkProps {
  /** Rendered cap height. Everything scales from it. */
  readonly height?: string;
  readonly className?: string;
}

export function Wordmark({ height, className }: WordmarkProps) {
  // Scoped, so two marks on one page cannot claim the same mask.
  const id = useId();
  const maskId = `${id}-trace`;
  const fadeId = `${id}-fade`;
  const fadeMaskId = `${id}-fademask`;
  const channelId = `${id}-channel`;

  return (
    <svg
      className={className ? `wordmark ${className}` : 'wordmark'}
      viewBox={`0 0 ${WIDTH} ${CAP}`}
      style={height ? { height } : undefined}
      role="img"
      aria-label="NeuroGrip"
    >
      <defs>
        {/*
          Two ramps of the same shape, opposite polarity, because a mask reads
          white as keep and black as remove and the two jobs want opposite
          things. `fade` keeps the trace through the middle and dissolves it at
          the ends; `channel` removes the letters through the middle and leaves
          them whole where the trace has gone.
        */}
        <linearGradient
          id={fadeId}
          gradientUnits="userSpaceOnUse"
          x1="0"
          y1="0"
          x2={WIDTH}
          y2="0"
        >
          <stop offset="0" stopColor="black" />
          <stop offset={FADE} stopColor="white" />
          <stop offset={1 - FADE} stopColor="white" />
          <stop offset="1" stopColor="black" />
        </linearGradient>

        <linearGradient
          id={channelId}
          gradientUnits="userSpaceOnUse"
          x1="0"
          y1="0"
          x2={WIDTH}
          y2="0"
        >
          <stop offset="0" stopColor="white" />
          <stop offset={FADE} stopColor="black" />
          <stop offset={1 - FADE} stopColor="black" />
          <stop offset="1" stopColor="white" />
        </linearGradient>

        <mask id={maskId}>
          <rect x="0" y="0" width={WIDTH} height={CAP} fill="white" />
          {/* Black hides: this is the channel the trace runs down. */}
          <path
            d={TRACE_PATH}
            fill="none"
            stroke={`url(#${channelId})`}
            strokeWidth={TRACE_CHANNEL}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </mask>

        <mask id={fadeMaskId}>
          <rect x="0" y="0" width={WIDTH} height={CAP} fill={`url(#${fadeId})`} />
        </mask>

        {LAID_OUT.glyphs.map(({ glyph, x, key }) =>
          glyph.notch ? (
            <clipPath key={key} id={`${id}-${key}`} clipPathUnits="userSpaceOnUse">
              <path
                d={`M0,0h${glyph.width}v${CAP}h${-glyph.width}z${glyph.notch}`}
                clipRule="evenodd"
              />
            </clipPath>
          ) : null,
        )}
      </defs>

      <g mask={`url(#${maskId})`}>
        {LAID_OUT.glyphs.map(({ glyph, x, key }) => (
          <g
            key={key}
            transform={`translate(${x} 0)`}
            clipPath={glyph.notch ? `url(#${id}-${key})` : undefined}
          >
            {glyph.path ? <path d={glyph.path} fill="currentColor" /> : null}
            {glyph.openPath ? (
              <path d={glyph.openPath} fill="currentColor" fillRule="evenodd" />
            ) : null}
          </g>
        ))}
      </g>

      <path
        d={TRACE_PATH}
        fill="none"
        stroke="currentColor"
        strokeWidth={TRACE_STROKE}
        strokeLinecap="round"
        strokeLinejoin="round"
        mask={`url(#${fadeMaskId})`}
      />
    </svg>
  );
}
