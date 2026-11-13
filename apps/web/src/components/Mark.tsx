/**
 * The NeuroGrip mark: a forearm cut through at the electrode band.
 *
 * A thin skin ring carries eight dome electrodes, the hardware this project
 * decodes; inside it sit the radius (the tilted oval) and the ulna (the small
 * round), so the ring reads as a limb rather than a dial. Redesigned
 * 2026-11-08 for the clinical direction, replacing the Ring G.
 *
 * Constructed on a 256 grid: skin ring radii 89 and 99, electrodes r 15 on
 * the 94 circle at 45-degree stops starting at twelve o'clock, radius bone an
 * ellipse 30 x 22 at -25 degrees, ulna r 15. Generated once, pasted here so
 * the mark has one source of truth and no runtime trigonometry.
 *
 * Monochrome, always: it inherits currentColor. Colour in this interface
 * marks the accent or data, and the mark is neither.
 */

const RING =
  'M29 128 A99 99 0 1 0 227 128 A99 99 0 1 0 29 128 Z M39 128 A89 89 0 1 1 217 128 A89 89 0 1 1 39 128 Z';

const ELECTRODES = [
  [128, 34],
  [194.47, 61.53],
  [222, 128],
  [194.47, 194.47],
  [128, 222],
  [61.53, 194.47],
  [34, 128],
  [61.53, 61.53],
] as const;

const BONES =
  'M84.81 152.68 A30 22 -25 1 0 139.19 127.32 A30 22 -25 1 0 84.81 152.68 Z M145 118 A15 15 0 1 0 175 118 A15 15 0 1 0 145 118 Z';

export function BrandMark({ size = 32, title }: { size?: number; title?: string }) {
  const accessibility = title
    ? ({ role: 'img', 'aria-label': title } as const)
    : ({ 'aria-hidden': true, focusable: false } as const);
  return (
    <svg className="brand-mark" width={size} height={size} viewBox="0 0 256 256" {...accessibility}>
      <path d={RING} fillRule="evenodd" />
      {ELECTRODES.map(([x, y]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r={15} />
      ))}
      <path d={BONES} />
    </svg>
  );
}

/** Mark plus name. The name is live text in the interface face, so it is crisp at every size and readable by assistive technology. */
export function BrandLockup({ size = 28 }: { size?: number }) {
  return (
    <span className="brand-lockup">
      <BrandMark size={size} />
      <span className="brand-name">NeuroGrip</span>
    </span>
  );
}
