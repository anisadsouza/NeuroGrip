/**
 * The forearm cross-section, with each electrode shaded by what it is reading.
 *
 * This is where carmine earns its place in the palette: the ring is the one
 * view where colour intensity IS the measurement. An electrode over a working
 * muscle fills toward carmine; a quiet one stays at paper.
 *
 * Geometry comes from the same anatomy the simulator mixes through, so what is
 * drawn here corresponds to where the signal actually came from rather than
 * being a decorative circle of dots.
 */

export interface ActivationRingProps {
  /** Per-channel RMS in volts, in electrode order. */
  readonly channelRms: readonly number[];
  /** RMS that counts as full activation. */
  readonly fullScaleVolts?: number;
  readonly size?: number;
  /** Electrode flagged by the signal-quality gate, if any. */
  readonly failedChannel?: number | null;
}

const SKIN_RADIUS = 40;
const ELECTRODE_RADIUS = 4.6;

export function ActivationRing({
  channelRms,
  fullScaleVolts = 5e-4,
  size = 148,
  failedChannel = null,
}: ActivationRingProps) {
  const count = channelRms.length;
  const box = SKIN_RADIUS * 2 + ELECTRODE_RADIUS * 2 + 6;
  const centre = box / 2;

  return (
    <figure className="ring">
      <figcaption className="ring-caption">Activation</figcaption>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${box} ${box}`}
        role="img"
        aria-label={`Forearm cross-section, ${count} electrodes, shaded by activation`}
      >
        <circle
          cx={centre}
          cy={centre}
          r={SKIN_RADIUS}
          fill="none"
          stroke="var(--ng-rule-strong)"
          strokeWidth="1"
        />
        {channelRms.map((rms, index) => {
          const angle = (index / count) * Math.PI * 2 - Math.PI / 2;
          const cx = centre + SKIN_RADIUS * Math.cos(angle);
          const cy = centre + SKIN_RADIUS * Math.sin(angle);
          const level = Math.max(0, Math.min(1, rms / fullScaleVolts));
          const failed = failedChannel === index;

          return (
            <g key={index}>
              <circle
                cx={cx}
                cy={cy}
                r={ELECTRODE_RADIUS}
                fill={failed ? 'var(--ng-paper)' : 'var(--ng-muscle)'}
                fillOpacity={failed ? 1 : level}
                stroke={failed ? 'var(--ng-stop)' : 'var(--ng-ink-muted)'}
                strokeWidth={failed ? 1.6 : 1}
              />
              <text
                x={centre + (SKIN_RADIUS - 11) * Math.cos(angle)}
                y={centre + (SKIN_RADIUS - 11) * Math.sin(angle) + 3}
                textAnchor="middle"
                className="ng-num ring-index"
                fill="var(--ng-ink-faint)"
              >
                {index + 1}
              </text>
            </g>
          );
        })}
      </svg>
      <table className="visually-hidden">
        <caption>Per-channel activation, microvolts RMS</caption>
        <tbody>
          {channelRms.map((rms, index) => (
            <tr key={index}>
              <th scope="row">Channel {index + 1}</th>
              <td>{(rms * 1e6).toFixed(0)} µV</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
