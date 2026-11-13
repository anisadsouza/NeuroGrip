/**
 * The Overview hero: one gesture travelling the whole pipeline, as a
 * Remotion composition (12 s at 30 fps, looping).
 *
 *   0.0 - 2.0 s   forearm cross-section at rest
 *   1.0 - 4.0 s   muscles fire; the electrodes nearest them light up
 *   3.0 - 7.0 s   eight surface-EMG traces rise out of the noise floor
 *   5.0 - 9.5 s   evidence for "close fist" accrues; the hand starts moving
 *                 at the motion-onset mark and keeps closing
 *   9.5 - 11.3 s  the gesture latches; the hand holds
 *  11.3 - 12.0 s  release, so the loop starts from rest again
 *
 * The hand is the app's own VirtualHand, driven by the same commitment
 * fraction the commitment bar shows, so the film is a picture of the real
 * instrument and not an illustration of it. The signal is drawn from a
 * seeded noise source shaped like rectified surface EMG; it is a depiction,
 * labelled as simulated in the caption under the player.
 *
 * Colours are the design tokens, read through CSS variables, so the film
 * follows the light and dark themes with the rest of the page.
 */

import { AbsoluteFill, interpolate, random, useCurrentFrame, Easing } from 'remotion';
import { DEFAULT_EVIDENCE_CONFIG } from '@neurogrip/core';
import { VirtualHand } from '../components/VirtualHand.js';

export const HERO_FPS = 30;
export const HERO_FRAMES = 360;
export const HERO_WIDTH = 1280;
export const HERO_HEIGHT = 720;
/** A frame where the hand is latched: the poster, and the still under reduced motion. */
export const HERO_POSTER_FRAME = 312;

const ELECTRODES = 8;
const CHANNEL_GAIN = [0.95, 0.7, 0.35, 0.2, 0.25, 0.45, 0.85, 1.0];
const CLAMP = { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' } as const;

const STAGES = [
  { from: 0, label: 'Muscles fire' },
  { from: 90, label: 'Electrodes record' },
  { from: 150, label: 'Evidence builds' },
  { from: 285, label: 'Hand latches' },
] as const;

/** Commitment toward "close fist": 0 at rest, 1 at the latch. */
function commitmentAt(frame: number): number {
  const rise = interpolate(frame, [150, 285], [0, 1], { ...CLAMP, easing: Easing.inOut(Easing.cubic) });
  const release = interpolate(frame, [340, 356], [1, 0], CLAMP);
  return Math.min(rise, release);
}

/** Muscle excitation envelope, 0-1. */
function excitationAt(frame: number): number {
  const on = interpolate(frame, [30, 120], [0, 1], { ...CLAMP, easing: Easing.out(Easing.quad) });
  const off = interpolate(frame, [336, 356], [1, 0], CLAMP);
  return Math.min(on, off);
}

function CrossSection({ frame }: { frame: number }) {
  const excitation = excitationAt(frame);
  const cx = 200;
  const cy = 380;
  const R = 150;
  return (
    <g>
      <text x={cx} y={196} textAnchor="middle" className="hero-label">
        Forearm, cut across
      </text>
      <circle cx={cx} cy={cy} r={R} className="hero-skin" />
      {/* Muscle bellies: flexors on the palm side (lower), extensors on the
          back (upper). A fist is a flexor gesture, so the flexors fill. */}
      {[
        { x: cx - 70, y: cy + 70, rx: 42, ry: 30, a: -30, flexor: true },
        { x: cx + 6, y: cy + 98, rx: 46, ry: 30, a: 0, flexor: true },
        { x: cx + 82, y: cy + 58, rx: 38, ry: 28, a: 35, flexor: true },
        { x: cx - 74, y: cy - 66, rx: 36, ry: 24, a: 35, flexor: false },
        { x: cx + 40, y: cy - 92, rx: 40, ry: 24, a: -10, flexor: false },
      ].map((m) => (
        <ellipse
          key={`${m.x}-${m.y}`}
          cx={m.x}
          cy={m.y}
          rx={m.rx}
          ry={m.ry}
          transform={`rotate(${m.a} ${m.x} ${m.y})`}
          className="hero-muscle"
          style={{ opacity: m.flexor ? 0.15 + 0.7 * excitation : 0.15 + 0.1 * excitation }}
        />
      ))}
      <ellipse cx={cx - 26} cy={cy + 12} rx={34} ry={25} transform={`rotate(-25 ${cx - 26} ${cy + 12})`} className="hero-bone" />
      <circle cx={cx + 36} cy={cy - 12} r={17} className="hero-bone" />
      {Array.from({ length: ELECTRODES }, (_, index) => {
        const angle = -Math.PI / 2 + (index * 2 * Math.PI) / ELECTRODES;
        const x = cx + R * Math.cos(angle);
        const y = cy + R * Math.sin(angle);
        const lit = excitation * (CHANNEL_GAIN[index] ?? 0);
        const delay = interpolate(frame, [40 + index * 6, 70 + index * 6], [0, 1], CLAMP);
        return (
          <g key={index}>
            <circle cx={x} cy={y} r={17} className="hero-electrode" />
            <circle cx={x} cy={y} r={17} className="hero-electrode-lit" style={{ opacity: lit * delay }} />
            <text x={x} y={y + 5} textAnchor="middle" className="hero-electrode-index">
              {index + 1}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function Traces({ frame }: { frame: number }) {
  const left = 420;
  const width = 420;
  const top = 210;
  const lane = 42;
  const samples = 120;
  const excitation = excitationAt(frame);
  return (
    <g>
      <text x={left} y={190} className="hero-label">Eight sEMG channels</text>
      {Array.from({ length: ELECTRODES }, (_, channel) => {
        const mid = top + lane * channel + lane / 2;
        const amplitude = 2 + 17 * excitation * (CHANNEL_GAIN[channel] ?? 0);
        let d = '';
        for (let i = 0; i <= samples; i += 1) {
          const t = frame * 2 + i;
          const value = (random(`ch${channel}-${t}`) - 0.5) * 2;
          const x = left + (i / samples) * width;
          const y = mid + value * amplitude;
          d += `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
        }
        return (
          <g key={channel}>
            <line x1={left} x2={left + width} y1={mid} y2={mid} className="hero-baseline" />
            <path d={d} className="hero-trace" />
            <text x={left - 12} y={mid + 4} textAnchor="end" className="hero-channel">
              {channel + 1}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function EvidenceBar({ frame }: { frame: number }) {
  const left = 420;
  const width = 800;
  const y = 600;
  const height = 34;
  const commitment = commitmentAt(frame);
  const latched = commitment >= 0.999;
  const onset = DEFAULT_EVIDENCE_CONFIG.motionOnset;
  return (
    <g>
      <text x={left} y={y - 14} className="hero-label">
        Evidence for close fist
      </text>
      <text x={left + width} y={y - 14} textAnchor="end" className="hero-readout">
        {latched ? 'Latched' : commitment >= onset ? 'Moving, reversible' : 'Waiting'}
      </text>
      <rect x={left} y={y} width={width} height={height} rx={6} className="hero-track" />
      <rect
        x={left}
        y={y}
        width={Math.max(0, width * commitment)}
        height={height}
        rx={6}
        className={latched ? 'hero-fill hero-fill-latched' : 'hero-fill'}
      />
      <line x1={left + width * onset} x2={left + width * onset} y1={y - 4} y2={y + height + 4} className="hero-mark" />
      <text x={left + width * onset} y={y + height + 22} textAnchor="middle" className="hero-tick">
        hand starts moving
      </text>
      <text x={left + width} y={y + height + 22} textAnchor="end" className="hero-tick">
        locks
      </text>
    </g>
  );
}

function StageRail({ frame }: { frame: number }) {
  const current = STAGES.reduce((found, stage, index) => (frame >= stage.from ? index : found), 0);
  return (
    <g>
      {STAGES.map((stage, index) => {
        const x = 40 + index * 300;
        const active = index === current;
        return (
          <g key={stage.label}>
            <rect x={x} y={44} width={280} height={4} rx={2} className={index <= current ? 'hero-rail hero-rail-done' : 'hero-rail'} />
            <text x={x} y={92} className={active ? 'hero-stage hero-stage-active' : 'hero-stage'}>
              {index + 1}  {stage.label}
            </text>
          </g>
        );
      })}
    </g>
  );
}

export function HeroFilm() {
  const frame = useCurrentFrame();
  const commitment = commitmentAt(frame);
  const latched = commitment >= 0.999;
  const excitation = excitationAt(frame);
  return (
    <AbsoluteFill className="hero-film">
      <svg viewBox={`0 0 ${HERO_WIDTH} ${HERO_HEIGHT}`} width="100%" height="100%" aria-hidden="true">
        <StageRail frame={frame} />
        <CrossSection frame={frame} />
        <Traces frame={frame} />
        <EvidenceBar frame={frame} />
      </svg>
      <div className="hero-hand">
        <VirtualHand
          gesture="fist"
          label="Close fist"
          commitment={commitment}
          latched={latched}
          reversible={!latched && commitment >= DEFAULT_EVIDENCE_CONFIG.motionOnset}
          timedOut={false}
          drive={{
            digitFlexor: excitation * 0.9,
            digitExtensor: excitation * 0.15,
            wristFlexor: excitation * 0.3,
            wristExtensor: excitation * 0.1,
          }}
        />
      </div>
    </AbsoluteFill>
  );
}
