/**
 * Eight channels of sEMG, drawn as stacked lanes on a graticule.
 *
 * Canvas rather than SVG: this redraws at up to 50 Hz and an SVG polyline per
 * channel would churn the DOM. Canvas also lets the trace be drawn with the
 * calibration marks a chart recorder would print, which is the point of the
 * graticule -- a trace without a stated scale is decoration.
 *
 * Channels share one vertical scale so their relative amplitudes stay
 * comparable. Per-channel autoscaling would make a quiet channel look as
 * active as a loud one, which is exactly the comparison the wearer needs.
 */

import { useEffect, useRef } from 'react';
import { DISPLAY_POINTS } from '../worker/protocol.js';

export interface OscilloscopeProps {
  /** nChannels x DISPLAY_POINTS, in volts. */
  readonly display: Float32Array | null;
  readonly nChannels: number;
  /**
   * Full-scale deflection for one lane, in volts. 600 uV puts a strong
   * contraction near the top of its lane without clipping the spikes; the
   * calibration label reports whatever this is set to, so the trace is never
   * shown without its scale.
   */
  readonly fullScaleVolts?: number;
  readonly windowMs?: number;
}

function cssVar(element: HTMLElement, name: string, fallback: string): string {
  return getComputedStyle(element).getPropertyValue(name).trim() || fallback;
}

export function Oscilloscope({
  display,
  nChannels,
  fullScaleVolts = 6e-4,
  windowMs = 200,
}: OscilloscopeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width === 0 || height === 0) return;

    if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const trace = cssVar(canvas, '--ng-muscle', '#a8324a');
    const rule = cssVar(canvas, '--ng-rule', '#d8dbd8');
    const laneHeight = height / nChannels;

    // Lane separators, so eight channels do not read as one tangled trace.
    context.strokeStyle = rule;
    context.lineWidth = 1;
    for (let c = 1; c < nChannels; c++) {
      const y = Math.round(c * laneHeight) + 0.5;
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }

    if (!display) return;

    context.strokeStyle = trace;
    context.lineWidth = 1.1;
    context.lineJoin = 'miter';
    context.lineCap = 'butt';

    const step = width / (DISPLAY_POINTS - 1);
    const amplitude = (laneHeight * 0.44) / fullScaleVolts;

    for (let c = 0; c < nChannels; c++) {
      const centre = laneHeight * (c + 0.5);
      context.beginPath();
      for (let p = 0; p < DISPLAY_POINTS; p++) {
        const value = display[c * DISPLAY_POINTS + p] ?? 0;
        const y = centre - Math.max(-laneHeight / 2, Math.min(laneHeight / 2, value * amplitude));
        if (p === 0) context.moveTo(0, y);
        else context.lineTo(p * step, y);
      }
      context.stroke();
    }
  }, [display, nChannels, fullScaleVolts]);

  return (
    <figure className="scope">
      <figcaption className="scope-caption">
        <span>Signal</span>
        <span className="ng-num scope-cal">
          {(fullScaleVolts * 1e6).toFixed(0)} µV · {windowMs} ms
        </span>
      </figcaption>
      <div className="scope-bed ng-graticule">
        <canvas ref={canvasRef} className="scope-canvas" />
      </div>
      <p className="visually-hidden">
        {nChannels} channels of surface electromyography, {windowMs} millisecond
        window, full scale {(fullScaleVolts * 1e6).toFixed(0)} microvolts.
      </p>
    </figure>
  );
}
