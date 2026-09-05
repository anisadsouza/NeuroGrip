/**
 * Live decoding.
 *
 * The wearer picks an intended gesture, the replay feeds that signal, and the
 * decoder responds in real time. Four things are on screen at once and they
 * answer four different questions:
 *
 *   Signal      is my hardware working?
 *   Activation  which muscles am I using?
 *   Commitment  how much evidence is there, and against what boundary?
 *   Hand        what is the prosthesis doing right now, and can I still stop it?
 *
 * The last two share a row because they are one story told twice: the bar is
 * the quantity, the hand is its physical consequence, and both are driven by
 * the same commitment fraction. They get the most weight on the screen because
 * they are the only parts that answer a question about the future.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { DEFAULT_EVIDENCE_CONFIG } from '@neurogrip/core';
import { ActivationRing } from '../components/ActivationRing.js';
import { CommitmentBar } from '../components/CommitmentBar.js';
import { Oscilloscope } from '../components/Oscilloscope.js';
import { NO_DRIVE, VirtualHand, type MuscleDrive } from '../components/VirtualHand.js';
import { Icon } from '../components/Icon.js';
import { ReplaySource, loadReplayBundle, type ReplayBundle } from '../sources/replaySource.js';
import type { DecisionResponse, WorkerRequest, WorkerResponse } from '../worker/protocol.js';

/**
 * Commit cost per gesture, in the model's class order.
 *
 * This is a safety judgement, not a tuning parameter. The question for each
 * gesture is how hard the mistake is to undo: a power grip closing on
 * something fragile, or on a person, is the expensive error. Opening a hand
 * drops what you were holding, which is recoverable. Rest costs nothing, which
 * is why it is also the fallback when nothing else commits.
 */
const COMMIT_COST: Readonly<Record<string, number>> = {
  rest: 0,
  fist: 1.0,
  spherical_grip: 1.0,
  pinch: 0.6,
  two_finger: 0.6,
  wrist_flexion: 0.3,
  wrist_extension: 0.3,
  point: 0.2,
  thumb_up: 0.2,
  open_hand: 0.1,
};

/** Plain-language names. The model's identifiers are not wearer-facing copy. */
const GESTURE_LABEL: Readonly<Record<string, string>> = {
  rest: 'Rest',
  fist: 'Close fist',
  open_hand: 'Open hand',
  pinch: 'Pinch',
  point: 'Point',
  wrist_flexion: 'Bend wrist in',
  wrist_extension: 'Bend wrist back',
  thumb_up: 'Thumb up',
  two_finger: 'Two-finger grip',
  spherical_grip: 'Hold a ball',
};

/**
 * Channel RMS that reads as full activation on the electrode ring, in volts.
 *
 * Measured off the replay bundle rather than assumed: across every gesture,
 * a single channel peaks at 953 uV and a resting one sits at 16 uV. One
 * millivolt puts the loudest electrode just under saturation and leaves rest
 * essentially dark.
 *
 * Defined here and passed to the ring rather than left to the ring's own
 * default, so the two views of this signal cannot drift apart.
 */
const FULL_SCALE_VOLTS = 1e-3;

/**
 * The same idea for a muscle group rather than a single electrode.
 *
 * A group's drive is a weighted mean over the ring. The weights concentrate on
 * the electrodes nearest that muscle, so the mean tracks the loudest channel
 * fairly closely rather than being dragged down by the far side of the
 * forearm -- measured, the group drive peaks at 656 uV against a per-channel
 * peak of 953. Seven hundred puts the loudest gesture in the vocabulary at
 * about 0.94 and pins nothing.
 *
 * Two earlier values were guessed at from the simulator's stated RMS range and
 * both saturated on a power grip, which made a strong contraction and a
 * fatigued one draw the same colour. This one comes from the signal.
 */
const FULL_DRIVE_VOLTS = 7e-4;

interface Status {
  ready: boolean;
  threaded: boolean;
  error: string | null;
  rejected: { reason: string; channel: number | null } | null;
}

export function Live() {
  const workerRef = useRef<Worker | null>(null);
  const sourceRef = useRef<ReplaySource | null>(null);

  const [bundle, setBundle] = useState<ReplayBundle | null>(null);
  const [gestures, setGestures] = useState<readonly string[]>([]);
  const [decision, setDecision] = useState<DecisionResponse | null>(null);
  const [intended, setIntended] = useState(0);
  // Mirrored in a ref because the transport callback reads it. Reading the
  // state value there captures whatever React last rendered, which is stale
  // if the wearer picks a gesture and presses start within the same task.
  const intendedRef = useRef(0);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<Status>({
    ready: false,
    threaded: false,
    error: null,
    rejected: null,
  });
  const [latency, setLatency] = useState<number[]>([]);

  const risks = useMemo(
    () => gestures.map((name) => COMMIT_COST[name] ?? 0.5),
    [gestures],
  );
  const gestureLabels = useMemo(
    () => gestures.map((name) => GESTURE_LABEL[name] ?? name),
    [gestures],
  );

  /**
   * Muscle-group drive, straight off the electrodes.
   *
   * This deliberately does not pass through the decoder. The hand's shape is
   * what the decoder concluded; its colour is what the muscles are doing, and
   * the two are worth being able to disagree. The weighting comes from the
   * replay manifest, which computes it from the same forearm anatomy the
   * simulator mixes through.
   */
  const drive = useMemo<MuscleDrive>(() => {
    const groups = bundle?.manifest.muscleGroups;
    const rms = decision?.channelRms;
    if (!groups || !rms) return NO_DRIVE;

    const project = (weights: readonly number[]) => {
      let total = 0;
      for (let i = 0; i < weights.length; i++) total += weights[i]! * (rms[i] ?? 0);
      return Math.min(1, Math.max(0, total / FULL_DRIVE_VOLTS));
    };

    return {
      digitFlexor: project(groups.digit_flexor),
      digitExtensor: project(groups.digit_extensor),
      wristFlexor: project(groups.wrist_flexor),
      wristExtensor: project(groups.wrist_extensor),
    };
  }, [bundle, decision]);

  useEffect(() => {
    let disposed = false;

    const worker = new Worker(new URL('../worker/inference.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      if (disposed) return;
      const message = event.data;
      switch (message.type) {
        case 'ready':
          setGestures(message.gestures);
          setStatus((s) => ({ ...s, ready: true, threaded: message.threaded }));
          break;
        case 'decision':
          setDecision(message);
          setStatus((s) => (s.rejected ? { ...s, rejected: null } : s));
          setLatency((previous) => {
            const next = [...previous, message.latencyMs];
            return next.length > 200 ? next.slice(-200) : next;
          });
          break;
        case 'rejected':
          setStatus((s) => ({
            ...s,
            rejected: { reason: message.reason, channel: message.failedChannel },
          }));
          break;
        case 'error':
          setStatus((s) => ({ ...s, error: message.message }));
          break;
      }
    };

    loadReplayBundle(`${import.meta.env.BASE_URL}replay`)
      .then((loaded) => {
        if (disposed) return;
        setBundle(loaded);
        const request: WorkerRequest = {
          type: 'init',
          modelUrl: `${import.meta.env.BASE_URL}models/decoder.onnx`,
          nChannels: loaded.manifest.nChannels,
          samplingRateHz: loaded.manifest.samplingRateHz,
          risks: loaded.manifest.gestures.map((name) => COMMIT_COST[name] ?? 0.5),
        };
        worker.postMessage(request);
      })
      .catch((error: unknown) => {
        if (!disposed) {
          setStatus((s) => ({ ...s, error: String(error) }));
        }
      });

    return () => {
      disposed = true;
      sourceRef.current?.stop();
      worker.terminate();
    };
  }, []);

  const toggleRunning = useCallback(() => {
    const worker = workerRef.current;
    if (!worker || !bundle) return;

    if (sourceRef.current?.isRunning) {
      sourceRef.current.stop();
      setRunning(false);
      return;
    }

    if (!sourceRef.current) {
      sourceRef.current = new ReplaySource(bundle, (samples, nChannels) => {
        worker.postMessage({ type: 'samples', samples, nChannels } satisfies WorkerRequest, [
          samples.buffer,
        ]);
      });
    }
    sourceRef.current.selectGesture(intendedRef.current);
    sourceRef.current.start();
    setRunning(true);
  }, [bundle]);

  const chooseGesture = useCallback((index: number) => {
    intendedRef.current = index;
    setIntended(index);
    sourceRef.current?.selectGesture(index);
    workerRef.current?.postMessage({ type: 'reset' } satisfies WorkerRequest);
  }, []);

  const p95 = useMemo(() => {
    if (latency.length < 20) return null;
    const sorted = [...latency].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length * 0.95)]!;
  }, [latency]);

  const nChannels = bundle?.manifest.nChannels ?? 8;
  const correct = decision !== null && decision.latched && decision.latchedClass === intended;

  if (status.error) {
    return (
      <div className="panel notice notice-stop">
        <Icon name="alert" label="Error" />
        <div>
          <h2>The decoder could not start</h2>
          <p className="ng-num">{status.error}</p>
          <p>
            Run <code>npm run build:assets</code> to regenerate the model and replay
            bundle, then reload.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="live">
      <div className="live-main">
        <Oscilloscope
          display={decision?.display ?? null}
          nChannels={nChannels}
          windowMs={200}
        />

        {status.rejected ? (
          <p className="notice notice-stop" role="status">
            <Icon name="alert" label="Signal rejected" />
            <span>
              {status.rejected.reason}
              {status.rejected.channel !== null
                ? ` on channel ${status.rejected.channel + 1}`
                : ''}
              . No gesture is decoded from a rejected window.
            </span>
          </p>
        ) : null}

        <div className="actuation">
          {/* The hand is shown from the first frame, at rest, so the wearer
              sees what it looks like before anything is decoded. */}
          <VirtualHand
            gesture={decision ? (gestures[decision.leader] ?? '') : 'rest'}
            label={
              decision
                ? (GESTURE_LABEL[gestures[decision.leader] ?? ''] ?? 'this gesture')
                : 'Rest'
            }
            commitment={decision?.commitment ?? 0}
            latched={decision?.latched ?? false}
            reversible={decision?.reversible ?? false}
            timedOut={decision?.timedOut ?? false}
            drive={drive}
          />

          <div className="actuation-readout">
            {gestures.length > 0 && decision ? (
              <CommitmentBar
                gestures={gestures}
                labels={gestureLabels}
                leader={decision.leader}
                commitment={decision.commitment}
                latched={decision.latched}
                reversible={decision.reversible}
                timedOut={decision.timedOut}
                evidence={decision.evidence}
                effectiveThreshold={decision.effectiveThreshold}
                risks={risks}
                baseThreshold={DEFAULT_EVIDENCE_CONFIG.baseThreshold}
                riskWeight={DEFAULT_EVIDENCE_CONFIG.riskWeight}
                motionOnset={DEFAULT_EVIDENCE_CONFIG.motionOnset}
              />
            ) : (
              <p className="panel muted">
                {status.ready ? 'Press start to begin decoding.' : 'Loading the decoder…'}
              </p>
            )}

            <div className="panel">
              <h2>Decoder</h2>
              <dl className="readout-list">
                <div>
                  <dt>Decoded</dt>
                  <dd data-correct={decision?.latched ? correct : undefined}>
                    {decision?.latched
                      ? decision.timedOut
                        ? 'No clear intent'
                        : (GESTURE_LABEL[gestures[decision.latchedClass ?? 0] ?? ''] ?? '—')
                      : '—'}
                  </dd>
                </div>
                <div>
                  <dt>Latency, this window</dt>
                  <dd className="ng-num">
                    {decision ? `${decision.latencyMs.toFixed(2)} ms` : '—'}
                  </dd>
                </div>
                <div>
                  <dt>Latency, P95</dt>
                  <dd className="ng-num">{p95 ? `${p95.toFixed(2)} ms` : 'measuring…'}</dd>
                </div>
                <div>
                  <dt>WASM threads</dt>
                  <dd className="ng-num">{status.threaded ? 'enabled' : 'single'}</dd>
                </div>
              </dl>
            </div>
          </div>
        </div>
      </div>

      <aside className="live-side">
        <div className="panel">
          <h2>Intended gesture</h2>
          <p className="muted small">
            Choose what you are attempting. The decoder is not told your choice.
          </p>
          <div className="gesture-choices" role="radiogroup" aria-label="Intended gesture">
            {(bundle?.manifest.gestures ?? []).map((name, index) => (
              <button
                key={name}
                type="button"
                role="radio"
                aria-checked={index === intended}
                className="gesture-choice"
                data-selected={index === intended || undefined}
                onClick={() => chooseGesture(index)}
              >
                {GESTURE_LABEL[name] ?? name}
              </button>
            ))}
          </div>
        </div>

        <div className="panel">
          <ActivationRing
            channelRms={decision?.channelRms ?? new Array(nChannels).fill(0)}
            fullScaleVolts={FULL_SCALE_VOLTS}
            failedChannel={status.rejected?.channel ?? null}
          />
        </div>

      </aside>

      <div className="live-transport">
        <button
          type="button"
          className="transport"
          onClick={toggleRunning}
          disabled={!status.ready}
        >
          <Icon name={running ? 'pause' : 'play'} />
          {running ? 'Pause' : 'Start decoding'}
        </button>
        <button
          type="button"
          className="transport transport-quiet"
          onClick={() => workerRef.current?.postMessage({ type: 'reset' } satisfies WorkerRequest)}
          disabled={!status.ready}
        >
          <Icon name="reset" />
          Release
        </button>
      </div>
    </div>
  );
}
