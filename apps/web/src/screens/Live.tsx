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
import { VirtualHand, type MuscleDrive } from '../components/VirtualHand.js';
import { GestureChoices } from '../components/GestureChoices.js';
import { Icon } from '../components/Icon.js';
import { GESTURE_LABEL, labelsFor, risksFor } from '../decode/commitCost.js';
import { percentile95, trimRing } from '../decode/latency.js';
import { projectDrive } from '../decode/muscleDrive.js';
import { ReplaySource, loadReplayBundle, type ReplayBundle } from '../sources/replaySource.js';
import type { DecisionResponse, WorkerRequest, WorkerResponse } from '../worker/protocol.js';



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

  const risks = useMemo(() => risksFor(gestures), [gestures]);
  // The worker's class list is authoritative once it has loaded -- it is the
  // order the decoder actually emits. Until then the manifest's list stands in,
  // so the wearer can pick a gesture while the model is still loading. The
  // compatibility gate guarantees the two agree.
  const gestureLabels = useMemo(
    () => labelsFor(gestures.length > 0 ? gestures : (bundle?.manifest.gestures ?? [])),
    [gestures, bundle],
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
  const drive = useMemo<MuscleDrive>(
    () => projectDrive(bundle?.manifest.muscleGroups, decision?.channelRms),
    [bundle, decision],
  );

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
          setLatency((previous) => trimRing(previous, message.latencyMs));
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
          risks: risksFor(loaded.manifest.gestures),
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

  const p95 = useMemo(() => percentile95(latency), [latency]);

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
              <p className="panel muted" aria-busy={!status.ready}>
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
          <GestureChoices
            labels={gestureLabels}
            selected={intended}
            onSelect={chooseGesture}
            legend="Intended gesture"
          />
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
