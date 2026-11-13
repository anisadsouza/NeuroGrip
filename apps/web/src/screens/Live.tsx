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
import { Screen } from '../components/Screen.js';
import { Term } from '../components/Term.js';
import { labelsFor, risksFor } from '../decode/commitCost.js';
import { decisionOutputFor } from '../decode/decisionOutput.js';
import { percentile95, trimRing } from '../decode/latency.js';
import { projectDrive } from '../decode/muscleDrive.js';
import { virtualCommandFor } from '../decode/virtualCommand.js';
import { ReplaySource, loadReplayBundle, replaySourceLabel, type ReplayBundle } from '../sources/replaySource.js';
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

type ReplayMode = 'simulated' | 'ninapro';

export function Live() {
  const workerRef = useRef<Worker | null>(null);
  const sourceRef = useRef<ReplaySource | null>(null);

  const [bundle, setBundle] = useState<ReplayBundle | null>(null);
  const [gestures, setGestures] = useState<readonly string[]>([]);
  const [replayMode, setReplayMode] = useState<ReplayMode>('simulated');
  // Whether a recorded replay is installed in this build. Checked once, so the
  // option can say so up front instead of failing after it is chosen.
  const [recordedAvailable, setRecordedAvailable] = useState<boolean | null>(null);
  const [fallbackNotice, setFallbackNotice] = useState<string | null>(null);
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
  const restCalibration = bundle?.manifest.signalQuality?.restCalibration;
  // The worker's class list is authoritative once it has loaded -- it is the
  // order the decoder actually emits. Until then the manifest's list stands in,
  // so the wearer can pick a gesture while the model is still loading. The
  // compatibility gate guarantees the two agree.
  const gestureLabels = useMemo(
    () => {
      const names = gestures.length > 0 ? gestures : (bundle?.manifest.gestures ?? []);
      return labelsFor(names).map((label, index) => {
        const name = names[index];
        return name ? (bundle?.manifest.gestureLabels?.[name] ?? label) : label;
      });
    },
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
    const controller = new AbortController();
    void fetch(`${import.meta.env.BASE_URL}replay/ninapro/emg-replay.json`, {
      signal: controller.signal,
    })
      .then((response) =>
        setRecordedAvailable(
          response.ok && !(response.headers.get('content-type') ?? '').includes('text/html'),
        ),
      )
      .catch(() => setRecordedAvailable(false));
    return () => controller.abort();
  }, []);

  // A recorded replay that cannot be opened is not a reason to lose the whole
  // screen: fall back to the simulated signal and say what happened.
  const fallBackToSimulated = useCallback(() => {
    setFallbackNotice(
      'The recorded replay could not be opened in this build, so the simulated signal is playing instead.',
    );
    setRecordedAvailable(false);
    setReplayMode('simulated');
  }, []);

  useEffect(() => {
    let disposed = false;
    setStatus({ ready: false, threaded: false, error: null, rejected: null });
    setBundle(null);
    setGestures([]);
    setDecision(null);
    setLatency([]);
    setRunning(false);
    sourceRef.current?.stop();
    sourceRef.current = null;

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
          if (replayMode === 'ninapro') fallBackToSimulated();
          else setStatus((s) => ({ ...s, error: message.message }));
          break;
      }
    };

    const replayBase = replayMode === 'ninapro'
      ? `${import.meta.env.BASE_URL}replay/ninapro`
      : `${import.meta.env.BASE_URL}replay`;
    const modelUrl = replayMode === 'ninapro'
      ? `${import.meta.env.BASE_URL}models/ninapro/decoder.onnx`
      : `${import.meta.env.BASE_URL}models/decoder.onnx`;

    loadReplayBundle(replayBase)
      .then((loaded) => {
        if (disposed) return;
        setBundle(loaded);
        const request: WorkerRequest = {
          type: 'init',
          modelUrl,
          nChannels: loaded.manifest.nChannels,
          samplingRateHz: loaded.manifest.samplingRateHz,
          risks: risksFor(loaded.manifest.gestures),
          ...(loaded.manifest.signalQuality
            ? {
                minChannelRmsByChannel:
                  loaded.manifest.signalQuality.minChannelRmsByChannel,
              }
            : {}),
        };
        worker.postMessage(request);
      })
      .catch((error: unknown) => {
        if (!disposed) {
          if (replayMode === 'ninapro') fallBackToSimulated();
          else setStatus((s) => ({ ...s, error: String(error) }));
        }
      });

    return () => {
      disposed = true;
      sourceRef.current?.stop();
      worker.terminate();
    };
  }, [replayMode, fallBackToSimulated]);

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
  const output = decision
    ? decisionOutputFor({
        gestures,
        leader: decision.leader,
        latched: decision.latched,
        latchedClass: decision.latchedClass,
        timedOut: decision.timedOut,
        commitment: decision.commitment,
      })
    : { index: -1, gesture: '', handCommitment: 0 };
  const outputGesture = output.gesture;
  const outputState = decision?.timedOut
    ? 'fallback'
    : decision?.latched
      ? 'latched'
      : decision?.moving
        ? 'reversible'
        : decision
          ? 'candidate'
          : '';
  const virtualCommand = outputGesture ? virtualCommandFor(outputGesture) : null;
  const modelProbability = decision?.posteriors[decision.leader];

  const transport = (
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
  );

  const errorStatus = status.error ? (
    <div className="panel notice notice-stop">
      <Icon name="alert" label="Error" />
      <div>
        <h3>The live decoder could not start</h3>
        <p>
          This build is missing the decoder or its sample signal. Every other
          screen still works; reload the page to try again.
        </p>
        <p className="small ng-num">Detail: {status.error}</p>
      </div>
    </div>
  ) : undefined;

  const decodedLabel = decision?.latched
    ? decision.timedOut
      ? 'No clear intent'
      : (gestureLabels[decision.latchedClass ?? 0] ?? '—')
    : '—';

  const ring = (
    <ActivationRing
      channelRms={decision?.channelRms ?? new Array(nChannels).fill(0)}
      fullScaleVolts={FULL_SCALE_VOLTS}
      failedChannel={status.rejected?.channel ?? null}
    />
  );

  const body = (
    <div className="live">
      <aside className="live-controls" aria-label="Decoder controls">
        <section className="live-step" aria-labelledby="live-step-gesture">
          <h3 id="live-step-gesture">
            <span className="step-index ng-num" aria-hidden="true">1</span>
            Gesture
          </h3>
          <GestureChoices
            labels={gestureLabels}
            selected={intended}
            onSelect={chooseGesture}
            legend="Intended gesture"
          />
        </section>

        <section className="live-step replay-source" aria-labelledby="live-step-source">
          <h3 id="live-step-source">
            <span className="step-index ng-num" aria-hidden="true">2</span>
            <label htmlFor="replay-source">Signal</label>
          </h3>
          <select
            id="replay-source"
            value={replayMode}
            onChange={(event) => {
              const { value } = event.currentTarget;
              if (value === 'simulated' || value === 'ninapro') setReplayMode(value);
            }}
            disabled={running}
          >
            <option value="simulated">Simulated forearm</option>
            <option value="ninapro" disabled={recordedAvailable === false}>
              {recordedAvailable === false
                ? 'Recorded person (not in this build)'
                : 'Recorded person (NinaPro DB2)'}
            </option>
          </select>
          {fallbackNotice ? (
            <p className="notice small" role="status">{fallbackNotice}</p>
          ) : null}
          <p className="small muted" role="status">
            {bundle ? replaySourceLabel(bundle.manifest) : 'Loading replay bundle…'}
          </p>
          {replayMode === 'ninapro' && restCalibration?.flaggedChannels.length ? (
            <p className="notice small" role="status">
              Rest screen flagged channel
              {restCalibration.flaggedChannels.length === 1 ? '' : 's'}{' '}
              {restCalibration.flaggedChannels.join(', ')} above{' '}
              {(restCalibration.maxChannelRms * 1e6).toFixed(0)} µV RMS. Replay
              continues; this is not a validated fault diagnosis.
            </p>
          ) : null}
        </section>

        <section className="live-step live-run" aria-labelledby="live-step-run">
          <h3 id="live-step-run">
            <span className="step-index ng-num" aria-hidden="true">3</span>
            Run it
          </h3>
          {transport}
        </section>
      </aside>

      <div className="live-center">
        {/* The hand is shown from the first frame, at rest, so the viewer
            sees what it looks like before anything is decoded. */}
        <VirtualHand
          gesture={outputGesture || 'rest'}
          label={output.index >= 0 ? (gestureLabels[output.index] ?? 'this gesture') : 'Rest'}
          commitment={output.handCommitment}
          latched={decision?.latched ?? false}
          reversible={decision?.reversible ?? false}
          timedOut={decision?.timedOut ?? false}
          drive={drive}
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
          <div className="live-idle" aria-busy={!status.ready}>
            <p className="live-idle-title">
              {status.ready ? 'Ready. Press Start decoding.' : 'Loading the decoder…'}
            </p>
          </div>
        )}
      </div>

      <div className="live-output">
        <Oscilloscope
          display={decision?.display ?? null}
          nChannels={nChannels}
          windowMs={200}
        />

        <div className="live-ring">{ring}</div>

        <section className="live-decoder" aria-labelledby="live-decoder-heading">
          <h3 id="live-decoder-heading" className="visually-hidden">Decoder</h3>
          <dl className="readout-list">
            <div className="readout-primary">
              <dt>Decided gesture</dt>
              <dd data-correct={decision?.latched ? correct : undefined}>{decodedLabel}</dd>
            </div>
            <div>
              <dt>Posterior</dt>
              <dd className="ng-num">
                {typeof modelProbability === 'number'
                  ? `${(modelProbability * 100).toFixed(1)}%`
                  : '—'}
              </dd>
            </div>
            <div>
              <dt>Hand action</dt>
              <dd>
                {decision
                  ? `${virtualCommand ?? 'Unmapped class'}${outputState ? ` · ${outputState}` : ''}`
                  : '—'}
              </dd>
            </div>
            <div>
              <dt><Term id="latency">Latency</Term></dt>
              <dd className="ng-num">
                {decision ? `${decision.latencyMs.toFixed(1)} ms` : '—'}
                {p95 ? ` · P95 ${p95.toFixed(1)}` : ''}
              </dd>
            </div>
          </dl>
          <p className="small muted">
            {status.threaded ? 'WASM threads on' : 'Single-threaded WASM'} · no motor attached
          </p>
        </section>
      </div>
    </div>
  );

  return (
    <Screen
      title="Live decoding"
      className="screen-live"
      page={0}
      onPage={() => undefined}
      status={errorStatus}
      pages={[
        {
          id: 'live',
          label: 'Live',
          summary:
            'Pick a gesture, press Start, watch the hand.',
          content: body,
        },
      ]}
    />
  );
}
