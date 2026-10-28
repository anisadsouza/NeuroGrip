/**
 * The browser latency bench.
 *
 * CLAUDE.md's rule is that every number in the docs traces to an artifact
 * written by a runnable command. The in-browser P95 was the one figure that
 * did not: it was read off the Live screen by hand. This harness produces it.
 *
 * Three design decisions worth stating:
 *
 * It is a second Vite entry rather than a hook in the application. A
 * production build compiles the worker to a hashed asset, so a script injected
 * into the page cannot construct it; and a test hook in Live.tsx would be
 * shipped code that exists only for a test.
 *
 * It drives the worker in lockstep -- push exactly one hop, wait for exactly
 * one decision -- rather than through the replay source. Wall-clock pacing
 * would take twenty seconds of real time to reach a thousand windows and would
 * measure the scheduler as much as the decoder.
 *
 * It reports latency only, and deliberately not TTUM. Measuring the hop at
 * which motion begins would need a clean trial boundary, and this harness has
 * none: `reset` clears the accumulator but not the ring, so for the first ten
 * hops after a switch the window still holds the previous gesture. The replay
 * signal is also in-sample. TTUM is reported from the corpus run instead,
 * where the posteriors are out-of-fold and the trials are real trials.
 */

import { risksFor } from '../decode/commitCost.js';
import { loadReplayBundle } from '../sources/replaySource.js';
import type { DecisionResponse, WorkerRequest, WorkerResponse } from '../worker/protocol.js';

/** Windows to measure. The spec asks for a thousand. */
const WINDOWS = 1000;

/** Hops per trial, matching one repetition in the training corpus. */
const HOPS_PER_TRIAL = 41;

export interface BenchResult {
  readonly windowsProcessed: number;
  readonly rejectedWindows: number;
  readonly n: number;
  readonly threaded: boolean;
  readonly backend: string;
  readonly meanMs: number;
  readonly p50Ms: number;
  readonly p95Ms: number;
  readonly p99Ms: number;
  readonly maxMs: number;
  readonly featureP95Ms: number;
  readonly inferenceP95Ms: number;
  readonly hopMs: number;
  readonly note: string;
}

/** Nearest rank, matching the estimator the Live screen reports. */
function percentile(samples: readonly number[], fraction: number): number {
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))]!;
}

async function main(): Promise<void> {
  const status = document.getElementById('bench-status')!;
  const recorded = new URLSearchParams(location.search).get('source') === 'ninapro';
  const replayUrl = recorded
    ? `${import.meta.env.BASE_URL}replay/ninapro`
    : `${import.meta.env.BASE_URL}replay`;
  const modelUrl = recorded
    ? `${import.meta.env.BASE_URL}models/ninapro/decoder.onnx`
    : `${import.meta.env.BASE_URL}models/decoder.onnx`;
  const bundle = await loadReplayBundle(replayUrl);
  const { nChannels, samplingRateHz, samplesPerGesture, gestures } = bundle.manifest;

  const worker = new Worker(new URL('../worker/inference.worker.ts', import.meta.url), {
    type: 'module',
  });

  /** Resolve on the next message of the given type, or reject on an error. */
  const next = <T extends WorkerResponse['type']>(type: T) =>
    new Promise<Extract<WorkerResponse, { type: T }>>((resolve, reject) => {
      const listener = (event: MessageEvent<WorkerResponse>) => {
        const message = event.data;
        if (message.type === 'error') {
          worker.removeEventListener('message', listener);
          reject(new Error(message.message));
        } else if (message.type === type) {
          worker.removeEventListener('message', listener);
          resolve(message as Extract<WorkerResponse, { type: T }>);
        }
      };
      worker.addEventListener('message', listener);
    });

  const nextDecision = () =>
    new Promise<DecisionResponse | null>((resolve, reject) => {
      const listener = (event: MessageEvent<WorkerResponse>) => {
        const message = event.data;
        if (message.type === 'error') {
          worker.removeEventListener('message', listener);
          reject(new Error(message.message));
        } else if (message.type === 'decision' || message.type === 'rejected') {
          worker.removeEventListener('message', listener);
          resolve(message.type === 'decision' ? message : null);
        }
      };
      worker.addEventListener('message', listener);
    });

  const readyPromise = next('ready');
  worker.postMessage({
    type: 'init',
    modelUrl,
    nChannels,
    samplingRateHz,
    risks: risksFor(gestures),
    ...(bundle.manifest.signalQuality
      ? {
          minChannelRmsByChannel:
            bundle.manifest.signalQuality.minChannelRmsByChannel,
        }
      : {}),
  } satisfies WorkerRequest);
  const ready = await readyPromise;

  const windowSamples = Math.round((200 / 1000) * samplingRateHz);
  const hopSamples = Math.round((20 / 1000) * samplingRateHz);

  /** A channel-major chunk of `count` samples from `gesture`, from `cursor`. */
  function chunkFrom(gesture: number, cursor: number, count: number): Float32Array {
    const segment = bundle.segments[gesture]!;
    const chunk = new Float32Array(nChannels * count);
    for (let c = 0; c < nChannels; c++) {
      for (let i = 0; i < count; i++) {
        const t = (cursor + i) % samplesPerGesture;
        chunk[c * count + i] = segment[c * samplesPerGesture + t]!;
      }
    }
    return chunk;
  }

  /** Send samples without expecting a decision. Used only to fill the ring. */
  function pushOnly(gesture: number, cursor: number, count: number): void {
    const chunk = chunkFrom(gesture, cursor, count);
    worker.postMessage({ type: 'samples', samples: chunk, nChannels } satisfies WorkerRequest, [
      chunk.buffer,
    ]);
  }

  /** Push exactly one hop and wait for the single decision it produces. */
  function push(gesture: number, cursor: number, count: number): Promise<DecisionResponse | null> {
    const decision = nextDecision();
    pushOnly(gesture, cursor, count);
    return decision;
  }

  const latencies: number[] = [];
  const featureMs: number[] = [];
  const inferenceMs: number[] = [];
  let rejectedWindows = 0;

  let gesture = 0;
  let cursor = 0;
  let hopInTrial = 0;

  // Prime the ring one hop short of a full window, and do not wait for a
  // decision: the worker only decodes once the ring holds a whole window, so
  // this produces none. A full window here would produce ten decisions at once
  // and the lockstep this bench depends on would be broken from the first
  // measurement.
  pushOnly(gesture, cursor, windowSamples - hopSamples);
  cursor += windowSamples - hopSamples;

  for (let i = 0; i < WINDOWS; i++) {
    const decision = await push(gesture, cursor, hopSamples);
    cursor = (cursor + hopSamples) % samplesPerGesture;

    if (decision) {
      latencies.push(decision.latencyMs);
      featureMs.push(decision.featureMs);
      inferenceMs.push(decision.inferenceMs);
    } else {
      rejectedWindows++;
    }

    hopInTrial++;
    if (hopInTrial >= HOPS_PER_TRIAL) {
      // Cycle the gesture so the run exercises the whole vocabulary and the
      // accumulator's reset path, rather than measuring one class for a
      // thousand windows. The ring keeps its samples, so the next few windows
      // straddle the transition -- which is what a real switch looks like.
      hopInTrial = 0;
      gesture = (gesture + 1) % gestures.length;
      cursor = 0;
      worker.postMessage({ type: 'reset' } satisfies WorkerRequest);
    }

    if (i % 100 === 0) status.textContent = `${i} / ${WINDOWS}`;
  }

  if (latencies.length === 0) {
    throw new Error(`all ${WINDOWS} windows were rejected by signal-quality checks`);
  }

  const result: BenchResult = {
    windowsProcessed: WINDOWS,
    rejectedWindows,
    n: latencies.length,
    threaded: ready.threaded,
    backend: ready.backend,
    meanMs: latencies.reduce((a, b) => a + b, 0) / latencies.length,
    p50Ms: percentile(latencies, 0.5),
    p95Ms: percentile(latencies, 0.95),
    p99Ms: percentile(latencies, 0.99),
    maxMs: Math.max(...latencies),
    featureP95Ms: percentile(featureMs, 0.95),
    inferenceP95Ms: percentile(inferenceMs, 0.95),
    hopMs: 20,
    note:
      `${recorded ? `Recorded NinaPro DB2 subject ${bundle.manifest.subject ?? 'unknown'}` : 'Simulated signal'}. ` +
      (recorded
        ? "Recorded channel floors are calibrated from this subject's rest segment. "
        : '') +
      'Latency covers feature extraction plus ONNX inference in the worker, ' +
      'measured one window at a time. Windows are pushed in lockstep rather ' +
      'than in real time, so this measures the decode path and not the scheduler. ' +
      'Latency excludes the postMessage round trip and the UI thread: it is ' +
      'what the worker spent, which is the part under the ten-millisecond budget. ' +
      'Signal-quality rejects are counted and excluded from latency percentiles.',
  };

  status.textContent = 'done';
  document.getElementById('bench-result')!.textContent = JSON.stringify(result, null, 2);
  (window as unknown as { __benchResult: BenchResult }).__benchResult = result;
}

main().catch((error: unknown) => {
  document.getElementById('bench-status')!.textContent = 'failed';
  (window as unknown as { __benchError: string }).__benchError =
    error instanceof Error ? error.message : String(error);
});
